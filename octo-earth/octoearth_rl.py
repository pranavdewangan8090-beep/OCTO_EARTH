#!/usr/bin/env python3
"""
OCTO-EARTH prototype: an RL agent ejects asteroid reaction mass to move Earth from ~1 AU
toward a stable, near-circular orbit at 2.7 AU within a mission time limit (default 10,000 y).

ALL Python logic lives in this file:
  physics constants and configuration, analytic feasibility gate, initial conditions,
  FAST training model (exact Kepler propagation of the Earth-Moon centre around the Sun AND of the
  Moon around Earth, impulsive momentum-conserving ejections that push Earth),
  HIGH-FIDELITY model (N-body Sun/Earth/Moon + explicit fragments, Yoshida-4 integrator,
  impact / escape / merge events with energy, momentum and angular-momentum ledgers),
  Gymnasium environment, reward, baseline policies, PPO training (Stable-Baselines3),
  best-model selection, evaluation and comparison, explanation and report generation,
  physics self-checks, and a small HTTP API that serves the Three.js viewer.

Run:   python octoearth_rl.py               train + compare + viewer (browser opens)
       python octoearth_rl.py --smoke       very short pipeline test
       python octoearth_rl.py --check       physics checks only
       python octoearth_rl.py --headless    no server/browser; prints the report
       python octoearth_rl.py --preset realistic    (see PRESETS)
On Windows simply double-click RUN_OCTOEARTH.bat.

HOW TO READ THIS FILE
---------------------
The file reads top to bottom like the story of one mission. Every function starts with a one-line
sentence saying what it does; numbers carry their unit in the name or in a comment (SI unless stated).

  1. CONSTANTS        Physical constants (G, masses, AU, year, sunlight at 1 AU, ...).
  2. CONFIGURATION    The Config dataclass: every knob a user may turn, with its default.
                      (The plain-language description of every knob is in APPENDIX A at the end.)
  3. FEASIBILITY      Before simulating anything: is the move possible at all with this energy and mass?
  3a. SOLAR POWER     Where the launch energy comes from: collectors near the Sun -> microwaves -> the arms.
  3b. CLIMATE         Two-box climate model. Far from the Sun Earth would freeze (about 152 K at 2.7 AU), so a
                      ring of thin SPACE MIRRORS (enabler E3) reflects extra sunlight onto Earth. The agent
                      chooses how much (action a4). life_indicators() turns the temperature into an ice line
                      and a growing area, which the viewer's Weather & life view shows.
  3c. COLLECTOR BELT  Do the collectors stay on their orbits? (12 are simulated with and without station keeping.)
  4. INITIAL STATE    Where the Sun, Earth, Moon and Mars start.
  5. NUMBA KERNELS    The fast inner loops: Kepler motion, N-body steps, ejections that push Earth.
  6. ENGINES          Two physics models with the same interface: FastEngine (used for training) and
                      HifiEngine (N-body with every fragment, used to validate).
  7. ENVIRONMENT      OctoEarthEnv: what the agent sees (31 numbers), what it decides (5 numbers), the reward,
                      and when a mission ends. Physics metrics are kept separate from the reward.
  8. BASELINES        "none" (do nothing) and a hand-designed heuristic to compare the agent with.
  9. PHYSICS CHECKS   Self-tests run before training (conservation laws, convergence, Moon, Mars, energy ledger ...).
  10. APPLICATION     Training (PPO), best-model selection, comparison, playback for the viewer, the report.
  11. HTTP API        Small web server for the Three.js viewer (Python is the only physics source).
  APPENDIX A          Documentation tables: parameters, outputs, glossary, assumptions, enablers, problems.
  12. MAIN            Command-line entry point.

Where to start: run_episode() runs one mission; OctoEarthEnv.step() is one decision; FastEngine._run_raw()
is the physics of one decision interval; OctoEarthEnv._climate_step() keeps Earth habitable with the mirrors.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass, fields, replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _dependency_guard():
    """Stop with a friendly message if a required Python package is missing."""
    import importlib.util
    need = ["numpy", "numba", "gymnasium", "stable_baselines3", "torch"]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        print("\n[OCTO-EARTH] Missing Python packages: " + ", ".join(missing) +
              "\nDouble-click RUN_OCTOEARTH.bat (it installs them into .venv), or run:\n"
              "    python -m pip install -r requirements.txt\n")
        sys.exit(2)


if __name__ == "__main__":
    _dependency_guard()

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
OUTPUTS = ROOT / "outputs"
APP_NAME = "OCTO-EARTH prototype"

# =============================================================================================
# 1. CONSTANTS (SI)
# =============================================================================================
G = 6.67430e-11
AU = 1.495978707e11
YEAR = 365.25 * 86400.0                 # Julian year [s]
GM_SUN = 1.32712440041e20
GM_EARTH = 3.986004418e14
GM_MOON = 4.9028e12
GM_MARS = 4.282837e13                   # Mars (+ its moons) [m^3/s^2]
M_SUN = GM_SUN / G
M_EARTH = GM_EARTH / G
M_MOON = GM_MOON / G
M_MARS = GM_MARS / G
R_SUN = 6.957e8
R_EARTH = 6.371e6
R_MOON = 1.7374e6
R_MARS = 3.3895e6
# Other planets (update p35): Mercury, Venus and Jupiter. Planets beyond Jupiter (Saturn, Uranus, Neptune) and the
# dwarf planets / asteroids are NOT modelled.
GM_MERCURY = 2.2031868551e13            # [m^3/s^2]
GM_VENUS = 3.24858592e14
GM_JUPITER = 1.26712764e17              # Jupiter + its moons
R_MERCURY, R_VENUS, R_JUPITER = 2.4397e6, 6.0518e6, 6.9911e7
NMAJ = 7                                # high-fidelity major bodies: 0 Sun, 1 Earth(+asteroid), 2 Moon, 3 Mars,
                                        # 4 Mercury, 5 Venus, 6 Jupiter; fragments from index 7
# Mars J2000 mean elements (ecliptic): a [AU], e, i, longitude of node, longitude of perihelion, mean longitude [deg]
MARS_EL = (1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959, -4.55343205)
# Same for the other planets (JPL "approximate positions of the planets", J2000 mean elements, ecliptic)
PLANET_NAMES = ("Mercury", "Venus", "Jupiter")
PLANET_EL = np.array([(0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350),
                      (0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950),
                      (5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644051)])
PLANET_GM = np.array([GM_MERCURY, GM_VENUS, GM_JUPITER])
PLANET_R = np.array([R_MERCURY, R_VENUS, R_JUPITER])
N_PL = 3
PL_SUBSTEP_S = 12.0 * 86400.0           # fast model: longest kick step while the planets pull (resolves Mercury/Venus)
MOON_A = 3.844e8
# Launcher geometry (update p36; enabler E1).  The asteroid sits on the AXIS of the launch ring, so every one of the
# 8 arm roots is the same distance from it.  Lengths in Earth radii; the same numbers are used by the viewer
# (frontend/launcher.js; a test compares them).
ARM_LENGTH_RE = 2.0          # length of each arm = 2 Earth radii (12,742 km)
RING_RADIUS_RE = 1.25        # radius of the ring that carries the 8 arm roots (0.25 R_E = 1,593 km above the surface)
GRAB_HEIGHT_RE = 1.25        # grab point on the ring axis (the asteroid's Earth-facing side rests on it)
AST_DENSITY = 2000.0         # assumed asteroid density [kg/m^3] (sizes for the geometry and the drawing only)
L_SUN = 3.828e26                        # W
S_1AU = 1361.0                          # W/m^2 solar constant
C_LIGHT = 299792458.0
EARTH_INTERCEPT_W = S_1AU * math.pi * R_EARTH ** 2   # sunlight intercepted by Earth [W]

# Ledger layout (float64 array). Every discrete event (ejection, impact, escape, merge) is
# measured as (state after - state before) so that conservation residuals isolate the
# numerical integration error.
LG_MASS_EJECTED, LG_N_PACKETS, LG_MASS_FALLBACK, LG_LAUNCH_KE, LG_E_DRAWN = 0, 1, 2, 3, 4
LG_DE_EJECT, LG_DE_EJECT_THEORY = 5, 6
LG_ESC_MASS, LG_ESC_PX, LG_ESC_PY, LG_ESC_PZ, LG_ESC_E = 7, 8, 9, 10, 11
LG_EARTH_IMPACT_M, LG_MOON_IMPACT_M, LG_SUN_IMPACT_M, LG_DE_IMPACT = 12, 13, 14, 15
LG_N_MERGES, LG_DE_MERGE = 16, 17
LG_PKT_BOUND_SUN_M, LG_PKT_UNBOUND_SUN_M = 18, 19
LG_PKT_PX, LG_PKT_PY, LG_PKT_PZ = 20, 21, 22
LG_MAX_IMPULSE, LG_N_LAUNCH_MOON_HITS, LG_N_FALLBACK = 23, 24, 25
LG_IMPULSE_TOTAL = 26                    # sum of momentum delivered to the Earth system [N s]
LG_MARS_IMPACT_M = 27                    # fragment mass that hit Mars (high-fidelity)
LG_SHIELD_OVERRIDES = 28                 # ejection events whose packet size the Mars safety shield changed
LG_EV_DE = 29                            # sum of all event energy changes
LG_EV_DP = 30                            # 30..32 sum of all event momentum changes
LG_EV_DL = 33                            # 33..35 sum of all event angular-momentum changes
LG_PLANET_IMPACT_M = 36                  # fragment mass that hit Mercury, Venus or Jupiter (high-fidelity)
LEDGER_N = 40

# Viewer sample = the physics state at ONE ejection instant (one row per ejection event).
# Between two consecutive rows the motion is purely ballistic (no event), so the viewer can
# interpolate with a cubic Hermite curve using v_post of the earlier row and v_pre of the later row.
S_T = 0              # simulated time [yr]
S_R = 1              # 1..3  Earth (fast model: Earth-Moon barycentre) heliocentric position [AU]
S_VPRE = 4           # 4..6  heliocentric velocity just BEFORE the ejection [km/s]
S_VPOST = 7          # 7..9  heliocentric velocity just AFTER the ejection (recoil applied) [km/s]
S_DM = 10            # packet mass [kg] (0 = no ejection at this event)
S_U = 11             # launch speed relative to Earth [m/s]
S_VREL = 12          # packet speed relative to Earth after climbing out of Earth's gravity [m/s]
S_VP = 13            # 13..15 packet heliocentric velocity at hand-over [km/s]
S_FATE = 16          # 0 none, 1 leaves the Solar System, 2 stays bound to the Sun, 3 fell back, 4 hit the Moon
S_MAST = 17          # asteroid mass remaining after the event [kg]
S_EREM = 18          # propulsion energy remaining after the event [J]
S_NFRAG = 19         # active explicit N-body fragments (high-fidelity only)
S_DV = 20            # recoil speed change of Earth from this packet [m/s]
S_EDRAW = 21         # energy drawn for this packet [J]
S_EHAT = 22          # 22..24 packet direction (unit vector, inertial frame)
S_MARS = 25          # 25..27 Mars heliocentric position [AU]
S_MARSV = 28         # 28..30 Mars heliocentric velocity [km/s]
S_DMARS = 31         # Earth-Mars distance at this event [AU]
S_SHIELD = 32        # Mars safety shield mode at this event: 0 agent's command, 1 coast, 2 full thrust, 3 full reverse (brake)
S_PL = 33            # 33..41 Mercury, Venus, Jupiter heliocentric positions [AU] (3 each)
S_PLV = 42           # 42..50 their heliocentric velocities [km/s]
S_DPL = 51           # 51..53 Earth-Mercury, Earth-Venus, Earth-Jupiter distance [AU]
N_SAMPLE_COLS = 54
MOON_SUB = 8         # high-fidelity: Moon state recorded 8x per ejection event (t, rel pos m, rel vel m/s)
# times after release at which packet positions are computed for the viewer [days]
PACKET_OFFSETS_DAYS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96,
                                128, 160, 192, 224, 256, 288, 320, 365.25])
FATE_NAMES = {0: "no ejection", 1: "leaves the Solar System", 2: "stays bound to the Sun",
              3: "fell back to Earth (too slow)", 4: "hit the Moon"}


# =============================================================================================
# 2. CONFIGURATION (all user-tweakable parameters)
# =============================================================================================
@dataclass
class Config:
    preset: str = "hypothetical"
    # --- resource and launcher -------------------------------------------------------------
    asteroid_mass_kg: float = 3.0e22          # reaction-mass reservoir docked at Earth
    max_packet_mass_kg: float = 1.0e18        # per ejection event
    max_ejection_speed_ms: float = 5.0e6      # launch speed relative to Earth (non-relativistic)
    min_ejection_speed_ms: float = 2.0e4      # below Earth escape speed (11.2 km/s) packets fall back
    energy_budget_j: float = 3.0e35           # total launch energy available
    launch_efficiency: float = 0.5            # kinetic energy delivered / energy drawn
    ejections_per_year: float = 8.0           # ejection events per simulated year
    # --- mission ----------------------------------------------------------------------------
    target_au: float = 2.7
    mission_years: float = 10000.0
    decision_interval_years: float = 100.0    # one RL decision per interval
    tol_a_au: float = 0.05
    tol_e: float = 0.05
    tol_inc_deg: float = 5.0
    validate_years: float = 200.0             # ballistic check after the mission (high-fidelity)
    randomize_phase: bool = True
    # --- Mars and planetary safety -------------------------------------------------------------
    mars_gravity: bool = True                 # False: Mars is massless (it cannot pull on Earth / Moon)
    mars_max_tide_ratio: float = 0.01         # Mars's tide on Earth <= this x the Moon's tide
    mars_max_moon_perturbation: float = 1e-3  # Mars's tidal pull on the Moon's orbit <= this x Earth's pull on the Moon
    mars_min_distance_km: float = 2.0e6       # hard minimum Earth-Mars distance
    mars_shield: bool = True                  # automatic safety shield (forecast + coast / full-thrust override)
    shield_caution_factor: float = 2.0        # shield acts when the forecast closest approach < factor x safe distance
    shield_horizon_years: float = 10.0        # forecast length of the shield
    other_planets: bool = True                # Mercury, Venus, Jupiter pull on Earth, Moon, Mars and each other (p35)
    # --- solar collectors + microwave power beaming (energy supply) ------------------------------
    solar_collectors: bool = True             # collectors between Sun and Earth power the launcher
    collector_fraction: float = 0.30          # fraction of the Sun's total output the collectors intercept
    collector_orbit_au: float = 0.5           # distance of the collector belt from the Sun (inside Earth's orbit)
    collector_count: float = 5.0e6            # number of individual collectors (sets the size of each one)
    collector_areal_density: float = 0.1      # kg per m^2 of collector (material estimate only)
    pv_efficiency: float = 0.25               # sunlight -> electricity on the collectors
    microwave_efficiency: float = 0.80        # electricity -> microwave beam on the collectors
    microwave_freq_ghz: float = 5.8           # beam frequency (wavelength 5.2 cm)
    transmitter_diameter_km: float = 30.0     # phased-array antenna of each beam on the collector side
    rectenna_diameter_km: float = 1000.0      # receiving antenna (rectenna) on each of the 8 arm tips
    rectenna_efficiency: float = 0.85         # microwaves -> electricity at the arms
    # --- exact energy mapping of the collector panels -------------------------------------------
    pv_temp_coeff: float = 0.002              # relative PV efficiency loss per K above 25 C (pv_efficiency is rated at 25 C)
    collector_absorptance: float = 0.90       # fraction of sunlight the panels absorb (rest reflected)
    collector_emissivity: float = 0.85        # thermal emissivity of each face (both faces radiate)
    pv_degradation_per_year: float = 0.002    # relative efficiency loss per year of panel age
    collector_lifetime_years: float = 50.0    # panels are replaced after this age (steady-state fleet)
    collector_recycle_frac: float = 0.95      # share of replaced panel mass that is recycled
    collector_initial_built: float = 0.3      # share of the belt already built at year 0
    collector_build_years: float = 400.0      # years to finish building the belt (linear build-up)
    collector_embodied_energy_jkg: float = 1.0e8   # energy to make 1 kg of collector (taken from the collectors' own output)
    earth_lane_clear: bool = True             # keep a clear lane in the belt so Earth (and Mars) see the whole Sun
    collector_sk_authority: float = 0.5       # station keeping: share of the light-pressure push usable for control (sail tilt)
    # --- climate + space mirrors ---------------------------------------------------------------------
    climate_model: bool = True                # two-box energy-balance climate model, part of the RL task
    mirrors: bool = True                      # ring of thin-film space mirrors that reflect extra sunlight onto Earth
    climate_target_K: float = 288.0           # target global-mean surface temperature (today ~288 K)
    climate_tol_K: float = 5.0                # success needs |T_surface - target| <= this
    mirror_area_m2: float = 1.6e15            # total mirror area (12.5 x Earth's cross-section)
    mirror_reflectivity: float = 0.90         # share of the sunlight hitting a mirror that it reflects (aluminium film)
    mirror_orbit_re: float = 20.0             # radius of the mirrors' orbit around Earth, in Earth radii
    mirror_areal_density: float = 0.005       # mirror film mass per area [kg/m^2] (5 g/m^2, solar-sail class)
    arm_heat_to_earth_frac: float = 1e-9      # share of the arms' waste heat that reaches Earth (ENABLER: directional radiators)
    beam_spill_to_earth_frac: float = 1e-9    # share of the microwaves missing the dishes that hits Earth (ENABLER: spill control)
    # --- training ---------------------------------------------------------------------------
    train_episodes: int = 1000
    ppo_n_steps: int = 1000
    ppo_batch_size: int = 200
    ppo_epochs: int = 10
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.995
    eval_every_episodes: int = 50
    seed: int = 1
    # --- reward weights (reward is NOT a physical metric) ------------------------------------
    w_progress: float = 100.0
    w_eccentricity: float = 1.0
    w_inclination: float = 1.0
    w_energy: float = 10.0
    w_mass: float = 5.0
    w_time: float = 0.02
    w_success: float = 100.0
    w_terminal_error: float = 50.0
    w_failure: float = 100.0
    w_mars_safety: float = 20.0
    w_climate: float = 2.0
    # --- high-fidelity model -----------------------------------------------------------------
    hifi_dt_hours: float = 6.0
    max_fragments: int = 48
    release_radius_m: float = 2.0e9           # patched-conic hand-over radius (> Earth Hill radius)
    fragment_escape_au: float = 60.0

    def to_dict(self):
        """Return the configuration as a plain dictionary (for JSON files and the viewer)."""
        return asdict(self)

    @property
    def max_steps(self) -> int:
        """Number of agent decisions that fit into the mission time limit."""
        return int(math.ceil(self.mission_years / self.decision_interval_years - 1e-9))



PRESETS = {
    # Physically HYPOTHETICAL: chosen so that the transfer is just feasible.  Requires ~0.4
    # lunar masses ejected at ~1% of light speed and ~5e23 W average launch power.
    "hypothetical": dict(preset="hypothetical", asteroid_mass_kg=3.0e22, max_packet_mass_kg=1.0e18,
                         max_ejection_speed_ms=5.0e6, energy_budget_j=3.0e35, launch_efficiency=0.5),
    # Generous "realistic-ish": a Ceres-mass asteroid, a 30 km/s electromagnetic launcher, and
    # ALL sunlight intercepted by Earth for 10,000 years as the energy budget.
    "realistic": dict(preset="realistic", asteroid_mass_kg=9.4e20, max_packet_mass_kg=1.0e14,
                      max_ejection_speed_ms=3.0e4, energy_budget_j=EARTH_INTERCEPT_W * 1e4 * YEAR,
                      launch_efficiency=0.5, solar_collectors=False),
}


# ---------------------------------------------------------------------------------------------
# 2b. PARAMETER / OUTPUT DOCUMENTATION -> moved to APPENDIX A at the end of the file (long tables).
# ---------------------------------------------------------------------------------------------


def validate_params(cfg_dict: dict) -> list:
    """Return a list of human-readable problems (empty = valid). Values are never silently changed."""
    errs = []
    for d in PARAM_DOCS:
        k, v = d["key"], cfg_dict.get(d["key"])
        if d["min"] is None or v is None or isinstance(v, bool):
            continue
        if not (d["min"] <= v <= d["max"]):
            errs.append(f"{d['label']} = {v:g} {d['unit']} is outside the allowed range {d['min']:g} .. {d['max']:g}")
    c = cfg_dict
    if c["min_ejection_speed_ms"] >= c["max_ejection_speed_ms"]:
        errs.append("Min ejection speed must be lower than max ejection speed")
    if c["max_packet_mass_kg"] > c["asteroid_mass_kg"]:
        errs.append("Max packet mass cannot exceed the asteroid mass")
    if c["ppo_batch_size"] > c["ppo_n_steps"]:
        errs.append("PPO mini-batch size cannot exceed PPO steps per update")
    steps = c["mission_years"] / c["decision_interval_years"]
    if steps < 2 or steps > 1000:
        errs.append(f"Mission / decision interval = {steps:g} decisions; must be between 2 and 1,000")
    return errs


def make_config(_preset: str = "hypothetical", **overrides) -> Config:
    """Preset values first, then overrides (strings are converted to the field type).  If an
    override changes one of the preset's physical parameters, the preset label becomes 'custom'."""
    name = _preset if _preset in PRESETS else "hypothetical"
    base = replace(Config(), **PRESETS[name])
    valid = {f.name for f in fields(Config)}
    clean = {}
    for k, v in overrides.items():
        if k in valid and k != "preset" and v is not None and v != "":
            cur = getattr(base, k)
            if isinstance(cur, bool):
                clean[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            else:
                clean[k] = type(cur)(float(v)) if isinstance(cur, int) else type(cur)(v)
    unknown = [k for k in overrides if k not in valid]
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    cfg = replace(base, **clean)
    if any(k in PRESETS[name] and getattr(cfg, k) != PRESETS[name][k] for k in clean):
        cfg = replace(cfg, preset="custom")
    elif _preset == "custom" or overrides.get("preset") == "custom":
        cfg = replace(cfg, preset="custom") if any(getattr(cfg, k) != v for k, v in PRESETS[name].items() if k != "preset") else cfg
    return cfg


def checked_config(_preset: str = "hypothetical", **overrides) -> Config:
    """make_config + range / consistency validation.  Invalid values raise ValueError; they are
    never clipped or reinterpreted."""
    cfg = make_config(_preset, **overrides)
    errs = validate_params(cfg.to_dict())
    if errs:
        raise ValueError("; ".join(errs))
    return cfg


def smoke_config(preset="hypothetical") -> Config:
    """A small, fast configuration for the --smoke pipeline test (few episodes, short missions)."""
    return make_config(preset, train_episodes=30, mission_years=4000.0, decision_interval_years=100.0,
                       ppo_n_steps=200, ppo_batch_size=50, eval_every_episodes=10, validate_years=50.0)


# =============================================================================================
# 3. ANALYTIC FEASIBILITY GATE
# =============================================================================================
def launcher_geometry(cfg: "Config" = None, m_ast: float = None) -> dict:
    """Geometry of the launcher (update p36): ring with 8 arm roots, the asteroid on the ring's AXIS.

    Earth's centre is the origin, the ring lies in the plane z = 0 (perpendicular to Earth's orbit normal), the roots
    are at radius RING_RADIUS_RE every 45 degrees and the grab point is on the axis at z = GRAB_HEIGHT_RE.  Because
    the grab point is on the axis, the 8 root-to-grab distances are identical.  The asteroid rests on the grab point
    (centre at GRAB_HEIGHT_RE + r_ast) and the docking clamps lower it as it shrinks, so the grab point never moves.
    The 'stroke' is how far the joint throw can carry a packet from the grab point (in the worst direction in the
    ring plane) while every one of the 8 arms (length ARM_LENGTH_RE) still holds it.  All lengths in metres unless
    the key ends in _re (Earth radii).  Physics is not changed by this geometry: the escape energy is computed from
    Earth's surface (conservative), and the asteroid's mass is merged with Earth's (its 47 km offset of the common
    centre of mass and its small tidal effect on the Moon are computed here and neglected)."""
    m_a = (cfg.asteroid_mass_kg if cfg is not None else 3.0e22) if m_ast is None else m_ast
    L, rho, zg = ARM_LENGTH_RE, RING_RADIUS_RE, GRAB_HEIGHT_RE
    ang = np.arange(8) * (np.pi / 4)
    roots = np.stack([rho * np.cos(ang), rho * np.sin(ang), np.zeros(8)], 1)
    grab = np.array([0.0, 0.0, zg])
    d_grab = np.linalg.norm(roots - grab, axis=1)
    r_ast = (3.0 * m_a / (4.0 * math.pi * AST_DENSITY)) ** (1.0 / 3.0) / R_EARTH
    z_c = zg + r_ast
    d_centre = float(np.linalg.norm(roots[0] - [0, 0, z_c]))

    def stroke(e):
        lo, hi = 0.0, L
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(roots - (grab + mid * e), axis=1).max() <= L:
                lo = mid
            else:
                hi = mid
        return lo
    strokes = [stroke(np.array([math.cos(a), math.sin(a), 0.0])) for a in np.linspace(0, np.pi / 4, 46)]
    s_min, s_max = min(strokes), max(strokes)
    r_rel = math.hypot(zg, s_min)                    # release point distance from Earth's centre (worst direction)
    m_l = M_EARTH + m_a
    u_typ = 2.48e6
    u_max = cfg.max_ejection_speed_ms if cfg is not None else 5.0e6
    de_rel = (G * m_l / R_EARTH) * (1.0 - 1.0 / r_rel) / (0.5 * u_typ ** 2)
    g_dock = G * M_EARTH / (z_c * R_EARTH) ** 2
    com_shift = m_a / m_l * z_c * R_EARTH
    # tidal error on the Moon of merging the asteroid with Earth (Moon in the ring plane and on the axis; worst)
    zE, zA = -m_a / m_l * z_c * R_EARTH, (M_EARTH / m_l) * z_c * R_EARTH
    worst = 0.0
    for moon in (np.array([MOON_A, 0.0, 0.0]), np.array([0.0, 0.0, MOON_A])):
        a_true = sum(G * m * (np.array([0, 0, z]) - moon) / np.linalg.norm(np.array([0, 0, z]) - moon) ** 3
                     for m, z in ((M_EARTH, zE), (m_a, zA)))
        a_pt = G * m_l * (-moon) / np.linalg.norm(moon) ** 3
        worst = max(worst, float(np.linalg.norm(a_true - a_pt) / np.linalg.norm(a_pt)))
    return dict(arm_length_re=L, ring_radius_re=rho, grab_height_re=zg, root_to_grab_re=float(d_grab.mean()),
                root_to_grab_spread_re=float(d_grab.max() - d_grab.min()), asteroid_radius_re=r_ast, asteroid_centre_re=z_c,
                root_to_asteroid_centre_re=d_centre, asteroid_clearance_km=(zg - 1.0) * R_EARTH / 1e3,
                ring_altitude_km=(rho - 1.0) * R_EARTH / 1e3, stroke_re=s_min, stroke_max_re=s_max, stroke_km=s_min * R_EARTH / 1e3,
                release_radius_re=r_rel, escape_energy_saving_rel=de_rel,
                accel_typ_ms2=u_typ ** 2 / (2 * s_min * R_EARTH), accel_max_ms2=u_max ** 2 / (2 * s_min * R_EARTH),
                stroke_time_typ_s=2 * s_min * R_EARTH / u_typ, u_typ_ms=u_typ, u_max_ms=u_max,
                clamp_load_n=m_a * g_dock, g_at_dock_ms2=g_dock, com_shift_km=com_shift / 1e3, moon_pull_error_rel=worst)


def launch_vinf(u: float, m_launcher: float) -> float:
    """Asymptotic relative speed of a packet launched at speed u from Earth's surface."""
    v2 = u * u - 2.0 * G * m_launcher / R_EARTH
    return math.sqrt(v2) if v2 > 0 else 0.0


# ---------------------------------------------------------------------------------------------
# 3a. SOLAR COLLECTORS + MICROWAVE POWER BEAMING (energy supply of the launcher)
# ---------------------------------------------------------------------------------------------
N_ARMS = 8
SIGMA_SB = 5.670374419e-8
_BAND_CACHE = {}


def _collector_points(cfg) -> np.ndarray:
    """Sample points of the collector belt [m]: 720 longitudes x 9 latitude rows, uniform in area, covering
    |sin(latitude)| <= collector_fraction on a sphere of radius collector_orbit_au (that band holds exactly the
    fraction f of the sphere's area, i.e. it intercepts the fraction f of all sunlight)."""
    key = (float(cfg.collector_fraction), float(cfg.collector_orbit_au))
    if key not in _BAND_CACHE:
        nl, nb = 720, 9
        psi = (np.arange(nl) + 0.5) / nl * 2 * np.pi
        sb = ((np.arange(nb) + 0.5) / nb * 2 - 1) * cfg.collector_fraction
        cb = np.sqrt(1 - sb ** 2)
        pts = np.stack([np.outer(np.cos(psi), cb).ravel(), np.outer(np.sin(psi), cb).ravel(), np.tile(sb, nl)], 1)
        _BAND_CACHE[key] = pts * cfg.collector_orbit_au * AU
    return _BAND_CACHE[key]


def beam_capture_efficiency(cfg, d_m):
    """Goubau relation for a focused microwave beam: tau = D_t D_r / (lambda d); captured fraction 1 - exp(-tau^2)."""
    lam = C_LIGHT / (cfg.microwave_freq_ghz * 1e9)
    tau = (cfg.transmitter_diameter_km * 1e3) * (cfg.rectenna_diameter_km * 1e3) / (lam * np.asarray(d_m, dtype=float))
    return 1.0 - np.exp(-tau * tau)


def collector_operating(cfg) -> dict:
    """Panel temperature and OPERATING solar-cell efficiency at the belt. Energy balance of a panel facing the Sun:
    absorbed sunlight = electricity + heat radiated from both faces, with eta(T) = eta_25C (1 - k (T - 298.15 K));
    the fleet's average age-degradation factor is 1 - d x lifetime / 2 (panels replaced at the end of their life)."""
    S = S_1AU / cfg.collector_orbit_au ** 2
    a, eps = cfg.collector_absorptance, cfg.collector_emissivity
    T = 400.0
    eta = cfg.pv_efficiency
    for _ in range(80):
        eta = max(0.0, cfg.pv_efficiency * (1.0 - cfg.pv_temp_coeff * (T - 298.15)))
        T = (max(a * S - eta * S, 1e-9) / (2.0 * eps * SIGMA_SB)) ** 0.25
    degr = max(0.0, 1.0 - cfg.pv_degradation_per_year * cfg.collector_lifetime_years / 2.0)
    return dict(S_wm2=S, T_panel_K=T, T_idle_K=(a * S / (2.0 * eps * SIGMA_SB)) ** 0.25, eta_temp=eta,
                temp_factor=eta / cfg.pv_efficiency if cfg.pv_efficiency > 0 else 0.0, degr_factor=degr, eta_op=eta * degr)


def collector_built(cfg, t_years) -> float:
    """Share of the belt that exists at time t (linear build-up)."""
    if cfg.collector_build_years <= 0:
        return 1.0
    return min(1.0, cfg.collector_initial_built + (1.0 - cfg.collector_initial_built) * max(0.0, t_years) / cfg.collector_build_years)


def collector_built_mean(cfg, t0, years) -> float:
    """Average fraction of the collector belt that is built during the interval [t0, t0 + years]."""
    ts = t0 + np.linspace(0.0, max(years, 0.0), 41)
    return float(np.mean([collector_built(cfg, t) for t in ts]))


def lane_fraction(cfg, r_au, r_body) -> float:
    """Share of the belt that must stay empty so a body at r_au (radius r_body) sees the whole solar disc."""
    R = cfg.collector_orbit_au * AU
    if r_au <= cfg.collector_orbit_au:
        return 0.0
    rho = R_SUN * (r_au - cfg.collector_orbit_au) / r_au + r_body * cfg.collector_orbit_au / r_au
    return rho * rho / (4 * R * R) / max(cfg.collector_fraction, 1e-12)


def earth_shading(cfg, r_earth) -> float:
    """Share of the solar disc hidden from Earth by the belt: 0 with the clear lane; otherwise 1 when Earth's line of sight
    to the Sun crosses the belt (|sin(latitude)| < collector_fraction), else 0."""
    if not cfg.solar_collectors or cfg.earth_lane_clear or cfg.collector_fraction <= 0:
        return 0.0
    r = np.asarray(r_earth, dtype=float)
    return 1.0 if abs(r[2]) / max(float(np.linalg.norm(r)), 1.0) < cfg.collector_fraction else 0.0


def collector_power(cfg, r_earth, built=1.0) -> dict:
    """Power chain Sun -> collectors -> microwave beams -> rectennas on Earth's arms, for Earth at heliocentric
    position r_earth [m] and the built share of the belt. Sunlight on the built collectors (minus the clear lanes for Earth
    and Mars) x OPERATING solar-cell efficiency (temperature + age, see collector_operating) x microwave efficiency x
    capture (Goubau, per collector distance) x rectenna efficiency. Beams passing within 2 solar radii of the Sun are blocked."""
    pts = _collector_points(cfg)
    r = np.asarray(r_earth, dtype=float)
    e = np.array([math.hypot(r[0], r[1]), 0.0, r[2]])          # the belt is symmetric about the ecliptic axis
    d = e - pts
    dist = np.linalg.norm(d, axis=1)
    t = np.clip(-(pts * d).sum(1) / (dist * dist), 0.0, 1.0)
    closest = np.linalg.norm(pts + d * t[:, None], axis=1)
    blocked = closest < 2.0 * R_SUN
    eta = beam_capture_efficiency(cfg, dist)
    eta[blocked] = 0.0
    ec = float(eta.mean())
    op = collector_operating(cfg)
    r_au = float(np.linalg.norm(r)) / AU
    lanes = (lane_fraction(cfg, r_au, R_EARTH) + lane_fraction(cfg, 1.524, R_MARS)) if cfg.earth_lane_clear else 0.0
    l_cap = L_SUN * cfg.collector_fraction * built * (1.0 - lanes)
    p_pv = l_cap * op["eta_op"]
    p_mw = p_pv * cfg.microwave_efficiency
    p_in = p_mw * ec
    return dict(L_captured_W=l_cap, P_pv_W=p_pv, P_mw_W=p_mw, eta_capture=ec, P_incident_W=p_in,
                P_elec_W=p_in * cfg.rectenna_efficiency, d_min_au=float(dist.min() / AU), d_max_au=float(dist.max() / AU),
                blocked_frac=float(blocked.mean()), built=built, lane_frac=lanes, eta_pv_op=op["eta_op"],
                T_panel_K=op["T_panel_K"], temp_factor=op["temp_factor"], degr_factor=op["degr_factor"])


def collector_design(cfg) -> dict:
    """Size, material and light-pressure estimates of the collector belt (display / report only)."""
    R = cfg.collector_orbit_au * AU
    area = cfg.collector_fraction * 4 * math.pi * R * R
    n = max(float(cfg.collector_count), 1.0)
    beta = L_SUN / (4 * math.pi * C_LIGHT * GM_SUN * cfg.collector_areal_density)

    def lane(r_au, r_body):                 # sky area Earth (or Mars) needs clear to see the whole solar disc
        """Width of the clear lane the belt must leave so that a body at r_au is not shaded."""
        return lane_fraction(cfg, r_au, r_body)
    op = collector_operating(cfg)
    m_belt = area * cfg.collector_areal_density
    repl = m_belt / max(cfg.collector_lifetime_years, 1e-9)               # kg/yr replaced (steady state, full belt)
    fresh_total = m_belt + repl * (1 - cfg.collector_recycle_frac) * cfg.mission_years
    pw1 = collector_power(cfg, [AU, 0.0, 0.0]) if cfg.collector_fraction > 0 else None
    lam = C_LIGHT / (cfg.microwave_freq_ghz * 1e9)
    return dict(area_m2=area, count=n, each_area_m2=area / n, each_side_km=math.sqrt(area / n) / 1e3,
                mass_kg=area * cfg.collector_areal_density, mass_moon_frac=area * cfg.collector_areal_density / M_MOON,
                light_pressure_ratio=beta, half_width_deg=math.degrees(math.asin(min(1.0, cfg.collector_fraction))),
                sunlight_at_collectors_wm2=S_1AU / cfg.collector_orbit_au ** 2,
                orbit_period_yr=cfg.collector_orbit_au ** 1.5 / math.sqrt(max(1e-9, 1 - min(beta, 0.999))),
                lane_frac_earth=lane(1.0, R_EARTH), lane_frac_mars=lane(1.524, R_MARS), wavelength_m=lam,
                rectenna_area_m2=math.pi * (cfg.rectenna_diameter_km * 500.0) ** 2,
                beam_intensity_wm2_1au=(pw1["P_incident_W"] / N_ARMS / (math.pi * (cfg.rectenna_diameter_km * 500.0) ** 2)) if pw1 else 0.0,
                p_elec_1au_W=pw1["P_elec_W"] if pw1 else 0.0, T_panel_K=op["T_panel_K"], T_idle_K=op["T_idle_K"],
                eta_pv_op=op["eta_op"], temp_factor=op["temp_factor"], degr_factor=op["degr_factor"],
                replace_kg_per_yr=repl, fresh_mass_total_kg=fresh_total, fresh_mass_mercury_frac=fresh_total / 3.3011e23,
                build_energy_J=cfg.collector_embodied_energy_jkg * (m_belt * (1 - cfg.collector_initial_built) + repl * cfg.mission_years),
                belt_pull=belt_pull_on_earth(cfg))


def belt_pull_on_earth(cfg) -> dict:
    """Net gravity of the whole belt (mass spread over the band sample) on Earth at 1 AU and at the target, compared with
    the Sun's pull. The band surrounds the Sun symmetrically, so the pulls of opposite collectors largely cancel."""
    pts = _collector_points(cfg)
    gm_each = G * cfg.collector_fraction * 4 * math.pi * (cfg.collector_orbit_au * AU) ** 2 * cfg.collector_areal_density / len(pts)
    out = {}
    for tag, r_au in (("1au", 1.0), ("target", cfg.target_au)):
        r = np.array([r_au * AU, 0.0, 0.0])
        d = pts - r
        acc = (gm_each * d / (np.linalg.norm(d, axis=1) ** 3)[:, None]).sum(0)
        out["acc_" + tag] = float(np.linalg.norm(acc))
        out["ratio_" + tag] = float(np.linalg.norm(acc) / (GM_SUN / (r_au * AU) ** 2))
    out["gm_belt"] = gm_each * len(pts)
    return out


def solar_meta(cfg) -> dict:
    """Everything the viewer needs to draw the collector belt and the power chain."""
    if not cfg.solar_collectors:
        return dict(on=False)
    des = collector_design(cfg)
    pw = collector_power(cfg, [AU, 0.0, 0.0])
    return dict(on=True, orbit_au=cfg.collector_orbit_au, fraction=cfg.collector_fraction, half_width_deg=des["half_width_deg"],
                count=des["count"], each_side_km=des["each_side_km"], mass_kg=des["mass_kg"], mass_moon_frac=des["mass_moon_frac"],
                area_m2=des["area_m2"], light_pressure_ratio=des["light_pressure_ratio"], lane_frac_earth=des["lane_frac_earth"],
                pv=cfg.pv_efficiency, mw=cfg.microwave_efficiency, rx=cfg.rectenna_efficiency, launch_eff=cfg.launch_efficiency,
                freq_ghz=cfg.microwave_freq_ghz, wavelength_m=des["wavelength_m"], dt_km=cfg.transmitter_diameter_km,
                dr_km=cfg.rectenna_diameter_km, L_sun_W=L_SUN, L_captured_W=pw["L_captured_W"], P_pv_W=pw["P_pv_W"],
                P_mw_W=pw["P_mw_W"], P_elec_1au_W=pw["P_elec_W"], eta_1au=pw["eta_capture"], n_arms=N_ARMS, earth_sunlight_W=EARTH_INTERCEPT_W, rectenna_area_m2=des["rectenna_area_m2"],
                pv_rated=cfg.pv_efficiency, eta_pv_op=des["eta_pv_op"], T_panel_K=des["T_panel_K"], T_idle_K=des["T_idle_K"],
                temp_factor=des["temp_factor"], degr_factor=des["degr_factor"], built0=cfg.collector_initial_built,
                build_years=cfg.collector_build_years, lifetime_years=cfg.collector_lifetime_years,
                replace_kg_per_yr=des["replace_kg_per_yr"], fresh_mass_total_kg=des["fresh_mass_total_kg"],
                fresh_mass_mercury_frac=des["fresh_mass_mercury_frac"], build_energy_J=des["build_energy_J"],
                lane_clear=bool(cfg.earth_lane_clear), belt_pull=des["belt_pull"], orbit_period_yr=des["orbit_period_yr"],
                sk_authority=cfg.collector_sk_authority, each_mass_kg=des["mass_kg"] / des["count"])


def feasibility(cfg: Config) -> dict:
    """Analytic go / no-go check before any simulation: velocity change, energy and time needed vs. available."""
    m_sys = M_EARTH + cfg.asteroid_mass_kg + M_MOON          # Earth + reservoir + Moon (moved together)
    m_launch = M_EARTH + cfg.asteroid_mass_kg
    mu = GM_SUN + G * m_sys
    a0, at = 1.0 * AU, cfg.target_au * AU
    v0, vt = math.sqrt(mu / a0), math.sqrt(mu / at)
    dv_spiral = abs(v0 - vt)                                  # slow-spiral delta-v (circular->circular)
    a_tr = 0.5 * (a0 + at)                                    # Hohmann (two impulses) reference
    dv_hoh = abs(math.sqrt(mu * (2 / a0 - 1 / a_tr)) - v0) + abs(vt - math.sqrt(mu * (2 / at - 1 / a_tr)))
    dE_orb = G * M_SUN * m_sys * (1 / (2 * a0) - 1 / (2 * at))
    dL_orb = m_sys * (math.sqrt(mu * at) - math.sqrt(mu * a0))
    T = cfg.mission_years * YEAR
    vinf_max = launch_vinf(cfg.max_ejection_speed_ms, m_launch)
    vesc = math.sqrt(2 * G * m_launch / R_EARTH)
    # mass-limited capacity at maximum speed
    ratio = m_sys / (m_sys - cfg.asteroid_mass_kg)
    dv_cap_mass = vinf_max * math.log(ratio)
    mass_needed_umax = m_sys * (1 - math.exp(-dv_spiral / vinf_max)) if vinf_max > 0 else float("inf")
    # minimum-energy strategy: use ALL reaction mass at the lowest sufficient speed
    vinf_req = dv_spiral / math.log(ratio)
    u_req = math.sqrt(vinf_req ** 2 + vesc ** 2)
    e_kin_min = 0.5 * cfg.asteroid_mass_kg * u_req ** 2
    e_draw_min = e_kin_min / cfg.launch_efficiency
    # thrust-rate limit (packet size x frequency at max speed)
    dv_rate_max = cfg.ejections_per_year * cfg.max_packet_mass_kg * vinf_max / m_sys     # m/s per year
    years_min = dv_spiral / dv_rate_max if dv_rate_max > 0 else float("inf")
    # combined: slowest launch speed that satisfies BOTH the mass limit and the time limit,
    # and the energy that this strategy needs
    vinf_time = dv_spiral * m_sys / (cfg.mission_years * cfg.ejections_per_year * cfg.max_packet_mass_kg)
    vinf_comb = max(vinf_req, vinf_time)
    u_comb = math.sqrt(vinf_comb ** 2 + vesc ** 2)
    m_comb = m_sys * (1 - math.exp(-dv_spiral / vinf_comb))
    e_draw_comb = 0.5 * m_comb * u_comb ** 2 / cfg.launch_efficiency
    # energy available: starting budget (+ the collectors' beam power over the whole mission, at its weakest point)
    p_peak = cfg.ejections_per_year * cfg.max_packet_mass_kg * 0.5 * cfg.max_ejection_speed_ms ** 2 / cfg.launch_efficiency / YEAR
    if cfg.solar_collectors:
        pw0 = collector_power(cfg, [AU, 0.0, 0.0])
        pwt = collector_power(cfg, [max(cfg.target_au, 1.0) * AU, 0.0, 0.0])
        p_sup = min(pw0["P_elec_W"], pwt["P_elec_W"])
        e_avail = cfg.energy_budget_j + p_sup * T
    else:
        pw0 = pwt = None
        p_sup = 0.0
        e_avail = cfg.energy_budget_j
    checks = {
        "mass": dict(ok=u_req <= cfg.max_ejection_speed_ms,
                     detail=f"needs launch speed {u_req:.3e} m/s with ALL {cfg.asteroid_mass_kg:.2e} kg; "
                            f"max allowed {cfg.max_ejection_speed_ms:.3e} m/s "
                            f"(mass-limited delta-v capacity {dv_cap_mass:.4g} m/s vs required {dv_spiral:.4g} m/s)"),
        "energy": dict(ok=e_draw_min <= e_avail,
                       detail=f"minimum energy drawn {e_draw_min:.3e} J vs available {e_avail:.3e} J" +
                              (f" (reserve {cfg.energy_budget_j:.3e} J + collectors {p_sup:.3e} W x {cfg.mission_years:,.0f} y)"
                               if cfg.solar_collectors else " (budget)")),
        "rate": dict(ok=years_min <= cfg.mission_years,
                     detail=f"at max packet rate and speed the transfer needs >= {years_min:,.0f} years "
                            f"(mission {cfg.mission_years:,.0f} y)"),
        "combined": dict(ok=u_comb <= cfg.max_ejection_speed_ms and e_draw_comb <= e_avail
                         and m_comb <= cfg.asteroid_mass_kg,
                         detail=f"slowest speed meeting mass AND time limits {u_comb:.3e} m/s -> needs "
                                f"{m_comb:.3e} kg and {e_draw_comb:.3e} J"),
    }
    feasible = all(c["ok"] for c in checks.values())
    shortfall = dv_cap_mass / dv_spiral
    lines = [
        f"Required slow-spiral delta-v 1 -> {cfg.target_au} AU: {dv_spiral:,.0f} m/s "
        f"(Hohmann two-impulse: {dv_hoh:,.0f} m/s)",
        f"Orbital energy change of Earth+Moon: {dE_orb:.3e} J; angular momentum change {dL_orb:.3e} kg m^2/s",
        f"Average power just for the orbital energy change over {cfg.mission_years:,.0f} y: "
        f"{dE_orb / T:.3e} W  (= {dE_orb / T / EARTH_INTERCEPT_W:.2e} x all sunlight intercepted by Earth)",
        f"Minimum launch energy with this reservoir: {e_draw_min:.3e} J -> average power {e_draw_min / T:.3e} W "
        f"(= {e_draw_min / T / L_SUN:.2e} x total solar luminosity)",
        f"Reaction mass needed at max speed: {mass_needed_umax:.3e} kg "
        f"({mass_needed_umax / M_MOON:.3g} lunar masses); available {cfg.asteroid_mass_kg:.3e} kg",
        f"Max launch speed = {cfg.max_ejection_speed_ms / C_LIGHT * 100:.3g}% of c "
        f"(non-relativistic model; error grows above ~10% c)",
    ]
    if cfg.solar_collectors:
        lines.append(f"Solar collectors: {cfg.collector_fraction:.0%} of the Sun's output = {pw0['L_captured_W']:.3e} W intercepted at "
                     f"{cfg.collector_orbit_au} AU; electricity delivered to the arms {pw0['P_elec_W']:.3e} W with Earth at 1 AU, "
                     f"{pwt['P_elec_W']:.3e} W at {cfg.target_au} AU; launcher peak demand (max packet rate and speed) {p_peak:.3e} W "
                     f"-> the beams supply {p_sup / p_peak:.2f}x the peak demand")
    for k, c in checks.items():
        lines.append(f"[{'OK ' if c['ok'] else 'FAIL'}] {k}: {c['detail']}")
    verdict = ("FEASIBLE within this model (physically hypothetical scale)" if feasible
               else "INFEASIBLE with these parameters (failed: " +
               ", ".join(k for k, c in checks.items() if not c["ok"]) +
               (f"; mass-limited delta-v reaches {shortfall:.2e} of the requirement)" if shortfall < 1 else ")"))
    plain = [
        f"To move Earth from 1 AU to {cfg.target_au} AU slowly, its orbital speed must be changed by about "
        f"{dv_spiral / 1000:,.1f} km/s in total (it ends up slower, but on a much bigger orbit).",
        ("Mass: OK - throwing the whole asteroid at " if checks["mass"]["ok"] else
         "Mass: NOT ENOUGH - even throwing the whole asteroid at ") +
        f"{min(u_req, cfg.max_ejection_speed_ms) / 1000:,.0f} km/s " +
        ("gives enough push." if checks["mass"]["ok"] else
         f"the maximum speed gives only {shortfall:.2%} of the required push."),
        ("Energy: OK - " if checks["energy"]["ok"] else "Energy: NOT ENOUGH - ") +
        f"the cheapest way needs {e_draw_min:.2e} J; " +
        (f"the solar collectors deliver {p_sup:.2e} W ({p_sup / p_peak:.1f}x the launcher's peak demand), i.e. up to {e_avail:.2e} J "
         f"over the mission, plus a reserve of {cfg.energy_budget_j:.2e} J." if cfg.solar_collectors else
         f"the budget is {cfg.energy_budget_j:.2e} J."),
        ("Time: OK - " if checks["rate"]["ok"] else "Time: TOO SHORT - ") +
        f"at full thrust the transfer takes at least {years_min:,.0f} years; the mission allows {cfg.mission_years:,.0f}.",
        ("All limits together: OK." if checks["combined"]["ok"] else
         "All limits together: NOT met (meeting the time limit needs faster, more energy-hungry launches)."),
        "This is an analytic estimate for a slow spiral; the simulation then shows what actually happens.",
    ]
    return dict(feasible=feasible, verdict=verdict, lines=lines, plain=plain, dv_required=dv_spiral, dv_hohmann=dv_hoh,
                dE_orbit=dE_orb, dL_orbit=dL_orb, dv_capacity_mass=dv_cap_mass, u_required=u_req,
                energy_min_draw=e_draw_min, mass_needed_at_umax=mass_needed_umax, years_min=years_min,
                u_combined=u_comb, energy_combined=e_draw_comb, mass_combined=m_comb,
                avg_power_min=e_draw_min / T, checks=checks, vesc=vesc, vinf_max=vinf_max,
                energy_available=e_avail, solar_power_min=p_sup, launcher_peak_power=p_peak)


def habitability(a_au: float, e: float) -> dict:  # noqa: D401  (quick indicator; the climate model is climate_integrate)
    """Simplified energy balance (NOT a climate model): orbit-averaged insolation and the
    blackbody equilibrium temperature for Bond albedo 0.3 plus today's ~33 K greenhouse offset."""
    if not (a_au > 0) or not math.isfinite(a_au) or e >= 1:
        return dict(insolation=float("nan"), t_eq=float("nan"), t_surface_naive=float("nan"))
    s = S_1AU / (a_au ** 2 * math.sqrt(1 - e * e))
    t_eq = (s * (1 - 0.3) / (4 * 5.670374419e-8)) ** 0.25
    return dict(insolation=s, insolation_rel=s / S_1AU, t_eq=t_eq, t_surface_naive=t_eq + 33.0)


# ---------------------------------------------------------------------------------------------
# 3b. CLIMATE: two-box energy-balance model + space mirrors (part of the RL task)
# ---------------------------------------------------------------------------------------------
CLIM_C_S = 8.0          # heat capacity of the surface layer (atmosphere + ocean mixed layer) [W yr m^-2 K^-1]
CLIM_C_D = 100.0        # heat capacity of the deep ocean [W yr m^-2 K^-1]
CLIM_GAMMA = 0.7        # surface <-> deep-ocean heat exchange [W m^-2 K^-1]
CLIM_ALB_WARM, CLIM_ALB_ICE = 0.30, 0.60
CLIM_T_ICE0, CLIM_T_ICE1 = 280.0, 255.0       # albedo rises linearly from 0.30 (>= 280 K) to 0.60 (<= 255 K)
CLIM_EPS = S_1AU * (1 - CLIM_ALB_WARM) / 4.0 / (SIGMA_SB * 288.0 ** 4)   # effective emissivity: 288 K today (fixed greenhouse)
CLIM_LIVABLE_K = 15.0   # 'livable' band = target +- this (reported: years outside it)
CLIM_DT_YR = 0.0625


@njit(cache=True)
def _albedo(T):
    """Earth's reflectivity at surface temperature T: 0.30 when warm, rising to 0.60 when ice-covered."""
    if T >= 280.0:
        return 0.30
    if T <= 255.0:
        return 0.60
    return 0.30 + 0.30 * (280.0 - T) / 25.0


@njit(cache=True)
def climate_integrate(t_start, ts, r_au, Ts, Td, s_set_rel, mirror_area_eff, mirror_refl, shade, q_w, mirrors_on,
                      t_target, livable, out):
    """Step the two-box climate through one decision interval.

    ts [yr] and r_au are Earth's simulated times and distances from the Sun. The space mirrors add reflected sunlight so
    the total light on Earth reaches s_set_rel x today's (they cannot take sunlight away). Their most possible light is
    mirror_area_eff (area facing the Sun, m^2) x mirror_refl x the local sunlight. q_w [W] = waste heat reaching Earth.
    Explicit Euler with sub-steps of at most 0.0625 yr (the surface reacts in ~2 yr).
    out[i] = (Ts, Td, sunlight relative to today, mirror light on Earth [W]) at each sample.
    Returns (Ts, Td, mirror light delivered [J], sunlight caught by the mirrors [J], years outside the livable band,
    largest |Ts - target|)."""
    area_disc = 3.141592653589793 * 6.371e6 ** 2
    area_sph = 4.0 * area_disc
    E_light = 0.0
    E_msun = 0.0
    out_years = 0.0
    dev_max = 0.0
    tp = t_start
    for i in range(ts.shape[0]):
        dt = ts[i] - tp
        n = int(dt / 0.0625) + 1
        h = dt / n
        s_sun = 1361.0 / (r_au[i] * r_au[i]) * (1.0 - shade)      # sunlight at Earth [W/m^2]
        p_mir = 0.0
        if mirrors_on:
            p_cap = mirror_area_eff * mirror_refl * s_sun           # most light the mirrors can send [W]
            p_mir = (s_set_rel * 1361.0 - s_sun) * area_disc        # light needed to reach the set-point [W]
            if p_mir < 0.0:
                p_mir = 0.0
            if p_mir > p_cap:
                p_mir = p_cap
            E_msun += mirror_area_eff * s_sun * dt * 365.25 * 86400.0
        s_tot = s_sun + p_mir / area_disc
        for _ in range(n):
            f_in = s_tot * (1.0 - _albedo(Ts)) / 4.0 + q_w / area_sph
            dTs = (f_in - 0.6105 * 5.670374419e-8 * Ts ** 4 - 0.7 * (Ts - Td)) / 8.0
            dTd = 0.7 * (Ts - Td) / 100.0
            Ts += h * dTs
            Td += h * dTd
            if abs(Ts - t_target) > livable:
                out_years += h
        if abs(Ts - t_target) > dev_max:
            dev_max = abs(Ts - t_target)
        E_light += p_mir * dt * 365.25 * 86400.0
        out[i, 0] = Ts
        out[i, 1] = Td
        out[i, 2] = s_sun / 1361.0
        out[i, 3] = p_mir
        tp = ts[i]
    return Ts, Td, E_light, E_msun, out_years, dev_max


def climate_equilibrium(s_rel, q_w=0.0):
    """Warm-branch equilibrium surface temperature for insolation s_rel x S_1AU (frozen branch if no warm one exists)."""
    def f(T):
        """Energy imbalance at temperature T (absorbed minus emitted); its zero is the equilibrium."""
        return s_rel * S_1AU * (1 - float(_albedo(T))) / 4 + q_w / (4 * math.pi * R_EARTH ** 2) - CLIM_EPS * SIGMA_SB * T ** 4
    lo, hi = (280.0, 450.0) if f(280.0) > 0 else (100.0, 280.0)     # warm branch if it exists, else the frozen state
    for _ in range(100):
        T = 0.5 * (lo + hi)
        lo, hi = (T, hi) if f(T) > 0 else (lo, T)
    return 0.5 * (lo + hi)


# Space mirrors (soletta): a ring of mirrors in a 'terminator' orbit around Earth (orbit plane at right angles to the Sun
# direction). Each mirror sees the Sun at 90 degrees from Earth, so it is tilted 45 degrees and catches cos 45 of the
# sunlight on its area.
MIRROR_TILT_COS = math.cos(math.radians(45.0))


def mirror_design(cfg, r_au=1.0) -> dict:
    """Size, mass and capacity of the space mirrors with Earth at r_au from the Sun."""
    area_eff = cfg.mirror_area_m2 * MIRROR_TILT_COS                       # area facing the Sun [m^2]
    cap_W = area_eff * cfg.mirror_reflectivity * S_1AU / r_au ** 2 if cfg.mirrors else 0.0
    r_orbit = cfg.mirror_orbit_re * R_EARTH
    sun_angle = 2 * R_SUN / (r_au * AU)                                     # angular size of the Sun [rad]
    return dict(on=bool(cfg.mirrors), area_m2=cfg.mirror_area_m2, area_eff_m2=area_eff, reflectivity=cfg.mirror_reflectivity,
                cap_W=cap_W, cap_rel=cap_W / EARTH_INTERCEPT_W, mass_kg=cfg.mirror_area_m2 * cfg.mirror_areal_density,
                orbit_re=cfg.mirror_orbit_re, orbit_days=2 * math.pi * math.sqrt(r_orbit ** 3 / (G * M_EARTH)) / 86400.0,
                spot_km=sun_angle * r_orbit / 1000.0, disc_radius_km=math.sqrt(cfg.mirror_area_m2 / math.pi) / 1000.0,
                tilt_cos=MIRROR_TILT_COS)


# Life indicators: a simple zonal (latitude) temperature profile around the global mean of the climate model.
LIFE_DT_EQ_POLE = 45.0        # equator-to-pole difference of the annual-mean temperature [K] (today's Earth)
LIFE_T_ICE = 263.0            # permanent ice where the annual mean is below -10 C [K]
LIFE_T_GROW = (278.0, 313.0)  # plants can grow where the annual mean is 5-40 C [K]


def _sinlat_where(T_mean, T_level):
    """sin(latitude) where the zonal profile T(x) = T_mean + dT/3 - dT x^2 equals T_level (clipped to 0..1)."""
    x2 = (T_mean + LIFE_DT_EQ_POLE / 3.0 - T_level) / LIFE_DT_EQ_POLE
    return math.sqrt(min(1.0, max(0.0, x2)))


def life_indicators(T_mean) -> dict:
    """What a global-mean temperature means for life: equator / pole temperature, ice line, liquid and growing area.

    Uses T(lat) = T_mean + dT/3 - dT sin^2(lat), whose area average is exactly T_mean. Shares are of Earth's surface."""
    x_ice = _sinlat_where(T_mean, LIFE_T_ICE)          # ice poleward of this
    x_cold = _sinlat_where(T_mean, LIFE_T_GROW[0])     # too cold to grow poleward of this
    x_hot = _sinlat_where(T_mean, LIFE_T_GROW[1])      # too hot to grow equatorward of this
    grow = max(0.0, x_cold - x_hot)
    if x_ice <= 0.0:
        state = "snowball (frozen to the equator)"
    elif grow >= 0.5:
        state = "temperate, like today"
    elif x_hot >= 1.0:
        state = "hothouse"
    else:
        state = "cold, ice to mid-latitudes" if x_ice < 0.7 else "cool"
    return dict(T_equator_K=T_mean + LIFE_DT_EQ_POLE / 3.0, T_pole_K=T_mean - 2.0 * LIFE_DT_EQ_POLE / 3.0,
                ice_line_deg=math.degrees(math.asin(x_ice)), ice_frac=1.0 - x_ice, liquid_frac=x_ice, grow_frac=grow,
                state=state)


# ---------------------------------------------------------------------------------------------
# 3c. COLLECTOR BELT: do the collectors stay on their orbits? (viewer + check 19; no feedback on Earth)
# ---------------------------------------------------------------------------------------------
BELT_N_TRACK = 12           # simulated collectors: inclination 0 / half / full band width x 4 node directions
BELT_DT_DAYS = 0.5
BELT_OUT_N = 20             # deviation snapshots per decision interval sent to the viewer


@njit(cache=True, nogil=True)
def _herm(ts, X, V, W, t, k):
    """Cubic Hermite interpolation of position X (m) at t (s) from samples at ts (s), using the velocity just AFTER the
    event at the left end (V) and just BEFORE the event at the right end (W) [m/s] - motion between events is ballistic."""
    n = ts.shape[0]
    while k < n - 2 and ts[k + 1] < t:
        k += 1
    h = ts[k + 1] - ts[k]
    out = np.zeros(3)
    if h <= 0.0:
        for j in range(3):
            out[j] = X[k, j]
        return out, k
    s_ = (t - ts[k]) / h
    if s_ < 0.0:
        s_ = 0.0
    if s_ > 1.0:
        s_ = 1.0
    h00 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1
    h10 = s_ ** 3 - 2 * s_ ** 2 + s_
    h01 = -2 * s_ ** 3 + 3 * s_ ** 2
    h11 = s_ ** 3 - s_ ** 2
    for j in range(3):
        out[j] = h00 * X[k, j] + h10 * h * V[k, j] + h01 * X[k + 1, j] + h11 * h * W[k + 1, j]
    return out, k


@njit(cache=True, nogil=True)
def _belt_acc(t, d, dv, P, Q, R, n, u0, mu, rE, rM, gmE, gmM, a_rec, auth, wc, sk, rsun2):
    """Acceleration of the deviation d = r - r_ref (Encke form: exact differential gravity of the Sun, so the analytic
    reference orbit carries no integration error) + planets (heliocentric, with indirect terms) + beam recoil (away from
    Earth, zero if the beam would pass the Sun) + station keeping (PD, limited to the sail's authority)."""
    N = d.shape[0]
    acc = np.zeros((N, 3))
    ctrl = np.zeros(N)
    pl = np.zeros(N)
    rE3 = (rE[0] ** 2 + rE[1] ** 2 + rE[2] ** 2) ** 1.5
    rM3 = (rM[0] ** 2 + rM[1] ** 2 + rM[2] ** 2) ** 1.5
    for i in range(N):
        th = u0[i] + n * t
        c, s_ = np.cos(th), np.sin(th)
        rr = np.zeros(3)
        r = np.zeros(3)
        for j in range(3):
            rr[j] = R * (c * P[i, j] + s_ * Q[i, j])
            r[j] = rr[j] + d[i, j]
        nr3 = (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 1.5
        nrr3 = R ** 3
        g = np.zeros(3)
        for j in range(3):
            g[j] = -mu * (r[j] / nr3 - rr[j] / nrr3)
        # planets (direct + indirect)
        e = r - rE
        ne3 = (e[0] ** 2 + e[1] ** 2 + e[2] ** 2) ** 1.5
        m = r - rM
        nm3 = (m[0] ** 2 + m[1] ** 2 + m[2] ** 2) ** 1.5
        ap = np.zeros(3)
        for j in range(3):
            ap[j] = -gmE * (e[j] / ne3 + rE[j] / rE3) - gmM * (m[j] / nm3 + rM[j] / rM3)
        pl[i] = np.sqrt(ap[0] ** 2 + ap[1] ** 2 + ap[2] ** 2)
        # beam recoil: the beam goes from the collector to Earth -> push away from Earth
        ne = np.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)
        tt = -(r[0] * (-e[0]) + r[1] * (-e[1]) + r[2] * (-e[2])) / (ne * ne)     # closest point of the segment to the Sun
        if tt < 0.0:
            tt = 0.0
        if tt > 1.0:
            tt = 1.0
        cl = 0.0
        for j in range(3):
            cl += (r[j] - tt * e[j]) ** 2
        rec = a_rec if cl > rsun2 else 0.0
        a = np.zeros(3)
        for j in range(3):
            a[j] = g[j] + ap[j] + rec * e[j] / ne
        if sk:
            ac = np.zeros(3)
            for j in range(3):
                ac[j] = -wc * wc * d[i, j] - 2.0 * wc * dv[i, j]
            na = np.sqrt(ac[0] ** 2 + ac[1] ** 2 + ac[2] ** 2)
            lim = auth / (nr3 ** (2.0 / 3.0))
            if na > lim:
                for j in range(3):
                    ac[j] *= lim / na
                na = lim
            ctrl[i] = na
            for j in range(3):
                a[j] += ac[j]
        for j in range(3):
            acc[i, j] = a[j]
    return acc, ctrl, pl


@njit(cache=True, nogil=True)
def belt_advance(t0, t1, dt, P, Q, R, n, u0, mu, d, dv, ts, EX, EV, EW, MX, MV, gmE, gmM, a_rec, auth, wc, sk, rsun2,
                 frozen, out_t, out_d, st):
    """RK4 over [t0, t1] (s). st accumulates: 0 max|d|, 1 sum|d|^2, 2 count, 3 delta-v of control, 4 max planet accel,
    5 max recoil accel (applied), 6 max |radial deviation|, 7 max |out-of-plane deviation|."""
    N = d.shape[0]
    nsteps = int((t1 - t0) / dt) + 1
    h = (t1 - t0) / nsteps
    kE = 0
    kM = 0
    nout = out_t.shape[0]
    io = 0
    for sidx in range(nsteps + 1):
        t = t0 + sidx * h
        if io < nout and t >= t0 + (t1 - t0) * io / (nout - 1) - 0.5 * h:
            out_t[io] = t
            for i in range(N):
                for j in range(3):
                    out_d[io, i, j] = d[i, j]
            io += 1
        if sidx == nsteps:
            break
        rE1, kE = _herm(ts, EX, EV, EW, t, kE)
        rM1, kM = _herm(ts, MX, MV, MV, t, kM)
        rE2, kE = _herm(ts, EX, EV, EW, t + 0.5 * h, kE)
        rM2, kM = _herm(ts, MX, MV, MV, t + 0.5 * h, kM)
        rE3, kE = _herm(ts, EX, EV, EW, t + h, kE)
        rM3, kM = _herm(ts, MX, MV, MV, t + h, kM)
        a1, c1, p1 = _belt_acc(t, d, dv, P, Q, R, n, u0, mu, rE1, rM1, gmE, gmM, a_rec, auth, wc, sk, rsun2)
        d2 = d + 0.5 * h * dv
        v2 = dv + 0.5 * h * a1
        a2, c2, p2 = _belt_acc(t + 0.5 * h, d2, v2, P, Q, R, n, u0, mu, rE2, rM2, gmE, gmM, a_rec, auth, wc, sk, rsun2)
        d3 = d + 0.5 * h * v2
        v3 = dv + 0.5 * h * a2
        a3, c3, p3 = _belt_acc(t + 0.5 * h, d3, v3, P, Q, R, n, u0, mu, rE2, rM2, gmE, gmM, a_rec, auth, wc, sk, rsun2)
        d4 = d + h * v3
        v4 = dv + h * a3
        a4, c4, p4 = _belt_acc(t + h, d4, v4, P, Q, R, n, u0, mu, rE3, rM3, gmE, gmM, a_rec, auth, wc, sk, rsun2)
        for i in range(N):
            if frozen[i]:
                continue
            for j in range(3):
                d[i, j] += h / 6.0 * (dv[i, j] + 2 * v2[i, j] + 2 * v3[i, j] + v4[i, j])
                dv[i, j] += h / 6.0 * (a1[i, j] + 2 * a2[i, j] + 2 * a3[i, j] + a4[i, j])
            nd = np.sqrt(d[i, 0] ** 2 + d[i, 1] ** 2 + d[i, 2] ** 2)
            if nd > 1.5 * 1.495978707e11:          # left the Sun's neighbourhood of its slot entirely: stop following
                frozen[i] = True
            th = u0[i] + n * (t + h)
            c_, s_ = np.cos(th), np.sin(th)
            rad = 0.0
            nor = 0.0
            for j in range(3):
                rad += d[i, j] * (c_ * P[i, j] + s_ * Q[i, j])
            nx = P[i, 1] * Q[i, 2] - P[i, 2] * Q[i, 1]
            ny = P[i, 2] * Q[i, 0] - P[i, 0] * Q[i, 2]
            nz = P[i, 0] * Q[i, 1] - P[i, 1] * Q[i, 0]
            nor = d[i, 0] * nx + d[i, 1] * ny + d[i, 2] * nz
            if abs(rad) > st[6]:
                st[6] = abs(rad)
            if abs(nor) > st[7]:
                st[7] = abs(nor)
            if nd > st[0]:
                st[0] = nd
            st[1] += nd * nd
            st[2] += 1.0
            st[3] += h / 6.0 * (c1[i] + 2 * c2[i] + 2 * c3[i] + c4[i])
            if p1[i] > st[4]:
                st[4] = p1[i]
    st[5] = a_rec


class BeltKeeper:
    """Follows BELT_N_TRACK collectors twice - with and without station keeping - along Earth's and Mars's simulated
    motion of each decision interval. Deviations are measured from each collector's assigned circular orbit (radius
    collector_orbit_au, inclination 0 / half / full band width, Sun gravity reduced by the light-pressure ratio)."""

    def __init__(self, cfg):
        """Set up the 12 followed collectors on their assigned slots, with and without station keeping."""
        des = collector_design(cfg)
        self.cfg = cfg
        self.R = cfg.collector_orbit_au * AU
        self.beta = des["light_pressure_ratio"]
        self.mu = GM_SUN * (1 - self.beta)
        self.n = math.sqrt(self.mu / self.R ** 3)
        self.count = des["count"]
        self.m_each = des["mass_kg"] / des["count"]
        hw = math.radians(des["half_width_deg"])
        inc, node, u0 = [], [], []
        for k in range(BELT_N_TRACK):
            inc.append([0.0, 0.5 * hw, hw][k % 3])
            node.append(math.pi / 2 * (k // 3))
            u0.append(2 * math.pi * k / BELT_N_TRACK)
        self.inc, self.node, self.u0 = np.array(inc), np.array(node), np.array(u0)
        cO, sO, ci, si = np.cos(self.node), np.sin(self.node), np.cos(self.inc), np.sin(self.inc)
        self.P = np.stack([cO, sO, np.zeros_like(cO)], 1)
        self.Q = np.stack([-sO * ci, cO * ci, si], 1)
        self.wc = 4.0 * self.n
        self.auth = cfg.collector_sk_authority * self.beta * GM_SUN
        N = 2 * BELT_N_TRACK
        self.d = np.zeros((N, 3))
        self.dv = np.zeros((N, 3))
        self.frozen = np.zeros(N, dtype=np.bool_)
        self.st = {True: np.zeros(8), False: np.zeros(8)}
        self.spacing = math.sqrt(des["each_area_m2"])
        self.t = 0.0
        self.years = 0.0
        self.last_row = None
        self.prev_dth = np.zeros(2 * BELT_N_TRACK)
        spacing = math.sqrt(des["each_area_m2"])
        self.neighbour_ratio = G * self.m_each / spacing ** 2 / (GM_SUN / self.R ** 2)

    def tracked(self):
        """Assigned orbits of the 12 followed collectors (sent to the viewer)."""
        return dict(R_au=self.cfg.collector_orbit_au, n_rad_yr=self.n * YEAR, inc_deg=np.degrees(self.inc).round(3).tolist(),
                    node_deg=np.degrees(self.node).round(3).tolist(), u0_rad=self.u0.round(6).tolist(),
                    beta=self.beta, period_days=2 * math.pi / self.n / 86400)

    def advance(self, samples, p_mw_W, t_start_years):
        """Integrate both sets over the samples of one interval. p_mw_W: mean microwave power sent by the whole belt."""
        if len(samples) < 2:
            return None
        smp = np.asarray(samples, dtype=np.float64)
        if self.last_row is not None:      # continue exactly where the previous interval ended (no gap, same phase)
            smp = np.vstack([self.last_row[None, :], smp])
        else:                              # first interval: start at t_start, 2nd-order backward step from the first sample
            r0 = smp[0, S_R:S_R + 3] * AU
            v0 = smp[0, S_VPRE:S_VPRE + 3] * 1e3
            dt0 = (t_start_years - smp[0, S_T]) * YEAR
            row = smp[0].copy()
            row[S_T] = t_start_years
            a0 = -GM_SUN * r0 / np.linalg.norm(r0) ** 3
            row[S_R:S_R + 3] = (r0 + v0 * dt0 + 0.5 * a0 * dt0 ** 2) / AU
            row[S_VPOST:S_VPOST + 3] = (v0 + a0 * dt0) / 1e3
            rm, vm = smp[0, S_MARS:S_MARS + 3] * AU, smp[0, S_MARSV:S_MARSV + 3] * 1e3
            am = -GM_SUN * rm / np.linalg.norm(rm) ** 3
            row[S_MARS:S_MARS + 3] = (rm + vm * dt0 + 0.5 * am * dt0 ** 2) / AU
            row[S_MARSV:S_MARSV + 3] = (vm + am * dt0) / 1e3
            smp = np.vstack([row[None, :], smp])
        self.last_row = smp[-1].copy()
        ts = smp[:, S_T] * YEAR
        EX = smp[:, S_R:S_R + 3] * AU
        EV = smp[:, S_VPOST:S_VPOST + 3] * 1e3
        EW = smp[:, S_VPRE:S_VPRE + 3] * 1e3
        MX = smp[:, S_MARS:S_MARS + 3] * AU
        MV = smp[:, S_MARSV:S_MARSV + 3] * 1e3
        gmE = GM_EARTH + GM_MOON
        a_rec = max(p_mw_W, 0.0) / self.count / (C_LIGHT * self.m_each)
        out = {}
        tr = {}
        sets = ((True, slice(0, BELT_N_TRACK)), (False, slice(BELT_N_TRACK, 2 * BELT_N_TRACK)))
        work = {}
        for sk, sl in sets:
            work[sk] = (self.d[sl].copy(), self.dv[sl].copy(), self.frozen[sl].copy(), np.zeros(BELT_OUT_N),
                        np.zeros((BELT_OUT_N, BELT_N_TRACK, 3)), np.zeros(8))

        def run(sk):
            """One numba RK4 call (releases the GIL, so the two independent sets run on two CPU cores at once)."""
            d, dv, fr, out_t, out_d, st_int = work[sk]
            belt_advance(ts[0], ts[-1], BELT_DT_DAYS * 86400.0, self.P, self.Q, self.R, self.n, self.u0, self.mu, d, dv, ts,
                         EX, EV, EW, MX, MV, gmE, GM_MARS, a_rec, self.auth, self.wc, sk, (2 * R_SUN) ** 2, fr, out_t, out_d, st_int)
        th = threading.Thread(target=run, args=(False,), daemon=True)
        th.start()
        run(True)
        th.join()
        for sk, sl in sets:
            d, dv, fr, out_t, out_d, st_int = work[sk]
            self.d[sl], self.dv[sl], self.frozen[sl] = d, dv, fr
            g = self.st[sk]
            g[0] = max(g[0], st_int[0]); g[1] += st_int[1]; g[2] += st_int[2]; g[3] += st_int[3]; g[4] = max(g[4], st_int[4])
            g[5] = max(g[5], st_int[5]); g[6] = max(g[6], st_int[6]); g[7] = max(g[7], st_int[7])
            yrs = (ts[-1] - ts[0]) / YEAR
            el = self._elements(d, dv, (ts[-1]))
            out["on" if sk else "off"] = dict(max_km=st_int[0] / 1e3, rms_km=math.sqrt(st_int[1] / max(st_int[2], 1)) / 1e3,
                                              now_km=float(np.linalg.norm(d, axis=1).max() / 1e3),
                                              dv_ms_per_yr=st_int[3] / max(yrs, 1e-9) / BELT_N_TRACK,
                                              cum_max_km=g[0] / 1e3, frozen=int(fr.sum()), radial_km=st_int[6] / 1e3,
                                              normal_km=st_int[7] / 1e3, cum_radial_km=g[6] / 1e3, cum_normal_km=g[7] / 1e3,
                                              slots=float(np.linalg.norm(d, axis=1).max()) / self.spacing, **el)
            tr["on" if sk else "off"] = self._curvilinear(out_t, out_d, sl)
        self.years = (ts[-1] / YEAR)
        g0 = GM_SUN / self.R ** 2
        out.update(t=[round(float(x) / YEAR, 4) for x in out_t], track=tr, recoil_accel=a_rec, recoil_ratio=a_rec / g0,
                   planet_ratio=self.st[True][4] / g0, control_limit=self.auth / self.R ** 2, control_ratio=self.auth / self.R ** 2 / g0,
                   neighbour_ratio=self.neighbour_ratio, light_ratio=self.beta)
        return out

    def _curvilinear(self, ts, dd, sl):
        """Deviation snapshots as [radius - R (km), angle along the orbit relative to the slot (rad, unwrapped), height above
        the assigned orbital plane (km)] - smooth in time, so the viewer can interpolate them (Cartesian deviations rotate
        with the orbit every ~130 days)."""
        Nv = np.cross(self.P, self.Q)
        prev = self.prev_dth[sl]
        out = []
        for k in range(len(ts)):
            th = self.u0 + self.n * ts[k]
            rr = self.R * (np.cos(th)[:, None] * self.P + np.sin(th)[:, None] * self.Q)
            r = rr + dd[k]
            x, y, z = (r * self.P).sum(1), (r * self.Q).sum(1), (r * Nv).sum(1)
            dth = np.arctan2(y, x) - th
            dth = prev + (dth - prev + np.pi) % (2 * np.pi) - np.pi        # unwrap against the previous snapshot
            prev = dth
            out.append([[int(round((math.hypot(x[i], y[i]) - self.R) / 1e3)), float(f"{dth[i]:.9g}"), int(round(z[i] / 1e3))]
                        for i in range(BELT_N_TRACK)])
        self.prev_dth[sl] = prev
        return out

    def _elements(self, d, dv, t):
        """Largest changes of semi-major axis, eccentricity and inclination vs. the assigned orbits; swing_km = |da| + a e is
        the largest change of the distance from the Sun (the orbit's shape), independent of the drift along the orbit."""
        da = de = di = 0.0
        for i in range(min(d.shape[0], BELT_N_TRACK)):
            th = self.u0[i] + self.n * t
            c, s_ = math.cos(th), math.sin(th)
            rr = self.R * (c * self.P[i] + s_ * self.Q[i])
            vr = self.R * self.n * (-s_ * self.P[i] + c * self.Q[i])
            r, v = rr + d[i], vr + dv[i]
            rn = float(np.linalg.norm(r))
            a = 1.0 / (2.0 / rn - float(v @ v) / self.mu)
            h = np.cross(r, v)
            ev = np.cross(v, h) / self.mu - r / rn
            inc = math.acos(max(-1.0, min(1.0, h[2] / float(np.linalg.norm(h)))))
            da, de, di = max(da, abs(a - self.R)), max(de, float(np.linalg.norm(ev))), max(di, abs(inc - self.inc[i]))
        return dict(da_km=da / 1e3, e_max=de, di_deg=math.degrees(di), swing_km=(da + self.R * de) / 1e3)

    def summary(self):
        """Largest slot deviations and control effort so far, with and without station keeping."""
        res = {}
        for sk in (True, False):
            g = self.st[sk]
            sl = slice(0, BELT_N_TRACK) if sk else slice(BELT_N_TRACK, 2 * BELT_N_TRACK)
            res["on" if sk else "off"] = dict(max_km=g[0] / 1e3, rms_km=math.sqrt(g[1] / max(g[2], 1)) / 1e3,
                                              dv_ms_per_yr=g[3] / max(self.years, 1e-9) / BELT_N_TRACK,
                                              radial_km=g[6] / 1e3, normal_km=g[7] / 1e3,
                                              slots=g[0] / self.spacing, **self._elements(self.d[sl], self.dv[sl], self.years * YEAR))
        g0 = GM_SUN / self.R ** 2
        res.update(recoil_ratio_max=self.st[True][5] / g0, planet_ratio=self.st[True][4] / g0, neighbour_ratio=self.neighbour_ratio,
                   years=self.years, spacing_km=self.spacing / 1e3, control_ratio=self.auth / self.R ** 2 / g0)
        return res


def mars_tide_ratio(d_m: float) -> float:
    """Mars's tidal acceleration on Earth relative to the Moon's (tide ~ G M / d^3)."""
    return (GM_MARS / GM_MOON) * (MOON_A / d_m) ** 3


def mars_moon_perturbation(d_m: float) -> float:
    """Mars's tidal pull on the Earth-Moon orbit relative to Earth's pull on the Moon: 2 (M_mars/M_earth) (d_EM/d)^3."""
    return 2.0 * (GM_MARS / GM_EARTH) * (MOON_A / d_m) ** 3


def mars_safety_limits(cfg) -> dict:
    """Planetary-safety distances (see the 'Planetary safety (Mars)' parameters).  The safe distance is the
    LARGEST of: the tide limit, the Moon-orbit limit and the hard minimum distance."""
    d_tide = MOON_A * (GM_MARS / GM_MOON / cfg.mars_max_tide_ratio) ** (1.0 / 3.0)
    d_moon = MOON_A * (2.0 * GM_MARS / GM_EARTH / cfg.mars_max_moon_perturbation) ** (1.0 / 3.0)
    d_min = cfg.mars_min_distance_km * 1e3
    d_safe = max(d_tide, d_moon, d_min)
    governing = "tide on Earth" if d_safe == d_tide else ("Moon's orbit" if d_safe == d_moon else "hard minimum distance")
    return dict(d_tide_m=d_tide, d_moon_m=d_moon, d_min_m=d_min, d_safe_m=d_safe, d_safe_au=d_safe / AU,
                caution_m=cfg.shield_caution_factor * d_safe, caution_au=cfg.shield_caution_factor * d_safe / AU,
                governing=governing, d_collision_m=R_EARTH + R_MARS,
                d_tide_equal_moon_m=MOON_A * (GM_MARS / GM_MOON) ** (1.0 / 3.0),
                roche_m=2.44 * R_EARTH * (5514.0 / 3933.0) ** (1.0 / 3.0))


def space_environment(r_au: float, m_sys_kg: float) -> dict:
    """Display-only space-environment estimates at Earth's simulated Sun distance r_au (Earth-POV view).
    Standard simplified models (identical formulas in frontend/envmodel.js; a test compares them):
    steady solar wind 400 km/s with 5 protons/cm^3 at 1 AU (density ~ 1/r^2); Parker-spiral IMF
    (radial 3.5 nT at 1 AU); Shue et al. (1998) magnetopause with IMF Bz = 0; bow shock 1.29 x the
    magnetopause stand-off (Mach 8 assumed); Hill radius a (m / 3 M_sun)^(1/3) with a = r.
    These numbers do NOT feed back into the orbit simulation."""
    n = 5.0 / (r_au * r_au)
    v = 400.0
    dp = 1.6726e-6 * n * v * v
    bz = 0.0
    r0 = (10.22 + 1.29 * math.tanh(0.184 * (bz + 8.14))) * dp ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * bz) * (1.0 + 0.024 * math.log(dp))
    br = 3.5 / (r_au * r_au)
    wind = 2.865e-6 * r_au * 1.495978707e8 / v
    hill_re = r_au * AU / R_EARTH * (m_sys_kg / (3.0 * M_SUN)) ** (1.0 / 3.0)
    return dict(n_cm3=n, v_kms=v, dp_npa=dp, r0_re=r0, alpha=alpha, rbs_re=1.29 * r0, alpha_bs=0.8,
                imf_nt=br * math.sqrt(1.0 + wind * wind), spiral_deg=math.degrees(math.atan(wind)), hill_re=hill_re,
                insolation_wm2=S_1AU / (r_au * r_au), sun_ang_rad=R_SUN / (r_au * AU))


# =============================================================================================
# 4. INITIAL CONDITIONS (J2000 mean elements, ecliptic frame)
# =============================================================================================
def elements_to_state(a, e, inc, raan, argp, M, mu):
    """Turn orbital elements (a, e, i, node, perihelion, mean anomaly) into position and velocity."""
    E = M
    for _ in range(50):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    cE, sE = math.cos(E), math.sin(E)
    r = a * (1 - e * cE)
    rp = np.array([a * (cE - e), a * math.sqrt(1 - e * e) * sE, 0.0])
    vp = math.sqrt(mu * a) / r * np.array([-sE, math.sqrt(1 - e * e) * cE, 0.0])
    cO, sO, ci, si, cw, sw = math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc), math.cos(argp), math.sin(argp)
    Rm = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                   [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                   [sw * si, cw * si, ci]])
    return Rm @ rp, Rm @ vp


def initial_conditions(cfg: Config, seed: int) -> dict:
    """Start positions and velocities of the Sun, Earth, Moon and Mars for one mission (seeded)."""
    rng = np.random.default_rng(seed)
    dL = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
    moon_phase = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
    m_e = M_EARTH + cfg.asteroid_mass_kg
    mu_h = GM_SUN + G * (m_e + M_MOON)
    varpi, L0 = math.radians(102.93768193), math.radians(100.46457166)
    r_emb, v_emb = elements_to_state(1.00000261 * AU, 0.01671123, 0.0, 0.0, varpi, (L0 - varpi + dL) % (2 * math.pi), mu_h)
    r_rel, v_rel = elements_to_state(MOON_A, 0.0549, math.radians(5.145), math.radians(125.08),
                                     math.radians(318.15), moon_phase, G * (m_e + M_MOON))
    # Mars: J2000 orbit with its own random orbital phase (third random draw, so Earth/Moon phases are unchanged)
    mars_dL = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
    a_m, e_m, i_m, node_m, varpi_m, L_m = MARS_EL
    node_m, varpi_m = math.radians(node_m), math.radians(varpi_m)
    r_mars, v_mars = elements_to_state(a_m * AU, e_m, math.radians(i_m), node_m, varpi_m - node_m,
                                       (math.radians(L_m) - varpi_m + mars_dL) % (2 * math.pi), GM_SUN + GM_MARS)
    # Mercury, Venus, Jupiter: J2000 orbits, each with its own random phase (drawn after the earlier draws, so the
    # Earth / Moon / Mars phases of a seed are unchanged)
    r_pl, v_pl, ph_pl = np.zeros((N_PL, 3)), np.zeros((N_PL, 3)), np.zeros(N_PL)
    for k in range(N_PL):
        ph_pl[k] = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
        a_k, e_k, i_k, node_k, varpi_k, L_k = PLANET_EL[k]
        node_k, varpi_k = math.radians(node_k), math.radians(varpi_k)
        r_pl[k], v_pl[k] = elements_to_state(a_k * AU, e_k, math.radians(i_k), node_k, varpi_k - node_k,
                                             (math.radians(L_k) - varpi_k + ph_pl[k]) % (2 * math.pi), GM_SUN + PLANET_GM[k])
    return dict(r_emb=r_emb, v_emb=v_emb, r_rel=r_rel, v_rel=v_rel, m_earth=m_e, seed=seed,
                phase_offset=dL, moon_phase=moon_phase, r_mars=r_mars, v_mars=v_mars, mars_phase=mars_dL,
                r_pl=r_pl, v_pl=v_pl, pl_phase=ph_pl)


# =============================================================================================
# 5. NUMBA KERNELS
# =============================================================================================
@njit(cache=True, nogil=True)
def _stumpC(z):
    """Stumpff function C(z) used by the universal-variable Kepler solver."""
    if z > 1e-8:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z)
    return 0.5 - z / 24.0


@njit(cache=True, nogil=True)
def _stumpS(z):
    """Stumpff function S(z) used by the universal-variable Kepler solver."""
    if z > 1e-8:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s * s * s)
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s * s * s)
    return 1.0 / 6.0 - z / 120.0


@njit(cache=True, nogil=True)
def kepler_step(r, v, mu, dt):
    """Exact two-body propagation (universal variables), in place."""
    r0 = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v0sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    smu = math.sqrt(mu)
    alpha = 2.0 / r0 - v0sq / mu
    x = smu * abs(alpha) * dt if alpha > 1e-30 else smu * dt / r0
    for _ in range(80):
        z = alpha * x * x
        C = _stumpC(z)
        S = _stumpS(z)
        F = rv / smu * x * x * C + (1.0 - alpha * r0) * x * x * x * S + r0 * x - smu * dt
        dF = rv / smu * x * (1.0 - z * S) + (1.0 - alpha * r0) * x * x * C + r0
        dx = F / dF
        x -= dx
        if abs(dx) <= 1e-14 * abs(x) + 1e-12:
            break
    z = alpha * x * x
    C = _stumpC(z)
    S = _stumpS(z)
    f = 1.0 - x * x / r0 * C
    g = dt - x * x * x / smu * S
    nr0 = f * r[0] + g * v[0]
    nr1 = f * r[1] + g * v[1]
    nr2 = f * r[2] + g * v[2]
    rn = math.sqrt(nr0 * nr0 + nr1 * nr1 + nr2 * nr2)
    fd = smu / (rn * r0) * x * (z * S - 1.0)
    gd = 1.0 - x * x / rn * C
    v0, v1, v2 = v[0], v[1], v[2]
    v[0] = fd * r[0] + gd * v0
    v[1] = fd * r[1] + gd * v1
    v[2] = fd * r[2] + gd * v2
    r[0], r[1], r[2] = nr0, nr1, nr2


@njit(cache=True, nogil=True)
def orbit_elements(r, v, mu):
    """Returns (a, e, inc, specific_energy). a = inf for unbound orbits."""
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    hm = math.sqrt(hx * hx + hy * hy + hz * hz)
    rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    eps = 0.5 * vsq - mu / rm
    ex = (v[1] * hz - v[2] * hy) / mu - r[0] / rm
    ey = (v[2] * hx - v[0] * hz) / mu - r[1] / rm
    ez = (v[0] * hy - v[1] * hx) / mu - r[2] / rm
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    inc = math.acos(max(-1.0, min(1.0, hz / hm))) if hm > 0 else 0.0
    a = -mu / (2.0 * eps) if eps < 0 else np.inf
    return a, e, inc, eps


@njit(cache=True, nogil=True)
def rtn_to_inertial(r, v, d_rtn):
    """Turn a direction given as (radial, along-track, normal) into the inertial frame."""
    rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    R = r / rm
    h = np.cross(r, v)
    N = h / math.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
    T = np.cross(N, R)
    return d_rtn[0] * R + d_rtn[1] * T + d_rtn[2] * N


@njit(cache=True, nogil=True)
def eject_core(M, m_ast, E_rem, dm_req, u, eta, r_launch, phi_extra):
    """Shared launch physics.  Returns (dm, v_inf, E_draw, E_kin, fell_back).
    phi_extra = additional gravitational potential depth [J/kg] the packet must climb out of
    (fast model: the Moon, G M_moon / d_EM; high-fidelity: 0 because the Moon is explicit).
    Two-body Earth(+reservoir) / packet momentum exchange with reduced mass mu = dm(M-dm)/M:
    kinetic energy given by the launcher in the centre-of-mass frame E_kin = mu u^2 / 2,
    energy drawn from the budget E_kin / eta.  Packets slower than escape speed fall back."""
    dm = min(dm_req, m_ast)
    if dm <= 0.0 or E_rem <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mu_red = dm * (M - dm) / M
    E_kin = 0.5 * mu_red * u * u
    E_draw = E_kin / eta
    if E_draw > E_rem:
        dm = dm * E_rem / E_draw
        mu_red = dm * (M - dm) / M
        E_kin = 0.5 * mu_red * u * u
        E_draw = min(E_kin / eta, E_rem)
    if dm <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    vesc2 = 2.0 * G * M / r_launch + 2.0 * phi_extra
    if u * u <= vesc2:
        return dm, 0.0, E_draw, E_kin, 1
    return dm, math.sqrt(u * u - vesc2), E_draw, E_kin, 0


# ---------------------------------------------------------------- Mars: forecast and shield ---
@njit(cache=True, nogil=True)
def _closest_on_chord(a0, a1, a2, b0, b1, b2):
    """Smallest |rel| on the straight chord from relative position a to b (relative motion is nearly straight
    over one sub-step near a close pass).  Returns (distance, fraction of the step)."""
    d0, d1, d2 = b0 - a0, b1 - a1, b2 - a2
    dd = d0 * d0 + d1 * d1 + d2 * d2
    ts = 0.0
    if dd > 0.0:
        ts = -(a0 * d0 + a1 * d1 + a2 * d2) / dd
        ts = min(1.0, max(0.0, ts))
    c0, c1, c2 = a0 + ts * d0, a1 + ts * d1, a2 + ts * d2
    return math.sqrt(c0 * c0 + c1 * c1 + c2 * c2), ts


@njit(cache=True, nogil=True)
def _pair_kick(rE, vE, rM, vM, gm_e, gm_m, h):
    """Earth(-Moon) <-> Mars gravity kick in heliocentric coordinates (direct + indirect terms)."""
    d0, d1, d2 = rM[0] - rE[0], rM[1] - rE[1], rM[2] - rE[2]
    d2s = d0 * d0 + d1 * d1 + d2 * d2
    id3 = 1.0 / (d2s * math.sqrt(d2s))
    rm2 = rM[0] * rM[0] + rM[1] * rM[1] + rM[2] * rM[2]
    im3 = 1.0 / (rm2 * math.sqrt(rm2))
    re2 = rE[0] * rE[0] + rE[1] * rE[1] + rE[2] * rE[2]
    ie3 = 1.0 / (re2 * math.sqrt(re2))
    for c in range(3):
        dc = rM[c] - rE[c]
        vE[c] += (gm_m * dc * id3 - gm_m * rM[c] * im3) * h
        vM[c] += (-gm_e * dc * id3 - gm_e * rE[c] * ie3) * h


@njit(cache=True, nogil=True)
def mars_forecast(rE0, vE0, muE, rM0, vM0, horizon, dt_event, dv_evt, d_rtn, gm_e, gm_m):
    """Closest Earth-Mars distance over `horizon` seconds if Earth receives a push of dv_evt [m/s] along the
    RTN direction d_rtn at every ejection event (dv_evt = 0: coasting).  Sun gravity exactly (Kepler drifts)
    plus the mutual Earth-Mars pull (kicks), so gravitational focusing during slow passes is included.
    gm_e = G (M_earth+asteroid+Moon), gm_m = G M_mars (0 = massless Mars)."""
    rE = rE0.copy(); vE = vE0.copy(); rM = rM0.copy(); vM = vM0.copy()
    muM = GM_SUN + GM_MARS
    h = 0.5 * dt_event
    n = int(horizon / h) + 1
    best = math.sqrt((rE[0] - rM[0]) ** 2 + (rE[1] - rM[1]) ** 2 + (rE[2] - rM[2]) ** 2)
    for j in range(n):
        if dv_evt > 0.0 and (j % 2) == 1:
            rm_ = math.sqrt(rE[0] * rE[0] + rE[1] * rE[1] + rE[2] * rE[2])
            R0, R1, R2 = rE[0] / rm_, rE[1] / rm_, rE[2] / rm_
            h0 = rE[1] * vE[2] - rE[2] * vE[1]
            h1 = rE[2] * vE[0] - rE[0] * vE[2]
            h2 = rE[0] * vE[1] - rE[1] * vE[0]
            hm = math.sqrt(h0 * h0 + h1 * h1 + h2 * h2)
            N0, N1, N2 = h0 / hm, h1 / hm, h2 / hm
            T0, T1, T2 = N1 * R2 - N2 * R1, N2 * R0 - N0 * R2, N0 * R1 - N1 * R0
            vE[0] += dv_evt * (d_rtn[0] * R0 + d_rtn[1] * T0 + d_rtn[2] * N0)
            vE[1] += dv_evt * (d_rtn[0] * R1 + d_rtn[1] * T1 + d_rtn[2] * N1)
            vE[2] += dv_evt * (d_rtn[0] * R2 + d_rtn[1] * T2 + d_rtn[2] * N2)
        _pair_kick(rE, vE, rM, vM, gm_e, gm_m, 0.5 * h)
        a0, a1, a2 = rE[0] - rM[0], rE[1] - rM[1], rE[2] - rM[2]
        kepler_step(rE, vE, muE, h)
        kepler_step(rM, vM, muM, h)
        _pair_kick(rE, vE, rM, vM, gm_e, gm_m, 0.5 * h)
        d, _ = _closest_on_chord(a0, a1, a2, rE[0] - rM[0], rE[1] - rM[1], rE[2] - rM[2])
        if d < best:
            best = d
    return best


@njit(cache=True, nogil=True)
def shield_decide(rE, vE, muE, M_launch, rM, vM, dm_req, dm_full, u, d_rtn, dt_event, horizon, caution, gm_m):
    """Mars safety shield.  Returns (mode, forecast closest distance [m]).  Modes: 0 keep the agent's command,
    1 coast (no packets), 2 full thrust in the agent's direction, 3 full thrust in the OPPOSITE direction (brake).
    Only acts while Earth's orbit overlaps Mars's orbit (perihelion/aphelion ranges within 0.15 AU).  If the
    agent's command is forecast to stay beyond the caution distance it is kept; otherwise the first option in
    the order coast, full, brake that is forecast to be safe is used (cheapest first); if none is, the option
    with the largest forecast distance."""
    aE, eE, _, _ = orbit_elements(rE, vE, muE)
    aM, eM, _, _ = orbit_elements(rM, vM, GM_SUN + GM_MARS)
    margin = 0.15 * AU
    if not (aE < 1e300) or aE * (1.0 + eE) < aM * (1.0 - eM) - margin or aE * (1.0 - eE) > aM * (1.0 + eM) + margin:
        return 0, 1e300
    vinf2 = u * u - 2.0 * G * M_launch / R_EARTH
    vinf = math.sqrt(vinf2) if vinf2 > 0.0 else 0.0
    m_sys = M_launch + M_MOON
    gm_e = G * m_sys
    dA = mars_forecast(rE, vE, muE, rM, vM, horizon, dt_event, dm_req * vinf / m_sys, d_rtn, gm_e, gm_m)
    if dA >= caution:
        return 0, dA
    best_mode, best = 0, dA
    if dm_req > 0.0:
        dC = mars_forecast(rE, vE, muE, rM, vM, horizon, dt_event, 0.0, d_rtn, gm_e, gm_m)
        if dC >= caution:
            return 1, dC
        if dC > best:
            best_mode, best = 1, dC
    if dm_full > dm_req:
        dF = mars_forecast(rE, vE, muE, rM, vM, horizon, dt_event, dm_full * vinf / m_sys, d_rtn, gm_e, gm_m)
        if dF >= caution:
            return 2, dF
        if dF > best:
            best_mode, best = 2, dF
    dB = mars_forecast(rE, vE, muE, rM, vM, horizon, dt_event, dm_full * vinf / m_sys, -d_rtn, gm_e, gm_m)
    if dB >= caution or dB > best:
        return 3, dB
    return best_mode, best


# ---------------------------------------------------------------- FAST model kernel ----------
@njit(cache=True, nogil=True)
def _fast_eject(st, ledger, k, samples, dm_req, u, d_rtn, eta, r_launch):
    """One packet launch in the FAST model.  The packet leaves EARTH (not the barycentre):
    Earth(+docked asteroid) recoils, the Moon keeps its velocity, and the new Earth-Moon
    barycentre and Earth-Moon relative orbit follow from momentum conservation."""
    M = st[6]
    mm = M_MOON
    r = st[0:3]
    v = st[3:6]
    rrel = st[11:14]
    vrel = st[14:17]
    d_em = math.sqrt(rrel[0] ** 2 + rrel[1] ** 2 + rrel[2] ** 2)
    phi_moon = G * mm / d_em
    dm, vinf, Ed, Ek, fb = eject_core(M, st[7], st[8], dm_req, u, eta, r_launch, phi_moon)
    if dm <= 0.0:
        return
    st[8] -= Ed
    ledger[LG_E_DRAWN] += Ed
    ledger[LG_LAUNCH_KE] += Ek
    samples[k, S_DM] = dm
    samples[k, S_U] = u
    samples[k, S_EDRAW] = Ed
    if fb == 1:
        ledger[LG_N_FALLBACK] += 1.0
        ledger[LG_MASS_FALLBACK] += dm
        samples[k, S_FATE] = 3
        return
    fe = mm / (M + mm)
    fm = M / (M + mm)
    rE = r - fe * rrel
    vE = v - fe * vrel
    rM = r + fm * rrel
    vM = v + fm * vrel
    ehat = -rtn_to_inertial(r, v, d_rtn)
    vp = vE + ((M - dm) / M) * vinf * ehat           # packet velocity after escaping Earth+Moon
    vE2 = vE - (dm / M) * vinf * ehat                # Earth(+asteroid) recoil
    M2 = M - dm
    v_old = v.copy()
    for c in range(3):
        st[0 + c] = (M2 * rE[c] + mm * rM[c]) / (M2 + mm)
        st[3 + c] = (M2 * vE2[c] + mm * vM[c]) / (M2 + mm)
        st[14 + c] = vM[c] - vE2[c]
    st[6] = M2
    st[7] -= dm
    mu_red = dm * (M - dm) / M
    ledger[LG_MASS_EJECTED] += dm
    ledger[LG_N_PACKETS] += 1.0
    for c in range(3):
        ledger[LG_PKT_PX + c] += dm * vp[c]
    imp = mu_red * vinf
    ledger[LG_IMPULSE_TOTAL] += imp
    if dm * u > ledger[LG_MAX_IMPULSE]:
        ledger[LG_MAX_IMPULSE] = dm * u
    ledger[LG_DE_EJECT_THEORY] += Ek - G * dm * (M - dm) / r_launch - dm * phi_moon
    rm = math.sqrt(rE[0] * rE[0] + rE[1] * rE[1] + rE[2] * rE[2])
    if 0.5 * (vp[0] ** 2 + vp[1] ** 2 + vp[2] ** 2) - GM_SUN / rm < 0:
        ledger[LG_PKT_BOUND_SUN_M] += dm
        samples[k, S_FATE] = 2
    else:
        ledger[LG_PKT_UNBOUND_SUN_M] += dm
        samples[k, S_FATE] = 1
    samples[k, S_VREL] = vinf
    dv2 = 0.0
    for c in range(3):
        dv2 += (st[3 + c] - v_old[c]) ** 2
    samples[k, S_DV] = math.sqrt(dv2)
    for c in range(3):
        samples[k, S_VP + c] = vp[c] / 1000.0
        samples[k, S_EHAT + c] = ehat[c]


@njit(cache=True, nogil=True)
def _mars_kick(st, h, gm_m):
    """Velocity kicks over time h from the mutual gravity of the Earth-Moon barycentre (EMB) and Mars,
    written in heliocentric coordinates (so it includes the 'indirect' terms: both bodies also accelerate
    the Sun), plus Mars's tidal kick on the Earth-Moon relative orbit.  gm_m = G M_mars (0 = massless Mars:
    Mars no longer pulls, but it still feels the EMB's pull)."""
    gm_e = G * (st[6] + M_MOON)
    d0, d1, d2 = st[17] - st[0], st[18] - st[1], st[19] - st[2]
    d2s = d0 * d0 + d1 * d1 + d2 * d2
    id3 = 1.0 / (d2s * math.sqrt(d2s))
    rm2 = st[17] * st[17] + st[18] * st[18] + st[19] * st[19]
    im3 = 1.0 / (rm2 * math.sqrt(rm2))
    re2 = st[0] * st[0] + st[1] * st[1] + st[2] * st[2]
    ie3 = 1.0 / (re2 * math.sqrt(re2))
    dv2 = 0.0
    for c in range(3):
        aE = gm_m * (st[17 + c] - st[c]) * id3 - gm_m * st[17 + c] * im3
        aM = -gm_e * (st[17 + c] - st[c]) * id3 - gm_e * st[c] * ie3
        st[3 + c] += aE * h
        st[20 + c] += aM * h
        dv2 += (aE * h) ** 2
    st[27] += math.sqrt(dv2)
    if gm_m > 0.0:
        M = st[6]
        fe = M_MOON / (M + M_MOON)
        fm = M / (M + M_MOON)
        e0, e1, e2 = st[17] - (st[0] - fe * st[11]), st[18] - (st[1] - fe * st[12]), st[19] - (st[2] - fe * st[13])
        m0, m1, m2 = st[17] - (st[0] + fm * st[11]), st[18] - (st[1] + fm * st[12]), st[19] - (st[2] + fm * st[13])
        de = e0 * e0 + e1 * e1 + e2 * e2
        dmm = m0 * m0 + m1 * m1 + m2 * m2
        ide, idm = 1.0 / (de * math.sqrt(de)), 1.0 / (dmm * math.sqrt(dmm))
        st[14] += gm_m * (m0 * idm - e0 * ide) * h
        st[15] += gm_m * (m1 * idm - e1 * ide) * h
        st[16] += gm_m * (m2 * idm - e2 * ide) * h


@njit(cache=True, nogil=True)
def _planet_kicks(st, pl, gpl, gm_m, h):
    """Velocity kicks over time h between the other planets (Mercury, Venus, Jupiter = rows of pl) and the Earth-Moon
    barycentre (EMB) and Mars, and among the three planets themselves. Heliocentric coordinates with direct + indirect
    terms (same form as _mars_kick), so every pair pulls both ways and all orbits change. gpl = their G M
    (all 0 = planets switched off). The EMB-Mars pair is handled by _mars_kick. pl[k, 8] sums the kicks planet k
    gives the EMB [m/s]. Not included in the FAST model (but in the high-fidelity one): the planets' tide on the
    Moon's orbit around Earth (at least ~1e-6 of Earth's pull on the Moon)."""
    rb = np.empty((2 + N_PL, 3))
    dv = np.zeros((2 + N_PL, 3))
    gm = np.empty(2 + N_PL)
    for c in range(3):
        rb[0, c] = st[c]
        rb[1, c] = st[17 + c]
    gm[0] = G * (st[6] + M_MOON)
    gm[1] = gm_m
    for k in range(N_PL):
        gm[2 + k] = gpl[k]
        for c in range(3):
            rb[2 + k, c] = pl[k, c]
    ir3 = np.empty(2 + N_PL)
    for i in range(2 + N_PL):
        r2 = rb[i, 0] * rb[i, 0] + rb[i, 1] * rb[i, 1] + rb[i, 2] * rb[i, 2]
        ir3[i] = 1.0 / (r2 * math.sqrt(r2))
    for j in range(2, 2 + N_PL):
        for i in range(j):
            d0, d1, d2 = rb[j, 0] - rb[i, 0], rb[j, 1] - rb[i, 1], rb[j, 2] - rb[i, 2]
            d2s = d0 * d0 + d1 * d1 + d2 * d2
            id3 = 1.0 / (d2s * math.sqrt(d2s))
            e2 = 0.0
            for c in range(3):
                dc = rb[j, c] - rb[i, c]
                ai = gm[j] * (dc * id3 - rb[j, c] * ir3[j])
                aj = gm[i] * (-dc * id3 - rb[i, c] * ir3[i])
                dv[i, c] += ai * h
                dv[j, c] += aj * h
                e2 += (ai * h) ** 2
            if i == 0:
                pl[j - 2, 8] += math.sqrt(e2)
    for c in range(3):
        st[3 + c] += dv[0, c]
        st[20 + c] += dv[1, c]
        for k in range(N_PL):
            pl[k, 3 + c] += dv[2 + k, c]


@njit(cache=True, nogil=True)
def _fast_drift(st, T, gm_m, t0, pl, gpl):
    """Advance the fast model by T seconds: exact Kepler drifts (EMB about the Sun, Moon about Earth, Mars about
    the Sun) alternated with Earth-Mars gravity kicks (kick-drift-kick).  The number of sub-steps adapts to the
    Earth-Mars distance so that close passes are resolved.  Tracks the closest Earth-Mars distance in st[23]
    (time st[24]).  Returns 3 if Earth and Mars collide, else 0."""
    d0, d1, d2 = st[17] - st[0], st[18] - st[1], st[19] - st[2]
    dist = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    w0, w1, w2 = st[20] - st[3], st[21] - st[4], st[22] - st[5]
    vrel = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
    n = int(math.ceil(T * vrel / (0.05 * dist) - 1e-12))
    pon = gpl[0] + gpl[1] + gpl[2] > 0.0
    if pon:                       # other planets pull: resolve Mercury's / Venus's motion and close-ish passes
        n = max(n, int(math.ceil(T / PL_SUBSTEP_S - 1e-12)))
        for k in range(N_PL):
            q0, q1, q2 = pl[k, 0] - st[0], pl[k, 1] - st[1], pl[k, 2] - st[2]
            u0, u1, u2 = pl[k, 3] - st[3], pl[k, 4] - st[4], pl[k, 5] - st[5]
            dk = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2)
            n = max(n, int(math.ceil(T * math.sqrt(u0 * u0 + u1 * u1 + u2 * u2) / (0.2 * dk) - 1e-12)))
    n = min(max(n, 1), 4000)
    h = T / n
    pre = np.empty((N_PL, 3))
    r = st[0:3]
    v = st[3:6]
    rrel = st[11:14]
    vrel_m = st[14:17]
    rm = st[17:20]
    vm = st[20:23]
    for j in range(n):
        _mars_kick(st, 0.5 * h, gm_m)
        if pon:
            _planet_kicks(st, pl, gpl, gm_m, 0.5 * h)
        a0, a1, a2 = r[0] - rm[0], r[1] - rm[1], r[2] - rm[2]
        for k in range(N_PL):
            for c in range(3):
                pre[k, c] = r[c] - pl[k, c]
        mu_m = G * (st[6] + M_MOON)
        kepler_step(r, v, GM_SUN + mu_m, h)
        kepler_step(rrel, vrel_m, mu_m, h)
        kepler_step(rm, vm, GM_SUN + GM_MARS, h)
        for k in range(N_PL):
            kepler_step(pl[k, 0:3], pl[k, 3:6], GM_SUN + PLANET_GM[k], h)
        d, ts = _closest_on_chord(a0, a1, a2, r[0] - rm[0], r[1] - rm[1], r[2] - rm[2])
        if d < st[23]:
            st[23] = d
            st[24] = t0 + (j + ts) * h
        if d < R_EARTH + R_MARS:
            return 3
        for k in range(N_PL):
            dk, tk = _closest_on_chord(pre[k, 0], pre[k, 1], pre[k, 2], r[0] - pl[k, 0], r[1] - pl[k, 1], r[2] - pl[k, 2])
            if dk < pl[k, 6]:
                pl[k, 6] = dk
                pl[k, 7] = t0 + (j + tk) * h
            if dk < R_EARTH + PLANET_R[k]:
                return 4
        _mars_kick(st, 0.5 * h, gm_m)
        if pon:
            _planet_kicks(st, pl, gpl, gm_m, 0.5 * h)
    return 0


@njit(cache=True, nogil=True)
def fast_interval(st, ledger, n_events, dt_event, dm_req, u, d_rtn, eta, r_launch, samples, moon, mp, pl, gpl):
    """FAST model (hierarchical two-body): the Earth-Moon barycentre (EMB) moves on an exact
    Kepler orbit about the Sun, and the Moon moves on an exact Kepler orbit about Earth(+docked
    asteroid) under their mutual gravity G (M_earth + m_asteroid + M_moon).  Between packets both
    motions are exact; each packet launch is an instantaneous momentum-conserving event applied to
    EARTH (see _fast_eject).  Neglected here (kept in the high-fidelity model): the Sun's tidal
    perturbation of the Moon's orbit and the packets' gravity after they have escaped.
    MARS: Mars moves on its own Kepler orbit about the Sun; Earth(-Moon) and Mars pull on each other through
    kick-drift-kick steps (see _fast_drift), so BOTH orbits change.  The Mars safety shield (shield_decide)
    may replace the agent's packet size by 0 or the maximum once per check.
    st = [r_emb(3), v_emb(3), m_earth+asteroid, m_ast, E_rem, t, unused, r_rel(3), v_rel(3), r_mars(3),
          v_mars(3), closest Earth-Mars distance, its time, shield mode, shield overrides, sum of Mars kicks on
          the EMB [m/s], last shield forecast [m], events until the next shield check]
    mp = [G M_mars (0 = massless), shield on (1/0), horizon [s], caution distance [m], max packet [kg],
          events per shield check].
    OTHER PLANETS (p35): pl = Mercury, Venus, Jupiter rows [r(3), v(3), closest distance to the EMB, its time,
    sum of kicks on the EMB, unused]; gpl = their G M (0 = switched off: they still move on their orbits but
    nobody feels them). They move on Kepler orbits and pull / are pulled through _planet_kicks.  Status 4 = Earth
    hit one of them.
    (r_rel, v_rel = Moon minus Earth).  If moon has rows, MOON_SUB Moon states per event are
    recorded for the viewer (same format as the high-fidelity model)."""
    r = st[0:3]
    v = st[3:6]
    rrel = st[11:14]
    vrel = st[14:17]
    rec = moon.shape[0] > 0
    half = 0.5 * dt_event
    hs = MOON_SUB // 2
    sub = dt_event / MOON_SUB
    status = 0
    n_done = 0
    rb = np.zeros(3)
    vb = np.zeros(3)
    gm_m = mp[0]
    for k in range(n_events):
        for c in range(10, N_SAMPLE_COLS):
            samples[k, c] = 0.0
        for h in range(2):
            if h == 1:
                vpre = v.copy()
                dm_eff = dm_req
                if mp[1] > 0.0:
                    if st[29] <= 0.0:
                        mode, dpred = shield_decide(r, v, GM_SUN + G * (st[6] + M_MOON), st[6], st[17:20], st[20:23],
                                                    dm_req, mp[4], u, d_rtn, dt_event, mp[2], mp[3], gm_m)
                        st[25] = mode
                        st[28] = dpred
                        st[29] = mp[5]
                    st[29] -= 1.0
                    if st[25] == 1.0:
                        dm_eff = 0.0
                    elif st[25] >= 2.0:
                        dm_eff = mp[4]
                    if dm_eff != dm_req or st[25] == 3.0:
                        st[26] += 1.0
                        ledger[LG_SHIELD_OVERRIDES] += 1.0
                if dm_eff > 0.0:
                    _fast_eject(st, ledger, k, samples, dm_eff, u, -d_rtn if st[25] == 3.0 and mp[1] > 0.0 else d_rtn,
                                eta, r_launch)
                # record the state AT the ejection instant (position continuous; velocity jumps)
                samples[k, S_T] = (st[9] + half) / YEAR
                for c in range(3):
                    samples[k, S_R + c] = r[c] / AU
                    samples[k, S_VPRE + c] = vpre[c] / 1000.0
                    samples[k, S_VPOST + c] = v[c] / 1000.0
                samples[k, S_MAST] = st[7]
                samples[k, S_EREM] = st[8]
                for c in range(3):
                    samples[k, S_MARS + c] = st[17 + c] / AU
                    samples[k, S_MARSV + c] = st[20 + c] / 1000.0
                samples[k, S_DMARS] = math.sqrt((st[17] - r[0]) ** 2 + (st[18] - r[1]) ** 2 + (st[19] - r[2]) ** 2) / AU
                samples[k, S_SHIELD] = st[25] if mp[1] > 0.0 else 0.0
                for q in range(N_PL):
                    for c in range(3):
                        samples[k, S_PL + 3 * q + c] = pl[q, c] / AU
                        samples[k, S_PLV + 3 * q + c] = pl[q, 3 + c] / 1000.0
                    samples[k, S_DPL + q] = math.sqrt((pl[q, 0] - r[0]) ** 2 + (pl[q, 1] - r[1]) ** 2 + (pl[q, 2] - r[2]) ** 2) / AU
            mu_m = G * (st[6] + M_MOON)
            if rec:      # viewer samples: exact Kepler from the half-interval start (state untouched)
                for j in range(hs):
                    for c in range(3):
                        rb[c] = rrel[c]
                        vb[c] = vrel[c]
                    kepler_step(rb, vb, mu_m, (j + 1) * sub)
                    row = k * MOON_SUB + h * hs + j
                    moon[row, 0] = (st[9] + h * half + (j + 1) * sub) / YEAR
                    for c in range(3):
                        moon[row, 1 + c] = rb[c]
                        moon[row, 4 + c] = vb[c]
            dstat = _fast_drift(st, half, gm_m, st[9] + h * half, pl, gpl)
            if dstat != 0:
                status = dstat
        st[9] += dt_event
        if status == 3 or status == 4:
            n_done = k + 1
            break
        n_done = k + 1
        mu = GM_SUN + G * (st[6] + M_MOON)
        rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if rm < 0.2 * AU:
            status = 2
            break
        if rm > 30.0 * AU or 0.5 * vsq - mu / rm >= 0.0:
            status = 1
            break
    return n_done, status


# ---------------------------------------------------------------- HIGH-FIDELITY kernels ------
# bodies: 0 Sun, 1 Earth (+ docked reservoir), 2 Moon, 3 Mars, 4.. fragments (NMAJ = 4 major bodies).
# Pairs: every major body with every body; fragment-fragment gravity is neglected (documented).
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = -(2.0 ** (1.0 / 3.0)) * _W1
YC1 = _W1 / 2.0
YC2 = (_W0 + _W1) / 2.0
YD1 = _W1
YD2 = _W0


@njit(cache=True, nogil=True)
def nb_accel(pos, mass, nb, acc):
    """Gravitational acceleration of every body from every other body (direct N-body sum)."""
    for i in range(nb):
        acc[i, 0] = 0.0
        acc[i, 1] = 0.0
        acc[i, 2] = 0.0
    for i in range(min(NMAJ, nb)):
        for j in range(i + 1, nb):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            inv3 = 1.0 / (r2 * math.sqrt(r2))
            fi = G * mass[j] * inv3
            fj = G * mass[i] * inv3
            acc[i, 0] += fi * dx
            acc[i, 1] += fi * dy
            acc[i, 2] += fi * dz
            acc[j, 0] -= fj * dx
            acc[j, 1] -= fj * dy
            acc[j, 2] -= fj * dz


@njit(cache=True, nogil=True)
def yoshida_step(pos, vel, mass, nb, h, acc):
    """Advance all bodies by one time step with the 4th-order Yoshida symplectic integrator."""
    for i in range(nb):
        for c in range(3):
            pos[i, c] += YC1 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD1 * h * acc[i, c]
            pos[i, c] += YC2 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD2 * h * acc[i, c]
            pos[i, c] += YC2 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD1 * h * acc[i, c]
            pos[i, c] += YC1 * h * vel[i, c]


@njit(cache=True, nogil=True)
def nb_totals(pos, vel, mass, nb, out):
    """out = [E, Px, Py, Pz, Lx, Ly, Lz, M] with the same pair set as nb_accel."""
    for k in range(8):
        out[k] = 0.0
    for i in range(nb):
        m = mass[i]
        out[0] += 0.5 * m * (vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
        for c in range(3):
            out[1 + c] += m * vel[i, c]
        out[4] += m * (pos[i, 1] * vel[i, 2] - pos[i, 2] * vel[i, 1])
        out[5] += m * (pos[i, 2] * vel[i, 0] - pos[i, 0] * vel[i, 2])
        out[6] += m * (pos[i, 0] * vel[i, 1] - pos[i, 1] * vel[i, 0])
        out[7] += m
    for i in range(min(NMAJ, nb)):
        for j in range(i + 1, nb):
            d = math.sqrt((pos[j, 0] - pos[i, 0]) ** 2 + (pos[j, 1] - pos[i, 1]) ** 2 + (pos[j, 2] - pos[i, 2]) ** 2)
            out[0] -= G * mass[i] * mass[j] / d


@njit(cache=True, nogil=True)
def _record_event(before, after, ledger, slot_e):
    """Book an impact / merge event: change of energy and momentum before vs. after."""
    de = after[0] - before[0]
    ledger[LG_EV_DE] += de
    if slot_e >= 0:
        ledger[slot_e] += de
    for c in range(3):
        ledger[LG_EV_DP + c] += after[1 + c] - before[1 + c]
        ledger[LG_EV_DL + c] += after[4 + c] - before[4 + c]
    return de


@njit(cache=True, nogil=True)
def _remove(pos, vel, mass, nbarr, k):
    """Delete body k from the N-body arrays (after it merged into another body)."""
    last = nbarr[0] - 1
    if k != last:
        for c in range(3):
            pos[k, c] = pos[last, c]
            vel[k, c] = vel[last, c]
        mass[k] = mass[last]
    mass[last] = 0.0
    nbarr[0] = last


@njit(cache=True, nogil=True)
def _merge_into(pos, vel, mass, nbarr, k, b, ledger, tb, ta):
    """Perfectly inelastic impact of fragment k on body b (mass and momentum conserved)."""
    nb_totals(pos, vel, mass, nbarr[0], tb)
    mf, mb = mass[k], mass[b]
    for c in range(3):
        vel[b, c] = (mb * vel[b, c] + mf * vel[k, c]) / (mb + mf)
    mass[b] = mb + mf
    if b == 1:
        ledger[LG_EARTH_IMPACT_M] += mf
    elif b == 2:
        ledger[LG_MOON_IMPACT_M] += mf
    elif b == 3:
        ledger[LG_MARS_IMPACT_M] += mf
    elif b >= 4:
        ledger[LG_PLANET_IMPACT_M] += mf
    else:
        ledger[LG_SUN_IMPACT_M] += mf
    _remove(pos, vel, mass, nbarr, k)
    nb_totals(pos, vel, mass, nbarr[0], ta)
    _record_event(tb, ta, ledger, LG_DE_IMPACT)


@njit(cache=True, nogil=True)
def _merge_closest_fragments(pos, vel, mass, nbarr, ledger, tb, ta):
    """When the fragment slots are full, merge the two closest fragments (momentum conserved)."""
    nb = nbarr[0]
    bi, bj, best = -1, -1, 1e300
    for i in range(NMAJ, nb):
        for j in range(i + 1, nb):
            d = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2 + (pos[i, 2] - pos[j, 2]) ** 2
            if d < best:
                best, bi, bj = d, i, j
    if bi < 0:
        return
    nb_totals(pos, vel, mass, nb, tb)
    mi, mj = mass[bi], mass[bj]
    for c in range(3):
        pos[bi, c] = (mi * pos[bi, c] + mj * pos[bj, c]) / (mi + mj)
        vel[bi, c] = (mi * vel[bi, c] + mj * vel[bj, c]) / (mi + mj)
    mass[bi] = mi + mj
    _remove(pos, vel, mass, nbarr, bj)
    nb_totals(pos, vel, mass, nbarr[0], ta)
    ledger[LG_N_MERGES] += 1.0
    _record_event(tb, ta, ledger, LG_DE_MERGE)


@njit(cache=True, nogil=True)
def hifi_advance(pos, vel, mass, nbarr, ledger, T, dt_max, prev, acc, esc_r, tb, ta, mt):
    """mt = [closest Earth-Mars distance, its time, current time, closest Earth-Mercury / -Venus / -Jupiter distance,
    their times] (updated every step; Mars = body 3, Mercury 4, Venus 5, Jupiter 6)."""
    n = max(1, int(math.ceil(T / dt_max - 1e-9)))
    h = T / n
    radii = np.array([R_SUN, R_EARTH, R_MOON, R_MARS, R_MERCURY, R_VENUS, R_JUPITER])
    pre = np.empty((N_PL, 3))
    for _ in range(n):
        nb = nbarr[0]
        if nb > NMAJ:
            for i in range(nb):
                for c in range(3):
                    prev[i, c] = pos[i, c]
        a0, a1, a2 = pos[1, 0] - pos[3, 0], pos[1, 1] - pos[3, 1], pos[1, 2] - pos[3, 2]
        for q in range(N_PL):
            for c in range(3):
                pre[q, c] = pos[1, c] - pos[4 + q, c]
        yoshida_step(pos, vel, mass, nb, h, acc)
        d, ts = _closest_on_chord(a0, a1, a2, pos[1, 0] - pos[3, 0], pos[1, 1] - pos[3, 1], pos[1, 2] - pos[3, 2])
        if d < mt[0]:
            mt[0] = d
            mt[1] = mt[2] + ts * h
        for q in range(N_PL):
            d, ts = _closest_on_chord(pre[q, 0], pre[q, 1], pre[q, 2], pos[1, 0] - pos[4 + q, 0],
                                      pos[1, 1] - pos[4 + q, 1], pos[1, 2] - pos[4 + q, 2])
            if d < mt[3 + q]:
                mt[3 + q] = d
                mt[3 + N_PL + q] = mt[2] + ts * h
        mt[2] += h
        if nb > NMAJ:
            k = NMAJ
            while k < nbarr[0]:
                hit = -1
                for b in range(NMAJ):
                    r00 = prev[k, 0] - prev[b, 0]
                    r01 = prev[k, 1] - prev[b, 1]
                    r02 = prev[k, 2] - prev[b, 2]
                    d0 = (pos[k, 0] - pos[b, 0]) - r00
                    d1 = (pos[k, 1] - pos[b, 1]) - r01
                    d2 = (pos[k, 2] - pos[b, 2]) - r02
                    dd = d0 * d0 + d1 * d1 + d2 * d2
                    ts = 0.0
                    if dd > 0:
                        ts = -(r00 * d0 + r01 * d1 + r02 * d2) / dd
                        ts = min(1.0, max(0.0, ts))
                    cx, cy, cz = r00 + ts * d0, r01 + ts * d1, r02 + ts * d2
                    if cx * cx + cy * cy + cz * cz < radii[b] * radii[b]:
                        hit = b
                        break
                if hit >= 0:
                    _merge_into(pos, vel, mass, nbarr, k, hit, ledger, tb, ta)
                else:
                    k += 1
    # escape bookkeeping: unbound fragments beyond esc_r leave the simulated subsystem
    k = NMAJ
    while k < nbarr[0]:
        rx, ry, rz = pos[k, 0] - pos[0, 0], pos[k, 1] - pos[0, 1], pos[k, 2] - pos[0, 2]
        vx, vy, vz = vel[k, 0] - vel[0, 0], vel[k, 1] - vel[0, 1], vel[k, 2] - vel[0, 2]
        rr = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rr > esc_r and 0.5 * (vx * vx + vy * vy + vz * vz) - GM_SUN / rr > 0:
            nb_totals(pos, vel, mass, nbarr[0], tb)
            ledger[LG_ESC_MASS] += mass[k]
            _remove(pos, vel, mass, nbarr, k)
            nb_totals(pos, vel, mass, nbarr[0], ta)
            de = _record_event(tb, ta, ledger, -1)
            ledger[LG_ESC_E] -= de
            for c in range(3):
                ledger[LG_ESC_PX + c] -= ta[1 + c] - tb[1 + c]
        else:
            k += 1


@njit(cache=True, nogil=True)
def eject_hifi(pos, vel, mass, nbarr, stv, ledger, dm_req, u, d_rtn, eta, r_launch, r_release, tb, ta, pk):
    """pk (output, len 12): dm, u, v_rel, vp(3) [barycentric m/s], fate, E_draw, ehat(3), unused."""
    nmax = mass.shape[0]
    M = mass[1]
    for c in range(pk.shape[0]):
        pk[c] = 0.0
    dm, vinf, Ed, Ek, fb = eject_core(M, stv[0], stv[1], dm_req, u, eta, r_launch, 0.0)
    if dm <= 0.0:
        return
    pk[0], pk[1], pk[7] = dm, u, Ed
    stv[1] -= Ed
    ledger[LG_E_DRAWN] += Ed
    ledger[LG_LAUNCH_KE] += Ek
    if fb == 1:
        ledger[LG_N_FALLBACK] += 1.0
        ledger[LG_MASS_FALLBACK] += dm
        pk[6] = 3
        return
    if nbarr[0] >= nmax:
        _merge_closest_fragments(pos, vel, mass, nbarr, ledger, tb, ta)
    me, mm = mass[1], mass[2]
    remb = (me * pos[1] + mm * pos[2]) / (me + mm) - pos[0]
    vemb = (me * vel[1] + mm * vel[2]) / (me + mm) - vel[0]
    ehat = -rtn_to_inertial(remb, vemb, d_rtn)
    nb_totals(pos, vel, mass, nbarr[0], tb)
    s = pos[2] - pos[1]
    p = s[0] * ehat[0] + s[1] * ehat[1] + s[2] * ehat[2]
    perp = s - p * ehat
    hit_moon = (p > r_launch) and (p < r_release) and \
        (math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2) < R_MOON)
    rr = p if hit_moon else r_release
    vrel2 = u * u - 2.0 * G * M * (1.0 / r_launch - 1.0 / rr)
    vrel = math.sqrt(max(vrel2, 0.0))
    v1 = vel[1].copy()
    p1 = pos[1].copy()
    vp = v1 + ((M - dm) / M) * vrel * ehat
    for c in range(3):
        vel[1, c] = v1[c] - (dm / M) * vrel * ehat[c]
    mass[1] = M - dm
    stv[0] -= dm
    pk[2] = vrel
    for c in range(3):
        pk[3 + c] = vp[c]
        pk[8 + c] = ehat[c]
    if hit_moon:
        pk[6] = 4
    else:
        rh = p1 + r_release * ehat - pos[0]
        vh = vp - vel[0]
        if 0.5 * (vh[0] ** 2 + vh[1] ** 2 + vh[2] ** 2) - GM_SUN / math.sqrt(rh[0] ** 2 + rh[1] ** 2 + rh[2] ** 2) < 0:
            pk[6] = 2
        else:
            pk[6] = 1
    if hit_moon:
        for c in range(3):
            vel[2, c] = (mm * vel[2, c] + dm * vp[c]) / (mm + dm)
        mass[2] = mm + dm
        ledger[LG_N_LAUNCH_MOON_HITS] += 1.0
        ledger[LG_MOON_IMPACT_M] += dm
    else:
        k = nbarr[0]
        nbarr[0] = k + 1
        for c in range(3):
            pos[k, c] = p1[c] + ((M - dm) / M) * r_release * ehat[c]
            pos[1, c] = p1[c] - (dm / M) * r_release * ehat[c]
            vel[k, c] = vp[c]
        mass[k] = dm
    nb_totals(pos, vel, mass, nbarr[0], ta)
    _record_event(tb, ta, ledger, LG_DE_EJECT)
    mu_red = dm * (M - dm) / M
    ledger[LG_MASS_EJECTED] += dm
    ledger[LG_N_PACKETS] += 1.0
    ledger[LG_IMPULSE_TOTAL] += mu_red * vinf
    if dm * u > ledger[LG_MAX_IMPULSE]:
        ledger[LG_MAX_IMPULSE] = dm * u
    ledger[LG_DE_EJECT_THEORY] += Ek - G * dm * (M - dm) / r_launch
    for c in range(3):
        ledger[LG_PKT_PX + c] += dm * vp[c]


@njit(cache=True, nogil=True)
def hifi_interval(pos, vel, mass, nbarr, stv, ledger, n_events, dt_event, dt_max, dm_req, u, d_rtn,
                  eta, r_launch, r_release, esc_r, samples, moon, prev, acc, mp):
    """stv = [m_ast, E_rem, t, closest Earth-Mars distance, its time, shield mode, shield overrides, unused,
    last shield forecast, events until next shield check, closest Earth-Mercury / -Venus / -Jupiter distance (3),
    their times (3)].  mp: see fast_interval.  Each event: 4 sub-chunks,
    ejection, 4 sub-chunks; one viewer sample at the ejection instant and MOON_SUB Moon states per event.
    Returns (n_samples, status); status 3 = Earth-Mars collision, 4 = Earth hit Mercury, Venus or Jupiter."""
    tb = np.zeros(8)
    ta = np.zeros(8)
    pk = np.zeros(12)
    mt = np.empty(3 + 2 * N_PL)
    mt[0], mt[1], mt[2] = stv[3], stv[4], stv[2]
    for q in range(2 * N_PL):
        mt[3 + q] = stv[10 + q]
    status = 0
    n_done = 0
    sub = dt_event / MOON_SUB
    for k in range(n_events):
        for j in range(MOON_SUB):
            if j == MOON_SUB // 2:
                vpre = vel[1] - vel[0]
                dm_eff = dm_req
                me, mm = mass[1], mass[2]
                if mp[1] > 0.0:
                    if stv[9] <= 0.0:
                        rE = (me * pos[1] + mm * pos[2]) / (me + mm) - pos[0]
                        vE = (me * vel[1] + mm * vel[2]) / (me + mm) - vel[0]
                        mode, dpred = shield_decide(rE, vE, G * (mass[0] + me + mm), me, pos[3] - pos[0], vel[3] - vel[0],
                                                    dm_req, mp[4], u, d_rtn, dt_event, mp[2], mp[3], G * mass[3])
                        stv[5] = mode
                        stv[8] = dpred
                        stv[9] = mp[5]
                    stv[9] -= 1.0
                    if stv[5] == 1.0:
                        dm_eff = 0.0
                    elif stv[5] >= 2.0:
                        dm_eff = mp[4]
                    if dm_eff != dm_req or stv[5] == 3.0:
                        stv[6] += 1.0
                        ledger[LG_SHIELD_OVERRIDES] += 1.0
                if dm_eff > 0.0:
                    eject_hifi(pos, vel, mass, nbarr, stv, ledger, dm_eff, u,
                               -d_rtn if stv[5] == 3.0 and mp[1] > 0.0 else d_rtn, eta, r_launch, r_release, tb, ta, pk)
                else:
                    for c in range(pk.shape[0]):
                        pk[c] = 0.0
                samples[k, S_T] = (stv[2] + 0.5 * dt_event) / YEAR
                for c in range(3):
                    samples[k, S_R + c] = (pos[1, c] - pos[0, c]) / AU
                    samples[k, S_VPRE + c] = vpre[c] / 1000.0
                    samples[k, S_VPOST + c] = (vel[1, c] - vel[0, c]) / 1000.0
                    samples[k, S_VP + c] = (pk[3 + c] - vel[0, c]) / 1000.0 if pk[0] > 0 and pk[6] != 3 else 0.0
                    samples[k, S_EHAT + c] = pk[8 + c]
                samples[k, S_DM] = pk[0]
                samples[k, S_U] = pk[1]
                samples[k, S_VREL] = pk[2]
                samples[k, S_FATE] = pk[6]
                samples[k, S_EDRAW] = pk[7]
                samples[k, S_MAST] = stv[0]
                samples[k, S_EREM] = stv[1]
                samples[k, S_NFRAG] = nbarr[0] - NMAJ
                for c in range(3):
                    samples[k, S_MARS + c] = (pos[3, c] - pos[0, c]) / AU
                    samples[k, S_MARSV + c] = (vel[3, c] - vel[0, c]) / 1000.0
                samples[k, S_DMARS] = math.sqrt((pos[3, 0] - pos[1, 0]) ** 2 + (pos[3, 1] - pos[1, 1]) ** 2 +
                                                (pos[3, 2] - pos[1, 2]) ** 2) / AU
                samples[k, S_SHIELD] = stv[5] if mp[1] > 0.0 else 0.0
                for q in range(N_PL):
                    for c in range(3):
                        samples[k, S_PL + 3 * q + c] = (pos[4 + q, c] - pos[0, c]) / AU
                        samples[k, S_PLV + 3 * q + c] = (vel[4 + q, c] - vel[0, c]) / 1000.0
                    samples[k, S_DPL + q] = math.sqrt((pos[4 + q, 0] - pos[1, 0]) ** 2 + (pos[4 + q, 1] - pos[1, 1]) ** 2 +
                                                      (pos[4 + q, 2] - pos[1, 2]) ** 2) / AU
                dvx = samples[k, S_VPOST] - samples[k, S_VPRE]
                dvy = samples[k, S_VPOST + 1] - samples[k, S_VPRE + 1]
                dvz = samples[k, S_VPOST + 2] - samples[k, S_VPRE + 2]
                samples[k, S_DV] = 1000.0 * math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
            hifi_advance(pos, vel, mass, nbarr, ledger, sub, dt_max, prev, acc, esc_r, tb, ta, mt)
            row = k * MOON_SUB + j
            moon[row, 0] = (stv[2] + (j + 1) * sub) / YEAR
            for c in range(3):
                moon[row, 1 + c] = pos[2, c] - pos[1, c]
                moon[row, 4 + c] = vel[2, c] - vel[1, c]
        stv[2] += dt_event
        stv[3] = mt[0]
        stv[4] = mt[1]
        for q in range(2 * N_PL):
            stv[10 + q] = mt[3 + q]
        n_done = k + 1
        if mt[0] < R_EARTH + R_MARS:
            status = 3
            break
        hitp = False
        for q in range(N_PL):
            if mt[3 + q] < R_EARTH + PLANET_R[q]:
                hitp = True
        if hitp:
            status = 4
            break
        me, mm = mass[1], mass[2]
        rx = (me * pos[1, 0] + mm * pos[2, 0]) / (me + mm) - pos[0, 0]
        ry = (me * pos[1, 1] + mm * pos[2, 1]) / (me + mm) - pos[0, 1]
        rz = (me * pos[1, 2] + mm * pos[2, 2]) / (me + mm) - pos[0, 2]
        rm = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rm < 0.2 * AU:
            status = 2
            break
        if rm > 30.0 * AU:
            status = 1
            break
    return n_done, status


@njit(cache=True, nogil=True)
def packet_arcs(r0, v0, offsets_s, esc_r, out, nvalid):
    """Viewer helper (Python-side physics): propagate each released packet on its exact two-body
    orbit about the Sun (universal-variable Kepler solution) and store its position at the given
    times after release.  Stops once the packet is beyond esc_r (it has left the tracked region)."""
    for i in range(r0.shape[0]):
        r = r0[i].copy()
        v = v0[i].copy()
        tprev = 0.0
        nvalid[i] = 0
        for j in range(offsets_s.shape[0]):
            if offsets_s[j] > tprev:
                kepler_step(r, v, GM_SUN, offsets_s[j] - tprev)
                tprev = offsets_s[j]
            for c in range(3):
                out[i, j, c] = r[c]
            nvalid[i] = j + 1
            if math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) > esc_r:
                break


# =============================================================================================
# 6. ENGINES (Python wrappers)
# =============================================================================================
def planet_gms(cfg) -> np.ndarray:
    """G M of Mercury, Venus, Jupiter as used by the physics (all 0 when 'other_planets' is off)."""
    return PLANET_GM.copy() if getattr(cfg, "other_planets", True) else np.zeros(N_PL)


def planets_meta(cfg) -> dict:
    """What the viewer needs to draw Mercury, Venus, Jupiter: names, G M [m^3/s^2], safe distances [AU], on/off."""
    lim = planet_safety_limits(cfg)
    return dict(on=bool(getattr(cfg, "other_planets", True)), names=list(PLANET_NAMES), gm=[float(x) for x in PLANET_GM],
                safe_au=[q["d_safe_au"] for q in lim], radius_km=[float(x) / 1e3 for x in PLANET_R])


def planet_safety_limits(cfg) -> list:
    """Safe distance of Earth from Mercury, Venus and Jupiter, with the SAME rules as for Mars (the 'Planetary
    safety' parameters: the planet's tide on Earth <= mars_max_tide_ratio x the Moon's tide, its tidal pull on the
    Moon's orbit <= mars_max_moon_perturbation x Earth's pull, and at least mars_min_distance_km). The tide limits
    scale with (G M)^(1/3), so Jupiter's safe distance is ~14 x Mars's."""
    out = []
    for k in range(N_PL):
        gm = PLANET_GM[k]
        d_tide = MOON_A * (gm / GM_MOON / cfg.mars_max_tide_ratio) ** (1.0 / 3.0)
        d_moon = MOON_A * (2.0 * gm / GM_EARTH / cfg.mars_max_moon_perturbation) ** (1.0 / 3.0)
        d_min = cfg.mars_min_distance_km * 1e3
        d_safe = max(d_tide, d_moon, d_min)
        out.append(dict(name=PLANET_NAMES[k], d_tide_m=d_tide, d_moon_m=d_moon, d_min_m=d_min, d_safe_m=d_safe,
                        d_safe_au=d_safe / AU, d_collision_m=R_EARTH + PLANET_R[k],
                        governing="tide on Earth" if d_safe == d_tide else ("Moon's orbit" if d_safe == d_moon else "hard minimum distance")))
    return out


def mars_params(cfg: Config, shield: bool = True) -> np.ndarray:
    """Parameter vector of the Mars model / safety shield passed to both physics kernels."""
    lim = mars_safety_limits(cfg)
    return np.array([GM_MARS if cfg.mars_gravity else 0.0, 1.0 if (cfg.mars_shield and shield) else 0.0,
                     cfg.shield_horizon_years * YEAR, lim["caution_m"], cfg.max_packet_mass_kg,
                     float(max(1, round(cfg.ejections_per_year)))])


def _planet_elements(r, v, k):
    """Orbit size and shape of Mercury (k=0), Venus (1) or Jupiter (2) from its heliocentric state."""
    a, e, inc, _ = orbit_elements(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64), GM_SUN + PLANET_GM[k])
    return dict(a_au=a / AU if math.isfinite(a) else float("inf"), e=e, inc_deg=math.degrees(inc))


def _mars_elements(r, v):
    """Mars's orbital size and shape (a, e) from its position and velocity around the Sun."""
    a, e, inc, _ = orbit_elements(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64), GM_SUN + GM_MARS)
    return dict(a_au=a / AU if math.isfinite(a) else float("inf"), e=e, inc_deg=math.degrees(inc))


class _PowerSupply:
    """Energy supply shared by both engines. With the solar collectors on, the beam energy that the collectors can deliver
    during a decision interval (P_elec x duration, P_elec at Earth's distance at the start of the interval) is made available
    to the launches of that interval, on top of the remaining reserve. Launches use beam energy first; whatever beam energy
    is not used is simply not sent (the beams are throttled to demand, nothing is stored). Ledger: beam energy used, reserve
    used, microwave energy absorbed by the rectennas, peak beam power."""

    LEDGER_KEYS = ("sun_on_collectors", "reflected", "collector_heat", "manufacturing", "panel_electricity", "transmitter_heat",
                   "microwaves_sent", "beam_spill", "captured", "rectenna_heat", "arm_electricity", "reserve_used",
                   "launcher_draw", "packet_ke", "launcher_heat", "mirror_sun", "mirror_to_earth", "mirror_to_space",
                   "mirror_heat", "heat_to_earth",
                   "fresh_mass_kg", "built_mass_kg")

    def _supply_init(self):
        """Set up the energy supply: collector belt, beams, starting reserve and the climate state."""
        self.sol = dict(E_beam_used_J=0.0, E_reserve_used_J=0.0, E_offered_J=0.0, E_incident_J=0.0, peak_beam_W=0.0,
                        eta_min=float("nan"), eta_max=float("nan"))
        self.last_power = None
        self.eled = {k: 0.0 for k in self.LEDGER_KEYS}
        self.eled_resid = 0.0
        self.last_eled = None
        self.last_drawn = 0.0

    def interval_power(self, years):
        """Collector power chain for the coming interval (built share averaged over the interval)."""
        r, _ = self.emb()
        return collector_power(self.cfg, r, collector_built_mean(self.cfg, self.t_years, years))

    def settle_interval(self, years, ke_launch, t0, mirror_sun_J=0.0, mirror_earth_J=0.0):
        """Close the energy books of the interval just run.

        Power chain: sunlight -> panels -> microwaves -> arms -> launcher, every loss a named sink.
        Mirrors: sunlight caught by the mirrors = light sent to Earth + light turned past Earth + heat in the mirrors.
        Returns the interval ledger [J] including the relative residual of the balance."""
        cfg = self.cfg
        lp = self.last_power
        dt = years * YEAR
        L = {k: 0.0 for k in self.LEDGER_KEYS}
        drawn = float(lp["E_drawn_J"]) if lp else float(self.last_drawn)
        launch_beam = float(lp["E_beam_used_J"]) if lp else 0.0
        launch_res = drawn - launch_beam
        if cfg.solar_collectors and lp:
            chain = cfg.microwave_efficiency * lp["eta_capture"] * cfg.rectenna_efficiency
            arm_el = launch_beam
            el_used = arm_el / chain if chain > 0 else 0.0
            b0, b1 = collector_built(cfg, t0), collector_built(cfg, t0 + years)
            m_belt = cfg.collector_fraction * 4 * math.pi * (cfg.collector_orbit_au * AU) ** 2 * cfg.collector_areal_density
            new_mass = m_belt * max(0.0, b1 - b0)
            repl_mass = m_belt * lp["built"] * years / max(cfg.collector_lifetime_years, 1e-9)
            sun = lp["L_captured_W"] * dt
            manuf = min(cfg.collector_embodied_energy_jkg * (new_mass + repl_mass), max(0.0, sun * lp["eta_pv_op"] - el_used))
            L.update(sun_on_collectors=sun, reflected=sun * (1 - cfg.collector_absorptance), manufacturing=manuf,
                     panel_electricity=el_used, collector_heat=sun * cfg.collector_absorptance - el_used - manuf,
                     transmitter_heat=el_used * (1 - cfg.microwave_efficiency), microwaves_sent=el_used * cfg.microwave_efficiency)
            L["captured"] = L["microwaves_sent"] * lp["eta_capture"]
            L["beam_spill"] = L["microwaves_sent"] - L["captured"]
            L["arm_electricity"] = L["captured"] * cfg.rectenna_efficiency
            L["rectenna_heat"] = L["captured"] - L["arm_electricity"]
            L["fresh_mass_kg"] = new_mass + repl_mass * (1 - cfg.collector_recycle_frac)
            L["built_mass_kg"] = new_mass + repl_mass
        L["reserve_used"] = launch_res
        L["launcher_draw"] = drawn
        L["packet_ke"] = ke_launch
        L["launcher_heat"] = drawn - ke_launch
        L["mirror_sun"] = mirror_sun_J                                   # sunlight caught by the mirrors
        L["mirror_to_earth"] = mirror_earth_J                            # reflected onto Earth
        L["mirror_heat"] = mirror_sun_J * (1.0 - cfg.mirror_reflectivity)   # absorbed by the film, radiated to space
        L["mirror_to_space"] = mirror_sun_J * cfg.mirror_reflectivity - mirror_earth_J   # reflected past Earth
        L["heat_to_earth"] = cfg.beam_spill_to_earth_frac * L["beam_spill"] + cfg.arm_heat_to_earth_frac * (L["rectenna_heat"] + L["launcher_heat"])
        e_in = L["sun_on_collectors"] + L["reserve_used"] + L["mirror_sun"]
        e_out = (L["reflected"] + L["collector_heat"] + L["manufacturing"] + L["transmitter_heat"] + L["beam_spill"] +
                 L["rectenna_heat"] + L["packet_ke"] + L["launcher_heat"] + L["mirror_to_earth"] + L["mirror_to_space"] +
                 L["mirror_heat"])
        L["residual_rel"] = abs(e_in - e_out) / e_in if e_in > 0 else 0.0
        for k in self.LEDGER_KEYS:
            self.eled[k] += L[k]
        self.eled_resid = max(self.eled_resid, L["residual_rel"])
        self.last_eled = L
        return L

    def _with_supply(self, years, run):
        """Run one interval while charging its launch energy to the beams / reserve (no free energy)."""
        cfg = self.cfg
        if not cfg.solar_collectors or years <= 0:
            self.last_power = None
            d0 = float(self.ledger[LG_E_DRAWN])
            out = run()
            self.last_drawn = float(self.ledger[LG_E_DRAWN]) - d0
            return out
        pw = self.interval_power(years)
        e_sol = pw["P_elec_W"] * years * YEAR
        res0 = float(self.E_rem)
        self._set_E(res0 + e_sol)
        samples, status = run()
        drawn = res0 + e_sol - float(self.E_rem)
        used = min(max(drawn, 0.0), e_sol)
        res_used = max(0.0, drawn - used)
        self._set_E(res0 - res_used)
        if len(samples):
            samples[:, S_EREM] = np.minimum(samples[:, S_EREM], res0)       # the reserve as seen by the viewer
            dt_ev = years * YEAR / max(1, int(round(cfg.ejections_per_year * years)))
            beam_el = np.minimum(samples[:, S_EDRAW], pw["P_elec_W"] * dt_ev)
            peak = float(beam_el.max() / cfg.rectenna_efficiency / dt_ev)
        else:
            dt_ev, peak = years * YEAR, 0.0
        inc = used / cfg.rectenna_efficiency
        S = self.sol
        S["E_beam_used_J"] += used; S["E_reserve_used_J"] += res_used; S["E_offered_J"] += e_sol; S["E_incident_J"] += inc
        S["peak_beam_W"] = max(S["peak_beam_W"], peak)
        S["eta_min"] = pw["eta_capture"] if not S["eta_min"] <= pw["eta_capture"] else S["eta_min"]
        S["eta_max"] = pw["eta_capture"] if not S["eta_max"] >= pw["eta_capture"] else S["eta_max"]
        self.last_power = dict(pw, years=years, dt_event_s=dt_ev, E_offered_J=e_sol, E_beam_used_J=used, E_drawn_J=drawn,
                               E_reserve_used_J=res_used, E_incident_J=inc, mean_beam_W=inc / (years * YEAR), peak_beam_W=peak)
        return samples, status


class FastEngine(_PowerSupply):
    """FAST training model (hierarchical two-body, see fast_interval): EMB on an exact Kepler
    orbit about the Sun + the Moon on an exact Kepler orbit about Earth(+docked asteroid) under
    their mutual gravity.  Ejections are impulsive, act on Earth and conserve momentum exactly.
    Packets are tracked as a ledger (mass, momentum, launch energy, bound/unbound to the Sun);
    their escape from Earth's AND the Moon's gravity is included in their final speed.
    Not modelled here (but in the high-fidelity model): the Sun's tide on the Moon's orbit,
    individual fragment trajectories after escape, packets hitting the Moon, fragment re-impacts."""
    fidelity = "fast"

    def __init__(self, cfg: Config, seed: int, record_moon: bool = True):
        """Fast training model: exact Kepler motion of the Earth-Moon centre and of the Moon around Earth."""
        self.cfg = cfg
        self.record_moon = record_moon
        ic = initial_conditions(cfg, seed)
        self.st = np.zeros(30)
        self.st[0:3], self.st[3:6] = ic["r_emb"], ic["v_emb"]
        self.st[17:20], self.st[20:23] = ic["r_mars"], ic["v_mars"]
        self.st[23] = 1e300
        self.st[6] = ic["m_earth"]
        self.st[7] = cfg.asteroid_mass_kg
        self.st[8] = cfg.energy_budget_j
        self.st[11:14], self.st[14:17] = ic["r_rel"], ic["v_rel"]
        self.pl = np.zeros((N_PL, 10))
        self.pl[:, 0:3], self.pl[:, 3:6] = ic["r_pl"], ic["v_pl"]
        self.pl[:, 6] = 1e300
        self.gpl = planet_gms(cfg)
        self.ledger = np.zeros(LEDGER_N)
        self.last_moon = np.zeros((0, 7))
        self._supply_init()

    t_years = property(lambda s: s.st[9] / YEAR)
    m_ast = property(lambda s: s.st[7])
    E_rem = property(lambda s: s.st[8])
    m_launcher = property(lambda s: s.st[6])

    def _set_E(self, x):
        """Store the Earth-Moon centre's state vector."""
        self.st[8] = x

    def mu(self):
        """Gravitational parameter of the Sun plus the Earth-Moon system plus the docked asteroid."""
        return GM_SUN + G * (self.st[6] + M_MOON)

    def emb(self):
        """Current position and velocity of the Earth-Moon centre."""
        return self.st[0:3].copy(), self.st[3:6].copy()

    def run_interval(self, years, dm, u, d_rtn, shield=True):
        """Run one decision interval, paying its energy from the supply."""
        return self._with_supply(years, lambda: self._run_raw(years, dm, u, d_rtn, shield))

    def _run_raw(self, years, dm, u, d_rtn, shield):
        """Run one decision interval: Kepler coasting between impulsive, momentum-conserving ejections."""
        n = max(1, int(round(self.cfg.ejections_per_year * years)))
        samples = np.zeros((n, N_SAMPLE_COLS))
        moon = np.zeros((n * MOON_SUB if self.record_moon else 0, 7))
        self.st[23], self.st[24], self.st[26], self.st[27], self.st[29] = 1e300, 0.0, 0.0, 0.0, 0.0
        self.pl[:, 6], self.pl[:, 7], self.pl[:, 8] = 1e300, 0.0, 0.0
        k, status = fast_interval(self.st, self.ledger, n, years * YEAR / n, float(dm), float(u),
                                  np.asarray(d_rtn, dtype=np.float64), self.cfg.launch_efficiency,
                                  R_EARTH, samples, moon, mars_params(self.cfg, shield), self.pl, self.gpl)
        self.last_moon = moon[:k * MOON_SUB]
        return samples[:k], int(status)

    def mars_state(self):
        """Heliocentric Mars position [m] and velocity [m/s]."""
        return self.st[17:20].copy(), self.st[20:23].copy()

    def planets_state(self):
        """Heliocentric positions [m] and velocities [m/s] of Mercury, Venus, Jupiter (arrays 3 x 3)."""
        return self.pl[:, 0:3].copy(), self.pl[:, 3:6].copy()

    def planets_interval(self):
        """Closest distance of the Earth-Moon centre to each other planet during the last run_interval [m], its time
        [yr], and the sum of that planet's gravity kicks on the Earth-Moon system [m/s]."""
        return [dict(name=PLANET_NAMES[k], dmin_m=float(self.pl[k, 6]), t_dmin_yr=float(self.pl[k, 7]) / YEAR,
                     kick_ms=float(self.pl[k, 8])) for k in range(N_PL)]

    def mars_interval(self):
        """Closest Earth-Mars distance during the last run_interval [m], its time [yr], shield overrides,
        last shield forecast [m], shield mode, sum of Mars gravity kicks on the Earth-Moon system [m/s]."""
        return dict(dmin_m=float(self.st[23]), t_dmin_yr=float(self.st[24]) / YEAR, overrides=int(self.st[26]),
                    forecast_m=float(self.st[28]), mode=int(self.st[25]), kick_ms=float(self.st[27]))

    def earth_state(self):
        """Viewer state: the fast model's rows are the Earth-Moon BARYCENTRE (4,700 km from Earth's
        centre, invisible at AU scale); third item = Moon relative to Earth (position, velocity)."""
        return self.st[0:3].copy(), self.st[3:6].copy(), (self.st[11:14].copy(), self.st[14:17].copy())

    def earth_moon(self):
        """Heliocentric Earth and Moon positions [m] (from barycentre + relative orbit)."""
        M = self.st[6]
        fe, fm = M_MOON / (M + M_MOON), M / (M + M_MOON)
        r, rr = self.st[0:3], self.st[11:14]
        return r - fe * rr, r + fm * rr

    def moon_info(self):
        """Moon distance and whether it is still bound to Earth."""
        rel, vrel = self.st[11:14], self.st[14:17]
        d = float(np.linalg.norm(rel))
        eps = 0.5 * float(vrel @ vrel) - G * (self.st[6] + M_MOON) / d
        hill = float(np.linalg.norm(self.st[0:3])) * ((self.st[6] + M_MOON) / (3 * M_SUN)) ** (1 / 3)
        return dict(modelled=True, bound=bool(eps < 0 and d < 0.5 * hill), dist_m=d, hill_m=hill)

    def conservation(self):
        """Momentum and energy bookkeeping errors of the fast model."""
        return dict(model="fast", note="impulses conserve momentum exactly by construction; "
                                        "orbital motion is exact two-body Kepler (Sun-EMB and Earth-Moon)")


class HifiEngine(_PowerSupply):
    """HIGH-FIDELITY model.  N-body Sun, Earth(+docked reservoir), Moon and explicit fragments
    integrated with 4th-order Yoshida (fixed dt, default 6 h); ejection, impact, escape and
    merge events; full energy / momentum / angular-momentum bookkeeping."""
    fidelity = "hifi"

    def __init__(self, cfg: Config, seed: int):
        """High-fidelity model: N-body Sun / Earth / Moon / Mars plus every thrown fragment."""
        self.cfg = cfg
        ic = initial_conditions(cfg, seed)
        nmax = NMAJ + cfg.max_fragments
        self.pos = np.zeros((nmax, 3))
        self.vel = np.zeros((nmax, 3))
        self.mass = np.zeros(nmax)
        me = ic["m_earth"]
        f_e, f_m = M_MOON / (me + M_MOON), me / (me + M_MOON)
        gpl = planet_gms(cfg)
        self.mass[:NMAJ] = [M_SUN, me, M_MOON, M_MARS if cfg.mars_gravity else 0.0] + [g / G for g in gpl]
        self.pos[1] = ic["r_emb"] - f_e * ic["r_rel"]
        self.vel[1] = ic["v_emb"] - f_e * ic["v_rel"]
        self.pos[2] = ic["r_emb"] + f_m * ic["r_rel"]
        self.vel[2] = ic["v_emb"] + f_m * ic["v_rel"]
        self.pos[3], self.vel[3] = ic["r_mars"], ic["v_mars"]
        self.pos[4:NMAJ], self.vel[4:NMAJ] = ic["r_pl"], ic["v_pl"]
        mt = self.mass[:NMAJ].sum()
        self.pos[:NMAJ] -= (self.mass[:NMAJ, None] * self.pos[:NMAJ]).sum(0) / mt
        self.vel[:NMAJ] -= (self.mass[:NMAJ, None] * self.vel[:NMAJ]).sum(0) / mt
        self.nbarr = np.array([NMAJ], dtype=np.int64)
        self.stv = np.array([cfg.asteroid_mass_kg, cfg.energy_budget_j, 0.0, 1e300, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                            + [1e300] * N_PL + [0.0] * N_PL)
        self.ledger = np.zeros(LEDGER_N)
        self.prev = np.zeros((nmax, 3))
        self.acc = np.zeros((nmax, 3))
        self.tot0 = self.totals()
        self.last_moon = np.zeros((0, 7))
        self._supply_init()

    t_years = property(lambda s: s.stv[2] / YEAR)
    m_ast = property(lambda s: s.stv[0])
    E_rem = property(lambda s: s.stv[1])
    m_launcher = property(lambda s: s.mass[1])

    def _set_E(self, x):
        """Overwrite Earth's state in the N-body arrays."""
        self.stv[1] = x

    def totals(self):
        """Total momentum and energy of all bodies (for the conservation checks)."""
        out = np.zeros(8)
        nb_totals(self.pos, self.vel, self.mass, int(self.nbarr[0]), out)
        return out

    def mu(self):
        """Gravitational parameter of the Sun plus Earth, Moon and the docked asteroid."""
        return G * (self.mass[0] + self.mass[1] + self.mass[2])

    def emb(self):
        """Current position and velocity of the Earth-Moon centre."""
        me, mm = self.mass[1], self.mass[2]
        r = (me * self.pos[1] + mm * self.pos[2]) / (me + mm) - self.pos[0]
        v = (me * self.vel[1] + mm * self.vel[2]) / (me + mm) - self.vel[0]
        return r, v

    def run_interval(self, years, dm, u, d_rtn, shield=True):
        """Run one decision interval, paying its energy from the supply."""
        return self._with_supply(years, lambda: self._run_raw(years, dm, u, d_rtn, shield))

    def _run_raw(self, years, dm, u, d_rtn, shield):
        """Run one decision interval with the N-body integrator and explicit fragments."""
        cfg = self.cfg
        n = max(1, int(round(cfg.ejections_per_year * years)))
        samples = np.zeros((n, N_SAMPLE_COLS))
        moon = np.zeros((n * MOON_SUB, 7))
        self.stv[3], self.stv[4], self.stv[6], self.stv[9] = 1e300, 0.0, 0.0, 0.0
        self.stv[10:10 + N_PL], self.stv[10 + N_PL:10 + 2 * N_PL] = 1e300, 0.0
        k, status = hifi_interval(self.pos, self.vel, self.mass, self.nbarr, self.stv, self.ledger, n,
                                  years * YEAR / n, cfg.hifi_dt_hours * 3600.0, float(dm), float(u),
                                  np.asarray(d_rtn, dtype=np.float64), cfg.launch_efficiency, R_EARTH,
                                  cfg.release_radius_m, cfg.fragment_escape_au * AU, samples, moon,
                                  self.prev, self.acc, mars_params(cfg, shield))
        self.last_moon = moon[:k * MOON_SUB]
        return samples[:k], int(status)

    def mars_state(self):
        """Mars's current position and velocity."""
        return self.pos[3] - self.pos[0], self.vel[3] - self.vel[0]

    def planets_state(self):
        """Heliocentric positions [m] and velocities [m/s] of Mercury, Venus, Jupiter (arrays 3 x 3)."""
        return self.pos[4:NMAJ] - self.pos[0], self.vel[4:NMAJ] - self.vel[0]

    def planets_interval(self):
        """Closest Earth distance to each other planet during the last run_interval [m] and its time [yr]."""
        return [dict(name=PLANET_NAMES[k], dmin_m=float(self.stv[10 + k]), t_dmin_yr=float(self.stv[10 + N_PL + k]) / YEAR,
                     kick_ms=float("nan")) for k in range(N_PL)]

    def mars_interval(self):
        """Closest Earth-Mars distance and Mars's kick during the last interval."""
        return dict(dmin_m=float(self.stv[3]), t_dmin_yr=float(self.stv[4]) / YEAR, overrides=int(self.stv[6]),
                    forecast_m=float(self.stv[8]), mode=int(self.stv[5]), kick_ms=float("nan"))

    def earth_state(self):
        """Earth's position and velocity, plus the Moon's relative state."""
        return self.pos[1] - self.pos[0], self.vel[1] - self.vel[0], (self.pos[2] - self.pos[1], self.vel[2] - self.vel[1])

    def moon_info(self):
        """Moon distance and whether it is still bound to Earth."""
        rel = self.pos[2] - self.pos[1]
        vrel = self.vel[2] - self.vel[1]
        d = float(np.linalg.norm(rel))
        eps = 0.5 * float(vrel @ vrel) - G * (self.mass[1] + self.mass[2]) / d
        r_emb, _ = self.emb()
        hill = float(np.linalg.norm(r_emb)) * ((self.mass[1] + self.mass[2]) / (3 * self.mass[0])) ** (1 / 3)
        return dict(modelled=True, bound=bool(eps < 0 and d < 0.5 * hill), dist_m=d, hill_m=hill)

    def conservation(self):
        """Relative drift of momentum, energy and angular momentum since the start."""
        now = self.totals()
        L = self.ledger
        e_ref = abs(self.tot0[0])
        p_ref = M_EARTH * 3e4
        l_ref = float(np.linalg.norm(self.tot0[4:7]))
        resid_e = (now[0] - self.tot0[0] - L[LG_EV_DE]) / e_ref
        resid_p = float(np.linalg.norm(now[1:4] - self.tot0[1:4] - L[LG_EV_DP:LG_EV_DP + 3])) / p_ref
        resid_l = float(np.linalg.norm(now[4:7] - self.tot0[4:7] - L[LG_EV_DL:LG_EV_DL + 3])) / l_ref
        mass_resid = (now[7] + L[LG_ESC_MASS] - self.tot0[7]) / self.tot0[7]
        return dict(model="hifi", energy_residual_rel=float(resid_e), momentum_residual_rel=resid_p,
                    angmom_residual_rel=resid_l, mass_residual_rel=float(mass_resid),
                    energy_injected_by_ejections_J=float(L[LG_DE_EJECT]),
                    energy_theory_ejections_J=float(L[LG_DE_EJECT_THEORY]),
                    energy_removed_by_escapes_J=float(L[LG_ESC_E]), impact_heat_J=float(-L[LG_DE_IMPACT]),
                    merge_energy_J=float(L[LG_DE_MERGE]), merges=int(L[LG_N_MERGES]),
                    active_fragments=int(self.nbarr[0] - NMAJ))


def make_engine(cfg: Config, fidelity: str, seed: int, record_moon: bool = True):
    """Create the fast or the high-fidelity physics engine for one mission."""
    return HifiEngine(cfg, seed) if fidelity == "hifi" else FastEngine(cfg, seed, record_moon)


# =============================================================================================
# 7. GYMNASIUM ENVIRONMENT
# =============================================================================================
OBS_DIM = 31
ACT_DIM = 5
REWARD_KEYS = ["progress", "eccentricity", "inclination", "energy_use", "mass_use", "time", "mars_safety", "climate",
               "success", "terminal_error", "failure"]


class OctoEarthEnv(gym.Env):
    """Action (Box[-1,1]^5):
        a0 throttle -> packet mass = throttle * max_packet (throttle < 0.02 -> no ejection)
        a1 in-plane thrust angle theta = a1*pi measured from prograde (+T) toward +R
        a2 out-of-plane thrust angle phi = a2*pi/2
        a3 launch speed u = u_min + (a3+1)/2 (u_max - u_min)
        a4 climate: insolation set-point = (1 + 0.5 a4) x today's sunlight; the space mirrors send extra sunlight to
           Earth (they never remove sunlight) up to their capacity so the total reaches the set-point
    Observation: 31 numbers (orbit, Moon, resources, capacities, previous action, Mars, climate: surface / deep-ocean
    temperature error, sunlight share, mirror capacity).
    The thrust direction is held fixed in the rotating RTN frame of the Earth-Moon barycentre
    during one decision interval; packets are ejected opposite to it at the configured
    frequency.  The agent never sets Earth's velocity: only packets are launched."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: Config | None = None, fidelity: str = "fast", record: bool = True,
                 record_moon: bool = True):
        """The RL environment: one episode = one mission; one step = one decision interval."""
        super().__init__()
        self.cfg = cfg or Config()
        self.fidelity = fidelity
        self.record = record
        self.record_moon = record_moon
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.engine = None
        self.last_trail = None
        self.last_decisions = []

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Start a new mission and return the first observation."""
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.seed_used = int(seed)
        cfg = self.cfg
        self.engine = make_engine(cfg, self.fidelity, self.seed_used, self.record_moon)
        self.steps = 0
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.m_ast0 = cfg.asteroid_mass_kg
        self.E0 = cfg.energy_budget_j
        self.mu_t = self.engine.mu()
        self.eps_t = -self.mu_t / (2 * cfg.target_au * AU)
        r, v = self.engine.emb()
        a0, e0, _, eps0 = orbit_elements(r, v, self.engine.mu())
        self.eps0 = eps0
        self.a0_au, self.e0 = a0 / AU, e0
        self.phi = self._potential()
        self.mars_lim = mars_safety_limits(cfg)
        rm, vm = self.engine.mars_state()
        me0 = _mars_elements(rm, vm)
        self.mars_a0, self.mars_e0 = me0["a_au"], me0["e"]
        self.mars_dmin_total, self.mars_dmin_t = float(np.linalg.norm(rm - r)), 0.0
        self.pl_lim = planet_safety_limits(cfg)
        rp, vp = self.engine.planets_state()
        self.pl_a0 = [_planet_elements(rp[k], vp[k], k)["a_au"] for k in range(N_PL)]
        self.pl_dmin = [float(np.linalg.norm(rp[k] - r)) for k in range(N_PL)]
        self.pl_dmin_t = [0.0] * N_PL
        self.pl_kick = [0.0] * N_PL
        self.shield_overrides = 0
        self.cum_reward = 0.0
        self.components = {k: 0.0 for k in REWARD_KEYS}
        self.decisions = []
        self.trail = []
        self.done_reason = None
        self.success = False
        self.validated = None
        self.dv_cum = 0.0
        s_rel0 = S_1AU / (float(np.linalg.norm(r)) / AU) ** 2 / S_1AU * (1.0 - earth_shading(cfg, r))
        T0 = climate_equilibrium(s_rel0)
        self.clim = dict(Ts=T0, Td=T0, s_sun_rel=s_rel0, s_set_rel=1.0, p_mirror_W=0.0, q_w_W=0.0, years_out=0.0,
                         dev_max=abs(T0 - cfg.climate_target_K), T0=T0,
                         cap_rel=mirror_design(cfg, float(np.linalg.norm(r)) / AU)["cap_rel"] * (1.0 - earth_shading(cfg, r)))
        return self._obs(), {}

    def climate_ok(self):
        """True when the surface temperature is within the target tolerance (needed for success)."""
        if not self.cfg.climate_model:
            return True
        return abs(self.clim["Ts"] - self.cfg.climate_target_K) <= self.cfg.climate_tol_K

    def _climate_obs(self):
        """The four climate numbers the agent sees (temperature error, sunlight, set-point, mirror capacity)."""
        c, cfg = self.clim, self.cfg
        if not cfg.climate_model:
            return np.zeros(4)
        return np.array([max(-5.0, min(5.0, (c["Ts"] - cfg.climate_target_K) / 10.0)),
                         max(-5.0, min(5.0, (c["Td"] - cfg.climate_target_K) / 10.0)),
                         min(2.0, c["s_sun_rel"]), min(5.0, c["cap_rel"]) / 5.0])

    # ------------------------------------------------------------------------------------------
    def elements(self):
        """Earth's current orbital elements (a, e, inclination, ...)."""
        r, v = self.engine.emb()
        a, e, inc, eps = orbit_elements(r, v, self.engine.mu())
        return dict(a_au=a / AU if math.isfinite(a) else float("inf"), e=e, inc_deg=math.degrees(inc),
                    r_au=float(np.linalg.norm(r)) / AU, speed=float(np.linalg.norm(v)), eps=eps)

    def _potential(self):
        """Progress measure used for reward shaping: how much velocity change is still needed."""
        el = self.elements()
        return -min(3.0, abs(el["eps"] - self.eps_t) / abs(self.eps0 - self.eps_t))

    def dv_needed(self, el=None):
        """Velocity change still needed to reach the target orbit (Hohmann estimate)."""
        el = el or self.elements()
        a = el["a_au"] * AU
        vc = math.sqrt(self.engine.mu() / a) if math.isfinite(a) and a > 0 else 0.0
        return abs(vc - math.sqrt(self.engine.mu() / (self.cfg.target_au * AU)))

    def mars_now(self):
        """Current Earth-Mars distance and Mars's orbit."""
        r, _ = self.engine.emb()
        rm, vm = self.engine.mars_state()
        return rm - r, float(np.linalg.norm(rm - r)), rm, vm

    def _obs(self):
        """Build the 31-number observation vector the agent sees (all scaled to about -1..1)."""
        cfg = self.cfg
        r, v = self.engine.emb()
        el = self.elements()
        moon = self.engine.moon_info()
        M = self.engine.m_launcher + M_MOON
        m_rem = self.engine.m_ast
        vinf_max = launch_vinf(cfg.max_ejection_speed_ms, self.engine.m_launcher)
        need = self.dv_needed(el) + 1.0
        cap_m = vinf_max * math.log(M / max(M - m_rem, 1.0)) + 1e-9
        cap_e = math.sqrt(2 * cfg.launch_efficiency * self._energy_available() * m_rem) / M + 1e-9
        a_rel = el["a_au"] / cfg.target_au if math.isfinite(el["a_au"]) else 5.0
        o = np.concatenate([
            r / (cfg.target_au * AU), v / 3.0e4,
            [min(a_rel, 5.0), min(el["e"] / cfg.tol_e, 10.0), min(el["inc_deg"] / cfg.tol_inc_deg, 10.0),
             max(-5.0, min(5.0, (el["eps"] - self.eps_t) / abs(self.eps0 - self.eps_t))),
             moon["dist_m"] / MOON_A, 1.0 if moon["bound"] else 0.0,
             m_rem / self.m_ast0, max(self.engine.E_rem, 0) / self.E0,
             max(0.0, 1.0 - self.engine.t_years / cfg.mission_years),
             max(-3.0, min(3.0, math.log10(cap_m / need))) / 3.0,
             max(-3.0, min(3.0, math.log10(cap_e / need))) / 3.0],
            self.prev_action, self._mars_obs(r, v), self._climate_obs()])
        return np.clip(np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)

    def _energy_available(self):
        """Energy the launcher can still get: the reserve, plus (collectors on) the beam power at Earth's current distance
        for the rest of the mission."""
        e = max(self.engine.E_rem, 0.0)
        if self.cfg.solar_collectors:
            r, _ = self.engine.emb()
            e += collector_power(self.cfg, r)["P_elec_W"] * max(0.0, self.cfg.mission_years - self.engine.t_years) * YEAR
        return e

    def _mars_obs(self, r, v):
        """5 Mars features: Mars position relative to Earth [AU] (3), log10(distance / safe distance) / 2 now, and
        the same for the closest approach forecast for the next shield horizon if Earth coasts."""
        rm, vm = self.engine.mars_state()
        rel = (rm - r) / AU
        d_safe = self.mars_lim["d_safe_m"]
        d_now = float(np.linalg.norm(rm - r))
        d_fc = mars_forecast(r, v, self.engine.mu(), rm, vm, self.cfg.shield_horizon_years * YEAR,
                             YEAR / self.cfg.ejections_per_year, 0.0, np.array([0.0, 1.0, 0.0]),
                             G * (self.engine.m_launcher + M_MOON), GM_MARS if self.cfg.mars_gravity else 0.0)
        return np.array([max(-5.0, min(5.0, rel[0])), max(-5.0, min(5.0, rel[1])), max(-5.0, min(5.0, rel[2])),
                         max(-1.5, min(1.5, 0.5 * math.log10(max(d_now, 1.0) / d_safe))),
                         max(-1.5, min(1.5, 0.5 * math.log10(max(d_fc, 1.0) / d_safe)))])

    # ------------------------------------------------------------------------------------------
    def decode_action(self, action):
        """Turn the agent's five numbers (-1..1) into throttle, launch direction and climate set-point."""
        cfg = self.cfg
        a_in = np.nan_to_num(np.asarray(action, dtype=np.float64).ravel(), nan=0.0)
        a = np.zeros(ACT_DIM)
        a[:min(ACT_DIM, a_in.size)] = a_in[:ACT_DIM]
        a = np.clip(a, -1.0, 1.0)
        throttle = 0.5 * (a[0] + 1.0)
        if throttle < 0.02:
            throttle = 0.0
        theta, phi = a[1] * math.pi, a[2] * math.pi / 2
        u = cfg.min_ejection_speed_ms + 0.5 * (a[3] + 1.0) * (cfg.max_ejection_speed_ms - cfg.min_ejection_speed_ms)
        d = np.array([math.sin(theta) * math.cos(phi), math.cos(theta) * math.cos(phi), math.sin(phi)])
        return dict(throttle=throttle, dm=throttle * cfg.max_packet_mass_kg, u=u, d_rtn=d,
                    theta_deg=math.degrees(theta), phi_deg=math.degrees(phi), clipped=a, s_set_rel=1.0 + 0.5 * a[4])

    def in_target(self, el=None):
        """True when Earth's orbit is inside the target tolerances (size and shape)."""
        el = el or self.elements()
        c = self.cfg
        return (math.isfinite(el["a_au"]) and abs(el["a_au"] - c.target_au) <= c.tol_a_au and
                el["e"] <= c.tol_e and el["inc_deg"] <= c.tol_inc_deg)

    def step(self, action):
        """Apply one decision: run the physics for one interval, then compute reward and termination."""
        cfg = self.cfg
        dec = self.decode_action(action)
        eng = self.engine
        years = min(cfg.decision_interval_years, cfg.mission_years - eng.t_years)
        led0 = eng.ledger.copy()
        m0, E0 = eng.m_ast, eng.E_rem
        t_int0 = eng.t_years
        samples, status = eng.run_interval(years, dec["dm"], dec["u"], dec["d_rtn"])
        self.steps += 1
        clim_pen, clim_track = self._climate_step(samples, t_int0, years, dec, led0)
        dm_used = m0 - eng.m_ast
        e_used = float(eng.ledger[LG_E_DRAWN] - led0[LG_E_DRAWN])   # all energy drawn (reserve + collector beams)
        el = self.elements()
        moon = eng.moon_info()
        phi_new = self._potential()
        frac = years / 100.0
        mi = eng.mars_interval()
        lim = self.mars_lim
        dmin = mi["dmin_m"]
        if dmin < self.mars_dmin_total:
            self.mars_dmin_total, self.mars_dmin_t = dmin, mi["t_dmin_yr"]
        self.shield_overrides += mi["overrides"]
        pinf = eng.planets_interval()
        pl_bad = False
        for k, q in enumerate(pinf):
            if q["dmin_m"] < self.pl_dmin[k]:
                self.pl_dmin[k], self.pl_dmin_t[k] = q["dmin_m"], q["t_dmin_yr"]
            if math.isfinite(q["kick_ms"]):
                self.pl_kick[k] += q["kick_ms"]
            pl_bad = pl_bad or q["dmin_m"] < self.pl_lim[k]["d_safe_m"]
        mars_pen = max(0.0, min(1.0, (lim["caution_m"] - dmin) / max(lim["caution_m"] - lim["d_safe_m"], 1.0)))
        rc = dict(progress=cfg.w_progress * (phi_new - self.phi),
                  eccentricity=-cfg.w_eccentricity * min(5.0, max(0.0, el["e"] - cfg.tol_e) / cfg.tol_e) * frac,
                  inclination=-cfg.w_inclination * min(5.0, max(0.0, el["inc_deg"] - cfg.tol_inc_deg) / cfg.tol_inc_deg) * frac,
                  energy_use=-cfg.w_energy * e_used / self.E0,
                  mass_use=-cfg.w_mass * dm_used / self.m_ast0,
                  time=-cfg.w_time, mars_safety=-cfg.w_mars_safety * mars_pen, climate=-cfg.w_climate * clim_pen * frac,
                  success=0.0, terminal_error=0.0, failure=0.0)
        self.phi = phi_new
        terminated = False
        reason = None
        if status == 1:
            terminated, reason = True, "escaped_or_unbound"
        elif status == 2:
            terminated, reason = True, "too_close_to_sun"
        elif status == 3 or dmin < lim["d_safe_m"]:
            terminated, reason = True, "mars_unsafe_approach"
        elif status == 4 or pl_bad:
            terminated, reason = True, "planet_unsafe_approach"
        elif not moon["bound"]:
            terminated, reason = True, "moon_lost"
        elif self.in_target(el) and self.climate_ok():
            terminated, reason = True, "success"
        elif eng.t_years >= cfg.mission_years - 1e-6:
            terminated, reason = True, "deadline"
        elif eng.m_ast <= 1e-9 * self.m_ast0 or (eng.E_rem <= 1e-9 * self.E0 and not cfg.solar_collectors):
            terminated, reason = True, "resources_exhausted"
        if terminated:
            if reason == "success":
                left = 0.5 * (eng.m_ast / self.m_ast0 + max(eng.E_rem, 0) / self.E0)
                rc["success"] = cfg.w_success * (1.0 + 0.5 * left)
                self.success = True
            elif reason in ("deadline", "resources_exhausted"):
                clim_err = min(3.0, abs(self.clim["Ts"] - cfg.climate_target_K) / cfg.climate_tol_K) if cfg.climate_model else 0.0
                rc["terminal_error"] = -cfg.w_terminal_error * (self.error_norm(el) + 0.3 * clim_err)
            else:
                rc["failure"] = -cfg.w_failure
        reward = float(sum(rc.values()))
        self.cum_reward += reward
        for k, v in rc.items():
            self.components[k] += v
        self.prev_action = dec["clipped"].astype(np.float32)
        led1 = eng.ledger
        impulse = led1[LG_IMPULSE_TOTAL] - led0[LG_IMPULSE_TOTAL]
        M_emb = eng.m_launcher + M_MOON
        dv_int = impulse / M_emb                 # speed change delivered to the Earth-Moon system
        self.dv_cum += dv_int
        rec = dict(step=self.steps, t_years=eng.t_years, a_au=el["a_au"], e=el["e"], inc_deg=el["inc_deg"],
                   r_au=el["r_au"], throttle=dec["throttle"], dm_kg=dm_used, u_ms=dec["u"],
                   theta_deg=dec["theta_deg"], phi_deg=dec["phi_deg"], d_rtn=dec["d_rtn"].tolist(),
                   impulse_Ns=impulse, energy_J=e_used, m_ast_frac=eng.m_ast / self.m_ast0,
                   energy_frac=max(eng.E_rem, 0) / self.E0, reward=reward, cum_reward=self.cum_reward,
                   moon_dist_m=moon["dist_m"], moon_bound=moon["bound"],
                   fallback=int(led1[LG_N_FALLBACK] - led0[LG_N_FALLBACK]),
                   n_packets=int(led1[LG_N_PACKETS] - led0[LG_N_PACKETS]),
                   moon_hits=int(led1[LG_N_LAUNCH_MOON_HITS] - led0[LG_N_LAUNCH_MOON_HITS]),
                   dv_ms=dv_int, dv_cum_ms=self.dv_cum, thrust_N=impulse / max(years * YEAR, 1.0),
                   impulse_cum_Ns=float(led1[LG_IMPULSE_TOTAL]), speed_kms=el["speed"] / 1000.0,
                   E_orb_J=el["eps"] * M_emb, E_used_cum_J=float(eng.ledger[LG_E_DRAWN]),
                   m_ast_kg=eng.m_ast, E_rem_J=max(eng.E_rem, 0.0), years=years, reason=reason,
                   **self._mars_record(mi), planets=self._planets_record(pinf), **self._power_record(),
                   **self._climate_record(clim_track))
        if eng.fidelity == "hifi":
            rec["conservation"] = eng.conservation()
        self.decisions.append(rec)
        self.last_samples = samples
        if self.record and len(samples):
            # every 3rd sample (3/8 year apart): [t (years), x, y, z (AU), vx, vy, vz (km/s, after the event)].
            # With the velocity the viewer can replay the episode with smooth, physically consistent motion.
            tr = samples[::3][:, [S_T, 1, 2, 3, S_VPOST, S_VPOST + 1, S_VPOST + 2]]
            self.trail.extend(np.column_stack([tr[:, :4].round(5), tr[:, 4:].round(4)]).tolist())
        if terminated:
            self.done_reason = reason
            self.last_trail = self.trail
            self.last_decisions = self.decisions
        info = dict(reason=reason, metrics=self.metrics(el), components=rc, decision=rec)
        if terminated:
            info["episode_summary"] = self.summary()
        return self._obs(), reward, terminated, False, info

    def _climate_step(self, samples, t0, years, dec, led0):
        """Climate over the interval just simulated, then close the interval's energy books.

        The mirrors add light up to the agent's set-point (limited by their capacity at Earth's distance). Returns
        (climate penalty 0-3, climate track for the viewer)."""
        cfg, eng, c = self.cfg, self.engine, self.clim
        if len(samples):
            ts = samples[:, S_T].astype(np.float64)
            r_au = np.linalg.norm(samples[:, S_R:S_R + 3].astype(np.float64), axis=1)
            r_mid = samples[len(samples) // 2, S_R:S_R + 3].astype(np.float64) * AU
        else:
            r, _ = eng.emb()
            ts, r_au, r_mid = np.array([eng.t_years]), np.array([float(np.linalg.norm(r)) / AU]), r
        shade = earth_shading(cfg, r_mid)
        ke = float(eng.ledger[LG_LAUNCH_KE] - led0[LG_LAUNCH_KE])
        lp = eng.last_power
        dt = years * YEAR
        # waste heat of the power chain that reaches Earth [W] (E1 arm radiators, E2 beam spill control)
        spill = rect = 0.0
        launch_heat = (float(lp["E_drawn_J"]) if lp else float(getattr(eng, "last_drawn", 0.0))) - ke
        if lp and cfg.solar_collectors:
            capt = lp["E_beam_used_J"] / cfg.rectenna_efficiency
            rect = capt - lp["E_beam_used_J"]
            spill = capt / max(lp["eta_capture"], 1e-12) * (1 - lp["eta_capture"])
        q_w = (cfg.beam_spill_to_earth_frac * spill + cfg.arm_heat_to_earth_frac * (rect + max(0.0, launch_heat))) / dt
        mirrors_on = bool(cfg.mirrors and cfg.climate_model)
        md = mirror_design(cfg)
        out = np.zeros((len(ts), 4))
        Ts, Td, E_earth, E_msun, yo, dmax = climate_integrate(float(t0), ts, r_au, c["Ts"], c["Td"], float(dec["s_set_rel"]),
                                                             md["area_eff_m2"], cfg.mirror_reflectivity, shade, q_w, mirrors_on,
                                                             cfg.climate_target_K, CLIM_LIVABLE_K, out)
        L = eng.settle_interval(years, ke, t0, float(E_msun), float(E_earth))
        cap_rel = mirror_design(cfg, float(r_au[-1]))["cap_rel"] * (1.0 - shade)
        c.update(Ts=float(Ts), Td=float(Td), s_sun_rel=float(out[-1, 2]), s_set_rel=float(dec["s_set_rel"]),
                 p_mirror_W=float(E_earth / dt), p_mirror_end_W=float(out[-1, 3]), cap_rel=float(cap_rel), q_w_W=float(q_w),
                 years_out=c["years_out"] + float(yo), dev_max=max(c["dev_max"], float(dmax)), shade=shade)
        if not cfg.climate_model:
            return 0.0, []
        dev = np.abs(out[:, 0] - cfg.climate_target_K)
        pen = float(np.mean(np.minimum(3.0, np.maximum(0.0, dev - 0.5 * cfg.climate_tol_K) / cfg.climate_tol_K)))
        step = max(1, len(ts) // 50)
        track = [[round(float(ts[i]), 3), round(float(out[i, 0]), 2), round(float(out[i, 2]), 4), _g(out[i, 3], 4)]
                 for i in range(0, len(ts), step)]
        if track and track[-1][0] != round(float(ts[-1]), 3):
            track.append([round(float(ts[-1]), 3), round(float(out[-1, 0]), 2), round(float(out[-1, 2]), 4), _g(out[-1, 3], 4)])
        self._last_ledger = L
        return pen, track

    def _climate_record(self, track):
        """Climate numbers stored in the decision record (temperature, light, mirrors, life)."""
        c = self.clim
        L = getattr(self, "_last_ledger", None) or {}
        life = life_indicators(c["Ts"])
        T_nomir = climate_equilibrium(c["s_sun_rel"], c["q_w_W"])
        return dict(T_surface_K=c["Ts"], T_deep_K=c["Td"], sun_rel=c["s_sun_rel"], insol_set_rel=c["s_set_rel"],
                    mirror_W=c["p_mirror_W"], mirror_rel=c["p_mirror_W"] / EARTH_INTERCEPT_W, mirror_cap_rel=c["cap_rel"],
                    mirror_end_rel=c.get("p_mirror_end_W", 0.0) / EARTH_INTERCEPT_W, waste_heat_earth_W=c["q_w_W"],
                    T_no_mirror_K=T_nomir, life={k: (_g(v, 4) if isinstance(v, float) else v) for k, v in life.items()},
                    life_no_mirror=life_indicators(T_nomir)["state"],
                    climate_ok=bool(self.climate_ok()), years_outside_livable=c["years_out"], shade=c.get("shade", 0.0),
                    clim_track=track, ledger={k: _g(v, 6) for k, v in L.items()},
                    built=self.engine.last_power["built"] if self.engine.last_power else None,
                    eta_pv_op=self.engine.last_power["eta_pv_op"] if self.engine.last_power else None)

    def _climate_metrics(self):
        """Mission-level climate metrics (final temperature, worst deviation, mirror energy, ...)."""
        c, eng = self.clim, self.engine
        led = dict(eng.eled)
        e_in = led["sun_on_collectors"] + led["reserve_used"] + led["mirror_sun"]
        life = life_indicators(c["Ts"])
        return dict(T_surface_K=c["Ts"], T_deep_K=c["Td"], climate_ok=bool(self.climate_ok()), climate_dev_max_K=c["dev_max"],
                    years_outside_livable=c["years_out"], mirror_now_W=c["p_mirror_W"], waste_heat_earth_W=c["q_w_W"],
                    energy_ledger=led, energy_ledger_residual=float(eng.eled_resid), energy_in_J=e_in,
                    mirror_energy_J=led["mirror_to_earth"], heat_to_earth_J=led["heat_to_earth"],
                    ice_line_deg=life["ice_line_deg"], grow_frac=life["grow_frac"],
                    T_no_mirror_K=climate_equilibrium(c["s_sun_rel"], c["q_w_W"]))

    def _planets_record(self, pinf):
        """Mercury, Venus, Jupiter numbers stored in the decision record (distance now, closest in this interval,
        orbit size and its change since the start)."""
        r, _ = self.engine.emb()
        rp, vp = self.engine.planets_state()
        out = []
        for k, q in enumerate(pinf):
            el = _planet_elements(rp[k], vp[k], k)
            out.append(dict(name=q["name"], dist_au=round(float(np.linalg.norm(rp[k] - r)) / AU, 6),
                            dmin_au=round(q["dmin_m"] / AU, 6), a_au=round(el["a_au"], 7),
                            da_au=float(el["a_au"] - self.pl_a0[k]), kick_ms=q["kick_ms"] if math.isfinite(q["kick_ms"]) else None))
        return out

    def _mars_record(self, mi):
        """Mars numbers stored in the decision record."""
        rel, d_now, rm, vm = self.mars_now()
        me = _mars_elements(rm, vm)
        dmin = mi["dmin_m"]
        return dict(mars_dist_au=d_now / AU, mars_dmin_au=dmin / AU, mars_dmin_t=mi["t_dmin_yr"],
                    mars_tide_ratio=mars_tide_ratio(dmin), mars_moon_perturb=mars_moon_perturbation(dmin),
                    mars_a_au=me["a_au"], mars_e=me["e"], mars_da_au=me["a_au"] - self.mars_a0, mars_de=me["e"] - self.mars_e0,
                    shield_overrides=mi["overrides"], shield_overrides_cum=self.shield_overrides, shield_mode=mi["mode"],
                    shield_forecast_au=mi["forecast_m"] / AU if mi["forecast_m"] < 1e299 else None,
                    mars_kick_ms=mi["kick_ms"], mars_safe_au=self.mars_lim["d_safe_au"],
                    mars_caution_au=self.mars_lim["caution_au"])

    def _power_record(self):
        """Power-supply numbers stored in the decision record."""
        lp = self.engine.last_power
        if not lp:
            return dict(beam_on=False)
        return dict(beam_on=True, P_avail_W=lp["P_elec_W"], P_beam_W=lp["mean_beam_W"], P_beam_peak_W=lp["peak_beam_W"],
                    eta_capture=lp["eta_capture"], E_beam_J=lp["E_beam_used_J"], E_reserve_J=lp["E_reserve_used_J"],
                    beam_d_min_au=lp["d_min_au"], beam_d_max_au=lp["d_max_au"],
                    beam_use_frac=lp["E_beam_used_J"] / lp["E_offered_J"] if lp["E_offered_J"] > 0 else 0.0)

    def _power_metrics(self):
        """Mission-level energy-supply metrics."""
        eng = self.engine
        drawn = float(eng.ledger[LG_E_DRAWN])
        if not self.cfg.solar_collectors:
            return dict(solar_on=False, energy_from_reserve_J=drawn, energy_from_collectors_J=0.0)
        S = eng.sol
        t = max(eng.t_years, 1e-9) * YEAR
        heat = (S["E_incident_J"] - S["E_beam_used_J"]) + drawn * (1 - self.cfg.launch_efficiency)
        m_emb = eng.m_launcher + M_MOON
        r, _ = eng.emb()
        pw = collector_power(self.cfg, r)
        return dict(solar_on=True, energy_from_collectors_J=S["E_beam_used_J"], energy_from_reserve_J=S["E_reserve_used_J"],
                    beam_energy_absorbed_J=S["E_incident_J"], beam_offered_J=S["E_offered_J"],
                    beam_use_frac=S["E_beam_used_J"] / S["E_offered_J"] if S["E_offered_J"] > 0 else 0.0,
                    beam_mean_W=S["E_incident_J"] / t, beam_peak_W=S["peak_beam_W"], beam_avail_now_W=pw["P_elec_W"],
                    eta_capture_min=S["eta_min"], eta_capture_max=S["eta_max"], eta_capture_now=pw["eta_capture"],
                    arm_heat_J=heat, arm_heat_mean_W=heat / t, arm_heat_vs_earth_sunlight=heat / t / EARTH_INTERCEPT_W,
                    radiator_area_1000K_m2=heat / t / (SIGMA_SB * 1000.0 ** 4),
                    beam_push_dv_ms=S["E_incident_J"] / (C_LIGHT * m_emb))

    def error_norm(self, el=None):
        """One number for how far the orbit is from the target (used to rank failed runs)."""
        el = el or self.elements()
        a = el["a_au"] if math.isfinite(el["a_au"]) else 10.0
        return min(3.0, abs(a - self.cfg.target_au) / (self.cfg.target_au - 1.0)) + 0.1 * min(10.0, el["e"] / self.cfg.tol_e)

    def metrics(self, el=None):
        """All physical metrics of the current state (kept separate from the reward)."""
        el = el or self.elements()
        eng = self.engine
        L = eng.ledger
        hab = habitability(el["a_au"], el["e"])
        return dict(t_years=eng.t_years, a_au=el["a_au"], e=el["e"], inc_deg=el["inc_deg"], r_au=el["r_au"],
                    shortfall_au=self.cfg.target_au - el["a_au"] if math.isfinite(el["a_au"]) else float("nan"),
                    mass_used_kg=self.m_ast0 - eng.m_ast, mass_used_frac=1 - eng.m_ast / self.m_ast0,
                    energy_used_J=float(L[LG_E_DRAWN]), energy_used_frac=float(L[LG_E_DRAWN]) / self.E0,
                    impulse_Ns=float(L[LG_IMPULSE_TOTAL]), n_packets=int(L[LG_N_PACKETS]),
                    fallback_packets=int(L[LG_N_FALLBACK]), max_packet_impulse_Ns=float(L[LG_MAX_IMPULSE]),
                    insolation_rel=hab.get("insolation_rel", float("nan")), t_eq_K=hab["t_eq"],
                    in_target=self.in_target(el), **self._mars_metrics(), **self._power_metrics(), **self._climate_metrics())

    def _mars_metrics(self):
        """Mission-level Mars metrics (closest approach, shield overrides, Mars's orbit change)."""
        _, d_now, rm, vm = self.mars_now()
        me = _mars_elements(rm, vm)
        dmin = self.mars_dmin_total
        return dict(mars_dist_au=d_now / AU, mars_min_dist_au=dmin / AU, mars_min_dist_t=self.mars_dmin_t,
                    mars_safe_au=self.mars_lim["d_safe_au"], mars_safe=bool(dmin >= self.mars_lim["d_safe_m"]),
                    mars_max_tide_ratio=mars_tide_ratio(dmin), mars_max_moon_perturb=mars_moon_perturbation(dmin),
                    mars_a_au=me["a_au"], mars_e=me["e"], mars_da_au=me["a_au"] - self.mars_a0, mars_de=me["e"] - self.mars_e0,
                    mars_a0_au=self.mars_a0, mars_e0=self.mars_e0, shield_overrides=int(self.shield_overrides),
                    **self._planet_metrics())

    def _planet_metrics(self):
        """Mission-level numbers for Mercury, Venus, Jupiter: closest approach vs its safe distance, orbit change, and
        (fast model) the total speed change their gravity gave the Earth-Moon system."""
        rp, vp = self.engine.planets_state()
        pls = []
        for k in range(N_PL):
            el = _planet_elements(rp[k], vp[k], k)
            pls.append(dict(name=PLANET_NAMES[k], min_dist_au=self.pl_dmin[k] / AU, min_dist_t=self.pl_dmin_t[k],
                            safe_au=self.pl_lim[k]["d_safe_au"], safe=bool(self.pl_dmin[k] >= self.pl_lim[k]["d_safe_m"]),
                            a_au=el["a_au"], da_au=el["a_au"] - self.pl_a0[k],
                            pull_sum_ms=self.pl_kick[k] if self.fidelity == "fast" else None))
        return dict(planets=pls, planets_safe=all(q["safe"] for q in pls), other_planets=bool(self.cfg.other_planets))

    def summary(self):
        """Final summary of the episode: outcome, metrics and the decision records."""
        el = self.elements()
        s = dict(fidelity=self.fidelity, seed=self.seed_used, steps=self.steps, reason=self.done_reason,
                 success=self.success, total_reward=self.cum_reward, components=dict(self.components),
                 **self.metrics(el))
        L = self.engine.ledger
        s["ledger"] = dict(mass_ejected=float(L[LG_MASS_EJECTED]), packets=int(L[LG_N_PACKETS]),
                           launch_ke=float(L[LG_LAUNCH_KE]), energy_drawn=float(L[LG_E_DRAWN]),
                           escaped_mass=float(L[LG_ESC_MASS]), earth_impact_mass=float(L[LG_EARTH_IMPACT_M]),
                           moon_impact_mass=float(L[LG_MOON_IMPACT_M]), sun_impact_mass=float(L[LG_SUN_IMPACT_M]),
                           packets_bound_to_sun_mass=float(L[LG_PKT_BOUND_SUN_M]),
                           packets_unbound_from_sun_mass=float(L[LG_PKT_UNBOUND_SUN_M]),
                           fallback_mass=float(L[LG_MASS_FALLBACK]), merges=int(L[LG_N_MERGES]),
                           n_fallback=int(L[LG_N_FALLBACK]), n_launch_moon_hits=int(L[LG_N_LAUNCH_MOON_HITS]),
                           mars_impact_mass=float(L[LG_MARS_IMPACT_M]), shield_override_events=int(L[LG_SHIELD_OVERRIDES]),
                           planet_impact_mass=float(L[LG_PLANET_IMPACT_M]))
        s["a0_au"], s["e0"] = self.a0_au, self.e0
        if self.fidelity == "hifi":
            s["conservation"] = self.engine.conservation()
            s["moon"] = self.engine.moon_info()
        return s

    def render(self):
        """Text rendering of the current state (Gymnasium interface)."""
        el = self.elements()
        return (f"t={self.engine.t_years:8.1f} y  a={el['a_au']:.4f} AU  e={el['e']:.4f}  "
                f"ast={self.engine.m_ast / self.m_ast0:6.1%}  E={max(self.engine.E_rem, 0) / self.E0:6.1%}")


gym.register(id="OctoEarth-v0", entry_point=lambda **kw: OctoEarthEnv(**kw))


# =============================================================================================
# 8. BASELINE POLICIES
# =============================================================================================
def policy_none(env, obs):
    """No ejection at all (baseline). The mirror set-point is at its minimum (0.5 x today), so the mirrors send no
    light while Earth stays near 1 AU."""
    return np.array([-1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)


def heuristic_climate(env) -> float:
    """Hand-designed climate rule: insolation set-point = today's sunlight, corrected in proportion to the surface-temperature
    error (dT/dS ~ 72 K per x today), clipped to the set-point range. Returns a4."""
    cfg, c = env.cfg, env.clim
    s_set = 1.0 + 0.8 * (cfg.climate_target_K - c["Ts"]) / 72.0 + 0.4 * (cfg.climate_target_K - c["Td"]) / 72.0
    return float(np.clip(2.0 * (s_set - 1.0), -1.0, 1.0))


def heuristic_speed(cfg: Config) -> float:
    """Hand-designed choice of launch speed: the lowest speed that (a) lets the whole reservoir
    supply the required delta-v (energy-optimal) and (b) still finishes within 85% of the
    mission at the maximum packet rate."""
    fz = feasibility(cfg)
    m_sys = M_EARTH + cfg.asteroid_mass_kg + M_MOON
    vinf_mass = fz["dv_required"] / math.log(m_sys / (m_sys - 0.95 * cfg.asteroid_mass_kg))
    vinf_time = fz["dv_required"] * m_sys / (0.85 * cfg.mission_years * cfg.ejections_per_year * cfg.max_packet_mass_kg)
    vinf = max(vinf_mass, vinf_time)
    u = math.sqrt(vinf ** 2 + fz["vesc"] ** 2)
    return min(max(u, cfg.min_ejection_speed_ms), cfg.max_ejection_speed_ms)


def policy_heuristic(env, obs):
    """Hand-designed strategy: prograde (tangential) thrust, speed from heuristic_speed(),
    throttle proportional to the remaining slow-spiral delta-v so it lands on the target."""
    cfg = env.cfg
    if not hasattr(env, "_heur_u"):
        env._heur_u = heuristic_speed(cfg)
    u = env._heur_u
    el = env.elements()
    vinf = launch_vinf(u, env.engine.m_launcher)
    M = env.engine.m_launcher + M_MOON
    n_ev = cfg.ejections_per_year * cfg.decision_interval_years
    dv_max = n_ev * cfg.max_packet_mass_kg * vinf / M
    need = env.dv_needed(el)
    a = el["a_au"]
    theta = 0.0 if a < cfg.target_au else 1.0               # retro-thrust if overshooting
    throttle = min(1.0, need / max(dv_max, 1e-9))
    if abs(a - cfg.target_au) < 0.25 * cfg.tol_a_au:
        throttle = 0.0
    a3 = 2 * (u - cfg.min_ejection_speed_ms) / (cfg.max_ejection_speed_ms - cfg.min_ejection_speed_ms) - 1
    return np.array([2 * throttle - 1, theta, 0.0, a3, heuristic_climate(env)], dtype=np.float32)


def model_policy(model):
    """Wrap a trained PPO model as a policy function (deterministic actions)."""
    def pol(env, obs):
        """Action of the trained model for this observation."""
        a, _ = model.predict(obs, deterministic=True)
        return a
    return pol


def run_episode(cfg: Config, policy, seed: int, fidelity: str = "fast", validate: bool = False,
                on_step=None, stop_event=None, on_reset=None) -> dict:
    """Run one complete mission with a policy and return its summary."""
    env = OctoEarthEnv(cfg, fidelity, record=True)
    obs, _ = env.reset(seed=seed)
    if on_reset:
        on_reset(env)
    done = False
    t0 = time.time()
    while not done:
        if stop_event is not None and stop_event.is_set():
            break
        obs, r, term, trunc, info = env.step(policy(env, obs))
        done = term or trunc
        if on_step:
            on_step(env, info)
    summ = env.summary()
    summ["decisions"] = env.decisions
    summ["wall_s"] = time.time() - t0
    if validate and fidelity == "hifi" and summ["reason"] in ("success", "deadline", "resources_exhausted"):
        summ["validation"] = validate_orbit(env)
    return summ


def validate_orbit(env: OctoEarthEnv) -> dict:
    """Ballistic (zero ejection) propagation after the mission in the HIGH-FIDELITY model; the
    orbit must stay within tolerances and the Moon must stay bound."""
    cfg = env.cfg
    chunks = 10
    worst_da, worst_e, moon_ok = 0.0, 0.0, True
    for _ in range(chunks):
        env.engine.run_interval(cfg.validate_years / chunks, 0.0, cfg.min_ejection_speed_ms, [0.0, 1.0, 0.0], shield=False)
        el = env.elements()
        da = abs(el["a_au"] - cfg.target_au) if math.isfinite(el["a_au"]) else float("inf")
        worst_da, worst_e = max(worst_da, da), max(worst_e, el["e"])
        moon_ok = moon_ok and env.engine.moon_info()["bound"]
    ok = worst_da <= cfg.tol_a_au and worst_e <= cfg.tol_e and moon_ok
    return dict(years=cfg.validate_years, max_abs_da_au=worst_da, max_e=worst_e, moon_bound=moon_ok, passed=bool(ok))


# =============================================================================================
# 9. PHYSICS SELF-CHECKS (run before training; also used by tests/test_octoearth.py)
# =============================================================================================
def _fresh_hifi(cfg, seed=3):
    """A new high-fidelity engine for the physics checks."""
    return HifiEngine(cfg, seed)


def run_physics_checks(full: bool = False, log=print) -> list:
    """Run all physics self-checks and return their PASS / FAIL results."""
    cfg = make_config("hypothetical", randomize_phase=False)
    out = []

    def check(name, fn):
        """Run one check, time it and record PASS / FAIL with its detail text."""
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as exc:  # pragma: no cover
            ok, detail = False, f"exception: {exc!r}"
        out.append(dict(name=name, passed=bool(ok), detail=detail, seconds=round(time.time() - t0, 2)))
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def kepler():
        """Check 1: two-body motion agrees with the exact Kepler solution."""
        ic = initial_conditions(cfg, 1)
        mu = GM_SUN + G * (ic["m_earth"] + M_MOON)
        r, v = ic["r_emb"].copy(), ic["v_emb"].copy()
        a, e, _, _ = orbit_elements(r, v, mu)
        P = 2 * math.pi * math.sqrt(a ** 3 / mu)
        r1, v1 = r.copy(), v.copy()
        for _ in range(37):
            kepler_step(r1, v1, mu, P / 37)
        closure = np.linalg.norm(r1 - r) / np.linalg.norm(r)
        # N-body integrator vs Kepler for an isolated Sun-Earth pair (Moon massless and far away)
        pos = np.zeros((4, 3)); vel = np.zeros((4, 3)); mass = np.array([M_SUN, M_EARTH, 0.0, 0.0])
        pos[1], vel[1] = r, v
        pos[2] = [60 * AU, 0, 0]
        nbarr = np.array([3], dtype=np.int64)
        led = np.zeros(LEDGER_N)
        hifi_advance(pos, vel, mass, nbarr, led, YEAR, 6 * 3600.0, np.zeros((4, 3)), np.zeros((4, 3)),
                     1e30, np.zeros(8), np.zeros(8), np.array([1e300, 0.0, 0.0]))
        rk, vk = r.copy(), v.copy()
        kepler_step(rk, vk, G * (M_SUN + M_EARTH), YEAR)
        err = np.linalg.norm((pos[1] - pos[0]) - rk)
        return closure < 1e-9 and err < 1e4, f"Kepler 1-orbit closure {closure:.1e}; N-body vs Kepler after 1 y: {err:.1f} m"

    def gravity():
        """Check 2: the N-body accelerations (Sun, Earth, Moon, asteroid, fragments) match Newton's law."""
        h = _fresh_hifi(cfg)
        acc = np.zeros_like(h.pos)
        def src(i, j):
            """Acceleration of body i caused by body j alone."""
            d = h.pos[j] - h.pos[i]
            return G * h.mass[j] * d / np.linalg.norm(d) ** 3
        sun_e = np.linalg.norm(src(1, 0)); moon_e = np.linalg.norm(src(1, 2)); earth_m = np.linalg.norm(src(2, 1))
        mars_e = np.linalg.norm(src(1, 3)); earth_mars = np.linalg.norm(src(3, 1))
        nb_accel(h.pos, h.mass, NMAJ, acc)
        tot_ok = np.allclose(acc[1], src(1, 0) + src(1, 2) + src(1, 3) + sum(src(1, k) for k in range(4, NMAJ)), rtol=1e-12)
        jup_e, earth_jup = np.linalg.norm(src(1, 6)), np.linalg.norm(src(6, 1))
        d_ej = np.linalg.norm(h.pos[6] - h.pos[1])
        jup_ok = abs(jup_e - GM_JUPITER / d_ej ** 2) / jup_e < 1e-9 and abs(earth_jup / jup_e - h.mass[1] / h.mass[6]) < 1e-6
        d_em = np.linalg.norm(h.pos[3] - h.pos[1])
        mars_ok = abs(mars_e - GM_MARS / d_em ** 2) / mars_e < 1e-9 and abs(earth_mars / mars_e - h.mass[1] / M_MARS) < 1e-6
        h0 = _fresh_hifi(replace(cfg, asteroid_mass_kg=0.0))
        ratio = np.linalg.norm(src(2, 1)) / (G * h0.mass[1] / np.linalg.norm(h.pos[2] - h.pos[1]) ** 2)
        ast_ok = abs(ratio - (M_EARTH + cfg.asteroid_mass_kg) / M_EARTH) < 1e-9
        # fragment gravity on Earth
        h.pos[NMAJ] = h.pos[1] + np.array([1e9, 0, 0]); h.mass[NMAJ] = 1e20
        acc2 = np.zeros_like(h.pos)
        nb_accel(h.pos, h.mass, NMAJ + 1, acc2)
        frag = np.linalg.norm(acc2[1] - acc[1])
        frag_ok = abs(frag - G * 1e20 / 1e18) / (G * 1e20 / 1e18) < 1e-6
        ok = (5.6e-3 < sun_e < 6.2e-3 and 2.8e-5 < moon_e < 3.8e-5 and 2.4e-3 < earth_m < 3.2e-3 and tot_ok and ast_ok
              and frag_ok and mars_ok and jup_ok)
        return ok, (f"Sun->Earth {sun_e:.3e}, Moon->Earth {moon_e:.3e}, Earth(+asteroid)->Moon {earth_m:.3e}, "
                    f"Mars->Earth {mars_e:.3e} at {d_em / AU:.2f} AU (Earth->Mars {earth_mars:.3e}, ratio = mass ratio) m/s^2; "
                    f"Jupiter->Earth {jup_e:.3e} at {d_ej / AU:.2f} AU, Venus->Earth {np.linalg.norm(src(1, 5)):.3e}, "
                    f"Mercury->Earth {np.linalg.norm(src(1, 4)):.3e}; "
                    f"asteroid mass scales Earth's pull by {ratio:.6f}; fragment(1e20 kg @1e6 km)->Earth {frag:.3e} m/s^2")

    def newton3():
        """Check 3: forces are equal and opposite (total momentum constant)."""
        h = _fresh_hifi(cfg)
        h.pos[NMAJ] = h.pos[1] + [3e9, 1e9, 0]; h.mass[NMAJ] = 1e19
        h.pos[NMAJ + 1] = h.pos[0] + [0, 2 * AU, 0]; h.mass[NMAJ + 1] = 1e18
        acc = np.zeros_like(h.pos)
        nb_accel(h.pos, h.mass, NMAJ + 2, acc)
        f = (h.mass[:NMAJ + 2, None] * acc[:NMAJ + 2])
        rel = np.linalg.norm(f.sum(0)) / np.abs(f).sum()
        return rel < 1e-13, f"|sum m a| / sum|m a| = {rel:.1e}"

    def eject_momentum():
        """Checks 4 + 6: an ejection conserves momentum and angular momentum."""
        h = _fresh_hifi(cfg)
        tb, ta = h.totals(), None
        eject_hifi(h.pos, h.vel, h.mass, h.nbarr, h.stv, h.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        ta = h.totals()
        dp = np.linalg.norm(ta[1:4] - tb[1:4]) / (M_EARTH * 3e4)       # vs Earth's orbital momentum
        dl = np.linalg.norm(ta[4:7] - tb[4:7]) / np.linalg.norm(tb[4:7])
        f = FastEngine(cfg, 3)
        memb = f.st[6] + M_MOON
        p0 = memb * f.st[3:6]
        f.run_interval(1e-3 / YEAR, 1e18, 3e6, [0, 1, 0])           # one packet, 1 ms of orbital motion
        p1 = (f.st[6] + M_MOON) * f.st[3:6] + f.ledger[LG_PKT_PX:LG_PKT_PX + 3]
        dpf = np.linalg.norm(p1 - p0) / (1e18 * 3e6)
        return dp < 1e-13 and dl < 1e-12 and dpf < 1e-4, (f"high-fidelity ejection: |dP|/(M_E v_E) = {dp:.1e} (round-off), |dL|/|L| = {dl:.1e}; "
                                                           f"fast model |dP|/(dm u) = {dpf:.1e} (incl. 1 ms of solar gravity)")

    def energy_accounting():
        """Check 5: launch energy equals the kinetic energy given to packet and Earth."""
        h = _fresh_hifi(cfg)
        eject_hifi(h.pos, h.vel, h.mass, h.nbarr, h.stv, h.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        L = h.ledger
        rel = abs(L[LG_DE_EJECT] - L[LG_DE_EJECT_THEORY]) / L[LG_LAUNCH_KE]
        draw_ok = abs(L[LG_E_DRAWN] - L[LG_LAUNCH_KE] / cfg.launch_efficiency) <= 1e-12 * L[LG_E_DRAWN]
        budget_ok = abs(cfg.energy_budget_j - h.stv[1] - L[LG_E_DRAWN]) <= 1e-9 * cfg.energy_budget_j
        hs = _fresh_hifi(replace(cfg, energy_budget_j=1e30))
        eject_hifi(hs.pos, hs.vel, hs.mass, hs.nbarr, hs.stv, hs.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        clip_ok = hs.stv[1] >= 0 and hs.ledger[LG_MASS_EJECTED] < 1e18
        return rel < 1e-6 and draw_ok and budget_ok and clip_ok, (
            f"measured system energy change vs 1/2 mu u^2 - G dm M/R: rel diff {rel:.1e}; drawn = KE/eta; "
            f"budget never exceeded (tiny budget -> packet reduced to {hs.ledger[LG_MASS_EJECTED]:.2e} kg)")

    def mass_accounting():
        """Check 7: the asteroid mass is finite and no fragment mass is created or lost."""
        c2 = replace(cfg, max_fragments=3)
        h = _fresh_hifi(c2)
        m0 = h.totals()[7]
        h.run_interval(3.0, 1e18, 3e4, [0.3, 0.9, 0.0])           # slow packets stay bound -> merges
        h.run_interval(2.0, 1e18, 3e6, [0, 1, 0])                  # fast packets escape
        c = h.conservation()
        small = HifiEngine(replace(cfg, asteroid_mass_kg=5e18), 3)
        small.run_interval(5.0, 1e18, 3e6, [0, 1, 0])
        ok = abs(c["mass_residual_rel"]) < 1e-14 and small.m_ast >= 0 and small.ledger[LG_MASS_EJECTED] <= 5e18 * (1 + 1e-12)
        return ok, (f"total mass + escaped mass - initial = {c['mass_residual_rel']:.1e} (rel); merges={c['merges']}, "
                    f"escaped {h.ledger[LG_ESC_MASS]:.2e} kg; reservoir cannot go negative "
                    f"(5e18 kg reservoir -> ejected {small.ledger[LG_MASS_EJECTED]:.2e} kg)")

    def reverse():
        """Check 8: reversing the ejection direction reverses the push."""
        res = {}
        for name, d in (("pro", [0, 1, 0]), ("retro", [0, -1, 0])):
            f = FastEngine(cfg, 3)
            a0 = orbit_elements(*f.emb(), f.mu())[0]
            f.run_interval(200, 5e17, 2.5e6, d)
            res[name] = orbit_elements(*f.emb(), f.mu())[0] - a0
        sym = abs(res["pro"] + res["retro"]) / abs(res["pro"])
        return res["pro"] > 0 > res["retro"] and sym < 0.2, (
            f"prograde da = {res['pro'] / AU:+.4f} AU, retrograde da = {res['retro'] / AU:+.4f} AU (asymmetry {sym:.2f})")

    def fragments():
        """Check 9: fragments fall back, escape or hit Earth according to their real velocities."""
        f = FastEngine(cfg, 3)
        v0 = f.st[3:6].copy()
        f.run_interval(1e-6, 1e18, 5e3, [0, 1, 0])               # 5 km/s < escape speed
        fb_ok = f.ledger[LG_N_FALLBACK] == 1 and f.m_ast == cfg.asteroid_mass_kg and f.ledger[LG_E_DRAWN] > 0
        h = _fresh_hifi(cfg)
        h.run_interval(0.125, 1e18, 3e6, [0, 1, 0])
        h.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        esc_ok = abs(h.ledger[LG_ESC_MASS] - 1e18) < 1e6
        g = _fresh_hifi(cfg)
        g.pos[NMAJ] = g.pos[1] + [2e8, 0, 0]; g.vel[NMAJ] = g.vel[1] + [-2e4, 0, 0]; g.mass[NMAJ] = 1e18; g.nbarr[0] = NMAJ + 1
        p0 = g.totals()[1:4].copy()
        g.run_interval(0.02, 0.0, 3e6, [0, 1, 0])
        p1 = g.totals()[1:4]
        imp_ok = g.ledger[LG_EARTH_IMPACT_M] == 1e18 and g.nbarr[0] == NMAJ and np.linalg.norm(p1 - p0) / (1e18 * 2e4) < 1e-6
        return fb_ok and esc_ok and imp_ok, (
            f"sub-escape packet falls back (no net recoil, energy lost): {fb_ok}; fast packet leaves the system "
            f"and is ledgered as escaped: {esc_ok}; fragment aimed at Earth merges with momentum conserved: {imp_ok}")

    def reproducible():
        """Check 11: the same seed gives the same result."""
        a = run_episode(replace(cfg, randomize_phase=True, mission_years=1000.0), policy_heuristic, 7)
        b = run_episode(replace(cfg, randomize_phase=True, mission_years=1000.0), policy_heuristic, 7)
        return a["a_au"] == b["a_au"] and a["total_reward"] == b["total_reward"], f"identical results for seed 7 (a={a['a_au']:.6f} AU)"

    def convergence():
        """Check 10: the integrator error shrinks 16x when the step is halved (4th order)."""
        errs = []
        def run(dt_h):
            """Earth's position after 10 years with time step dt_h hours."""
            h = HifiEngine(replace(cfg, hifi_dt_hours=dt_h), 3)
            h.run_interval(10.0, 0.0, 3e6, [0, 1, 0])
            return h.pos[1] - h.pos[0]
        ref = run(0.75)
        for dt in (12.0, 6.0, 3.0):
            errs.append(float(np.linalg.norm(run(dt) - ref)))
        r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
        return errs[2] < errs[1] < errs[0] and r2 > 6, (f"Earth position error after 10 y at dt 12/6/3 h: "
                                                          f"{errs[0]:.0f}/{errs[1]:.0f}/{errs[2]:.0f} m (ratios {r1:.1f}, {r2:.1f}; 4th order -> 16)")

    def energy_drift():
        """Check 10b: energy and angular momentum stay constant without thrust."""
        h = HifiEngine(cfg, 3)
        h.run_interval(100.0, 0.0, 3e6, [0, 1, 0])
        c = h.conservation()
        return abs(c["energy_residual_rel"]) < 1e-10 and c["angmom_residual_rel"] < 1e-12, (
            f"100 y ballistic: |dE/E| = {abs(c['energy_residual_rel']):.1e}, |dL|/|L| = {c['angmom_residual_rel']:.1e}")

    def fast_vs_hifi():
        """Same heuristic mission in both models.  (a) Physics agreement with the Mars shield off and Mars
        massless (no close-pass chaos): must agree to 0.1 %.  (b) Full model (Mars gravity + shield): the
        shield's yes/no decisions can differ between the models after tiny differences, so only a looser
        agreement is required (reported)."""
        c2 = replace(cfg, mission_years=1500.0, mars_shield=False, mars_gravity=False, mars_min_distance_km=1e4,
                     mars_max_tide_ratio=10.0, mars_max_moon_perturbation=0.1)
        a = run_episode(c2, policy_heuristic, 11, "fast")
        b = run_episode(c2, policy_heuristic, 11, "hifi")
        rel = abs(a["a_au"] - b["a_au"]) / b["a_au"]
        de = abs(a["e"] - b["e"])
        c3 = replace(cfg, mission_years=1500.0)
        a3 = run_episode(c3, policy_heuristic, 11, "fast")
        b3 = run_episode(c3, policy_heuristic, 11, "hifi")
        rel3 = abs(a3["a_au"] - b3["a_au"]) / b3["a_au"]
        return rel < 1e-3 and de < 5e-3 and rel3 < 2e-2, (
            f"heuristic, 1,500 y, shield off: a fast {a['a_au']:.5f} vs high-fidelity {b['a_au']:.5f} AU (rel {rel:.1e}), "
            f"|de| {de:.1e}; with Mars gravity + shield: {a3['a_au']:.4f} vs {b3['a_au']:.4f} AU (rel {rel3:.1e}; "
            f"{a3['shield_overrides']} vs {b3['shield_overrides']} shield overrides)")

    def polar_hermite(r0, v0, r1, v1, h, s_):
        """Same interpolation as frontend/app.js: cubic Hermite in (radius, angle, z) using the
        recorded velocities as end-point derivatives."""
        def pol(r, v):
            """Test policy that pushes along the orbit normal."""
            rho2 = r[0] ** 2 + r[1] ** 2
            rho = math.sqrt(rho2)
            return rho, math.atan2(r[1], r[0]), r[2], (r[0] * v[0] + r[1] * v[1]) / rho, \
                (r[0] * v[1] - r[1] * v[0]) / rho2, v[2]
        a0, a1 = pol(r0, v0), pol(r1, v1)
        th1 = a1[1]
        while th1 - a0[1] < -math.pi:
            th1 += 2 * math.pi
        while th1 - a0[1] > math.pi:
            th1 -= 2 * math.pi
        h00, h10, h01, h11 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1, s_ ** 3 - 2 * s_ ** 2 + s_, -2 * s_ ** 3 + 3 * s_ ** 2, s_ ** 3 - s_ ** 2
        rho = h00 * a0[0] + h10 * h * a0[3] + h01 * a1[0] + h11 * h * a1[3]
        th = h00 * a0[1] + h10 * h * a0[4] + h01 * th1 + h11 * h * a1[4]
        z = h00 * a0[2] + h10 * h * a0[5] + h01 * a1[2] + h11 * h * a1[5]
        return np.array([rho * math.cos(th), rho * math.sin(th), z])

    def visual_continuity():
        """The viewer interpolates between recorded states.  Verify that (a) consecutive fast-model
        rows are connected by a pure ballistic Kepler arc (no hidden jump), (b) the cubic Hermite
        curve the viewer draws stays within 0.001 AU of the exact arc, (c) the recoil recorded in
        each row equals |v_post - v_pre|, and (d) packet paths follow two-body solar gravity."""
        f = FastEngine(replace(cfg, mars_gravity=False, other_planets=False), 3)
        smp, _ = f.run_interval(5.0, 1e18, 3e6, [0.0, 1.0, 0.0])
        fm = FastEngine(cfg, 3)
        smp_m, _ = fm.run_interval(5.0, 1e18, 3e6, [0.0, 1.0, 0.0])
        mars_dev = float(np.abs(smp_m[:, S_R:S_R + 3] - smp[:, S_R:S_R + 3]).max())
        mu = f.mu()
        worst_join, worst_herm, worst_dv = 0.0, 0.0, 0.0
        for i in range(len(smp) - 1):
            a, b = smp[i], smp[i + 1]
            h = (b[S_T] - a[S_T]) * YEAR
            r = a[S_R:S_R + 3] * AU
            v = a[S_VPOST:S_VPOST + 3] * 1000.0
            r_end, v_end = r.copy(), v.copy()
            kepler_step(r_end, v_end, mu, h)
            worst_join = max(worst_join, float(np.linalg.norm(r_end - b[S_R:S_R + 3] * AU)) / AU)
            r_mid, v_mid = r.copy(), v.copy()
            kepler_step(r_mid, v_mid, mu, 0.5 * h)
            herm = polar_hermite(r, v, b[S_R:S_R + 3] * AU, b[S_VPRE:S_VPRE + 3] * 1000.0, h, 0.5)
            worst_herm = max(worst_herm, float(np.linalg.norm(herm - r_mid)) / AU)
            dv = np.linalg.norm(a[S_VPOST:S_VPOST + 3] - a[S_VPRE:S_VPRE + 3]) * 1000.0
            worst_dv = max(worst_dv, abs(dv - a[S_DV]) / a[S_DV])
        # packet arc: energy of the two-body path must be constant; 1-day position ~ straight line
        r0 = smp[0, S_R:S_R + 3][None, :] * AU
        v0 = smp[0, S_VP:S_VP + 3][None, :] * 1000.0
        offs = PACKET_OFFSETS_DAYS * 86400.0
        out = np.zeros((1, len(offs), 3))
        nv = np.zeros(1, dtype=np.int64)
        packet_arcs(r0, v0, offs, cfg.fragment_escape_au * AU, out, nv)
        j = int(np.searchsorted(PACKET_OFFSETS_DAYS, 1.0))
        straight = r0[0] + v0[0] * 86400.0
        dev = float(np.linalg.norm(out[0, j] - straight)) / float(np.linalg.norm(out[0, j] - r0[0]))
        ok = worst_join < 1e-9 and worst_herm < 1e-4 and worst_dv < 1e-6 and 0 < dev < 1e-3 and nv[0] > 3 and mars_dev < 1e-3
        return ok, (f"ballistic join between rows {worst_join:.1e} AU without the planets' pull (no hidden jumps); Mars + Mercury + Venus + Jupiter move "
                    f"Earth by up to {mars_dev:.1e} AU in 5 y (smooth, far below the drawing accuracy); viewer Hermite curve vs exact "
                    f"arc max {worst_herm:.1e} AU; recoil |v_post - v_pre| matches recorded value to {worst_dv:.1e}; "
                    f"packet path bends {dev:.1e} (relative) from a straight line after 1 day under solar gravity")

    def fast_moon():
        """The training model simulates the Moon as a gravitating body (Earth-Moon two-body orbit
        about their barycentre).  Compare with the N-body Moon for 1 year without thrust, check the
        Earth's wobble produced by the Moon's pull, and check that a packet launch changes the
        Earth-Moon relative velocity by exactly Earth's recoil (the Moon is not pushed directly)."""
        f, h = FastEngine(cfg, 3), _fresh_hifi(cfg)
        f.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        h.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        df = np.linalg.norm(f.last_moon[:, 1:4], axis=1)
        dh = np.linalg.norm(h.last_moon[:, 1:4], axis=1)
        dmax = float(np.abs(df - dh).max())
        a1, a2 = f.last_moon[-1, 1:4], h.last_moon[-1, 1:4]
        phase = math.degrees(math.acos(max(-1.0, min(1.0, float(a1 @ a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))))))
        emb = float(np.linalg.norm(f.emb()[0] - h.emb()[0]))
        rE, _ = f.earth_moon()
        wobble = float(np.linalg.norm(rE - f.st[0:3]))
        g, g0 = FastEngine(cfg, 3), FastEngine(cfg, 3)           # same interval with and without a packet
        g.run_interval(1e-9, 1e18, 3e6, [0, 1, 0])
        g0.run_interval(1e-9, 0.0, 3e6, [0, 1, 0])
        dvr = float(np.linalg.norm(g.st[14:17] - g0.st[14:17]))
        m_l = M_EARTH + cfg.asteroid_mass_kg
        vinf = math.sqrt(3e6 ** 2 - 2 * G * m_l / R_EARTH - 2 * G * M_MOON / np.linalg.norm(g0.st[11:14]))
        expect = 1e18 / m_l * vinf
        rel_dv = abs(dvr - expect) / expect
        ok = (f.moon_info()["bound"] and h.moon_info()["bound"] and dmax < 0.05 * MOON_A and emb < 1e6
              and 4.0e6 < wobble < 5.2e6 and rel_dv < 1e-6)
        return ok, (f"1 y without thrust: Earth-Moon distance fast {df.min() / 1e3:,.0f}-{df.max() / 1e3:,.0f} km vs N-body "
                    f"{dh.min() / 1e3:,.0f}-{dh.max() / 1e3:,.0f} km (max diff {dmax / 1e3:,.0f} km); Moon's orbital phase "
                    f"drifts {phase:.0f} deg/yr (Sun's tide on the Moon not in the fast model); barycentre diff "
                    f"{emb / 1e3:.0f} km; Earth wobble from the Moon's pull {wobble / 1e3:,.0f} km; recoil changes the "
                    f"Earth-Moon relative velocity by Earth's recoil only (rel. error {rel_dv:.1e}); Moon bound: yes")

    def mars_gravity_check():
        """Mars in both models: (a) the fast model's Mars and Earth agree with the N-body model after 20 years
        without thrust, (b) Mars's pull changes Earth's motion (Mars placed 0.02 AU from Earth for 1 year,
        gravity on vs off), (c) the Earth <-> Mars pull is equal and opposite (direct terms), so Mars's orbit
        is changed by Earth just as Earth's is by Mars."""
        f, h = FastEngine(cfg, 3), _fresh_hifi(cfg)
        f.run_interval(20.0, 0.0, 3e6, [0, 1, 0])
        h.run_interval(20.0, 0.0, 3e6, [0, 1, 0])
        dmars = float(np.linalg.norm(f.mars_state()[0] - h.mars_state()[0]))
        demb = float(np.linalg.norm(f.emb()[0] - h.emb()[0]))

        def near(grav):
            """Run with Mars placed close to Earth, gravity on or off."""
            g = FastEngine(replace(cfg, mars_gravity=grav, mars_shield=False), 3)
            r, v = g.st[0:3].copy(), g.st[3:6].copy()
            g.st[17:20] = r + 0.02 * AU * v / np.linalg.norm(v)
            g.st[20:23] = v.copy()
            g.run_interval(1.0, 0.0, 3e6, [0, 1, 0], shield=False)
            return g
        dv_e = float(np.linalg.norm(near(True).st[3:6] - near(False).st[3:6]))
        g3 = FastEngine(replace(cfg, mars_shield=False), 3)
        r, v = g3.st[0:3].copy(), g3.st[3:6].copy()
        g3.st[17:20] = r + np.array([0.03 * AU, 0.0, 0.0]); g3.st[20:23] = v.copy()
        p_e0, p_m0 = g3.st[3:6].copy(), g3.st[20:23].copy()
        _mars_kick(g3.st, 1000.0, GM_MARS)
        de, dm_ = g3.st[3:6] - p_e0, g3.st[20:23] - p_m0
        ind_e = -GM_MARS * g3.st[17:20] / np.linalg.norm(g3.st[17:20]) ** 3 * 1000.0
        ind_m = -G * (g3.st[6] + M_MOON) * g3.st[0:3] / np.linalg.norm(g3.st[0:3]) ** 3 * 1000.0
        pe = (g3.st[6] + M_MOON) * (de - ind_e)
        pm = M_MARS * (dm_ - ind_m)
        n3 = float(np.linalg.norm(pe + pm) / np.linalg.norm(pe))
        ratio = float(np.linalg.norm(dm_ - ind_m) / np.linalg.norm(de - ind_e))
        ok = dmars < 1e7 and demb < 1e7 and dv_e > 1.0 and n3 < 1e-9
        return ok, (f"20 y without thrust: Mars position fast vs N-body {dmars / 1e3:,.0f} km, Earth-Moon centre "
                    f"{demb / 1e3:,.0f} km; Mars placed 0.02 AU ahead of Earth, after 1 y, changes Earth's velocity by {dv_e:.1f} m/s "
                    f"(gravity on vs off); Earth<->Mars pull equal and opposite (momentum residual {n3:.1e}); Mars is "
                    f"kicked {ratio:.2f}x harder than Earth (= mass ratio), so Mars's orbit changes too")

    def mars_safety_check():
        """The safety limits: at the safe distance the tide / Moon-perturbation ratios equal the configured limits;
        the reported safe distance is the largest limit; the shield forecast reproduces a known coasting closest
        approach; a shielded heuristic mission (fast model, seed 5) never comes closer than the safe distance."""
        lim = mars_safety_limits(cfg)
        r1 = mars_tide_ratio(lim["d_tide_m"]) / cfg.mars_max_tide_ratio
        r2 = mars_moon_perturbation(lim["d_moon_m"]) / cfg.mars_max_moon_perturbation
        big = lim["d_safe_m"] == max(lim["d_tide_m"], lim["d_moon_m"], lim["d_min_m"])
        f = FastEngine(replace(cfg, mars_gravity=False, mars_shield=False), 4)
        r, v = f.emb(); rm, vm = f.mars_state()
        fc = mars_forecast(r, v, f.mu(), rm, vm, 10 * YEAR, YEAR / 8, 0.0, np.array([0.0, 1.0, 0.0]), G * (f.m_launcher + M_MOON), 0.0)
        f.run_interval(10.0, 0.0, 3e6, [0, 1, 0], shield=False)
        sim = f.mars_interval()["dmin_m"]
        fc_err = abs(fc - sim) / sim
        c5 = replace(cfg, randomize_phase=True)
        s5 = run_episode(c5, policy_heuristic, 5)
        ok = abs(r1 - 1) < 1e-12 and abs(r2 - 1) < 1e-12 and big and fc_err < 0.02 and s5["mars_safe"]
        return ok, (f"safe distance {lim['d_safe_m'] / 1e9:.2f} million km ({lim['d_safe_au']:.4f} AU, set by: {lim['governing']}); "
                    f"tide ratio at the tide limit = {r1:.12f} x limit; forecast vs simulated 10-y closest approach "
                    f"{fc / AU:.4f} vs {sim / AU:.4f} AU ({fc_err:.1e}); shielded heuristic mission (seed 5): closest Mars "
                    f"pass {s5['mars_min_dist_au']:.4f} AU, {s5['shield_overrides']} shield overrides, outcome {s5['reason']}")

    def solar_check():
        """Collectors: the power chain is the exact product of the configured factors; the capture efficiency follows the
        Goubau relation (tau = 1 -> 1 - 1/e); the belt sample covers the configured share of the sky; both engines never
        draw more than reserve + beam energy offered, and with a tiny reserve the launches are powered by the beams; with the
        collectors off the reserve is the only energy."""
        c = replace(cfg, solar_collectors=True)
        pw = collector_power(c, [AU, 0.0, 0.0])
        op = collector_operating(c)
        bal_panel = abs(c.collector_absorptance * op["S_wm2"] - op["eta_temp"] * op["S_wm2"]
                        - 2 * c.collector_emissivity * SIGMA_SB * op["T_panel_K"] ** 4) / op["S_wm2"]
        chain = (L_SUN * c.collector_fraction * (1 - pw["lane_frac"]) * c.pv_efficiency * op["temp_factor"] * op["degr_factor"] *
                 c.microwave_efficiency * pw["eta_capture"] * c.rectenna_efficiency)
        chain_err = abs(chain - pw["P_elec_W"]) / chain
        lam = C_LIGHT / (c.microwave_freq_ghz * 1e9)
        d1 = c.transmitter_diameter_km * 1e3 * c.rectenna_diameter_km * 1e3 / lam
        g_err = abs(float(beam_capture_efficiency(c, d1)) - (1 - math.exp(-1.0)))
        pts = _collector_points(c)
        band = float(np.abs(pts[:, 2]).max() / (c.collector_orbit_au * AU))      # max |sin(latitude)| of the sample
        band_ok = band <= c.collector_fraction + 1e-12
        c2 = replace(c, energy_budget_j=1e28)
        res = []
        for eng in (FastEngine(c2, 3), HifiEngine(c2, 3)):
            eng.run_interval(2.0, 1e18, 3e6, [0.0, 1.0, 0.0], shield=False)
            drawn, S = float(eng.ledger[LG_E_DRAWN]), eng.sol
            bal = abs(c2.energy_budget_j + S["E_beam_used_J"] - eng.E_rem - drawn) / drawn
            lim = S["E_beam_used_J"] <= S["E_offered_J"] * (1 + 1e-12) and eng.E_rem <= c2.energy_budget_j
            res.append((drawn, bal, lim))
        f0 = FastEngine(replace(c2, solar_collectors=False), 3)
        f0.run_interval(2.0, 1e18, 3e6, [0.0, 1.0, 0.0], shield=False)
        off_ok = f0.ledger[LG_E_DRAWN] <= c2.energy_budget_j * (1 + 1e-12)
        des = collector_design(c)
        p_peak = c.ejections_per_year * c.max_packet_mass_kg * 0.5 * c.max_ejection_speed_ms ** 2 / c.launch_efficiency / YEAR
        ok = (chain_err < 1e-12 and g_err < 1e-12 and band_ok and off_ok and bal_panel < 1e-9 and
              all(b < 1e-9 and lim and d > 100 * c2.energy_budget_j for d, b, lim in res))
        return ok, (f"power chain {L_SUN:.3e} W x {c.collector_fraction} x (1 - lanes {pw['lane_frac']:.1e}) x rated {c.pv_efficiency} "
                    f"x temperature {op['temp_factor']:.3f} (panels {op['T_panel_K']:.0f} K, energy balance {bal_panel:.0e}) x age "
                    f"{op['degr_factor']:.3f} x {c.microwave_efficiency} x capture "
                    f"{pw['eta_capture']:.3f} x {c.rectenna_efficiency} = {pw['P_elec_W']:.3e} W at 1 AU (exact to {chain_err:.0e}); "
                    f"Goubau tau=1 -> {1 - math.exp(-1):.4f} ({g_err:.0e}); belt covers |latitude| <= {des['half_width_deg']:.1f} deg; "
                    f"with a reserve of only 1e28 J the fast / N-body engines launched {res[0][0]:.2e} / {res[1][0]:.2e} J powered by "
                    f"the beams (bookkeeping residual {max(res[0][1], res[1][1]):.0e}, never more than offered); collectors off -> "
                    f"never more than the reserve: {off_ok}; {des['count']:.0e} collectors of {des['each_side_km']:,.0f} km each; "
                    f"Earth's clear sunlight lane {des['lane_frac_earth']:.1e} of the belt; beams supply {pw['P_elec_W'] / p_peak:.1f}x the "
                    f"launcher's peak demand")

    def climate_ledger_check():
        """Climate: 288 K today; without the space mirrors Earth at 2.7 AU freezes (snowball branch); with them the heuristic
        mission holds the temperature in tolerance; the energy ledger of every interval closes to round-off and the mirrors
        never send more light than they catch."""
        T_now = climate_equilibrium(1.0)
        T_far = climate_equilibrium(1 / 2.7 ** 2)
        c = make_config("hypothetical", randomize_phase=False)
        s_on = run_episode(c, policy_heuristic, 5)
        s_off = run_episode(replace(c, mirrors=False), policy_heuristic, 5)
        resid = max(float(s_on["energy_ledger_residual"]), max((d["ledger"].get("residual_rel", 0) for d in s_on["decisions"]), default=0))
        L = s_on["energy_ledger"]
        e_in = L["sun_on_collectors"] + L["reserve_used"] + L["mirror_sun"]
        mirror_ok = all(d["ledger"].get("mirror_to_space", 0) >= -1e-9 * max(d["ledger"].get("mirror_sun", 0), 1.0)
                        for d in s_on["decisions"])
        md = mirror_design(c, c.target_au)
        ok = (abs(T_now - 288.0) < 0.05 and T_far < 200 and s_on["reason"] == "success" and s_on["climate_ok"] and
              s_off["T_surface_K"] < 230 and s_off["reason"] != "success" and resid < 1e-9 and mirror_ok)
        return ok, (f"equilibrium today {T_now:.2f} K, at 2.7 AU without extra light {T_far:.0f} K (frozen); mirrors "
                    f"{md['area_m2']:.1e} m^2 ({md['mass_kg']:.0e} kg) can send {md['cap_rel']:.2f} x today's sunlight at "
                    f"{c.target_au} AU; heuristic mission with the space mirrors: {s_on['reason']}, final {s_on['T_surface_K']:.1f} K "
                    f"(max deviation {s_on['climate_dev_max_K']:.1f} K, "
                    f"{s_on['years_outside_livable']:.0f} y outside +-{CLIM_LIVABLE_K:.0f} K); without it: {s_off['reason']}, "
                    f"{s_off['T_surface_K']:.0f} K; ledger: in {e_in:.3e} J = reflected {L['reflected']:.2e} + collector heat "
                    f"{L['collector_heat']:.2e} + manufacturing {L['manufacturing']:.1e} + transmitter heat {L['transmitter_heat']:.2e} + "
                    f"spill {L['beam_spill']:.2e} + rectenna heat {L['rectenna_heat']:.2e} + packet KE {L['packet_ke']:.2e} + launcher "
                    f"heat {L['launcher_heat']:.2e}; mirrors: sunlight {L['mirror_sun']:.2e} = onto Earth {L['mirror_to_earth']:.2e} "
                    f"+ past Earth {L['mirror_to_space']:.2e} + mirror heat {L['mirror_heat']:.2e} (residual {resid:.0e}); "
                    f"heat reaching Earth {L['heat_to_earth']:.2e} J")

    def launcher_check():
        """Check 21 (p36): the asteroid sits on the launch ring's axis, so all 8 arm roots are equally far from the grab
        point; 2 R_E arms reach it (and the asteroid's centre); the joint throw has a non-zero stroke; and the
        simplifications that keep the physics unchanged are negligible (escape energy from the surface instead of the
        release point; asteroid merged with Earth: centre-of-mass offset and tidal error on the Moon)."""
        g = launcher_geometry(cfg)
        g_end = launcher_geometry(cfg, m_ast=0.0)
        ok = (g["root_to_grab_spread_re"] < 1e-12 and g["root_to_grab_re"] <= g["arm_length_re"]
              and g["root_to_asteroid_centre_re"] <= g["arm_length_re"] and g["grab_height_re"] - 1.0 > 0.1
              and g["ring_radius_re"] > 1.05 and g["stroke_re"] > 0.1 and g["escape_energy_saving_rel"] < 1e-4
              and g["moon_pull_error_rel"] < 1e-4 and g["com_shift_km"] < 100 and g_end["stroke_re"] == g["stroke_re"])
        return ok, (f"8 roots on a {g['ring_radius_re']:.2f} R_E ring, grab point {g['grab_height_re']:.2f} R_E up the axis: every root "
                    f"{g['root_to_grab_re']:.4f} R_E from it (spread {g['root_to_grab_spread_re']:.0e}) <= arm {g['arm_length_re']:.1f} R_E; "
                    f"asteroid (r {g['asteroid_radius_re']:.3f} R_E at {AST_DENSITY:.0f} kg/m^3) centre {g['root_to_asteroid_centre_re']:.3f} R_E "
                    f"from every root; {g['asteroid_clearance_km']:,.0f} km above the surface; joint-throw stroke "
                    f"{g['stroke_km']:,.0f} km (same when the asteroid is used up) -> {g['accel_typ_ms2']:.2e} m/s^2 for 2,480 km/s; "
                    f"escape energy from the surface instead of the release point ({g['release_radius_re']:.3f} R_E): "
                    f"+{g['escape_energy_saving_rel']:.1e} (conservative); asteroid merged with Earth: centre-of-mass offset "
                    f"{g['com_shift_km']:.0f} km, tidal error on the Moon {g['moon_pull_error_rel']:.1e} of Earth's pull")

    def planets_check():
        """Check 20 (p35): Mercury, Venus and Jupiter in both models. (a) 100 y without thrust: their positions and the
        Earth-Moon centre agree between the fast model and the N-body model; (b) their orbits stay put over 1,000 y in
        the fast model (Earth's pull changes them only very slightly); (c) Jupiter's pull on Earth vs Earth's on
        Jupiter = mass ratio (N-body); (d) a 1,500-year heuristic mission with and without the planets (fast
        model): how much they change Earth's final orbit; (e) safe distances with the Mars rules."""
        f, h = FastEngine(cfg, 3), _fresh_hifi(cfg)
        f.run_interval(100.0, 0.0, 3e6, [0, 1, 0])
        h.run_interval(100.0, 0.0, 3e6, [0, 1, 0])
        rf, _ = f.planets_state()
        rh, _ = h.planets_state()
        dpos = np.linalg.norm(rf - rh, axis=1) / AU
        demb = float(np.linalg.norm(f.emb()[0] - h.emb()[0])) / AU
        g = FastEngine(cfg, 4)
        rp0, vp0 = g.planets_state()
        a0 = [_planet_elements(rp0[k], vp0[k], k)["a_au"] for k in range(N_PL)]
        g.run_interval(1000.0, 0.0, 3e6, [0, 1, 0])
        rp1, vp1 = g.planets_state()
        da = [abs(_planet_elements(rp1[k], vp1[k], k)["a_au"] - a0[k]) / a0[k] for k in range(N_PL)]
        hh = _fresh_hifi(cfg)
        d = hh.pos[6] - hh.pos[1]
        aEJ = G * hh.mass[6] / np.linalg.norm(d) ** 2
        aJE = G * hh.mass[1] / np.linalg.norm(d) ** 2
        ratio_ok = abs(aJE / aEJ - hh.mass[1] / hh.mass[6]) < 1e-12
        c2 = replace(cfg, mission_years=1500.0)
        on = run_episode(c2, policy_heuristic, 11, "fast")
        off = run_episode(replace(c2, other_planets=False), policy_heuristic, 11, "fast")
        lim = planet_safety_limits(cfg)
        pl_on = {q["name"]: q for q in on["planets"]}
        ok = (dpos.max() < 2e-3 and demb < 2e-3 and max(da) < 1e-3 and ratio_ok and on["planets_safe"]
              and abs(on["a_au"] - off["a_au"]) < 0.05)
        return ok, (f"100 y without thrust, fast vs N-body: Mercury {dpos[0]:.1e}, Venus {dpos[1]:.1e}, Jupiter {dpos[2]:.1e} AU, "
                    f"Earth-Moon centre {demb:.1e} AU; orbit sizes after 1,000 y change by {max(da):.1e} (relative); Jupiter pulls "
                    f"Earth {aEJ:.2e} m/s^2 and Earth pulls Jupiter {aJE:.2e} (ratio = mass ratio: {ratio_ok}); 1,500-y heuristic "
                    f"mission: final a {on['a_au']:.4f} AU with the planets vs {off['a_au']:.4f} without, e {on['e']:.4f} vs "
                    f"{off['e']:.4f}; closest Venus {pl_on['Venus']['min_dist_au']:.3f} AU, Jupiter {pl_on['Jupiter']['min_dist_au']:.3f} AU; "
                    f"safe distances (Mars rules): Mercury {lim[0]['d_safe_au']:.4f}, Venus {lim[1]['d_safe_au']:.4f}, "
                    f"Jupiter {lim[2]['d_safe_au']:.4f} AU")

    def belt_check():
        """Collector belt along a heuristic mission's first 300 years: with station keeping the deviation stays small; without
        it the collectors drift much further; the belt's pull on Earth is negligible."""
        c = make_config("hypothetical", randomize_phase=False)
        env = OctoEarthEnv(c, "fast", record=True)
        env.reset(seed=5)
        bk = BeltKeeper(c)
        last = None
        for _ in range(3):
            t_s = env.engine.t_years
            env.step(policy_heuristic(env, None))
            L = env.engine.last_eled or {}
            last = bk.advance(env.last_samples, L.get("microwaves_sent", 0.0) / (c.decision_interval_years * YEAR), t_s)
        sm = bk.summary()
        bp = belt_pull_on_earth(c)
        ok = (sm["on"]["max_km"] < 5000 and sm["on"]["max_km"] < 0.1 * sm["spacing_km"] and
              sm["off"]["max_km"] > 10 * sm["on"]["max_km"] and bp["ratio_1au"] < 1e-6 and
              sm["recoil_ratio_max"] < c.collector_sk_authority * bk.beta)
        return ok, (f"{BELT_N_TRACK} collectors x 300 y: station keeping ON -> max distance from the assigned slot "
                    f"{sm['on']['max_km']:,.0f} km (rms {sm['on']['rms_km']:,.0f} km; spacing between collectors "
                    f"{sm['spacing_km']:,.0f} km), control {sm['on']['dv_ms_per_yr']:.0f} m/s per year each; OFF -> the orbit's "
                    f"shape changes little (semi-major axis {sm['off']['da_km']:,.0f} km, e {sm['off']['e_max']:.1e}, distance from "
                    f"the Sun swings by up to {sm['off']['swing_km']:,.0f} km) but the collectors drift along the orbit up to {sm['off']['max_km']:,.0f} km "
                    f"= {sm['off']['slots']:,.0f} slots (collisions); beam recoil "
                    f"{sm['recoil_ratio_max']:.1e} of the Sun's pull, planets {sm['planet_ratio']:.1e}, neighbour "
                    f"{sm['neighbour_ratio']:.0e}, control limit {c.collector_sk_authority * bk.beta:.1e}; belt's net pull on Earth "
                    f"{bp['ratio_1au']:.0e} (1 AU) / {bp['ratio_target']:.0e} (target) of the Sun's")

    log("[CHECK] physics checks" + (" (full)" if full else " (quick)"))
    check("1 two-body motion vs Kepler", kepler)
    check("2 gravity of Sun / Earth / Moon / asteroid / fragments", gravity)
    check("3 equal and opposite forces", newton3)
    check("4+6 momentum & angular momentum in ejection", eject_momentum)
    check("5 energy accounting of propulsion", energy_accounting)
    check("7 finite asteroid mass & fragment mass accounting", mass_accounting)
    check("8 reversed ejection direction", reverse)
    check("9 fragment fall-back / escape / Earth impact", fragments)
    check("11 reproducibility (fixed seed)", reproducible)
    check("13 viewer continuity (interpolation of recorded states)", visual_continuity)
    check("14 Moon in the fast (training) model", fast_moon)
    check("15 Mars gravity (both models, two-way)", mars_gravity_check)
    check("16 planetary-safety limits and Mars shield", mars_safety_check)
    check("17 solar collectors and microwave power beaming", solar_check)
    check("18 climate + space mirrors + energy ledger", climate_ledger_check)
    check("19 collector belt station keeping", belt_check)
    check("20 other planets: Mercury, Venus, Jupiter (both models, two-way)", planets_check)
    check("21 launcher geometry: asteroid on the ring axis, 2 R_E arms", launcher_check)
    if full:
        check("10 integrator convergence", convergence)
        check("10b energy / angular-momentum drift", energy_drift)
        check("12 fast model vs high-fidelity N-body", fast_vs_hifi)
    return out


# =============================================================================================
# 10. APPLICATION STATE, TRAINING, COMPARISON, EXPLANATION
# =============================================================================================
# Row layout sent to the browser (see frontend/app.js, constant R_*):
# t[yr], x,y,z[AU], v_pre(3)[km/s], v_post(3)[km/s], dm[kg], u[m/s], v_rel[m/s], fate, m_ast[kg],
# E_rem[J], n_frag, recoil dv[m/s], E_draw[J], ehat(3)
def _g(x, n=5):
    """Round x to n significant digits (smaller JSON files)."""
    return float(f"{float(x):.{n}g}")


N_VIEW_COLS = 55          # 34 + 21 planet columns (p35): Mercury/Venus/Jupiter positions [AU], velocities [km/s], Earth distances [AU]
POWER_METRIC_KEYS = ("solar_on", "energy_from_collectors_J", "energy_from_reserve_J", "beam_energy_absorbed_J", "beam_offered_J",
                     "beam_use_frac", "beam_mean_W", "beam_peak_W", "beam_avail_now_W", "eta_capture_min", "eta_capture_max",
                     "eta_capture_now", "arm_heat_J", "arm_heat_mean_W", "arm_heat_vs_earth_sunlight", "radiator_area_1000K_m2",
                     "beam_push_dv_ms")


def _power_cols(q, power, cfg):
    """Viewer columns 30-33: beam power available at the arms [W], microwave power absorbed by the rectennas in this
    event [W], capture efficiency, electric power drawn by the launcher in this event [W]."""
    if not power:
        return [0.0, 0.0, 0.0, 0.0]
    dt = power["dt_event_s"]
    e_draw = float(q[S_EDRAW])
    beam_el = min(e_draw, power["P_elec_W"] * dt)
    return [_g(power["P_elec_W"], 4), _g(beam_el / cfg.rectenna_efficiency / dt, 4), round(power["eta_capture"], 4), _g(e_draw / dt, 4)]


def _viewer_rows(samples, power=None, cfg=None):
    """Pack the engine's samples into compact rows for the viewer."""
    out = []
    for q in samples:
        out.append([round(float(q[S_T]), 6)] + [round(float(x), 7) for x in q[S_R:S_R + 3]] +
                   [round(float(x), 5) for x in q[S_VPRE:S_VPRE + 3]] + [round(float(x), 5) for x in q[S_VPOST:S_VPOST + 3]] +
                   [_g(q[S_DM], 4), int(q[S_U]), int(q[S_VREL]), int(q[S_FATE]), _g(q[S_MAST], 6), _g(q[S_EREM], 6),
                    int(q[S_NFRAG]), _g(q[S_DV], 4), _g(q[S_EDRAW], 4)] + [round(float(x), 4) for x in q[S_EHAT:S_EHAT + 3]] +
                   [round(float(x), 7) for x in q[S_MARS:S_MARS + 3]] + [round(float(x), 5) for x in q[S_MARSV:S_MARSV + 3]] +
                   [round(float(q[S_DMARS]), 6), int(q[S_SHIELD])] + _power_cols(q, power, cfg) + _planet_cols(q))
    return out


def _planet_cols(q):
    """Viewer columns 34..54: Mercury, Venus, Jupiter positions [AU] (9), velocities [km/s] (9), distances to Earth [AU] (3)."""
    return [round(float(x), 6) for x in q[S_PL:S_PL + 9]] + [round(float(x), 4) for x in q[S_PLV:S_PLV + 9]] + \
        [round(float(x), 5) for x in q[S_DPL:S_DPL + 3]]


def _initial_row(eng):
    """First viewer row: the state at year 0."""
    r, v, _ = eng.earth_state()
    return [0.0] + [round(float(x) / AU, 7) for x in r] + [round(float(x) / 1000, 5) for x in v] * 2 + \
        [0.0, 0, 0, 0, _g(eng.m_ast, 6), _g(eng.E_rem, 6), 0, 0.0, 0.0, 0.0, 0.0, 0.0] + _mars_row(eng, r) + _initial_power(eng, r) + \
        _initial_planets(eng, r)


def _initial_planets(eng, r):
    """Planet columns of the first viewer row (same layout as _planet_cols)."""
    rp, vp = eng.planets_state()
    q = np.zeros(N_SAMPLE_COLS)
    q[S_PL:S_PL + 9] = (rp / AU).ravel()
    q[S_PLV:S_PLV + 9] = (vp / 1000.0).ravel()
    q[S_DPL:S_DPL + 3] = np.linalg.norm(rp - r, axis=1) / AU
    return _planet_cols(q)


def _initial_power(eng, r):
    """Power-supply columns of the first viewer row."""
    if not eng.cfg.solar_collectors:
        return [0.0, 0.0, 0.0, 0.0]
    pw = collector_power(eng.cfg, r)
    return [_g(pw["P_elec_W"], 4), 0.0, round(pw["eta_capture"], 4), 0.0]


def _mars_row(eng, r_earth):
    """Mars's position and velocity for the viewer."""
    rm, vm = eng.mars_state()
    return [round(float(x) / AU, 7) for x in rm] + [round(float(x) / 1000, 5) for x in vm] + \
        [round(float(np.linalg.norm(rm - r_earth)) / AU, 6), 0]


def _moon_rows(moon):
    """The Moon's recorded positions for the viewer."""
    return [[round(float(m[0]), 6)] + [int(round(float(x) / 1000.0)) for x in m[1:4]] + [round(float(x), 1) for x in m[4:7]]
            for m in moon]


def _packet_records(samples, cfg, fidelity):
    """One record per ejected packet: [t, fate, dm, u, n_points, x0, y0, z0, x1, ...] with the
    packet's positions [AU] at PACKET_OFFSETS_DAYS after release, computed here with exact
    two-body solar gravity from the packet's release state (patched conic)."""
    idx = [i for i in range(len(samples)) if samples[i, S_DM] > 0]
    if not idx:
        return []
    sm = samples[idx]
    flying = (sm[:, S_FATE] == 1) | (sm[:, S_FATE] == 2)
    r0 = sm[:, S_R:S_R + 3] * AU
    if fidelity == "hifi":
        r0 = r0 + cfg.release_radius_m * sm[:, S_EHAT:S_EHAT + 3]
    v0 = sm[:, S_VP:S_VP + 3] * 1000.0
    offs = PACKET_OFFSETS_DAYS * 86400.0
    out = np.zeros((len(idx), len(offs), 3))
    nv = np.zeros(len(idx), dtype=np.int64)
    fl = np.where(flying)[0]
    if len(fl):
        o2 = np.zeros((len(fl), len(offs), 3))
        n2 = np.zeros(len(fl), dtype=np.int64)
        packet_arcs(np.ascontiguousarray(r0[fl]), np.ascontiguousarray(v0[fl]), offs, cfg.fragment_escape_au * AU, o2, n2)
        out[fl], nv[fl] = o2, n2
    recs = []
    for j, q in enumerate(sm):
        if flying[j]:
            pts = [round(float(x) / AU, 5) for x in out[j, :nv[j]].ravel()]
            n = int(nv[j])
        else:
            pts = [round(float(x), 5) for x in q[S_R:S_R + 3]]
            n = 1
        recs.append([round(float(q[S_T]), 6), int(q[S_FATE]), _g(q[S_DM], 4), int(q[S_U]), n] + pts)
    return recs


def _policy_label(name):
    """Readable name of a policy (none / heuristic / PPO)."""
    return {"none": "No ejection", "heuristic": "Hand-designed heuristic", "ppo": "Best PPO model"}.get(name, name)


def pick_device(request="auto"):
    """Devices that can be used for the PPO network: "cuda" only if it was asked for ("cuda"/"gpu"/"auto") and a
    usable NVIDIA GPU exists, else "cpu". For "auto" the training worker then MEASURES both (benchmark_devices) and
    keeps the faster one, because a GPU is usually SLOWER for this tiny network (see benchmark_devices)."""
    request = (request or "auto").lower()
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if request == "cpu":
        return "cpu"
    if request in ("cuda", "gpu", "auto") and has_cuda:
        return "cuda"
    return "cpu"


def benchmark_devices(cfg, devices=("cpu", "cuda")):
    """Time the network work of ONE PPO iteration on each device and return {device: seconds}.

    Why this exists: PPO asks the network for one action per simulation step (batch of ONE observation). On a GPU
    every such call has a fixed cost (copy to the GPU, start the kernels, wait, copy back) that is much larger than
    the work itself, and under Windows the GPU is also shared with the browser's 3D view. For a 64x64 network the
    CPU is therefore usually several times faster (Stable-Baselines3 itself warns that PPO with an MLP policy is
    meant for the CPU). Workload per iteration: ppo_n_steps single forward passes (collecting experience) +
    ppo_epochs * (ppo_n_steps / ppo_batch_size) mini-batch updates (learning)."""
    import torch
    import torch.nn as nn
    out = {}
    n_step = int(cfg.ppo_n_steps)
    n_upd = int(cfg.ppo_epochs * max(1, cfg.ppo_n_steps // max(1, cfg.ppo_batch_size)))
    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            torch.manual_seed(0)
            net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                nn.Linear(64, ACT_DIM + 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=3e-4)
            x1 = np.zeros((1, OBS_DIM), np.float32)
            xb = torch.zeros((int(cfg.ppo_batch_size), OBS_DIM), device=dev)

            def work(ns, nu):
                for _ in range(ns):
                    with torch.no_grad():
                        net(torch.as_tensor(x1, device=dev)).cpu().numpy()
                for _ in range(nu):
                    loss = net(xb).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    float(loss.item())
            work(20, 5)                                       # warm-up (CUDA start-up, memory allocation)
            t = time.perf_counter()
            work(n_step, n_upd)
            out[dev] = time.perf_counter() - t
        except Exception:
            continue
    return out


def gpu_report():
    """What PyTorch can see: version, whether it is a CUDA build, the GPU name, and a fix if a GPU is not used."""
    import shutil
    info = dict(torch=None, cuda_build=None, cuda_ok=False, gpu=None, nvidia_present=bool(shutil.which("nvidia-smi")),
                fix=None)
    try:
        import torch
        info.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_ok=bool(torch.cuda.is_available()))
        if info["cuda_ok"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = str(exc)
    if info["nvidia_present"] and not info["cuda_ok"]:
        info["fix"] = ("an NVIDIA GPU is installed but this PyTorch " +
                       ("is the CPU-only build" if not info["cuda_build"] else f"(CUDA {info['cuda_build']}) cannot use it - update the NVIDIA driver") +
                       ". Install the CUDA build:  .venv\\Scripts\\python -m pip install --force-reinstall --no-deps torch "
                       "--index-url https://download.pytorch.org/whl/cu126   (or delete .venv\\gpu_install_tried.txt and run "
                       "RUN_OCTOEARTH.bat again)")
    return info


def device_label(device):
    """Human-readable name of the training device, e.g. 'GPU (NVIDIA GeForce RTX 3060)' or 'CPU'."""
    if device == "cuda":
        try:
            import torch
            return f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "GPU (CUDA)"
    return "CPU (no CUDA GPU found - see README: Using a GPU)"


def _mp_context():
    """multiprocessing with 'spawn' (the Windows default; also safe with numba and torch on Linux)."""
    import multiprocessing as mp
    return mp.get_context("spawn")


def _worker_process(job, cfg, out_dir, device_request, state, q, stop_event, live_event=None):
    """Entry point of a worker process: run App.<job> (training or comparison) and report to the main process."""
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)      # Ctrl+C is handled by the main process (it sets stop_event)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        app = App(cfg, Path(out_dir))
        app._child_q = q
        app.stop_train = stop_event
        if live_event is not None:
            app.train_live = live_event
        app.auto_compare = False
        app.use_processes = False
        app.device_request = device_request
        app.train.update(state["train"])
        app.history, app.evals = list(state["history"]), list(state["evals"])
        getattr(app, job)()
    except Exception as exc:
        q.put(("log", time.strftime("%H:%M:%S ") + f"[ERROR] worker process failed: {exc}\n{traceback.format_exc()}"))
    finally:
        q.put(("done", job))


class App:
    def __init__(self, cfg: Config, out_dir: Path):
        """The application state shared by training, comparison, playback and the HTTP API."""
        from collections import deque
        self.lock = threading.RLock()
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.feas = feasibility(cfg)
        self.checks = []
        self.logbuf = deque(maxlen=400)
        self.train = dict(state="idle", episode=0, target=cfg.train_episodes, timesteps=0, ppo_iter=0,
                          losses={}, best=None, error=None, wall_s=0.0, eps_per_s=0.0)
        self.history = []
        self.evals = []
        self.device_request = "auto"          # --device auto | cpu | cuda (training network only)
        self.train_latest = None              # newest finished training episode (references only)
        self._child_q = None                  # set inside a worker process (training / comparison)
        self.use_processes = True             # run training and the comparison in separate processes
        self._procs = {}
        self.comparison = None
        self.compare_state = "idle"
        self.viewer = dict(run_id=0, mode="idle", label="", fidelity="", rows=[], rbase=0, moon=[], mbase=0,
                           packets=[], pbase=0, trail=[], decisions=[], done=True, produced_years=0.0, final=None)
        self.disp_years = 0.0
        self.stop_train = _mp_context().Event()        # shared with the training process
        self.train_live = _mp_context().Event()        # "show the training live": send finished episodes to the screen
        self.train_live.set()
        self.stop_play = threading.Event()
        self.shutdown = threading.Event()
        self.train_thread = None
        self.play_thread = None
        self.compare_thread = None
        self.auto_compare = True
        self._csv_path = self.out / "training_log.csv"
        self._load_previous()

    def _load_previous(self):
        """Show results of an earlier run (from outputs/) until a new training run replaces them."""
        comp = _read_json(self.out / "comparison.json")
        if comp:
            self.comparison = comp
            self.compare_state = "done (previous run)"
        best = _read_json(self.out / "best_model_info.json")
        if best and (self.out / "best_model.zip").exists():
            self.train["best"] = best
        try:
            with open(self._csv_path) as f:
                for r in csv.DictReader(f):
                    fl = lambda k: float(r[k]) if r[k] not in ("", "None") else None
                    self.history.append(dict(ep=int(r["episode"]), reward=fl("reward"), success=r["success"] == "1",
                                             a=fl("a_au"), e=fl("e"), mass=fl("mass_used_frac"),
                                             energy=fl("energy_used_frac"), years=fl("years"), reason=r["reason"]))
        except (OSError, KeyError, ValueError):
            pass
        ev = _read_json(self.out / "evals.json")
        if isinstance(ev, list):
            self.evals = ev

    # ------------------------------------------------------------------------------------------
    def log(self, msg):
        """Print a time-stamped message and keep it for the viewer's log."""
        line = time.strftime("%H:%M:%S ") + msg
        if self._child_q is not None:        # inside a worker process: the main process prints and keeps it
            self._child_q.put(("log", line))
            return
        print(line, flush=True)
        with self.lock:
            self.logbuf.append(line)

    def _emit(self, kind, payload):
        """Inside a worker process: send a state update to the main process (no-op in the main process)."""
        if self._child_q is not None:
            self._child_q.put((kind, payload))

    def set_config(self, updates: dict):
        """Apply parameter changes from the viewer after validating them."""
        with self.lock:
            if self.train["state"] == "running":
                raise RuntimeError("stop training before changing parameters")
            if updates.pop("reset_defaults", False):
                base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
                cfg = make_config(base)
                self.cfg, self.feas = cfg, feasibility(cfg)
                self.train["target"] = cfg.train_episodes
                self.log(f"[CONFIG] all parameters reset to the '{base}' defaults")
                return cfg
            preset = updates.pop("preset", None)
            if preset in PRESETS:             # switching preset resets the physical parameters
                keep = {k: v for k, v in self.cfg.to_dict().items() if k not in PRESETS[preset]}
                cfg = checked_config(preset, **{**keep, **updates})
            else:
                base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
                cur = {k: v for k, v in self.cfg.to_dict().items() if k not in PRESETS[base]}
                phys = {k: getattr(self.cfg, k) for k in PRESETS[base] if k != "preset"}
                cfg = checked_config(base, **{**cur, **phys, **updates})
            self.cfg = cfg
            self.feas = feasibility(cfg)
            self.train["target"] = cfg.train_episodes
        self.log(f"[CONFIG] preset={cfg.preset}  feasibility: {self.feas['verdict']}")
        return cfg

    # ---------------------------------------------------------------- training ----------------
    def set_live(self, on):
        """Turn the live training view on or off. Off: training runs without sending episodes to the screen, and the
        screen is free for mission playbacks (best PPO model, heuristic, no ejection). On: the newest finished
        training episode is shown (a running playback is stopped)."""
        with self.lock:
            if on:
                self.train_live.set()
                if self.viewer["mode"] == "playback":
                    self.stop_play.set()
                if self.viewer["mode"] != "training":
                    self.viewer = self._new_viewer("idle", "", "")
                    self.viewer["done"] = True
                    self._training_view_if_wanted(True)
            else:
                self.train_live.clear()
                self.train_latest = None
                if self.viewer["mode"] == "training":
                    self.viewer = self._new_viewer("idle", "", "")
                    self.viewer["done"] = True
        self.log(f"[VIEW] live training view {'ON: the newest finished training episode is shown' if on else 'OFF: training continues in the background; choose a mission to watch'}")

    def start_training(self, device=None):
        """Start PPO training in its own process (so the viewer and playbacks stay responsive)."""
        if device in ("auto", "cpu", "cuda"):
            self.device_request = device
        with self.lock:
            if self.train_thread and self.train_thread.is_alive():
                return False
            self.stop_train.clear()
            self.history, self.evals = [], []
            self.train.update(state="running", episode=0, timesteps=0, ppo_iter=0, losses={}, best=None,
                              error=None, target=self.cfg.train_episodes)
        self.train_thread = self._start_job("_train_worker", "ppo")
        return True

    # ---------------------------------------------------------------- worker processes ------------
    # Training and the comparison are pure number crunching. In the same process they compete with the web
    # server and with mission playbacks for Python's interpreter lock, which made the viewer lag. They therefore
    # run in their own processes; a small reader thread here applies the updates they send (log lines, training
    # progress, the newest finished episode for the live view, evaluation results, the comparison).
    def _start_job(self, job, name):
        """Run App.<job> in a worker process (fallback: a thread) and return the thread that follows it."""
        if self.use_processes:
            try:
                ctx = _mp_context()
                q = ctx.Queue()
                state = dict(train=dict(self.train), history=self.history, evals=self.evals)
                proc = ctx.Process(target=_worker_process, name=f"octo-{name}", daemon=True,
                                   args=(job, self.cfg, str(self.out), self.device_request, state, q, self.stop_train,
                                         self.train_live))
                proc.start()
                self._procs[job] = proc
                self.log(f"[{'TRAIN' if job == '_train_worker' else 'COMPARE'}] started in its own process "
                         f"(pid {proc.pid}) so the viewer stays responsive")
                t = threading.Thread(target=self._follow_job, args=(job, proc, q), daemon=True, name=name)
                t.start()
                return t
            except Exception as exc:
                self.log(f"[WARN] could not start a worker process ({exc}); running {job} in a thread instead")
        t = threading.Thread(target=self._job_in_thread, args=(job,), daemon=True, name=name)
        t.start()
        return t

    def _job_in_thread(self, job):
        """Fallback: run the job in a thread of this process."""
        getattr(self, job)()
        self._after_job(job)

    def _follow_job(self, job, proc, q):
        """Reader thread: apply the updates a worker process sends until it has finished."""
        import queue as _queue
        while True:
            try:
                kind, payload = q.get(timeout=0.3)
            except _queue.Empty:
                if not proc.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            if kind == "done":
                break
            self._apply_update(kind, payload)
        proc.join(10)
        if job == "_train_worker" and self.train.get("state") == "running":
            self.train.update(state="error", error=f"training process ended unexpectedly (exit code {proc.exitcode})")
            self.log(f"[ERROR] {self.train['error']}")
        if job == "_compare_worker" and self.compare_state == "running":
            self.compare_state = "error"
            self.log(f"[ERROR] comparison process ended unexpectedly (exit code {proc.exitcode})")
        self._after_job(job)

    def _apply_update(self, kind, payload):
        """One update from a worker process."""
        with self.lock:
            if kind == "log":
                print(payload, flush=True)
                self.logbuf.append(payload)
            elif kind == "train":
                self.train.update(payload)
            elif kind == "episode":
                self.history.append(payload)
            elif kind == "eval":
                self.evals.append(payload)
            elif kind == "latest":
                if not self.train_live.is_set():
                    return
                self.train_latest = payload
                if self.viewer["mode"] == "idle":          # nothing on screen yet: show it at once
                    self._training_view_if_wanted(True)
            elif kind == "comparison":
                self.comparison = payload
                self.compare_state = "done"
            elif kind == "compare_state":
                self.compare_state = payload

    def _after_job(self, job):
        """After training: start the comparison automatically (unless the program is shutting down)."""
        if job == "_train_worker" and self.auto_compare and not self.shutdown.is_set() \
                and self.train.get("state") in ("finished", "stopped"):
            self.start_comparison()

    def _eval_policy_fast(self, model, cfg, seeds=(1001, 1002)):
        """Evaluate a model deterministically on a few seeds with the fast model."""
        res = [run_episode(cfg, model_policy(model), s, "fast") for s in seeds]
        sr = float(np.mean([r["success"] for r in res]))
        err = float(np.mean([OctoEarthEnv.error_norm_static(cfg, r) for r in res]))
        ef = float(np.mean([r["energy_used_frac"] for r in res]))
        yrs = float(np.mean([r["t_years"] for r in res]))
        return dict(success_rate=sr, error=err, energy_frac=ef, years=yrs, reward=float(np.mean([r["total_reward"] for r in res])),
                    a_au=float(np.mean([r["a_au"] if math.isfinite(r["a_au"]) else 99 for r in res])),
                    e=float(np.mean([r["e"] for r in res])), score=(sr, -round(err, 4), -ef))

    def _train_worker(self):
        """Training thread: PPO for the configured episodes, periodic evaluation, best-model selection."""
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.vec_env import DummyVecEnv
            torch.set_num_threads(1)
            device = pick_device(self.device_request)
            bench = {}
            if device == "cuda":                       # a GPU exists: measure whether it is actually faster
                bench = benchmark_devices(self.cfg)
                if len(bench) == 2:
                    self.log(f"[DEVICE] network time per PPO iteration ({self.cfg.ppo_n_steps} steps + update): "
                             f"CPU {bench['cpu']:.2f} s, GPU {bench['cuda']:.2f} s "
                             f"-> the {'GPU' if bench['cuda'] < bench['cpu'] else 'CPU'} is "
                             f"{max(bench.values()) / max(min(bench.values()), 1e-9):.1f}x faster")
                    if self.device_request == "auto" and bench["cpu"] <= bench["cuda"]:
                        device = "cpu"
                        self.log("[DEVICE] Auto: training on the CPU because it is faster for this small network "
                                 "(choose 'GPU' in the Mission panel to force the GPU)")
                    elif self.device_request == "cuda" and bench["cpu"] < bench["cuda"]:
                        self.log("[DEVICE] GPU forced by your choice although the CPU measured faster; choose 'Auto' or "
                                 "'CPU' if training is slow")
            self.train["device"] = device
            self.train["bench"] = bench
            self._emit("train", dict(device=device, device_label=device_label(device), bench=bench,
                                     device_request=self.device_request))
            cfg = self.cfg
            app = self
            env = OctoEarthEnv(cfg, "fast", record=True, record_moon=False)   # same physics, no viewer Moon samples
            phys = [0.0]                               # wall time spent inside the physics (env.step / reset)

            class _Timed(gym.Wrapper):
                """Measures how much of the training time is physics (the rest is the network, PPO and reporting)."""
                def step(self, a):
                    t = time.perf_counter()
                    r = self.env.step(a)
                    phys[0] += time.perf_counter() - t
                    return r

                def reset(self, **kw):
                    t = time.perf_counter()
                    r = self.env.reset(**kw)
                    phys[0] += time.perf_counter() - t
                    return r
            self._phys = phys
            vec = DummyVecEnv([lambda: _Timed(env)])
            model = PPO("MlpPolicy", vec, n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
                        n_epochs=cfg.ppo_epochs, learning_rate=cfg.ppo_learning_rate, gamma=cfg.ppo_gamma,
                        gae_lambda=0.95, ent_coef=0.0, seed=cfg.seed, device=device, verbose=0,
                        policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5))
            hp = dict(algorithm="PPO (Stable-Baselines3)", net_arch=[64, 64], n_steps=cfg.ppo_n_steps,
                      batch_size=cfg.ppo_batch_size, n_epochs=cfg.ppo_epochs, learning_rate=cfg.ppo_learning_rate,
                      gamma=cfg.ppo_gamma, gae_lambda=0.95, log_std_init=-0.5, seed=cfg.seed)
            t0 = time.time()
            best = dict(score=None)
            last_view = [0.0]
            with open(self._csv_path, "w", newline="") as fcsv:
                wr = csv.writer(fcsv)
                wr.writerow(["episode", "timesteps", "reward", "success", "reason", "years", "a_au", "e",
                             "mass_used_frac", "energy_used_frac"])

                class CB(BaseCallback):
                    def _on_rollout_start(self):
                        """Before each PPO rollout: log the losses of the previous update."""
                        vals = dict(self.model.logger.name_to_value)
                        if "train/value_loss" in vals:
                            app.train["ppo_iter"] += 1
                            app.train["losses"] = {k.split("/")[1]: float(v) for k, v in vals.items() if k.startswith("train/")}
                            if device == "cuda":
                                import torch
                                app.train["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            app._emit("train", dict(ppo_iter=app.train["ppo_iter"], losses=app.train["losses"],
                                                    gpu_mem_mb=app.train.get("gpu_mem_mb")))
                            L = app.train["losses"]
                            app.log(f"[PPO update {app.train['ppo_iter']}] policy_loss={L.get('policy_gradient_loss', float('nan')):.4g} "
                                    f"value_loss={L.get('value_loss', float('nan')):.4g} entropy={L.get('entropy_loss', float('nan')):.4g} "
                                    f"approx_kl={L.get('approx_kl', float('nan')):.3g} explained_var={L.get('explained_variance', float('nan')):.3g}")

                    def _on_step(self):
                        """After every environment step: report finished episodes to the App (log, live view, evaluations)."""
                        for info, done in zip(self.locals["infos"], self.locals["dones"]):
                            if done and "episode_summary" in info:
                                app._on_episode(info["episode_summary"], self.num_timesteps, wr, t0, env, last_view)
                                ep = app.train["episode"]
                                if ep % cfg.eval_every_episodes == 0 or ep == cfg.train_episodes:
                                    ev = app._eval_policy_fast(self.model, cfg)
                                    ev.update(episode=ep, timesteps=self.num_timesteps)
                                    app.evals.append(ev)
                                    app._emit("eval", ev)
                                    (app.out / "evals.json").write_text(json.dumps(_clean(app.evals)))
                                    tag = ""
                                    if best["score"] is None or ev["score"] > best["score"]:
                                        best["score"] = ev["score"]
                                        self.model.save(str(app.out / "best_model.zip"))
                                        info_b = dict(episode=ep, timesteps=self.num_timesteps, eval=ev, hyperparameters=hp,
                                                      selection="lexicographic: success rate, then lowest terminal orbit "
                                                                "error, then lowest energy use (deterministic, seeds 1001-1002)")
                                        app.train["best"] = info_b
                                        app._emit("train", dict(best=info_b))
                                        (app.out / "best_model_info.json").write_text(json.dumps(info_b, indent=2, default=str))
                                        tag = "  << NEW BEST -> best_model.zip"
                                    app.log(f"[EVAL ep {ep}] deterministic: success={ev['success_rate']:.0%} a={ev['a_au']:.4f} AU "
                                            f"e={ev['e']:.4f} energy={ev['energy_frac']:.1%} years={ev['years']:.0f} "
                                            f"reward={ev['reward']:.1f}{tag}")
                        return not (app.stop_train.is_set() or app.train["episode"] >= cfg.train_episodes)

                total = int(cfg.train_episodes * cfg.max_steps * 1.05) + cfg.ppo_n_steps
                app.log(f"[TRAIN] PPO network on {device_label(device)}")
                app.log(f"[TRAIN] PPO started: target {cfg.train_episodes} episodes (<= {cfg.max_steps} decisions each, "
                        f"{cfg.decision_interval_years:g} y per decision) on the FAST model")
                model.learn(total_timesteps=total, callback=CB())
            model.save(str(self.out / "last_model.zip"))
            if not (self.out / "best_model.zip").exists():
                model.save(str(self.out / "best_model.zip"))
            self.log(f"[TRAIN] {'stopped' if self.stop_train.is_set() else 'finished'} after {self.train['episode']} episodes, "
                     f"{self.train['timesteps']} timesteps, {time.time() - t0:.0f} s. best_model.zip saved.")
            self.train.update(state="stopped" if self.stop_train.is_set() else "finished", wall_s=time.time() - t0)
            self._emit("train", dict(self.train))
        except Exception as exc:
            self.train.update(state="error", error=f"{type(exc).__name__}: {exc}")
            self._emit("train", dict(state="error", error=self.train["error"]))
            self.log(f"[ERROR] training failed: {exc}\n{traceback.format_exc()}")

    def _on_episode(self, s, timesteps, wr, t0, env, last_view=None):
        """Log one finished training episode and update the live view."""
        with self.lock:
            self.train["episode"] += 1
            self.train["timesteps"] = int(timesteps)
            ep = self.train["episode"]
            el = time.time() - t0
            self.train["wall_s"] = el
            self.train["eps_per_s"] = ep / max(el, 1e-9)
            self.train["s_per_ep"] = el / max(ep, 1)
            rec = dict(ep=ep, reward=round(s["total_reward"], 3), success=bool(s["success"]),
                       a=round(s["a_au"], 5) if math.isfinite(s["a_au"]) else None, e=round(s["e"], 5),
                       mass=round(s["mass_used_frac"], 4), energy=round(s["energy_used_frac"], 4),
                       years=round(s["t_years"], 1), reason=s["reason"])
            self.history.append(rec)
        self._emit("episode", rec)
        self._emit("train", {k: self.train[k] for k in ("episode", "timesteps", "wall_s", "eps_per_s", "s_per_ep")})
        wr.writerow([ep, timesteps, rec["reward"], int(rec["success"]), rec["reason"], rec["years"], rec["a"], rec["e"],
                     rec["mass"], rec["energy"]])
        # speed report: where does the wall time go? (every 10 episodes, or every episode when an episode is slow)
        lw = getattr(self, "_speed_mark", None) or (t0, 0.0, 0)
        ph = getattr(self, "_phys", [0.0])[0]
        dt_w, dt_p, n_ep = time.time() - lw[0], ph - lw[1], ep - lw[2]
        if n_ep >= 10 or (n_ep >= 1 and dt_w / n_ep > 5.0):
            self._speed_mark = (time.time(), ph, ep)
            self.train["phys_frac"] = dt_p / max(dt_w, 1e-9)
            self._emit("train", dict(phys_frac=self.train["phys_frac"]))
            self.log(f"[SPEED] last {n_ep} episode(s): {dt_w / n_ep:.2f} s per episode on the {'GPU' if self.train.get('device') == 'cuda' else 'CPU'} "
                     f"(physics {dt_p / n_ep:.2f} s, network + PPO update + reporting {(dt_w - dt_p) / n_ep:.2f} s)"
                     + (" - SLOW: choose device 'Auto' or 'CPU' in the Mission panel and close other GPU-heavy programs"
                        if dt_w / n_ep > 5.0 and (dt_w - dt_p) > dt_p else ""))
        if ep % 10 == 0 or ep == 1:
            rw = [h["reward"] for h in self.history[-50:]]
            sr = np.mean([h["success"] for h in self.history[-50:]])
            self.log(f"[EPISODE {ep}/{self.cfg.train_episodes}] reward={rec['reward']:.2f} avg50={np.mean(rw):.2f} "
                     f"success50={sr:.0%} final a={rec['a']} AU e={rec['e']:.4f} asteroid used={rec['mass']:.1%} "
                     f"energy used={rec['energy']:.1%} sim={rec['years']:.0f} y ({rec['reason']})")
        if env.last_trail and self.train_live.is_set():
            # The viewer picks the newest finished episode up when it has finished replaying the previous one
            # (see _training_view_if_wanted).
            latest = dict(ep=ep, trail=env.last_trail, t_years=env.engine.t_years, summary=s,
                          decisions=[{k: v for k, v in d.items() if k not in ("conservation", "belt")} for d in env.last_decisions]
                          if self._child_q is not None else env.last_decisions,
                          moon_rel_km=[round(float(x) / 1000.0) for x in env.engine.st[11:14]],
                          m_ast_left=float(env.engine.m_ast))
            if self._child_q is not None:
                now = time.time()                          # send at most one episode every 2 s (about 1 MB each)
                if now - getattr(self, "_last_latest", 0.0) > 2.0:
                    self._last_latest = now
                    latest["trail"] = np.asarray(env.last_trail, dtype=np.float64)
                    self._emit("latest", latest)
            else:
                with self.lock:
                    self.train_latest = latest
                    if self.viewer["mode"] == "idle":      # nothing on screen yet: show the first episode at once
                        self._training_view_if_wanted(True)

    def _training_view_if_wanted(self, want):
        """Put the newest finished training episode on screen (caller holds the lock).

        Called when the viewer reports that it has finished replaying the previous episode (want=True). A mission
        playback is never replaced. The episode is replayed by the browser from its recorded positions and
        velocities, so the overview keeps moving even while PPO is busy updating its network."""
        L = getattr(self, "train_latest", None)
        if not (want and L) or self.viewer["mode"] not in ("idle", "training") or not self.train_live.is_set():
            return
        if self.viewer["mode"] == "training" and self.viewer.get("episode") == L["ep"]:
            return                                          # already showing the newest one
        nv = self._new_viewer("training", f"training episode {L['ep']} (fast model, exploring)", "fast")
        rows = L["trail"].tolist() if isinstance(L["trail"], np.ndarray) else L["trail"]
        nv.update(rows=rows, episode=L["ep"],
                  decisions=[{k: v for k, v in d.items() if k not in ("conservation", "belt")} for d in L["decisions"]],
                  done=True, produced_years=L["t_years"],
                  final=dict(L["summary"], moon_rel_km=L["moon_rel_km"], m_ast_left=L["m_ast_left"]))
        self.viewer = nv

    # ---------------------------------------------------------------- comparison --------------
    def start_comparison(self):
        """Start the none / heuristic / PPO comparison in a background thread."""
        if self.compare_thread and self.compare_thread.is_alive():
            return False
        self.compare_state = "running"
        self.compare_thread = self._start_job("_compare_worker", "compare")
        return True

    def load_ppo(self):
        """Load best_model.zip if it exists (None otherwise)."""
        p = self.out / "best_model.zip"
        if not p.exists():
            return None
        from stable_baselines3 import PPO
        m = PPO.load(str(p), device="cpu")
        if m.observation_space.shape != (OBS_DIM,) or m.action_space.shape != (ACT_DIM,):
            self.log(f"[WARN] {p.name} was trained for an older environment (obs {m.observation_space.shape}, action "
                     f"{m.action_space.shape}; now ({OBS_DIM},) / ({ACT_DIM},)) - it is ignored. Retrain to get a PPO policy.")
            return None
        return m

    def _compare_worker(self):
        """Comparison thread: run every policy in both models and write comparison.json and the report."""
        try:
            cfg = self.cfg
            self.log("[COMPARE] evaluating policies on the FAST model (3 seeds) and the HIGH-FIDELITY N-body model "
                     f"(1 seed + {cfg.validate_years:g} y ballistic validation) ...")
            pols = [("none", policy_none), ("heuristic", policy_heuristic)]
            model = self.load_ppo()
            if model is not None:
                pols.append(("ppo", model_policy(model)))
            rows, traces = [], {}
            for name, pol in pols:
                fast = [run_episode(cfg, pol, s, "fast") for s in (2001, 2002, 2003)]
                belt = [BeltKeeper(cfg)] if cfg.solar_collectors else [None]

                def belt_step(env, info, belt=belt):
                    """After every decision of the hi-fi playback: advance the 12 followed belt collectors over the same interval."""
                    if belt[0] is not None and env.last_samples is not None and len(env.last_samples) > 1:
                        L = env.engine.last_eled or {}
                        yrs = cfg.decision_interval_years
                        belt[0].advance(env.last_samples, L.get("microwaves_sent", 0.0) / (yrs * YEAR), env.engine.t_years - yrs)
                hifi = run_episode(cfg, pol, 2001, "hifi", validate=True, on_step=belt_step)
                hifi["belt"] = belt[0].summary() if belt[0] is not None else None
                row = dict(policy=name, label=_policy_label(name),
                           fast=dict(success_rate=float(np.mean([f["success"] for f in fast])),
                                     reward=float(np.mean([f["total_reward"] for f in fast])),
                                     a_au=[f["a_au"] for f in fast], e=[f["e"] for f in fast],
                                     years=[f["t_years"] for f in fast], reasons=[f["reason"] for f in fast],
                                     mass_used_frac=float(np.mean([f["mass_used_frac"] for f in fast])),
                                     energy_used_frac=float(np.mean([f["energy_used_frac"] for f in fast])),
                                     mars_min_dist_au=[f["mars_min_dist_au"] for f in fast],
                                     T_surface_K=[f["T_surface_K"] for f in fast], climate_ok=[f["climate_ok"] for f in fast],
                                     shield_overrides=[f["shield_overrides"] for f in fast]),
                           hifi={k: hifi[k] for k in ("success", "reason", "total_reward", "a_au", "e", "inc_deg", "t_years",
                                                      "mass_used_kg", "mass_used_frac", "energy_used_J", "energy_used_frac",
                                                      "impulse_Ns", "n_packets", "insolation_rel", "t_eq_K", "shortfall_au",
                                                      "max_packet_impulse_Ns", "ledger", "conservation", "moon",
                                                      "mars_min_dist_au", "mars_min_dist_t", "mars_safe_au", "mars_safe",
                                                      "mars_max_tide_ratio", "mars_max_moon_perturb", "mars_a_au", "mars_e",
                                                      "mars_a0_au", "mars_e0", "mars_da_au", "mars_de", "shield_overrides",
                                                      "T_surface_K", "T_deep_K", "climate_ok", "climate_dev_max_K",
                                                      "years_outside_livable", "energy_ledger", "energy_ledger_residual",
                                                      "energy_in_J", "mirror_energy_J", "heat_to_earth_J", "belt",
                                                      "ice_line_deg", "grow_frac", "T_no_mirror_K", "planets", "planets_safe")
                                    if k in hifi} | {k: hifi[k] for k in POWER_METRIC_KEYS if k in hifi},
                           validation=hifi.get("validation"),
                           model_agreement_da_au=abs(fast[0]["a_au"] - hifi["a_au"]) if math.isfinite(hifi["a_au"]) else None)
                row["mission_success"] = bool(hifi["success"] and (hifi.get("validation") or {}).get("passed", False))
                rows.append(row)
                traces[name] = dict(fast=[dict(t=d["t_years"], a=d["a_au"], e=d["e"], m=d["m_ast_frac"], E=d["energy_frac"],
                                               u=d["u_ms"], th=d["theta_deg"], ph=d["phi_deg"], thr=d["throttle"],
                                               dm=d["dm_kg"], imp=d["impulse_Ns"], d_rtn=d["d_rtn"],
                                               dmars=d["mars_dmin_au"], ovr=d["shield_overrides"], T=d.get("T_surface_K"),
                                               S=d.get("insol_set_rel"), mir=d.get("mirror_rel"))
                                          for d in fast[0]["decisions"]],
                                    hifi=[_trace_rec(d) for d in hifi["decisions"]],
                                    hifi_a0=hifi.get("a0_au", 1.0), hifi_e0=hifi.get("e0", 0.0167))
                self.log(f"[COMPARE] {row['label']:24s} fast: success {row['fast']['success_rate']:.0%}, a={np.mean(row['fast']['a_au']):.4f} AU | "
                         f"high-fidelity: {hifi['reason']}, a={hifi['a_au']:.4f} AU e={hifi['e']:.4f}, "
                         f"validated={row['mission_success']}, asteroid used {hifi['mass_used_frac']:.1%}, "
                         f"energy {hifi['energy_used_J']:.3e} J, closest Mars pass {hifi['mars_min_dist_au']:.4f} AU "
                         f"(safe >= {hifi['mars_safe_au']:.4f}), shield overrides {hifi['shield_overrides']}, surface "
                         f"{hifi['T_surface_K']:.1f} K (climate ok {hifi['climate_ok']}, max deviation {hifi['climate_dev_max_K']:.1f} K)")
            expl = explain(rows, traces, cfg, self.feas, self.train.get("best"))
            best_info = self.train.get("best") or _read_json(self.out / "best_model_info.json")
            panel = best_panel(rows, traces, cfg, best_info, self.out / "best_model.zip", self.history, self.evals)
            comp = dict(rows=rows, traces=traces, explanation=expl, config=cfg.to_dict(), feasibility=self.feas,
                        best_model=best_info, best_panel=panel,
                        created=time.strftime("%Y-%m-%d %H:%M:%S"))
            with self.lock:
                self.comparison = comp
                self.compare_state = "done"
            self._emit("comparison", _clean(comp))
            (self.out / "comparison.json").write_text(json.dumps(_clean(comp), indent=1))
            (self.out / "report.md").write_text(make_report(self, comp))
            self.log(f"[COMPARE] done -> {self.out / 'comparison.json'} and report.md")
            for line in expl["summary"]:
                self.log("  " + line)
        except Exception as exc:
            self.compare_state = "error"
            self._emit("compare_state", "error")
            self.log(f"[ERROR] comparison failed: {exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- playback ----------------
    def start_playback(self, policy: str, fidelity: str):
        """Start computing a mission for the viewer to play (in a background thread)."""
        self.stop_play.set()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(5)
        self.stop_play.clear()
        if policy == "ppo":
            model = self.load_ppo()
            if model is None:
                raise RuntimeError("no best_model.zip yet - train first")
            pol = model_policy(model)
        else:
            pol = policy_none if policy == "none" else policy_heuristic
        with self.lock:
            self.disp_years = 0.0
            self.viewer = self._new_viewer("playback", f"{_policy_label(policy)} - "
                                           f"{'high-fidelity N-body' if fidelity == 'hifi' else 'fast model'}", fidelity)
            self.viewer["policy"] = policy
            run_id = self.viewer["run_id"]
        self.play_thread = threading.Thread(target=self._play_worker, args=(pol, fidelity, run_id), daemon=True)
        self.play_thread.start()

    def _new_viewer(self, mode, label, fidelity):
        """Fresh playback buffers for a new run."""
        return dict(run_id=self.viewer["run_id"] + 1, mode=mode, label=label, fidelity=fidelity, rows=[], rbase=0,
                    moon=[], mbase=0, packets=[], pbase=0, trail=[], decisions=[], belts=[], done=False, produced_years=0.0,
                    final=None)

    @staticmethod
    def _append_window(v, key, base_key, items, limit):
        """Append items to a playback buffer, keeping at most `limit` of them."""
        v[key].extend(items)
        if len(v[key]) > limit:            # keep memory bounded; the browser already has the old part
            drop = len(v[key]) - limit // 2
            del v[key][:drop]
            v[base_key] += drop

    def _play_worker(self, pol, fidelity, run_id):
        """Playback thread: run the mission step by step and stream its states to the viewer."""
        try:
            cfg = self.cfg
            first = [True]
            belt_q = queue.Queue()

            def belt_worker():
                """Collector-belt thread of this playback: advances BeltKeeper decision by decision (in order)."""
                bk = BeltKeeper(cfg)
                while True:
                    job = belt_q.get()
                    if job is None or self.viewer["run_id"] != run_id:
                        return
                    k, smp, p_mw, t_s = job
                    try:
                        b = _clean(bk.advance(smp, p_mw, t_s))
                    except Exception as exc:  # pragma: no cover
                        self.log(f"[WARN] belt simulation failed: {exc!r}")
                        return
                    with self.lock:
                        if self.viewer["run_id"] != run_id:
                            return
                        self.viewer.setdefault("belts", []).append([k, b])

            if cfg.solar_collectors:
                threading.Thread(target=belt_worker, daemon=True, name="belt").start()

            def on_step(env, info):
                """Called after every decision: append the new samples, events and decision record to the playback buffers."""
                rec = {k: v for k, v in info["decision"].items() if k != "conservation"}
                if "conservation" in info["decision"]:
                    rec["conservation"] = {k: v for k, v in info["decision"]["conservation"].items()
                                           if k.endswith("_rel") or k in ("active_fragments", "merges")}
                rows = _viewer_rows(env.last_samples, env.engine.last_power, cfg)
                if cfg.solar_collectors and env.last_samples is not None and len(env.last_samples) > 1:
                    # The 12+12 collector orbits (RK4, half-day steps) take ~1 s of computer time per decision - far
                    # slower than the mission physics. They run in their own thread (numba releases the GIL) and
                    # their results are attached to the decision records when ready, so playback never waits.
                    L = env.engine.last_eled or {}
                    yrs = cfg.decision_interval_years
                    belt_q.put((len(self.viewer["decisions"]), env.last_samples.copy(),
                                L.get("microwaves_sent", 0.0) / (yrs * YEAR), float(rec["t_years"]) - yrs))
                moon = _moon_rows(env.engine.last_moon)
                pk = _packet_records(env.last_samples, cfg, fidelity)
                with self.lock:
                    v = self.viewer
                    if v["run_id"] != run_id:
                        return
                    if first[0]:
                        rows.insert(0, env._first_row)
                        if env._first_moon is not None:
                            moon.insert(0, env._first_moon)
                        first[0] = False
                    self._append_window(v, "rows", "rbase", rows, 24000)
                    self._append_window(v, "moon", "mbase", moon, 64000)
                    self._append_window(v, "packets", "pbase", pk, 24000)
                    v["trail"].extend([[r[0]] + r[1:4] for r in rows[::8]])
                    v["decisions"].append(rec)
                    v["produced_years"] = env.engine.t_years
                while (not self.stop_play.is_set() and self.viewer["run_id"] == run_id and
                       env.engine.t_years - self.disp_years > max(3 * cfg.decision_interval_years,
                                                                  getattr(self, "ahead_years", 0.0))):
                    time.sleep(0.05)

            def first_state(env):
                """Called right after reset: publish the year-0 state so the viewer can draw it immediately."""
                with self.lock:
                    if self.viewer["run_id"] == run_id:
                        self.viewer["meta"] = dict(mu=float(env.engine.mu()), m_ast0=float(env.m_ast0), E0=float(env.E0),
                                                   ejections_per_year=cfg.ejections_per_year,
                                                   decision_interval_years=cfg.decision_interval_years,
                                                   target_au=cfg.target_au, tol_a_au=cfg.tol_a_au, tol_e=cfg.tol_e,
                                                   mission_years=cfg.mission_years, release_radius_m=cfg.release_radius_m,
                                                   offsets_days=PACKET_OFFSETS_DAYS.tolist(), seed=2001,
                                                   mars_safe_au=env.mars_lim["d_safe_au"], mars_caution_au=env.mars_lim["caution_au"],
                                                   mars_governing=env.mars_lim["governing"], mars_shield=bool(cfg.mars_shield),
                                                   planets=planets_meta(cfg),
                                                   solar=solar_meta(cfg),
                                                   belt=BeltKeeper(cfg).tracked() if cfg.solar_collectors else None,
                                                   climate=dict(on=bool(cfg.climate_model), target=cfg.climate_target_K,
                                                                tol=cfg.climate_tol_K, livable=CLIM_LIVABLE_K,
                                                                T0=float(env.clim["T0"]), earth_sun_W=EARTH_INTERCEPT_W,
                                                                mirrors=mirror_design(cfg), mirror_cap_target_rel=
                                                                mirror_design(cfg, cfg.target_au)["cap_rel"],
                                                                life=dict(dT=LIFE_DT_EQ_POLE, T_ice=LIFE_T_ICE,
                                                                          T_grow=list(LIFE_T_GROW))),
                                                   mars_gravity=bool(cfg.mars_gravity))
                env._first_row = _initial_row(env.engine)
                _, _, mo = env.engine.earth_state()
                env._first_moon = None if mo is None else \
                    [0.0] + [int(round(float(x) / 1000.0)) for x in mo[0]] + [round(float(x), 1) for x in mo[1]]

            try:
                s = run_episode(cfg, pol, 2001, fidelity, validate=(fidelity == "hifi"), on_step=on_step,
                                stop_event=self.stop_play, on_reset=first_state)
            finally:
                belt_q.put(None)
            s.pop("decisions", None)
            with self.lock:
                if self.viewer["run_id"] == run_id:
                    self.viewer["done"] = True
                    self.viewer["final"] = s
        except Exception as exc:
            self.log(f"[ERROR] playback failed: {exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- API views ---------------
    def state_view(self, run, since, msince, psince, dsince, disp, pstride=1, ahead=0.0, trnext=False, bsince=0):
        """What the viewer asks for every few hundred ms: new samples, events and decisions since last time."""
        with self.lock:
            self._training_view_if_wanted(trnext)
            if disp is not None:
                self.disp_years = disp
                self.ahead_years = min(1500.0, max(0.0, float(ahead)))
            v = self.viewer
            reset = run != v["run_id"]
            if reset:
                since = msince = psince = dsince = bsince = 0

            def chunk(key, base_key, start, n):
                """Slice of a playback buffer starting at `start` (at most n items), clipped to what is still kept."""
                st = max(start, v[base_key])
                return st, v[key][st - v[base_key]: st - v[base_key] + n]
            rs, rows = chunk("rows", "rbase", since, 60000 if v["mode"] == "training" else 3000)
            ms, moon = chunk("moon", "mbase", msince, 16000)
            pstride = max(1, int(pstride))
            ps, pk = chunk("packets", "pbase", psince, 3000 * pstride)
            pnext = ps + len(pk)
            if pstride > 1:                   # high display speed: send every k-th packet only
                pk = pk[::pstride]
            return dict(train=dict(self.train, best=None, live=self.train_live.is_set(), device_request=self.device_request),
                        best=self.train.get("best"), compare_state=self.compare_state,
                        gpu=getattr(self, "gpu_info", None),
                        init=self._init_view(),
                        viewer=dict(run_id=v["run_id"], mode=v["mode"], label=v["label"], fidelity=v["fidelity"],
                                    policy=v.get("policy"), done=v["done"], final=v["final"],
                                    produced_years=v["produced_years"], reset=reset,
                                    rstart=rs, rows=rows, mstart=ms, moon=moon, pstart=ps, pnext=pnext, pstride=pstride,
                                    packets=pk, meta=v.get("meta"),
                                    trail=v["trail"] if (reset or since < v["rbase"]) else None,
                                    dstart=dsince, decisions=v["decisions"][dsince:],
                                    bstart=bsince, belts=v.get("belts", [])[bsince:bsince + 50]),
                        log=list(self.logbuf)[-40:], feasible=self.feas["feasible"], verdict=self.feas["verdict"],
                        preset=self.cfg.preset, checks_ok=all(c["passed"] for c in self.checks) if self.checks else None)

    def _init_view(self):
        """Simulated start state (year 0, playback seed 2001) shown while nothing is running."""
        key = json.dumps(self.cfg.to_dict(), sort_keys=True, default=str)
        if getattr(self, "_init_key", None) != key:
            ic = initial_conditions(self.cfg, 2001)
            me = ic["m_earth"]
            r_e = ic["r_emb"] - M_MOON / (me + M_MOON) * ic["r_rel"]
            vh = ic["v_emb"] / np.linalg.norm(ic["v_emb"])
            self._init_cache = dict(earth=[round(float(x) / AU, 7) for x in r_e],
                                    moon_rel_km=[round(float(x) / 1000.0, 1) for x in ic["r_rel"]],
                                    moon_v_kms=[round(float(x) / 1000.0, 6) for x in ic["v_rel"]],
                                    ast_dir=[round(float(-x), 5) for x in vh], m_ast0=float(self.cfg.asteroid_mass_kg),
                                    earth_v_kms=[round(float(x) / 1000.0, 6)
                                                 for x in ic["v_emb"] - M_MOON / (me + M_MOON) * ic["v_rel"]],
                                    mars=[round(float(x) / AU, 7) for x in ic["r_mars"]],
                                    mars_v_kms=[round(float(x) / 1000.0, 6) for x in ic["v_mars"]],
                                    mars_safe_au=mars_safety_limits(self.cfg)["d_safe_au"],
                                    mars_caution_au=mars_safety_limits(self.cfg)["caution_au"],
                                    launcher=_clean(launcher_geometry(self.cfg)), planets=planets_meta(self.cfg), planets_r=[[round(float(x) / AU, 7) for x in rr] for rr in ic["r_pl"]],
                                    planets_v_kms=[[round(float(x) / 1000.0, 6) for x in vv] for vv in ic["v_pl"]],
                                    solar=solar_meta(self.cfg))
            self._init_key = key
        return self._init_cache

    def results_view(self):
        """Everything the viewer's Results / Guide pages need (docs, results, checks, comparison)."""
        with self.lock:
            base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
            return dict(config=self.cfg.to_dict(), defaults=make_config(base).to_dict(), param_spec=PARAM_SPEC,
                        param_docs=PARAM_DOCS, output_docs=OUTPUT_DOCS, glossary=GLOSSARY, reason_docs=REASON_DOCS,
                        check_docs=CHECK_DOCS, constants=CONSTANT_DOCS, speed_doc=SPEED_DOC, assumptions=ASSUMPTIONS,
                        enablers=ENABLERS, problems=MAJOR_PROBLEMS,
                        presets=list(PRESETS), feasibility=self.feas, checks=self.checks, history=self.history[-3000:],
                        evals=self.evals, comparison=self.comparison, train=dict(self.train),
                        training_outcomes=failure_counts(self.history), out_dir=str(self.out.resolve()),
                        packet_offsets_days=PACKET_OFFSETS_DAYS.tolist())


def _read_json(p):
    """Read a JSON file, or return None if it does not exist or is broken."""
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _error_norm_static(cfg, s):
    """Orbit error of a finished run (same formula as OctoEarthEnv.error_norm)."""
    a = s["a_au"] if math.isfinite(s["a_au"]) else 10.0
    return min(3.0, abs(a - cfg.target_au) / (cfg.target_au - 1.0)) + 0.1 * min(10.0, s["e"] / cfg.tol_e)


OctoEarthEnv.error_norm_static = staticmethod(_error_norm_static)


def _clean(o):
    """Make an object JSON-safe (numpy types to Python, NaN / inf to None)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _trace_rec(d):
    """Compact per-decision trace for the comparison charts."""
    return dict(t=d["t_years"], a=d["a_au"], e=d["e"], inc=d["inc_deg"], thr=d["throttle"], dm=d["dm_kg"], u=d["u_ms"],
                th=d["theta_deg"], ph=d["phi_deg"], d_rtn=d["d_rtn"], imp=d["impulse_Ns"], E=d["energy_J"],
                dv=d.get("dv_ms"), npk=d.get("n_packets"), fb=d.get("fallback"), mh=d.get("moon_hits"),
                m=d["m_ast_frac"], Ef=d["energy_frac"], rw=d["reward"], moon=d["moon_bound"], T=d.get("T_surface_K"),
                S=d.get("insol_set_rel"), mir=d.get("mirror_rel"))


def failure_counts(history):
    """Outcome counts over all training episodes (stochastic exploration, fast model)."""
    c = {}
    for h in history:
        c[h["reason"]] = c.get(h["reason"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def best_panel(rows, traces, cfg, best, path, history, evals):
    """Everything the 'Best model' panel shows, computed only from recorded simulation data."""
    by = {r["policy"]: r for r in rows}
    out = dict(available="ppo" in by)
    p = Path(path)
    out["file"] = dict(name=p.name, path=str(p.resolve()), exists=p.exists(),
                       size_kb=round(p.stat().st_size / 1024, 1) if p.exists() else None,
                       modified=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None)
    out["selection"] = best
    out["training_outcomes"] = failure_counts(history)
    out["training_episodes"] = len(history)
    out["failed_evals"] = [dict(episode=e.get("episode"), success_rate=e.get("success_rate"), a_au=e.get("a_au"),
                                reason=",".join(e.get("reasons", [])) if isinstance(e.get("reasons"), list) else e.get("reason"))
                           for e in evals if e.get("success_rate", 0) < 1]
    out["disclaimer"] = [
        "This is the best policy found in this training run, judged on a few deterministic test episodes. It is not "
        "proven to be globally optimal; other strategies (e.g. the heuristic) can be better on some measures.",
        "Reward is only the scoring rule used to train the agent. Mission success is judged separately from physics: "
        "final a, e and inclination, a ballistic stability check, and the Moon staying bound."]
    if "ppo" not in by:
        return out
    r = by["ppo"]
    h = r["hifi"]
    tr = traces["ppo"]["hifi"]
    # ----- criteria
    v = r.get("validation") or {}
    out["criteria"] = [
        dict(name="Semi-major axis a", value=h["a_au"], unit="AU", required=f"{cfg.target_au} +/- {cfg.tol_a_au}",
             passed=bool(abs(h["a_au"] - cfg.target_au) <= cfg.tol_a_au)),
        dict(name="Eccentricity e", value=h["e"], unit="", required=f"<= {cfg.tol_e}", passed=bool(h["e"] <= cfg.tol_e)),
        dict(name="Inclination", value=h["inc_deg"], unit="deg", required=f"<= {cfg.tol_inc_deg}",
             passed=bool(h["inc_deg"] <= cfg.tol_inc_deg)),
        dict(name="Arrival time", value=h["t_years"], unit="yr", required=f"<= {cfg.mission_years:g}",
             passed=bool(h["success"])),
        dict(name="Moon still bound", value=bool(h["moon"]["bound"]), unit="", required="yes", passed=bool(h["moon"]["bound"])),
        dict(name="Closest approach to Mars (planetary safety)", value=h["mars_min_dist_au"], unit="AU",
             required=f">= {h['mars_safe_au']:.4f} AU", passed=bool(h["mars_safe"])),
        dict(name=f"Stays in the target window for {cfg.validate_years:g} yr without thrust (largest deviation of a from target)",
             value=v.get("max_abs_da_au"), unit="AU", required=f"<= {cfg.tol_a_au} AU, e <= {cfg.tol_e}, Moon bound",
             passed=bool(v.get("passed", False))),
    ]
    if cfg.climate_model and "T_surface_K" in h:
        out["criteria"].append(dict(name="Surface temperature at the end (climate model, space mirrors aimed by the agent)",
                                    value=h["T_surface_K"], unit="K", required=f"{cfg.climate_target_K:g} +/- {cfg.climate_tol_K:g}",
                                    passed=bool(h["climate_ok"])))
        out["criteria"].append(dict(name="Years outside the livable band (target +/- 15 K) during the mission",
                                    value=h["years_outside_livable"], unit="yr", required="0 (reported)",
                                    passed=bool(h["years_outside_livable"] <= 0.0)))
    # ----- per-decision analysis (high-fidelity run)
    a_prev, e_prev = traces["ppo"].get("hifi_a0", 1.0), traces["ppo"].get("hifi_e0", 0.0167)
    acts, total_da = [], 0.0
    for d in tr:
        da, de = d["a"] - a_prev, d["e"] - e_prev
        if d["dm"] <= 0:
            cls, why = "coast", "no packets launched in this interval"
        elif da > 0 and (d["e"] <= cfg.tol_e or de <= 0):
            cls = "helped"
            why = (f"recoil {d['d_rtn'][1]:+.2f} along the motion (prograde cosine) raised the orbit by {da:+.4f} AU"
                   + (f" and kept e at {d['e']:.4f}" if d["e"] <= cfg.tol_e else f" while lowering e by {-de:.4f}"))
        elif da > 0:
            cls = "mixed"
            why = f"raised the orbit by {da:+.4f} AU but e grew to {d['e']:.4f} (above tolerance {cfg.tol_e})"
        else:
            cls = "hurt"
            why = f"orbit shrank by {da:+.4f} AU (prograde cosine {d['d_rtn'][1]:+.2f})"
        if d["a"] > cfg.target_au + cfg.tol_a_au and da > 0:
            cls, why = "hurt", f"overshot the target: a = {d['a']:.4f} AU"
        total_da += max(da, 0.0)
        acts.append(dict(t=d["t"], dt=cfg.decision_interval_years, thr=d["thr"], dm=d["dm"], u=d["u"], th=d["th"], ph=d["ph"],
                         d_rtn=d["d_rtn"], npk=d["npk"], dv=d["dv"], E=d["E"], da=da, de=de, a=d["a"], e=d["e"],
                         fb=d["fb"], cls=cls, why=why, rw=d["rw"]))
        a_prev, e_prev = d["a"], d["e"]
    for x in acts:
        x["share"] = x["da"] / total_da if total_da > 0 and x["da"] > 0 else 0.0
        x["da_per_1e19kg"] = x["da"] / (x["dm"] / 1e19) if x["dm"] > 0 else None
        x["da_per_1e33J"] = x["da"] / (x["E"] / 1e33) if x["E"] > 0 else None
    out["actions"] = acts
    cnt = {}
    for x in acts:
        cnt[x["cls"]] = cnt.get(x["cls"], 0) + 1
    out["action_classes"] = cnt
    act = [x for x in acts if x["dm"] > 0]
    out["top_helpful"] = sorted(act, key=lambda x: -x["da"])[:5]
    out["least_helpful"] = sorted(act, key=lambda x: x["da"])[:3]
    # ----- resources / constraints
    L = h["ledger"]
    out["constraints"] = [
        dict(name="Asteroid mass used", value=h["mass_used_frac"], unit="fraction",
             note="ran out" if h["reason"] == "resources_exhausted" and h["mass_used_frac"] > 0.999 else "within limit"),
        dict(name="Energy budget used", value=h["energy_used_frac"], unit="fraction",
             note="ran out" if h["reason"] == "resources_exhausted" and h["energy_used_frac"] > 0.999 else "within limit"),
        dict(name="Packets that fell back to Earth (wasted)", value=int(L.get("n_fallback", 0)), unit="packets",
             note="launch slower than escape speed"),
        dict(name="Packets that hit the Moon", value=int(L.get("n_launch_moon_hits", 0)), unit="packets", note=""),
        dict(name="Outcome of high-fidelity run", value=h["reason"], unit="", note=REASON_DOCS.get(h["reason"], "")),
    ]
    out["comparison"] = [dict(policy=x["label"], success=x["mission_success"], a=x["hifi"]["a_au"], e=x["hifi"]["e"],
                              years=x["hifi"]["t_years"], mass=x["hifi"]["mass_used_frac"], energy_J=x["hifi"]["energy_used_J"],
                              reward=x["hifi"]["total_reward"]) for x in rows]
    return out


def explain(rows, traces, cfg, feas, best):
    """Plain-language explanation of what the best PPO model learned, from its own decisions."""
    by = {r["policy"]: r for r in rows}
    out = dict(summary=[], ppo=None)
    if "ppo" not in by:
        out["summary"].append("No trained PPO model available yet - only baselines were compared.")
        return out

    def stats(tr):
        """Averages of the agent's actions over a trace."""
        imp = np.array([d["imp"] for d in tr]); D = np.array([d["d_rtn"] for d in tr]); dm = np.array([d["dm"] for d in tr])
        u = np.array([d["u"] for d in tr]); t = np.array([d["t"] for d in tr])
        tot = imp.sum()
        if tot <= 0:
            return dict(active=False)
        act = dm > 0
        return dict(active=True, prograde=float((imp * D[:, 1]).sum() / tot), radial=float((imp * np.abs(D[:, 0])).sum() / tot),
                    normal=float((imp * np.abs(D[:, 2])).sum() / tot), retro=float(imp[D[:, 1] < 0].sum() / tot),
                    u_mean=float((dm * u).sum() / dm.sum()), first=float(t[act][0]) if act.any() else None,
                    last=float(t[act][-1]) if act.any() else None, steps=len(tr), active_steps=int(act.sum()))

    sp, sh = stats(traces["ppo"]["fast"]), stats(traces["heuristic"]["fast"])
    p, h, n = by["ppo"], by["heuristic"], by["none"]
    hp = p["hifi"]
    lines = []
    if p["mission_success"]:
        lines.append(f"RESULT: the best PPO model reached the target in the high-fidelity N-body check: a = {hp['a_au']:.4f} AU, "
                     f"e = {hp['e']:.4f} after {hp['t_years']:,.0f} simulated years, and the orbit stayed within tolerance "
                     f"for {cfg.validate_years:g} ballistic years with the Moon bound.")
    else:
        sf = hp["shortfall_au"]
        lines.append(f"RESULT: the best PPO model did NOT complete the mission in the high-fidelity check "
                     f"({hp['reason']}); final a = {hp['a_au']:.4f} AU (short by {sf:+.4f} AU), e = {hp['e']:.4f}.")
    if sp.get("active"):
        lines.append(f"Direction: the impulse-weighted recoil direction on Earth has along-track (prograde) cosine "
                     f"{sp['prograde']:.2f}, |radial| {sp['radial']:.2f}, |out-of-plane| {sp['normal']:.2f}; "
                     f"{sp['retro']:.1%} of the impulse was retrograde. Along-track thrust raises orbital energy fastest "
                     f"(Gauss: da/dt = 2 a^2 v f_T / mu, with f_T the along-track acceleration); radial/normal components mostly waste reaction mass.")
        lines.append(f"Launch speed: mass-weighted mean {sp['u_mean']:.3e} m/s vs {feas['u_required']:.3e} m/s, the "
                     f"minimum-energy speed that uses the whole asteroid (heuristic uses {sh.get('u_mean', float('nan')):.3e} m/s). "
                     f"Momentum per joule falls as 2/u, so a lower speed saves energy but costs more mass.")
        tr = traces["ppo"]["fast"]
        a_prev = [1.0] + [d["a"] for d in tr[:-1]]
        eff = sorted(((d["a"] - a0, d) for d, a0 in zip(tr, a_prev) if d["dm"] > 0), key=lambda x: -x[0])
        if eff:
            fmt = lambda x: (f"t={x[1]['t']:.0f} y: da={x[0]:+.4f} AU (prograde cos {x[1]['d_rtn'][1]:.2f}, "
                             f"u={x[1]['u']:.2e} m/s, throttle {x[1]['thr']:.2f})")
            lines.append("Most effective decisions (largest da per 100-y interval, fast model): " +
                         "; ".join(fmt(x) for x in eff[:3]) + ". Least effective: " + fmt(eff[-1]) +
                         ". At similar throttle, gains per interval grow with a, because at constant thrust acceleration da/dt ~ a^1.5.")
        stopped = sp["active_steps"] < sp["steps"]
        succ = p["fast"]["success_rate"] > 0
        lines.append(f"Timing: ejected in {sp['active_steps']} of {sp['steps']} decision intervals "
                     f"(first {sp['first']:.0f} y, last {sp['last']:.0f} y)" +
                     ("; it coasts in some intervals." if stopped else ".") +
                     (" It keeps thrusting until the orbit enters the target window, where the episode ends."
                      if succ else " It never learned to coast; throttle control near the target is a weakness."
                      if not stopped else ""))
    lines.append(f"Resources (high-fidelity run): PPO used {hp['mass_used_frac']:.1%} of the asteroid and "
                 f"{hp['energy_used_frac']:.1%} of the energy budget; heuristic {h['hifi']['mass_used_frac']:.1%} / "
                 f"{h['hifi']['energy_used_frac']:.1%}; no-ejection baseline stays at a = {n['hifi']['a_au']:.4f} AU.")
    if h["mission_success"] and p["mission_success"]:
        better = "faster" if hp["t_years"] < h["hifi"]["t_years"] else "slower"
        lines.append(f"Versus the hand-designed heuristic: PPO arrives {better} ({hp['t_years']:,.0f} vs {h['hifi']['t_years']:,.0f} y) "
                     f"and uses {'less' if hp['energy_used_J'] < h['hifi']['energy_used_J'] else 'more'} energy.")
    elif h["mission_success"] and not p["mission_success"]:
        lines.append("The hand-designed heuristic succeeds where PPO does not: more training or reward tuning is needed.")
    lines.append("This is the best model found in the tested training run, not a proven global optimum.")
    out["summary"] = lines
    out["ppo"] = sp
    out["heuristic"] = sh
    return out


def make_report(app: "App", comp: dict) -> str:
    """Write report.md: training, comparison, checks, feasibility and honest conclusions."""
    cfg = app.cfg
    L = ["# OCTO-EARTH - results report", "", f"Generated {comp['created']}  |  preset **{cfg.preset}**", "",
         "## Feasibility gate (analytic)", "", f"**{app.feas['verdict']}**", ""]
    L += [f"- {x}" for x in app.feas["lines"]]
    L += ["", "## Physics checks", ""]
    L += [f"- {'PASS' if c['passed'] else 'FAIL'} - {c['name']}: {c['detail']}" for c in app.checks] or ["- not run"]
    tr = app.train
    if tr["state"] == "idle" and app.history:
        L += ["", "## Training", "", f"Loaded from a previous run in this folder: {len(app.history)} episodes "
              f"(training_log.csv); this session did not train (comparison re-run only)."]
    else:
        L += ["", "## Training", "", f"State {tr['state']}, {tr['episode']} episodes, {tr['timesteps']} timesteps, "
              f"{tr.get('wall_s', 0):.0f} s wall time (fast model)."]
    b = comp.get("best_model")
    if b:
        L += [f"Best model: episode {b['episode']}, selection = {b['selection']}.",
              f"Hyperparameters: {json.dumps(b['hyperparameters'])}", f"Deterministic eval at selection: {json.dumps(_clean(b['eval']))}"]
    L += ["", "## Comparison (physical metrics; reward shown separately)", "",
          "| policy | fast success (3 seeds) | hi-fi result | hi-fi a [AU] | e | years | asteroid used | energy used [J] | validated | reward (hi-fi) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp["rows"]:
        h = r["hifi"]
        L.append(f"| {r['label']} | {r['fast']['success_rate']:.0%} | {h['reason']} | {h['a_au']:.4f} | {h['e']:.4f} | "
                 f"{h['t_years']:,.0f} | {h['mass_used_frac']:.1%} | {h['energy_used_J']:.3e} | {r['mission_success']} | {h['total_reward']:.1f} |")
    L += ["", "## Mars safety (high-fidelity runs)", "",
          f"Safe distance {comp['rows'][0]['hifi']['mars_safe_au']:.4f} AU ({REASON_DOCS['mars_unsafe_approach']})", "",
          "| policy | closest Mars pass [AU] | at year | safe | Mars tide (x Moon's) | Mars pull on Moon orbit | shield overrides | Mars a: start -> end [AU] | Mars e |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in comp["rows"]:
        h = r["hifi"]
        L.append(f"| {r['label']} | {h['mars_min_dist_au']:.4f} | {h['mars_min_dist_t']:,.0f} | {h['mars_safe']} | "
                 f"{h['mars_max_tide_ratio']:.2e} | {h['mars_max_moon_perturb']:.2e} | {h['shield_overrides']} | "
                 f"{h['mars_a0_au']:.4f} -> {h['mars_a_au']:.4f} | {h['mars_e0']:.4f} -> {h['mars_e']:.4f} |")
    L += ["", "Mars's final orbit is chaotic (sensitive to small differences); see README, 'Mars'."]
    if cfg.solar_collectors:
        des, sm = collector_design(cfg), solar_meta(cfg)
        L += ["", "## Solar collectors and microwave power (high-fidelity runs)", "",
              f"Belt at {cfg.collector_orbit_au} AU intercepting {cfg.collector_fraction:.0%} of the Sun's output ({sm['L_captured_W']:.3e} W): "
              f"{des['area_m2']:.2e} m^2 of collectors (+/- {des['half_width_deg']:.1f} deg around Earth's orbital plane), "
              f"{des['count']:.0e} collectors of ~{des['each_side_km']:,.0f} km each, ~{des['mass_kg']:.1e} kg of material "
              f"({des['mass_moon_frac']:.1%} of the Moon) at {cfg.collector_areal_density} kg/m^2; light pressure = "
              f"{des['light_pressure_ratio']:.1%} of the Sun's gravity on them. Chain: solar cells {cfg.pv_efficiency:.0%} -> "
              f"microwaves {cfg.microwave_efficiency:.0%} ({cfg.microwave_freq_ghz} GHz) -> captured by the {N_ARMS} arm rectennas "
              f"(Goubau, {cfg.transmitter_diameter_km:g} km transmitters, {cfg.rectenna_diameter_km:g} km rectennas) -> electricity "
              f"{cfg.rectenna_efficiency:.0%}.", "",
              "| policy | energy from collectors [J] | from reserve [J] | microwaves absorbed [J] | mean / peak beam power [W] | "
              "capture efficiency | heat at the arms (mean) | microwave push (not applied) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in comp["rows"]:
            h = r["hifi"]
            if not h.get("solar_on"):
                continue
            L.append(f"| {r['label']} | {h['energy_from_collectors_J']:.3e} | {h['energy_from_reserve_J']:.3e} | "
                     f"{h['beam_energy_absorbed_J']:.3e} | {h['beam_mean_W']:.2e} / {h['beam_peak_W']:.2e} | "
                     f"{h['eta_capture_min']:.3f}-{h['eta_capture_max']:.3f} | {h['arm_heat_mean_W']:.2e} W "
                     f"(= {h['arm_heat_vs_earth_sunlight']:.1e} x the sunlight Earth intercepts) | {h['beam_push_dv_ms']:.0f} m/s |"
                     if h.get("eta_capture_min") == h.get("eta_capture_min") else
                     f"| {r['label']} | 0 | 0 | 0 | 0 / 0 | - | 0 | 0 m/s |")
        L += ["", "The microwave push is an estimate (not applied). The heat at the arms is radiated away from Earth by assumption "
              "(enabler E1: directional arm radiators); the share arm_heat_to_earth_frac that reaches Earth is part of the climate model.",
              "", f"Panels: rated {cfg.pv_efficiency:.0%} at 25 C, operating {des['eta_pv_op']:.3f} (panel temperature "
              f"{des['T_panel_K']:.0f} K -> x{des['temp_factor']:.3f}; age x{des['degr_factor']:.3f}). Build-up: "
              f"{cfg.collector_initial_built:.0%} at year 0, complete after {cfg.collector_build_years:,.0f} y. Replacement "
              f"{des['replace_kg_per_yr']:.2e} kg/yr ({cfg.collector_recycle_frac:.0%} recycled); fresh material over the mission "
              f"{des['fresh_mass_total_kg']:.2e} kg = {des['fresh_mass_mercury_frac']:.1%} of Mercury. Earth's sunlight lane "
              f"{'kept clear' if cfg.earth_lane_clear else 'NOT kept clear (Earth in the belt shadow)'}. Belt's pull on Earth "
              f"{des['belt_pull']['ratio_1au']:.0e} of the Sun's."]
        L += ["", "## Energy ledger (high-fidelity runs, whole mission)", "",
              "| policy | sunlight on collectors | reflected | collector heat | manufacturing | transmitter heat | beam spill | "
              "rectenna heat | packet kinetic energy | launcher heat | sunlight on mirrors | mirror light onto Earth | mirror light past Earth | mirror heat | reserve used | residual |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in comp["rows"]:
            g = r["hifi"].get("energy_ledger")
            if not g:
                continue
            L.append(f"| {r['label']} | " + " | ".join(f"{g[k]:.3e}" for k in (
                "sun_on_collectors", "reflected", "collector_heat", "manufacturing", "transmitter_heat", "beam_spill",
                "rectenna_heat", "packet_ke", "launcher_heat", "mirror_sun", "mirror_to_earth", "mirror_to_space", "mirror_heat",
                "reserve_used")) +
                f" | {r['hifi'].get('energy_ledger_residual', float('nan')):.0e} |")
        L += ["", "## Collector belt: do the collectors stay on their orbits? (high-fidelity runs)", "",
              "| policy | max deviation WITH station keeping | control delta-v | WITHOUT: drift along the orbit | WITHOUT: distance-from-Sun swing | beam recoil / Sun's pull |",
              "|---|---|---|---|---|---|"]
        for r in comp["rows"]:
            b = r["hifi"].get("belt")
            if not b:
                continue
            L.append(f"| {r['label']} | {b['on']['max_km']:,.0f} km (rms {b['on']['rms_km']:,.0f}) | {b['on']['dv_ms_per_yr']:.0f} m/s/yr | "
                     f"{b['off']['max_km']:,.0f} km ({b['off']['slots']:,.1f} slots of {b['spacing_km']:,.0f} km) | "
                     f"{b['off']['swing_km']:,.0f} km | {b['recoil_ratio_max']:.1e} |")
    L += ["", "## Climate (high-fidelity runs; part of the RL task)", "",
          f"Target {cfg.climate_target_K:.0f} K +- {cfg.climate_tol_K:g} K; livable band +- {CLIM_LIVABLE_K:.0f} K. "
          f"Space mirrors {'ON' if cfg.mirrors else 'OFF'} ({cfg.mirror_area_m2:.2g} m^2; at {cfg.target_au} AU they can send "
          f"{mirror_design(cfg, cfg.target_au)['cap_rel']:.2f} x today's sunlight).", "",
          "| policy | final surface T | climate ok | max deviation | years outside livable band | mirror light onto Earth | without mirrors it would settle at | ice line | growing area | waste heat reaching Earth |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp["rows"]:
        h = r["hifi"]
        if "T_surface_K" not in h:
            continue
        L.append(f"| {r['label']} | {h['T_surface_K']:.1f} K | {h['climate_ok']} | {h['climate_dev_max_K']:.1f} K | "
                 f"{h['years_outside_livable']:.0f} | {h['mirror_energy_J']:.2e} J | {h.get('T_no_mirror_K', float('nan')):.0f} K | "
                 f"{h.get('ice_line_deg', float('nan')):.0f} deg | {100 * h.get('grow_frac', float('nan')):.0f} % | "
                 f"{h['heat_to_earth_J']:.2e} J |")
    L += ["", "## Enablers (assumed technologies - NOT shown to be possible)", "",
          "| id | enabler | provides | parameters | gap to reality | modelled |", "|---|---|---|---|---|---|"]
    L += [f"| {e['id']} | {e['name']} | {e['provides']} | {e['params']} | {e['gap']} | {e['modelled']} |" for e in ENABLERS]
    L += ["", "Instruments of each enabler (all numbers: CALCULATIONS_REPORT, made by calc_report.py):", ""]
    L += [f"* {e['id']} - {ins['name']}: {ins['use']} ({ins['params']})" for e in ENABLERS for ins in e.get("instruments", [])]
    L += ["", "## Major problems", "", "| id | problem | status | what was done | what remains |", "|---|---|---|---|---|"]
    L += [f"| {m['id']} | {m['title']} | {m['status']} | {m['detail']} | {m['remaining']} |" for m in MAJOR_PROBLEMS]
    bp = comp.get("best_panel") or {}
    if bp.get("criteria"):
        L += ["", "## Best model: final orbit vs target (high-fidelity run)", "", "| criterion | value | required | result |", "|---|---|---|---|"]
        for k in bp["criteria"]:
            val = f"{k['value']:.5g}" if isinstance(k["value"], float) else str(k["value"])
            L.append(f"| {k['name']} | {val} {k['unit']} | {k['required']} | {'PASS' if k['passed'] else 'FAIL'} |")
        L += ["", "## Best model: decisions (high-fidelity run)", "",
              "Classes: " + ", ".join(f"{k} {v}" for k, v in bp["action_classes"].items()), "",
              "| years | throttle | packets | packet kg | speed km/s | prograde cos | dv m/s | da AU | e | effect |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a in bp["actions"]:
            L.append(f"| {a['t'] - a['dt']:.0f}-{a['t']:.0f} | {a['thr']:.2f} | {a['npk']} | {a['dm']:.3g} | {a['u'] / 1000:.0f} | "
                     f"{a['d_rtn'][1]:.2f} | {a['dv'] or 0:.1f} | {a['da']:+.4f} | {a['e']:.4f} | {a['cls']} |")
    if bp.get("training_outcomes"):
        L += ["", "## Training episode outcomes (stochastic exploration)", ""]
        L += [f"- {k}: {v} ({REASON_DOCS.get(k, '')})" for k, v in bp["training_outcomes"].items()]
    L += [""] + [f"> {d}" for d in bp.get("disclaimer", [])]
    L += ["", "## Why the best model behaves as it does", ""] + [f"- {x}" for x in comp["explanation"]["summary"]]
    L += ["", "## Conservation (high-fidelity runs)", ""]
    for r in comp["rows"]:
        c = r["hifi"]["conservation"]
        L.append(f"- {r['label']}: energy residual {c['energy_residual_rel']:.1e} (rel. to |E0|), momentum {c['momentum_residual_rel']:.1e}, "
                 f"angular momentum {c['angmom_residual_rel']:.1e}, mass {c['mass_residual_rel']:.1e}; fragment merges {c['merges']}")
    L += ["", "## Limitations", "", "See README.md - simplified prototype, not a validated engineering or climate model."]
    return "\n".join(L) + "\n"


# =============================================================================================
# 11. HTTP API (stdlib) - Python is the only physics source; the browser only renders
# =============================================================================================
import re as _re  # noqa: E402

_NONFINITE = _re.compile(r"\bNaN\b|-?\bInfinity\b")


def _dumps(obj) -> bytes:
    """JSON text of an object after _clean."""
    return _NONFINITE.sub("null", json.dumps(obj, default=lambda o: _clean(o) if not isinstance(o, Path) else str(o),
                                             separators=(",", ":"))).encode()


def make_handler(app: App):
    """Build the HTTP request handler class bound to this App."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            """Serve files from the frontend folder."""
            super().__init__(*a, directory=str(FRONTEND), **kw)

        def log_message(self, *a):
            """Silence the default per-request logging."""
            pass

        def end_headers(self):
            """Add no-cache headers so the viewer always gets fresh files."""
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _send(self, obj, code=200):
            """Send a JSON response."""
            body = _dumps(_clean(obj) if not isinstance(obj, (bytes,)) else obj)
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                   # browser tab closed / reloaded

        def do_GET(self):
            """GET /api/state, /api/results, or a frontend file."""
            u = urlparse(self.path)
            if not u.path.startswith("/api/"):
                return super().do_GET()
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/state":
                    disp = float(q["disp"]) if "disp" in q else None
                    return self._send(app.state_view(int(q.get("run", -1)), int(q.get("since", 0)),
                                                     int(q.get("msince", 0)), int(q.get("psince", 0)),
                                                     int(q.get("dsince", 0)), disp, int(q.get("pstride", 1)),
                                                     float(q.get("ahead", 0) or 0), q.get("trnext") == "1",
                                                     int(q.get("bsince", 0))))
                if u.path == "/api/results":
                    return self._send(app.results_view())
                if u.path == "/api/health":
                    return self._send(dict(ok=True, app=APP_NAME))
                return self._send(dict(error="not found"), 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return None
            except Exception as exc:
                app.log(f"[HTTP] GET {u.path} failed: {exc}")
                return self._send(dict(error=str(exc)), 500)

        def do_POST(self):
            """POST /api/config, /api/train, /api/play, /api/stop_play, /api/compare, /api/shutdown."""
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if u.path == "/api/config":
                    app.set_config(dict(body))
                    return self._send(dict(ok=True, config=app.cfg.to_dict(), feasibility=app.feas))
                if u.path == "/api/train":
                    if body.get("action") == "stop":
                        app.stop_train.set()
                        return self._send(dict(ok=True))
                    if body.get("action") == "live":
                        app.set_live(bool(body.get("on")))
                        return self._send(dict(ok=True, live=app.train_live.is_set()))
                    if body.get("action") == "device":
                        if body.get("device") in ("auto", "cpu", "cuda"):
                            app.device_request = body["device"]
                        return self._send(dict(ok=True, device_request=app.device_request))
                    return self._send(dict(ok=app.start_training(body.get("device"))))
                if u.path == "/api/compare":
                    return self._send(dict(ok=app.start_comparison()))
                if u.path == "/api/play":
                    app.start_playback(body.get("policy", "heuristic"), body.get("fidelity", "fast"))
                    return self._send(dict(ok=True))
                if u.path == "/api/stop_play":
                    app.stop_play.set()
                    return self._send(dict(ok=True))
                if u.path == "/api/shutdown":
                    app.shutdown.set()
                    return self._send(dict(ok=True))
                return self._send(dict(error="not found"), 404)
            except Exception as exc:
                return self._send(dict(ok=False, error=str(exc)), 400)

    return Handler


def find_port(host, start):
    """First free TCP port at or above `start`."""
    import socket
    for p in range(start, start + 30):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


# ---------------------------------------------------------------------------------------------
# APPENDIX A. PARAMETER / OUTPUT DOCUMENTATION (single source for the viewer's guide and tooltips)
# ---------------------------------------------------------------------------------------------
# Plain-language tables: what every parameter and output means, the glossary, the assumptions, the
# enablers (things we ASSUME can be built) and the major problems. Functions above use them only at run
# time, so they can live at the end of the file where they do not interrupt the code.
CAT_ENG = "Engineering assumption"
CAT_MIS = "Mission requirement"
CAT_SIM = "Simulation setting"
CAT_RL = "RL hyperparameter"
CAT_RW = "Reward weight (scoring only)"
CAT_CONST = "Physical constant"
CAT_VIEW = "Viewer setting"
CAT_SAFE = "Planetary safety (Mars)"
CAT_SOL = "Solar collectors (energy supply)"
CAT_CLIM = "Climate + space mirrors"


def _p(key, label, unit, cat, lo, hi, meaning, why, up, down, constraint=""):
    """One parameter description: meaning, unit, range, why it matters, what raising / lowering it does."""
    return dict(key=key, label=label, unit=unit, category=cat, min=lo, max=hi, meaning=meaning, why=why,
                increase=up, decrease=down, constraint=constraint)


PARAM_DOCS = [
    _p("asteroid_mass_kg", "Initial asteroid mass", "kg", CAT_ENG, 1e15, 1e23,
       "Mass of the asteroid that is docked to Earth and used as throw-away reaction mass. "
       "For scale: the Moon is 7.3e22 kg, Ceres 9.4e20 kg.",
       "Momentum is conserved: Earth can only gain speed by throwing mass the other way. The total "
       "mass available limits the total speed change (rocket equation).",
       "More speed change is possible, at the same launch speed.",
       "Less mass to throw; below about 1.4e22 kg the default target cannot be reached even at maximum speed.",
       "Cannot exceed 1e23 kg in this model (the asteroid is treated as a small attachment to Earth)."),
    _p("max_packet_mass_kg", "Max packet mass", "kg", CAT_ENG, 1e3, 1e20,
       "Largest piece of asteroid that one ejection event can throw. The agent chooses a throttle between 0 and 100 % of this.",
       "Together with the ejection frequency it sets the maximum thrust (mass per second x speed).",
       "Faster transfer possible; each push is bigger (coarser control).",
       "Gentler, finer control, but the transfer may not finish within the mission time.",
       "Must not exceed the asteroid mass."),
    _p("max_ejection_speed_ms", "Max ejection speed (exhaust velocity)", "m/s", CAT_ENG, 1.2e4, 3.0e7,
       "Fastest launch speed of a packet relative to Earth's surface. This plays the role of a rocket's exhaust velocity.",
       "Momentum per packet = mass x speed, so faster packets push harder per kg. But energy = 1/2 mass x speed^2, so they "
       "cost much more energy per kg.",
       "Less asteroid mass needed, much more energy needed per kg.",
       "More mass needed. Below Earth's escape speed (11.2 km/s) packets fall back and give no net push.",
       "Capped at 3e7 m/s (10 % of light speed) because the model is non-relativistic."),
    _p("min_ejection_speed_ms", "Min ejection speed", "m/s", CAT_ENG, 1e3, 3.0e7,
       "Slowest launch speed the agent may choose.",
       "Sets the lower end of the agent's speed range (action a3 maps linearly between min and max).",
       "The agent cannot use slow, energy-cheap launches.",
       "Allows slow launches; below 11.2 km/s packets fall back to Earth (wasting energy, no push).",
       "Must be lower than the max ejection speed."),
    _p("energy_budget_j", "Propulsion energy budget", "J", CAT_ENG, 1e20, 1e37,
       "Energy stored for the launcher at the start of the mission. With the solar collectors OFF this is all the energy there is; "
       "with them ON it is a starting reserve, used only when the beams cannot deliver enough. For scale: Earth receives about "
       "5.4e24 J of sunlight per year.",
       "Each packet needs 1/2 x mass x speed^2 / efficiency. When the budget is used up, no more packets can be launched. "
       "No free energy is ever created.",
       "More or faster packets possible.",
       "The mission may stop with asteroid mass left over ('resources exhausted').",
       "Never exceeded: packets are shrunk automatically so the budget cannot go negative."),
    _p("launch_efficiency", "Launch efficiency", "0-1", CAT_ENG, 0.01, 1.0,
       "Fraction of the energy drawn from the budget that ends up as packet kinetic energy (the rest is lost as heat).",
       "Energy drawn = kinetic energy / efficiency.",
       "More packets per joule.", "Wastes more energy per packet.", "Between 0.01 and 1 (100 %)."),
    _p("ejections_per_year", "Ejection frequency", "1/yr", CAT_ENG, 1.0, 52.0,
       "How many packets are launched per simulated year (evenly spaced). Each one is a separate recoil event.",
       "Sets the maximum mass flow (thrust) and the number of recorded ejection events.",
       "More thrust available and more, smaller recoil steps (smoother); slower simulation.",
       "Less thrust; the transfer may need longer than the mission allows.", "1 to 52 per year."),
    _p("target_au", "Target orbit radius", "AU", CAT_MIS, 1.1, 5.0,
       "Semi-major axis of the circular orbit Earth should reach (1 AU = today's Sun-Earth distance; Mars is at 1.52 AU, the asteroid belt 2.2-3.3 AU).",
       "Defines success and the orbital energy that must be added.",
       "Much more energy and mass needed; much less sunlight (insolation falls as 1/a^2).",
       "Easier transfer.", "1.1 to 5 AU."),
    _p("mission_years", "Mission duration", "yr", CAT_MIS, 100.0, 20000.0,
       "Simulated time allowed before the episode ends ('deadline').",
       "Limits how long thrust can act; together with max thrust it decides if the target is reachable in time.",
       "More time for slow, energy-efficient transfers; longer episodes (slower training).",
       "Forces higher thrust; may become infeasible.", "100 to 20,000 years; must be at least 2 decision intervals."),
    _p("decision_interval_years", "Decision interval", "yr", CAT_SIM, 1.0, 1000.0,
       "How often the agent chooses a new action. During one interval all packets use the same mass, speed and direction.",
       "It is the RL time step: an episode has mission / interval decisions.",
       "Fewer, coarser decisions (faster training, less precise control).",
       "Finer control, longer episodes, slower training.", "At most 1,000 decisions per episode."),
    _p("tol_a_au", "Tolerance on target radius", "AU", CAT_MIS, 0.001, 0.5,
       "How close the final semi-major axis must be to the target.",
       "Part of the success test.", "Easier to count as success.", "Stricter.", ""),
    _p("tol_e", "Tolerance on eccentricity", "-", CAT_MIS, 0.001, 0.3,
       "Largest allowed eccentricity (0 = perfect circle). Earth today has e = 0.0167.",
       "Part of the success test; 'stable circular orbit' means small e.", "Allows more elliptical orbits.",
       "Requires a rounder orbit.", "Must be above 0.0167 to be reachable easily."),
    _p("tol_inc_deg", "Tolerance on inclination", "deg", CAT_MIS, 0.1, 30.0,
       "Largest allowed tilt of the final orbit relative to today's orbital plane.",
       "Part of the success test.", "Allows out-of-plane drift.", "Stricter.", ""),
    _p("validate_years", "Stability check duration", "yr", CAT_MIS, 10.0, 2000.0,
       "After the mission, the high-fidelity model runs this long with NO ejections to check that the orbit stays within tolerance and the Moon stays bound.",
       "Confirms 'stable orbit' instead of a lucky snapshot.", "Stronger evidence, slower comparison.",
       "Weaker evidence.", ""),
    _p("randomize_phase", "Randomize starting position", "on/off", CAT_SIM, None, None,
       "If on, each training episode starts at a random point of Earth's orbit (and Moon phase).",
       "Prevents the agent from memorising one exact sequence.", "(on) more robust policies.",
       "(off) every episode is identical.", ""),
    _p("train_episodes", "Training episodes", "-", CAT_RL, 1, 100000,
       "Number of complete missions the agent practises on (fast model).",
       "More practice usually gives a better policy, with no guarantee.", "Longer training (about 0.15 s per episode).",
       "Quicker, usually worse policy.", ""),
    _p("ppo_n_steps", "PPO steps per update", "-", CAT_RL, 64, 20000,
       "Number of decisions collected before each neural-network update.",
       "Larger batches give more stable but less frequent updates.", "Smoother, slower learning.",
       "Faster, noisier learning.", "Must be at least the batch size."),
    _p("ppo_batch_size", "PPO mini-batch size", "-", CAT_RL, 16, 20000,
       "Decisions per gradient step inside an update.", "Affects the noise of each gradient step.",
       "Smoother gradients.", "Noisier gradients.", "At most the PPO steps per update."),
    _p("ppo_epochs", "PPO epochs per update", "-", CAT_RL, 1, 50,
       "How many passes over the collected data per update.", "More passes extract more from the data but can overfit.",
       "Faster learning per sample; risk of instability.", "Slower, more stable.", ""),
    _p("ppo_learning_rate", "Learning rate", "-", CAT_RL, 1e-6, 1e-2,
       "Step size of neural-network weight updates.", "Controls how fast the policy changes.",
       "Faster but possibly unstable learning.", "Slower, steadier learning.", ""),
    _p("ppo_gamma", "Discount factor (gamma)", "-", CAT_RL, 0.8, 0.9999,
       "How much the agent values future reward compared with immediate reward (per decision).",
       "A transfer takes ~40-100 decisions, so gamma must be close to 1 for the agent to care about arriving.",
       "More far-sighted.", "More short-sighted (may never learn to finish).", ""),
    _p("eval_every_episodes", "Evaluate every", "episodes", CAT_RL, 1, 10000,
       "How often the current policy is tested deterministically (no exploration) to pick the best model.",
       "Only evaluated policies can become best_model.zip.", "Fewer checkpoints considered.",
       "More checkpoints considered, slower training.", ""),
    _p("seed", "Random seed", "-", CAT_SIM, 0, 2 ** 31 - 1,
       "Starting value of the random-number generator.", "Same seed gives an identical run (reproducible).",
       "(no direction) different random run.", "(no direction) different random run.", ""),
    _p("w_progress", "Reward: progress weight", "points", CAT_RW, 0.0, 1000.0,
       "Points given for closing the gap in orbital energy toward the target (100 = full gap closed).",
       "Main learning signal. Reward is a scoring rule for training, NOT a physical result.",
       "Agent rushes outward more.", "Weaker signal; slower learning.", ""),
    _p("w_eccentricity", "Reward: eccentricity penalty", "points", CAT_RW, 0.0, 100.0,
       "Penalty per 100 years while e is above the tolerance.", "Encourages a round orbit.",
       "Stronger push for circular orbits.", "Agent may tolerate elliptical orbits.", ""),
    _p("w_inclination", "Reward: inclination penalty", "points", CAT_RW, 0.0, 100.0,
       "Penalty per 100 years while the orbit is tilted beyond tolerance.", "Discourages out-of-plane thrust.",
       "Stronger.", "Weaker.", ""),
    _p("w_energy", "Reward: energy-use penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty for spending propulsion energy (10 = whole budget costs 10 points).", "Encourages efficiency.",
       "Agent saves energy (may arrive slower).", "Agent spends energy freely.", ""),
    _p("w_mass", "Reward: mass-use penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty for using asteroid mass (5 = whole asteroid costs 5 points).", "Encourages saving mass.",
       "Agent saves mass (uses faster, energy-hungry launches).", "Agent uses mass freely.", ""),
    _p("w_time", "Reward: time penalty", "points/decision", CAT_RW, 0.0, 10.0,
       "Small penalty per decision.", "Encourages arriving sooner.", "Faster arrival preferred.",
       "No hurry.", ""),
    _p("w_success", "Reward: success bonus", "points", CAT_RW, 0.0, 10000.0,
       "Bonus when the orbit enters the target window (plus extra for leftover resources).",
       "Makes success clearly better than near-misses.", "Stronger focus on finishing.", "Weaker.", ""),
    _p("w_terminal_error", "Reward: final-miss penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty at the deadline or when resources run out, proportional to the remaining orbit error.",
       "Distinguishes close misses from bad misses.", "Harsher.", "Milder.", ""),
    _p("w_failure", "Reward: failure penalty", "points", CAT_RW, 0.0, 10000.0,
       "Penalty for catastrophic outcomes (Earth unbound, too close to the Sun, Moon lost).",
       "Teaches the agent to avoid disasters.", "Harsher.", "Milder.", ""),
    _p("w_mars_safety", "Reward: Mars-proximity penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty per decision when Earth came closer to Mars than the shield's caution distance: 0 at the caution "
       "distance, the full weight at the safe distance. Getting closer than the safe distance ends the mission as a failure.",
       "Teaches the agent to plan a Mars-safe route instead of relying on the shield.", "Stronger avoidance (may cost time or energy).",
       "Weaker avoidance.", ""),
    _p("mars_gravity", "Mars gravity on", "true/false", CAT_SAFE, None, None,
       "true: Mars pulls on Earth and the Moon and Earth pulls on Mars (both orbits change). false: Mars is treated as massless "
       "(it still moves and is still used for the safety checks), for comparison only.",
       "Earth must cross Mars's orbit (1.38-1.67 AU) on the way to 2.7 AU, so Mars's gravity matters during close passes.",
       "-", "-", "Boolean."),
    _p("other_planets", "Mercury, Venus, Jupiter on", "true/false", CAT_SAFE, None, None,
       "true (default, update p35): Mercury, Venus and Jupiter move on their real (J2000) orbits and pull on Earth, the Moon "
       "(high-fidelity model), Mars, the thrown fragments (high-fidelity) and on each other; Earth pulls back, so their orbits "
       "change too (very slightly). The same planetary-safety rules as for Mars apply to them (safe distances: Mercury "
       "0.020 AU, Venus 0.048 AU, Jupiter 0.35 AU); coming closer ends the mission (planet_unsafe_approach). Planets beyond "
       "Jupiter (Saturn, Uranus, Neptune), dwarf planets and asteroids are not modelled. false: they still move and are "
       "drawn but are massless (for comparison).",
       "Jupiter is 318 Earth masses; at 2.7 AU Earth is only ~2.5 AU from Jupiter's orbit, so its pull shapes Earth's final "
       "orbit (mainly the eccentricity). Venus is the strongest pull near today's orbit. The agent does not observe the "
       "other planets directly; it sees their effect in Earth's orbit.",
       "-", "-", "Boolean. Not in the Mars safety shield's forecast (Sun + Earth + Mars only)."),
    _p("mars_max_tide_ratio", "Max Mars tide on Earth (x Moon's tide)", "-", CAT_SAFE, 1e-4, 10.0,
       "Largest allowed tidal acceleration from Mars on Earth, relative to the Moon's tide today. Tides, not the overall pull, "
       "are what stresses the oceans and crust (the overall pull makes all of Earth fall together, so nobody feels it). "
       "Mars's tide equals the Moon's at about 0.0053 AU (790,000 km).",
       "Sets one of the three safe distances: d = 384,400 km x (GM_mars / GM_moon / ratio)^(1/3).",
       "Closer passes allowed (riskier).", "Larger safety distance (harder to route around Mars).",
       "0.01 (default) -> 3.67 million km; ocean tide change under ~1 cm. Since p35 the same limit is also applied to Mercury, "
       "Venus and Jupiter (their safe distances scale with (GM)^(1/3))."),
    _p("mars_max_moon_perturbation", "Max Mars pull on the Moon's orbit", "-", CAT_SAFE, 1e-6, 0.1,
       "Largest allowed tidal pull of Mars on the Moon's orbit around Earth, relative to Earth's own pull on the Moon. "
       "The Moon stabilises Earth's axis tilt (seasons); a strong disturbance could change its orbit or free it. For "
       "comparison the Sun's tide is about 0.011 today.",
       "Sets the second safe distance: d = 384,400 km x (2 GM_mars / GM_earth / ratio)^(1/3).",
       "Closer passes allowed.", "Larger safety distance.", "1e-3 (default) -> 2.3 million km. Since p35 also applied to "
       "Mercury, Venus and Jupiter."),
    _p("mars_min_distance_km", "Hard minimum Earth-Mars distance", "km", CAT_SAFE, 1.0e4, 1.0e8,
       "Third safe distance: a plain distance limit (collision, debris, Roche limit and Hill-sphere effects all lie far inside "
       "2 million km). The safe distance used is the LARGEST of the three limits.",
       "Gives a margin that does not depend on the tide models.", "More conservative.", "Less conservative.",
       "Collision would occur at 9,760 km (sum of radii). Since p35 also the minimum distance from Mercury, Venus, Jupiter."),
    _p("mars_shield", "Mars safety shield", "true/false", CAT_SAFE, None, None,
       "Automatic safety layer of the launcher control (a 'shield' in safe reinforcement learning). Once per simulated year, "
       "while Earth's orbit overlaps Mars's orbit, it forecasts the Earth-Mars distance for the next years (Sun gravity plus the "
       "mutual Earth-Mars pull) for four options - the agent's command, coasting, full thrust, full thrust backwards (brake) - "
       "and picks the cheapest safe one, overriding the packet size if the agent's command would "
       "bring Earth within the caution distance. It cannot create energy or momentum: it only changes how much mass is thrown.",
       "Earth's path always crosses Mars's orbit; without planning, 0.05 AU passes occur in every tested trajectory.",
       "-", "-", "Boolean. Overrides are counted and shown."),
    _p("shield_caution_factor", "Shield caution factor", "x safe distance", CAT_SAFE, 1.0, 10.0,
       "The shield acts when the forecast closest approach is below this multiple of the safe distance.",
       "A margin for forecast errors (the forecast ignores the Moon and the fragments, and the shield only re-checks once a year).",
       "Earlier, more frequent overrides.", "Later overrides, less margin.", ""),
    _p("shield_horizon_years", "Shield forecast horizon", "yr", CAT_SAFE, 1.0, 50.0,
       "How far ahead the shield forecasts the Earth-Mars distance.",
       "Shifting Earth's position along its orbit by Mars-safe distances takes years of thrust or coasting.",
       "Plans further ahead (slower simulation).", "Reacts later; may be too late to avoid a pass.", ""),
    _p("solar_collectors", "Solar collectors on", "true/false", CAT_SOL, None, None,
       "Millions of solar collectors orbit the Sun between the Sun and Earth, turn sunlight into electricity and beam it as "
       "microwaves to receivers (rectennas) on the tips of Earth's 8 launcher arms. Only the arms receive the beams.",
       "With collectors on, the launcher's energy comes from the beams; the propulsion energy budget becomes a starting "
       "reserve that is used only when the beams cannot deliver enough. Energy is never created: every joule comes from "
       "intercepted sunlight and passes through the efficiency chain.",
       "-", "-", "Boolean. Off = the original fixed energy budget only."),
    _p("collector_fraction", "Share of the Sun's output collected", "0-1", CAT_SOL, 0.0, 0.9,
       "Fraction of ALL the light the Sun emits (3.828e26 W) that the collectors intercept. 0.30 = 30 %.",
       "To intercept a fraction f of all sunlight, the collectors must cover the same fraction of the sky as seen from the Sun. "
       "They are arranged as a belt around the Sun reaching up to +/- asin(f) above and below Earth's orbital plane "
       "(+/- 17.5 deg for 30 %), so they sit 'between the Sun and Earth' all along Earth's orbit.",
       "More power (and many more collectors).", "Less power.",
       "Lanes in front of Earth and Mars are kept clear so their sunlight is not blocked (lost share ~1e-5)."),
    _p("collector_orbit_au", "Collector belt distance from the Sun", "AU", CAT_SOL, 0.1, 0.95,
       "Radius of the collector belt. 0.5 AU lies between Mercury (0.39 AU) and Venus (0.72 AU).",
       "Sets the total collector area (f x 4 pi R^2) and the beam distance to Earth.",
       "Larger collectors in total, longer beams early on, shorter at the end.",
       "Smaller total area, but hotter collectors (4x sunlight at 0.5 AU).", "Must stay inside Earth's starting orbit (1 AU)."),
    _p("collector_count", "Number of collectors", "-", CAT_SOL, 1.0, 1e18,
       "How many separate collectors form the belt. Only changes the size and mass of each one (the total area is fixed).",
       "With 5 million collectors each one is about 65,000 km wide - 'panels' far larger than Earth.",
       "Smaller individual collectors.", "Larger individual collectors.", "Display / estimate only."),
    _p("collector_areal_density", "Collector mass per area", "kg/m^2", CAT_SOL, 0.001, 100.0,
       "Mass of one square metre of collector (thin film, wiring, structure, antenna). Used only for the material estimate.",
       "Total material = area x this. Also sets how strongly sunlight pushes the collectors (light pressure).",
       "More material needed.", "Less material, but light pressure becomes comparable to the Sun's gravity below ~0.001 kg/m^2.",
       "Estimate only; the collectors' own orbits are not simulated."),
    _p("pv_efficiency", "Solar-cell efficiency", "0-1", CAT_SOL, 0.01, 0.9,
       "RATED solar-cell efficiency at 25 C (standard test conditions). CHANGED MEANING (p28): before, this value was used "
       "directly as the operating efficiency; now the operating efficiency is computed from it with the panel temperature "
       "at the belt (~450 K at 0.5 AU -> x0.70) and the fleet's average age degradation (x0.95), i.e. ~0.165 for the default 0.25.",
       "First link of the power chain.", "More power.", "Less power, more heat on the collectors.", ""),
    _p("microwave_efficiency", "Electricity-to-microwave efficiency", "0-1", CAT_SOL, 0.01, 1.0,
       "Fraction of the collectors' electricity that leaves as a microwave beam (transmitter efficiency).",
       "Second link of the power chain.", "More beam power.", "Less beam power.", ""),
    _p("microwave_freq_ghz", "Microwave frequency", "GHz", CAT_SOL, 1.0, 300.0,
       "Frequency of the power beams. Wavelength = c / f (5.8 GHz -> 5.2 cm).",
       "A beam spreads by diffraction; shorter wavelengths stay narrower over the same distance.",
       "Narrower beams (more power reaches the arms at long distances).", "Wider beams, more power misses the arms.", ""),
    _p("transmitter_diameter_km", "Transmitter antenna diameter", "km", CAT_SOL, 0.01, 1e5,
       "Diameter of the phased-array antenna that forms one beam on the collector side.",
       "Captured fraction of a beam: 1 - exp(-tau^2), tau = D_transmitter x D_receiver / (wavelength x distance) (Goubau).",
       "Narrower beams.", "Wider beams; at 2-3 AU less power reaches the arms.", ""),
    _p("rectenna_diameter_km", "Rectenna diameter (per arm)", "km", CAT_SOL, 0.01, 1e5,
       "Diameter of the receiving antenna (rectenna) on the tip of each of the 8 arms.",
       "Same Goubau relation as the transmitter. Also sets the beam intensity at the dish.",
       "More of each beam is caught; lower intensity on the dish.", "Less is caught; extremely high intensity.", ""),
    _p("rectenna_efficiency", "Rectenna efficiency", "0-1", CAT_SOL, 0.01, 1.0,
       "Fraction of the caught microwave power turned back into electricity for the launcher (the rest heats the arms).",
       "Last link of the power chain.", "More power, less heat at the arms.", "Less power, more heat at the arms.", ""),
    _p("pv_temp_coeff", "Solar-cell temperature coefficient", "1/K", CAT_SOL, 0.0, 0.01,
       "Relative loss of solar-cell efficiency per kelvin above 25 C. The 'Solar-cell efficiency' parameter is the RATED value at "
       "25 C; the operating efficiency is computed from the panels' equilibrium temperature at the belt (about 430 K at 0.5 AU).",
       "Hot panels convert less sunlight. At 0.5 AU sunlight is 4x stronger than at Earth, so the panels run hot.",
       "Less electricity from the same sunlight.", "Panels lose less to heat (0 = ignore temperature).", ""),
    _p("collector_absorptance", "Panel absorptance", "0-1", CAT_SOL, 0.3, 1.0,
       "Share of the sunlight hitting a panel that it absorbs; the rest is reflected into space.",
       "Sets the panel temperature (absorbed sunlight minus electricity must be radiated as heat).",
       "Hotter panels (lower operating efficiency).", "Cooler panels; must stay above the solar-cell efficiency.", ""),
    _p("collector_emissivity", "Panel emissivity", "0-1", CAT_SOL, 0.1, 1.0,
       "How well each face of a panel radiates heat (both faces radiate).", "Good radiators keep the panels cool.",
       "Cooler panels, higher operating efficiency.", "Hotter panels.", ""),
    _p("pv_degradation_per_year", "Panel degradation", "1/yr", CAT_SOL, 0.0, 0.05,
       "Relative efficiency lost per year of panel age (radiation and heat damage).",
       "Old panels deliver less; with replacement after the lifetime the fleet's average loss is degradation x lifetime / 2.",
       "Lower average efficiency.", "Panels age more slowly.", ""),
    _p("collector_lifetime_years", "Panel lifetime", "yr", CAT_SOL, 1.0, 1000.0,
       "Panels are replaced after this many years.", "Longer life = less replacement material but more degradation.",
       "Fewer replacements, older (weaker) panels.", "More material and manufacturing energy per year.", ""),
    _p("collector_recycle_frac", "Recycled share of replaced panels", "0-1", CAT_SOL, 0.0, 1.0,
       "Share of the mass of replaced panels that is recycled into new ones.",
       "Sets how much FRESH material (e.g. mined from Mercury) the belt consumes over the mission.",
       "Less fresh material.", "More fresh material.", ""),
    _p("collector_initial_built", "Belt built at year 0", "0-1", CAT_SOL, 0.0, 1.0,
       "Share of the collector belt that exists when the mission starts.",
       "Power available early in the mission (the launcher uses the reserve for any shortfall).",
       "More power early.", "Less power early; the reserve covers the gap.", ""),
    _p("collector_build_years", "Build-up time of the belt", "yr", CAT_SOL, 0.0, 10000.0,
       "Years to build the rest of the belt (linear). 0 = complete from the start.",
       "The beam power grows during the build-up.", "Slower growth of the beam power.", "Faster growth.", ""),
    _p("collector_embodied_energy_jkg", "Embodied energy of collectors", "J/kg", CAT_SOL, 0.0, 1e10,
       "Energy needed to mine, refine and make 1 kg of collector. Paid from the collectors' own electricity (it is subtracted "
       "in the energy ledger).", "Makes the build-up and the replacements cost energy, so nothing is free.",
       "More of the collectors' output used for manufacturing.", "Less.", ""),
    _p("earth_lane_clear", "Keep Earth's sunlight lane clear", "true/false", CAT_SOL, None, None,
       "The belt surrounds the Sun at 0.5 AU, so it lies between the Sun and Earth. When true, the collectors leave a small "
       "moving gap (lane) so Earth and Mars still see the whole Sun (the lane costs ~2e-5 of the belt). When false, Earth "
       "is in the belt's shadow (no sunlight at all).", "Shading of Earth by the collectors.",
       "-", "Earth (and the mirrors next to it) lose all direct sunlight: Earth freezes.", ""),
    _p("collector_sk_authority", "Station-keeping authority", "0-1", CAT_SOL, 0.0, 1.0,
       "Share of the sunlight push (light pressure) a collector can use for steering by tilting its panels (no propellant).",
       "Keeps each collector on its assigned orbit against the beam recoil and the planets' pulls (viewer belt simulation).",
       "Stronger corrections.", "Weaker; 0 = no station keeping (the collectors drift).", ""),
    _p("climate_model", "Climate model on", "true/false", CAT_CLIM, None, None,
       "Two-box energy-balance climate (surface layer + deep ocean, ice-albedo feedback, fixed greenhouse effect). When on, "
       "the surface temperature is part of the RL task: the agent aims the space mirrors, gets a climate reward term and "
       "success needs a habitable temperature.", "Habitability is optimised by the agent, not only reported.", "-", "-", ""),
    _p("mirrors", "Space mirrors on", "true/false", CAT_CLIM, None, None,
       "A ring of thin aluminium-film mirrors orbiting Earth (ENABLER E3, 'soletta'). Each mirror can be turned so that it "
       "reflects sunlight onto Earth or past it. The agent decides how much extra light Earth gets.",
       "At 2.7 AU Earth gets only 14 % of today's sunlight and would freeze (~150 K).", "-", "Earth freezes as it moves out.", ""),
    _p("climate_target_K", "Target surface temperature", "K", CAT_CLIM, 250.0, 320.0,
       "Global-mean surface temperature the climate should be held at (today ~288 K = 15 C).", "Defines 'habitable'.",
       "Warmer target.", "Colder target.", ""),
    _p("climate_tol_K", "Temperature tolerance", "K", CAT_CLIM, 0.5, 30.0,
       "Success needs |T_surface - target| <= tolerance; the climate penalty grows outside it.", "Strictness of the climate goal.",
       "Easier.", "Stricter.", ""),
    _p("mirror_area_m2", "Total mirror area", "m^2", CAT_CLIM, 0.0, 1e17,
       "Total area of all space mirrors. They sit in an orbit at right angles to the Sun direction, so each mirror is tilted "
       "45 degrees and catches cos(45) = 71 % of the sunlight on its area. Light sent to Earth = area x 0.71 x reflectivity "
       "x sunlight at Earth's distance. Default 1.6e15 m^2 (12.5 x Earth's cross-section, e.g. 1 million mirrors of 40 x 40 km).",
       "At 2.7 AU about 1.3e15 m^2 is needed to give Earth today's sunlight.", "More margin far from the Sun (and more mass).",
       "Earth may freeze before reaching 2.7 AU.", ""),
    _p("mirror_reflectivity", "Mirror reflectivity", "0-1", CAT_CLIM, 0.1, 1.0,
       "Share of the sunlight hitting a mirror that is reflected (aluminised polymer film ~0.9). The rest heats the mirror and "
       "is radiated into space.", "Sets how much mirror area is needed.", "Less area needed.", "More area needed.", ""),
    _p("mirror_orbit_re", "Mirror orbit radius", "Earth radii", CAT_CLIM, 2.0, 55.0,
       "Radius of the mirrors' orbit around Earth (a 'terminator' orbit, its plane at right angles to the Sun direction, so the "
       "mirrors are never in Earth's shadow). Used for the drawing, the orbit period and the size of the light spot on Earth; "
       "the mirrors' orbital motion is not simulated.", "Far enough to stay sunlit, well inside the Moon's orbit (60 R_E).",
       "Bigger light spots, slower orbit.", "Smaller light spots, faster orbit.", ""),
    _p("mirror_areal_density", "Mirror film mass per area", "kg/m^2", CAT_CLIM, 1e-4, 1.0,
       "Mass of 1 m^2 of mirror including its support (5 g/m^2 is solar-sail class; today's sails are ~10 g/m^2).",
       "Gives the mirrors' total mass (reported).", "Heavier mirrors.", "Lighter mirrors.", ""),
    _p("arm_heat_to_earth_frac", "Arm waste heat reaching Earth", "0-1", CAT_CLIM, 0.0, 1.0,
       "Share of the arms' waste heat (rectenna + launcher losses, ~1e24 W) that ends up on Earth. ENABLER: radiators that "
       "point away from Earth. Without it (geometric value ~0.08) Earth would receive ~5e5 x today's sunlight and boil.",
       "Connects the power chain to the climate honestly.", "More heating of Earth.", "Less heating.", ""),
    _p("beam_spill_to_earth_frac", "Beam spill reaching Earth", "0-1", CAT_CLIM, 0.0, 1.0,
       "Share of the microwaves that miss the rectenna dishes and hit Earth. ENABLER: beam spill control.",
       "Heats Earth (climate model).", "More heating.", "Less heating.", ""),
    _p("w_climate", "Reward: climate penalty", "points per 100 y", CAT_RW, 0.0, 100.0,
       "Penalty per 100 years for the surface temperature being outside the tolerance (scaled by how far, capped).",
       "Makes the agent keep Earth habitable while moving it.", "Climate matters more vs. speed and mass.", "Climate matters less.", ""),
    _p("hifi_dt_hours", "High-fidelity time step", "h", CAT_SIM, 0.5, 24.0,
       "Integration step of the N-body (Sun, Earth, Moon, fragments) validation model.",
       "Smaller steps are more accurate but slower. The Moon's 27-day orbit needs about 6 h or less.",
       "Faster, less accurate (Moon orbit errors grow quickly above 12 h).",
       "More accurate, slower.", "0.5 to 24 h; the check '10 integrator convergence' shows the error at 12/6/3 h."),
    _p("max_fragments", "Max tracked fragments", "-", CAT_SIM, 4, 200,
       "How many ejected packets the N-body model integrates individually at the same time. When full, the two closest fragments merge (momentum kept, energy loss recorded).",
       "Representative-fragment approximation of thousands of packets.", "More detail, slower.",
       "Faster, coarser.", ""),
    _p("release_radius_m", "Packet hand-over distance", "m", CAT_SIM, 1.0e9, 1.0e10,
       "Distance from Earth at which a launched packet becomes an explicit N-body fragment (its climb out of Earth's gravity is computed analytically before that).",
       "Patched-conic approximation; it must be outside Earth's gravitational sphere of influence (Hill radius about 1.5e9 m).",
       "Less near-Earth detail.", "Below the Hill radius the approximation is invalid.", "1e9 to 1e10 m."),
    _p("fragment_escape_au", "Fragment escape distance", "AU", CAT_SIM, 10.0, 1000.0,
       "Fragments that are unbound from the Sun and farther than this leave the simulated region; their mass, momentum and energy are recorded in the ledger (never silently deleted).",
       "Keeps the N-body model small.", "Tracks fragments longer.", "Removes them sooner.", ""),
]
PARAM_BY_KEY = {d["key"]: d for d in PARAM_DOCS}
PARAM_SPEC = [(d["key"], d["label"], d["unit"]) for d in PARAM_DOCS]

CONSTANT_DOCS = [
    dict(name="Gravitational constant G", value=G, unit="m^3 kg^-1 s^-2", meaning="Strength of gravity in Newton's law F = G m1 m2 / r^2."),
    dict(name="Sun mass", value=M_SUN, unit="kg", meaning="From GM_sun = 1.32712440041e20 m^3/s^2."),
    dict(name="Earth mass", value=M_EARTH, unit="kg", meaning="From GM_earth = 3.986004418e14 m^3/s^2."),
    dict(name="Moon mass", value=M_MOON, unit="kg", meaning="From GM_moon = 4.9028e12 m^3/s^2."),
    dict(name="Mars mass", value=M_MARS, unit="kg", meaning="From GM_mars = 4.282837e13 m^3/s^2 (about 1/9.3 of Earth)."),
    dict(name="Mercury mass", value=GM_MERCURY / G, unit="kg", meaning="From GM = 2.2032e13 m^3/s^2 (0.055 Earth masses); orbit 0.387 AU."),
    dict(name="Venus mass", value=GM_VENUS / G, unit="kg", meaning="From GM = 3.2486e14 m^3/s^2 (0.815 Earth masses); orbit 0.723 AU."),
    dict(name="Jupiter mass", value=GM_JUPITER / G, unit="kg", meaning="From GM = 1.2671e17 m^3/s^2 incl. its moons (318 Earth masses); orbit 5.203 AU. Planets beyond Jupiter are not modelled."),
    dict(name="Mars radius", value=R_MARS, unit="m", meaning="Used for fragment impacts on Mars and the collision distance."),
    dict(name="Astronomical unit (AU)", value=AU, unit="m", meaning="Mean Sun-Earth distance; the unit of orbit size."),
    dict(name="Year", value=YEAR, unit="s", meaning="Julian year (365.25 days); all 'simulated years' use it."),
    dict(name="Earth radius", value=R_EARTH, unit="m", meaning="Launch radius used for the escape energy (conservative: the real release point "
         "of the arms is ~1.29 R_E from Earth's centre, which would need ~5e-6 less energy); escape speed from it is ~11.2 km/s."),
    dict(name="Arm length", value=ARM_LENGTH_RE * R_EARTH, unit="m", meaning="Each of the 8 launcher arms is 2 Earth radii long (update p36)."),
    dict(name="Launch-ring radius", value=RING_RADIUS_RE * R_EARTH, unit="m", meaning="The 8 arm roots sit on this ring around Earth (1.25 R_E)."),
    dict(name="Grab point height on the ring axis", value=GRAB_HEIGHT_RE * R_EARTH, unit="m", meaning="The docked asteroid sits on the ring's "
         "axis with its Earth-facing side here, so all 8 arm roots are the same distance (1.77 R_E < 2 R_E) from it."),
    dict(name="Moon distance", value=MOON_A, unit="m", meaning="Mean Earth-Moon distance (initial Moon orbit)."),
    dict(name="Solar constant at 1 AU", value=S_1AU, unit="W/m^2", meaning="Sunlight intensity today; used for the insolation estimate."),
    dict(name="Speed of light", value=C_LIGHT, unit="m/s", meaning="Only used to warn that the model is non-relativistic."),
]

SPEED_DOC = _p("sim_speed", "Simulation speed (time multiplier)", "simulated time per real second", CAT_VIEW, None, None,
               "How much simulated time the 3-D view advances per second of real (wall-clock) time. It does not change the physics.",
               "10,000 years cannot be watched in real time.",
               "Faster overview: Earth circles the Sun many times per second, so watch the trail.",
               "Slower: individual packets and the Moon become visible (packets cross Earth's orbit in hours at ~1% of light speed).",
               "1 day/s to 500 yr/s. The physics runs ahead of the display by at most 3 decision intervals or 2.5 s of display "
               "time (max 1,500 years); the dim grey dots show that computed-ahead part.")


def _o(key, label, unit, kind, measures, how, better, success):
    """One output description: what it measures, how it is computed, what is better, success threshold."""
    return dict(key=key, label=label, unit=unit, kind=kind, measures=measures, how=how, better=better, success=success)


K_SIM, K_CALC, K_EST, K_PRED, K_SCORE = ("simulated state", "calculated from simulated state", "estimate (simplified)",
                                         "prediction", "training score")
OUTPUT_DOCS = [
    _o("a", "Semi-major axis a", "AU", K_CALC, "Size of Earth's current orbit (half the longest diameter of the ellipse).",
       "From Earth's simulated position r and velocity v: a = 1 / (2/|r| - |v|^2/mu), mu = G(M_sun + M_earth + M_moon).",
       "Closer to the target (2.7 AU)", "Success needs |a - target| <= tolerance."),
    _o("e", "Eccentricity e", "-", K_CALC, "How elongated the orbit is: 0 = circle, 0.0167 = Earth today, 1 = escape.",
       "Length of the eccentricity vector computed from r and v.", "Lower", "Success needs e <= tolerance (round, stable orbit)."),
    _o("inc", "Inclination", "deg", K_CALC, "Tilt of the orbit plane relative to today's orbit plane.",
       "Angle between the orbital angular-momentum vector and the original orbit normal.", "Lower",
       "Success needs inclination <= tolerance."),
    _o("r", "Distance from Sun", "AU", K_SIM, "Earth's current distance from the Sun.", "|r| from the physics state.",
       "-", "Oscillates during each orbit; a is the better progress measure."),
    _o("speed", "Orbital speed", "km/s", K_SIM, "Earth's speed around the Sun (29.8 km/s today).",
       "|v| from the physics state.", "- (it DEcreases as the orbit grows: 18.1 km/s at 2.7 AU)",
       "Shows the paradox of orbit raising: pushing forward makes the orbit bigger and the speed smaller."),
    _o("E_orb", "Orbital energy", "J", K_CALC, "Kinetic + gravitational energy of the Earth-Moon system in its orbit around the Sun.",
       "E = M (|v|^2/2 - mu/|r|) = -mu M / (2a). Negative = bound to the Sun.", "Closer to zero (higher) until the target value",
       "Raising the orbit from 1 to 2.7 AU requires about +1.7e33 J of orbital energy."),
    _o("dv", "Velocity change delivered (delta-v)", "m/s", K_CALC,
       "Total speed change the recoil has given the Earth-Moon system so far.",
       "Sum over packets of recoil momentum / system mass (= impulse / M).",
       "Enough to reach the target (about 11.7 km/s for a slow spiral), but no more",
       "Compare with the required 11.7 km/s from the feasibility gate."),
    _o("thrust", "Average thrust", "N", K_CALC, "Average push on Earth during the last decision interval.",
       "Impulse delivered in the interval / interval duration.", "Depends: more thrust = faster transfer, more resources",
       "Tiny compared with Earth's weight in the Sun's gravity (3.5e22 N), so the orbit changes slowly."),
    _o("impulse", "Momentum delivered (impulse)", "N s", K_SIM,
       "Total momentum transferred to the Earth-Moon system by the recoil of all packets.",
       "Sum of reduced-mass x relative packet speed for every packet (exact momentum conservation).",
       "Enough for the target", "Momentum is conserved: packets carry exactly the opposite momentum."),
    _o("m_ast", "Remaining asteroid mass", "kg (%)", K_SIM, "Reaction mass still available.",
       "Initial mass minus all ejected packets (finite reservoir; cannot go below zero).",
       "Higher at arrival = more efficient", "Zero before arrival means the mission ran out of mass."),
    _o("E_used", "Energy consumed", "J (%)", K_SIM, "Propulsion energy drawn from the budget so far.",
       "Sum over packets of (1/2 reduced mass x launch speed^2) / efficiency.", "Lower for the same result",
       "At 100 % no more packets can be launched."),
    _o("packets", "Packets ejected", "-", K_SIM, "Number of ejection events so far.", "Counted by the physics ledger.",
       "-", "Each one produces a small recoil."),
    _o("recoil", "Recoil per packet", "m/s", K_SIM, "Speed change of Earth from the latest single packet.",
       "|v after - v before| at the ejection instant.", "-", "Tiny (well below 1 m/s) but repeated tens of thousands of times."),
    _o("moon", "Moon bound", "yes/no", K_CALC, "Whether the Moon still orbits Earth.",
       "Both models: negative Earth-Moon two-body energy and distance < 1/2 Hill radius (the Moon is simulated in both).",
       "yes", "Losing the Moon ends the episode as a failure."),
    _o("insolation", "Insolation", "x today", K_EST, "Sunlight per square metre at the final orbit relative to today.",
       "1 / a^2 (orbit-averaged, ignoring e).", "Not a goal of the optimisation; ~0.14 at 2.7 AU",
       "Habitability indicator only; no climate model."),
    _o("t_eq", "Equilibrium temperature", "K", K_EST, "Black-body temperature of Earth with today's albedo at the final orbit.",
       "255 K x a^-1/2 (no greenhouse effect).", "-", "Rough indicator only; real temperature depends on atmosphere."),
    _o("reward", "Reward (per decision)", "points", K_SCORE, "Score the agent receives after each decision.",
       "Weighted sum of progress, penalties and bonuses (see reward weights).", "Higher",
       "A scoring rule for training only. It is NOT a physical quantity; physical results are shown separately."),
    _o("return", "Episode return", "points", K_SCORE, "Sum of all rewards in one episode.", "Sum of per-decision rewards.",
       "Higher", "Good returns usually mean success, but always check a, e and the validation."),
    _o("progress", "Training progress", "episodes", K_SIM, "Completed training episodes out of the target.",
       "Counted after each finished episode.", "-", "More episodes do not guarantee a better policy."),
    _o("success_rate", "Success rate", "%", K_CALC, "Fraction of test episodes that reached the target window.",
       "Deterministic evaluation episodes (no exploration noise).", "Higher", "Used first when choosing the best model."),
    _o("cons_E", "Energy conservation error", "fraction of |E|", K_CALC,
       "High-fidelity only: total energy change NOT explained by recorded events (launches, impacts, escapes, merges).",
       "(E_now - E_start - sum of event energy changes) / |E_start|.", "Closer to 0",
       "Measures numerical error. Near 1e-6 here mainly because near-Earth fragment fly-outs are under-resolved."),
    _o("cons_P", "Momentum conservation error", "fraction", K_CALC, "Unexplained change of total momentum.",
       "|P_now - P_start - sum of event changes| / (M_earth x 30 km/s).", "Closer to 0", "Round-off level (~1e-13) means momentum is conserved."),
    _o("cons_L", "Angular-momentum conservation error", "fraction", K_CALC, "Unexplained change of total angular momentum.",
       "|L_now - L_start - sum of event changes| / |L_start|.", "Closer to 0", "Round-off level means it is conserved."),
    _o("cons_M", "Mass accounting error", "fraction", K_CALC, "Mass not explained by escaped fragments.",
       "(M_now + M_escaped - M_start) / M_start.", "0", "Zero means no mass appears or disappears."),
    _o("shortfall", "Shortfall", "AU", K_CALC, "Target radius minus final semi-major axis.", "target - a.",
       "0 (within tolerance)", "Positive = did not get far enough; negative = overshoot."),
    # ---- Earth-POV view: display-only environment estimates (space_environment / frontend envmodel.js)
    _o("env_r", "Distance from Sun (POV)", "AU", K_SIM, "Earth's simulated distance from the Sun; all POV environment "
       "numbers below are evaluated at this distance.", "|r| from the physics state.", "Grows toward the target",
       "Environment values change because this distance changes."),
    _o("env_sw", "Solar wind density and speed", "protons/cm^3, km/s", K_EST, "The stream of charged particles "
       "(plasma) from the Sun that hits Earth's magnetic field.", "Average quiet wind: 5 protons/cm^3 at 1 AU falling as "
       "1/r^2; speed 400 km/s at every distance. No storms, coronal mass ejections or solar cycle.",
       "Informational (lower = gentler space weather)", "Not part of the mission criteria; it thins ~7x at 2.7 AU."),
    _o("env_dp", "Solar-wind dynamic pressure", "nPa", K_EST, "Push of the solar wind on Earth's magnetic field.",
       "P = m_p n v^2 = 1.6726e-6 x n[cm^-3] x v[km/s]^2 nPa.", "Informational",
       "Lower pressure lets the magnetosphere expand (larger magnetopause)."),
    _o("env_imf", "Interplanetary magnetic field", "nT", K_EST, "The Sun's magnetic field carried outward by the wind.",
       "Parker spiral: radial part 3.5 nT x (1 AU/r)^2, azimuthal part = radial x (Sun rotation rate x r / wind speed); "
       "magnitude ~5 nT at 1 AU.", "Informational", "Not part of the mission criteria."),
    _o("env_mp", "Magnetopause stand-off distance", "R_E (Earth radii)", K_EST, "Distance from Earth's centre to the "
       "boundary of its magnetic bubble on the Sun-facing side.", "Shue et al. (1998) with IMF Bz = 0: "
       "r0 = (10.22 + 1.29 tanh(0.184 x 8.14)) x P^(-1/6.6); shape r = r0 (2/(1+cos theta))^alpha, "
       "alpha = 0.58 (1 + 0.024 ln P).", "Larger = better shielding from the solar wind",
       "~10.9 R_E at 1 AU, ~14.7 R_E at 2.7 AU: moving out ENLARGES the magnetosphere."),
    _o("env_bs", "Bow-shock stand-off distance", "R_E", K_EST, "Where the supersonic solar wind is abruptly slowed "
       "before reaching the magnetopause.", "1.29 x magnetopause stand-off (gas-dynamic Farris-Russell relation with an "
       "assumed fast-mode Mach number of 8). Flank shape schematic.", "Informational", "Moves out with the magnetopause."),
    _o("env_bfield", "Earth's magnetic field strength", "nT", K_EST, "Field of Earth's internal dynamo, at the "
       "surface equator and at the magnetopause nose.", "Centred dipole, 31,200 nT at the surface equator, falling as "
       "1/r^3 (dipole only: the compression by the solar wind roughly doubles the real value at the nose).",
       "Informational", "Earth's own field is assumed unchanged by the mission."),
    _o("env_hill", "Hill sphere radius", "R_E and km", K_CALC, "Region where Earth's gravity dominates the Sun's "
       "tide; the Moon and anything that should stay with Earth must orbit well inside it.",
       "r_H = r x (M_earth+moon+asteroid / (3 M_sun))^(1/3) with r = current Sun distance.",
       "Larger = Moon more safely bound", "Grows from ~1.5 million km at 1 AU to ~4 million km at 2.7 AU."),
    _o("mars_dist", "Distance to Mars", "AU", K_SIM, "Current distance between Earth and Mars (both simulated, both pulled by each other).",
       "|r_earth - r_mars| from the physics state.", "Larger (must stay above the safe distance)",
       "Mars's average orbit is at 1.52 AU, so Earth's path to 2.7 AU must cross it."),
    _o("mars_min_dist", "Closest approach to Mars", "AU", K_SIM, "Smallest Earth-Mars distance so far in the mission "
       "(or in the last decision interval in the decision log).", "Minimum over every integration step, refined on each step's chord.",
       "Larger", "Must stay above the safe distance, otherwise the mission fails (reason: mars_unsafe_approach)."),
    _o("mars_safe", "Mars-safe distance", "AU", K_CALC, "The planetary-safety limit: the largest of (1) the distance where Mars's "
       "tide on Earth equals mars_max_tide_ratio x the Moon's tide, (2) the distance where Mars disturbs the Moon's orbit by "
       "mars_max_moon_perturbation, (3) the hard minimum distance.", "Formulas in the parameter explanations; default 0.0246 AU "
       "(3.67 million km, set by the tide).", "-", "The caution distance (shield trigger) is shield_caution_factor x this."),
    _o("mars_tide", "Mars's tide on Earth (at closest approach)", "x Moon's tide", K_CALC, "How strong Mars's tidal "
       "stretching of Earth is compared with the Moon's (which causes our ocean tides).",
       "(GM_mars / d^3) / (GM_moon / d_moon^3).", "Lower (default limit 0.01)", "1.0 would double tidal forces on Earth."),
    _o("planet_dist", "Distance to Mercury / Venus / Jupiter", "AU", K_SIM, "Simulated distance between Earth and each of the "
       "three other modelled planets (update p35). Saturn and beyond are not modelled.",
       "|r_planet - r_earth| from the simulated states (fast model: Kepler drift + mutual kicks; high-fidelity: N-body).",
       "Larger than the safe distance", "Closer than the safe distance (Mercury 0.020, Venus 0.048, Jupiter 0.35 AU with "
       "the default Mars rules scaled by (GM)^(1/3)) ends the mission as planet_unsafe_approach."),
    _o("planet_min", "Closest approach to Mercury / Venus / Jupiter", "AU", K_SIM, "Smallest Earth-planet distance during "
       "the mission so far, checked at every gravity sub-step (at most 12 simulated days apart, finer near a planet).",
       "Minimum over all sub-steps of |r_planet - r_earth|.", "Higher", "Must stay above the safe distance."),
    _o("planet_orbit", "Orbit of Mercury / Venus / Jupiter (a, e) and its change", "AU, -", K_CALC,
       "Earth pulls the planets too (two-way gravity), so their orbits change - in the tested missions by less than "
       "0.001 AU (Jupiter about +1e-4 AU).", "Osculating elements from the simulated position and velocity.",
       "Small changes", "Information only."),
    _o("planet_pull_sum", "Sum of the planet's velocity kicks on Earth", "m/s", K_CALC,
       "Fast model only: the sum of the SIZES of all velocity kicks this planet gave the Earth-Moon centre (including the "
       "Sun-wobble term). The kicks point in changing directions and mostly cancel, so this is NOT the net change of "
       "Earth's velocity; it shows how busy the planet's influence was.", "Sum over gravity sub-steps of |dv|.",
       "Information only", "Information only."),
    _o("mars_moon_perturb", "Mars's pull on the Moon's orbit (at closest approach)", "x Earth's pull", K_CALC,
       "How strongly Mars's tide disturbs the Earth-Moon orbit, relative to Earth's pull on the Moon.",
       "2 (M_mars / M_earth) (d_moon / d)^3.", "Lower (default limit 0.001)", "The Sun's tide on the Moon is ~0.011 for comparison."),
    _o("mars_orbit", "Mars's orbit (a, e) and its change", "AU, -", K_CALC, "Earth's pull changes Mars's orbit as well: "
       "these are Mars's current semi-major axis and eccentricity and the change since the start.",
       "Same formulas as for Earth, from Mars's simulated position and velocity (mu = G(M_sun + M_mars)).",
       "Informational (small changes = Mars left largely undisturbed)",
       "Repeated passes can change Mars's orbit noticeably; the outcome is chaotic (sensitive to small differences)."),
    _o("shield", "Mars shield overrides / mode", "count", K_SIM, "How often the safety shield changed the agent's packet size, "
       "and what it did: 0 = agent's command kept, 1 = coast, 2 = full thrust, 3 = full thrust backwards (brake).",
       "Counted per ejection event inside the physics kernel.", "Fewer (agent plans Mars-safe routes itself)",
       "Overrides never create energy or momentum: they only change how much asteroid mass is thrown."),
    _o("shield_forecast", "Shield forecast closest approach", "AU", K_PRED, "The closest Earth-Mars distance the shield "
       "predicted for the next shield_horizon_years for the option it chose.", "Two-planet forecast (Sun + mutual Earth-Mars "
       "gravity) with the chosen thrust option.", "Above the caution distance", "A prediction, not a measurement."),
    _o("power_avail", "Beam power available at the arms", "W", K_CALC, "Electric power the collectors could deliver to the "
       "launcher right now, after every loss in the chain.", "Sun's output x collected share x solar-cell efficiency x "
       "microwave efficiency x captured fraction (Goubau, averaged over the belt) x rectenna efficiency.",
       "Higher than the launcher needs", "Falls slowly as Earth moves away (beams spread more over longer distances)."),
    _o("power_beam", "Microwave power absorbed by the arms", "W", K_SIM, "Microwave power the 8 rectennas absorb at this moment. "
       "The beams are throttled to what the launcher actually uses: no launch, no beam.", "Electricity the launcher draws from "
       "the beam / rectenna efficiency, per ejection event.", "Informational", "Never more than the available beam power."),
    _o("power_capture", "Beam capture efficiency", "0-1", K_CALC, "Average fraction of a beam that lands on the rectennas.",
       "1 - exp(-tau^2), tau = D_t D_r / (wavelength x distance), averaged over all collectors (beams passing the Sun are blocked).",
       "Higher", "Diffraction: decreases with distance."),
    _o("energy_sources", "Energy from collectors / from the reserve", "J", K_SIM, "How much launch energy came through the "
       "microwave beams and how much from the starting reserve.", "Ledger kept by the engine every decision interval.",
       "Informational", "Total drawn = collectors + reserve (checked to round-off)."),
    _o("arm_heat", "Heat at the arms", "W", K_EST, "Power that ends up as heat in the arms: rectenna losses + launcher losses.",
       "(beam power absorbed - electricity) + electricity used x (1 - launch efficiency).", "Lower",
       "Millions of times the sunlight Earth absorbs. Only the share arm_heat_to_earth_frac reaches Earth (ENABLER E1, directional arm radiators: "
       "directional radiators); that share enters the climate model. The radiators themselves are not modelled."),
    _o("beam_push", "Microwave push on Earth (not applied)", "m/s", K_EST, "Absorbed microwaves carry momentum E / c and push "
       "the arms away from the collectors.", "Upper bound: total absorbed beam energy / (c x mass of Earth + Moon + asteroid).",
       "Informational", "Mostly along the Sun-Earth line, where a push changes the orbit size only weakly (~e x push); "
       "estimated, not applied to the orbit."),
    _o("T_surface", "Surface temperature (climate model)", "K", K_EST, "Global-mean surface temperature of Earth.",
       "Two-box energy balance integrated along Earth's simulated distance: dT_s/dt = [S (1 - albedo(T_s))/4 + waste heat/area "
       "- 0.61 sigma T_s^4 - 0.7 (T_s - T_deep)] / 8; S = sunlight x (1 - shading) + mirror light.",
       "Within target +- tolerance", "Success needs it; freezes (~150 K) at 2.7 AU without the space mirrors."),
    _o("T_deep", "Deep-ocean temperature", "K", K_EST, "Temperature of the slow heat reservoir (deep ocean).",
       "dT_d/dt = 0.7 (T_s - T_d) / 100 (time constant ~140 y).", "Near the target", "Gives the climate a memory of decades to centuries."),
    _o("mirror_light", "Mirror light on Earth", "W (x today's sunlight)", K_SIM, "Sunlight the space mirrors reflect onto Earth.",
       "min(capacity, (set-point - sunlight) x pi R_earth^2); capacity = mirror area x cos 45 x reflectivity x sunlight at "
       "Earth's distance; the set-point is chosen by the agent (action a4).",
       "As low as possible while the temperature is in tolerance", "Grows to ~0.86 x today's sunlight at 2.7 AU."),
    _o("mirror_cap", "Mirror capacity", "x today's sunlight", K_CALC, "Most light the mirrors could send to Earth right now.",
       "mirror area x cos 45 x reflectivity x 1361 W/m^2 / r^2, divided by the sunlight Earth intercepts today.",
       "Above what is needed", "Falls as 1/r^2 (like the sunlight); at 2.7 AU 1.10 x today with the default area."),
    _o("insol_set", "Insolation set-point (agent's climate action)", "x today", K_SIM, "Total light on Earth the agent asks for.",
       "1 + 0.5 a4.", "Informational", "About 1.0 keeps today's climate."),
    _o("T_no_mirror", "Temperature without the mirrors", "K", K_EST, "Where the surface temperature would settle if the mirrors "
       "were turned away now.", "Equilibrium of the climate model with sunlight only (frozen branch if no warm state exists).",
       "Informational", "Shows what the mirrors are doing: ~150 K (snowball) at 2.7 AU."),
    _o("ice_line", "Ice line latitude", "degrees", K_EST, "Latitude beyond which the surface is permanently frozen.",
       "Zonal temperature T(lat) = T_mean + dT/3 - dT sin^2(lat) with dT = 45 K equator-to-pole (today's Earth); ice where "
       "T(lat) < 263 K (-10 C).", "Near today's ~70 degrees", "0 = snowball Earth (ice to the equator)."),
    _o("grow_frac", "Share of the surface with growing temperatures", "%", K_EST,
       "Share of Earth's surface with an annual-mean temperature between 5 and 40 C (plants can grow; liquid water).",
       "Same zonal temperature profile as the ice line.", "Higher", "Today ~75 %; 0 % for a snowball Earth."),
    _o("waste_heat_earth", "Waste heat reaching Earth", "W", K_CALC, "Heat from the power chain that ends up on Earth.",
       "beam_spill_to_earth_frac x microwaves missing the dishes + arm_heat_to_earth_frac x (rectenna + launcher heat).",
       "Lower", "Both shares are required properties of the enablers (E1 arm radiators, E2 beam spill control); without them the oceans would boil."),
    _o("years_outside", "Years outside the livable band", "yr", K_EST, "Simulated years with the surface temperature more "
       "than 15 K from the target.", "Summed over the climate sub-steps.", "0", "A crude habitability measure for the whole mission."),
    _o("ledger", "Energy ledger", "J", K_SIM, "Where every joule went: sunlight on the collectors (+ reserve) = reflected + "
       "collector heat + manufacturing + transmitter heat + beam spill + rectenna heat + packet kinetic energy + launcher heat "
       "; and sunlight on the mirrors = mirror light onto Earth + light turned past Earth + mirror heat.",
       "Closed every decision interval from the engine's bookkeeping.",
       "Residual ~1e-16", "Checked by check 18."),
    _o("panel_T", "Collector panel temperature", "K", K_CALC, "Equilibrium temperature of a collector at 0.5 AU.",
       "absorptance x S = eta(T) x S + 2 emissivity sigma T^4 (solved by iteration).", "Lower", "~450 K: reduces the solar-cell efficiency."),
    _o("eta_op", "Operating solar-cell efficiency", "0-1", K_CALC, "Electricity / sunlight on the panels in operation.",
       "rated x (1 - temp_coeff (T - 298 K)) x (1 - degradation x lifetime / 2).", "Higher", "~0.165 for the rated 0.25."),
    _o("built", "Belt built", "0-1", K_CALC, "Share of the collector belt that exists.", "Linear build-up from collector_initial_built "
       "over collector_build_years.", "Informational", "Less power early in the mission."),
    _o("belt_dev", "Collector deviation from its slot", "km", K_SIM, "Distance of each simulated collector from its assigned "
       "position on its assigned orbit, with and without station keeping.", "Viewer belt simulation: Encke deviation equations "
       "with the Sun (light pressure included), Earth+Moon, Mars, the beam recoil and the sail-tilt controller; RK4, 0.5 day.",
       "Small (with station keeping)", "~600 km with station keeping (spacing 65,000 km); millions of km of drift without."),
    _o("env_gsun", "Sun's gravitational pull on Earth", "m/s^2", K_CALC, "Acceleration of Earth toward the Sun; this "
       "is what bends Earth's path into an orbit.", "GM_sun / r^2 at the simulated distance.", "Informational",
       "Falls as 1/r^2: 5.9e-3 m/s^2 at 1 AU, 8.1e-4 m/s^2 at 2.7 AU."),
    _o("env_gmoon", "Moon's gravitational pull on Earth", "m/s^2", K_CALC, "Acceleration of Earth toward the Moon.",
       "GM_moon / d^2 with d = simulated Earth-Moon distance.", "Informational",
       "~3.3e-5 m/s^2; it makes Earth wobble around the Earth-Moon centre of mass."),
    _o("env_insol", "Sunlight at Earth's current distance", "W/m^2", K_CALC, "Solar power per square metre arriving "
       "at Earth right now.", "1361 W/m^2 x (1 AU / r)^2.", "Close to 1361 for today's climate",
       "At 2.7 AU only ~14 % of today's sunlight (see habitability)."),
]
OUTPUT_BY_KEY = {d["key"]: d for d in OUTPUT_DOCS}

GLOSSARY = [
    ("Energy-balance climate model", "Computes a planet's average temperature from the energy it receives (sunlight, other "
     "light, waste heat) and the heat it radiates to space; here with a fast surface layer and a slow deep ocean."),
    ("Ice-albedo feedback / snowball Earth", "Ice reflects more sunlight than ocean or land, so a colder Earth absorbs less "
     "and cools further. Below ~255 K the model has a stable frozen 'snowball' state."),
    ("Space mirrors (soletta)", "Thin aluminium-film mirrors in orbit around Earth (enabler E3) that reflect extra sunlight "
     "onto Earth so it stays warm far from the Sun. Turning a mirror changes where its light goes, so the extra light can be "
     "dialled up or down. The agent chooses how much light Earth should get. Small versions have flown (Znamya-2, 1993)."),
    ("Ice line", "The latitude beyond which the ground and the sea stay frozen all year. Today about 70 degrees; a "
     "'snowball' Earth is frozen to the equator."),
    ("Weather view", "A close-up of Earth that shows clouds, winds, storms, ice caps, vegetation and city lights. The ice "
     "line, the green share and the temperatures come from the climate model; the cloud and wind PATTERNS are a picture of "
     "today's Earth (trade winds, westerlies, storm tracks) - weather is not simulated."),
    ("Energy ledger", "A table that follows every joule from sunlight to its final form (heat, reflected light, kinetic "
     "energy of the packets, light on Earth). Its sum must equal the input: conservation of energy."),
    ("Station keeping", "Small corrections that keep a spacecraft at its assigned place; here by tilting the collectors' "
     "panels so the sunlight push points slightly sideways (no fuel)."),
    ("Enabler", "A technology the model assumes without showing it can be built (exactly three, E1-E3, listed in the Guide)."),
    ("AU (astronomical unit)", "The average Sun-Earth distance, 149.6 million km. 2.7 AU is in the asteroid belt."),
    ("Orbit / ellipse", "The closed path of a planet around the Sun. It is described by its size (a), shape (e) and tilt (inclination)."),
    ("Semi-major axis (a)", "Half of the longest diameter of the orbit ellipse; the 'size' of the orbit. It fixes the orbital period and energy."),
    ("Eccentricity (e)", "How stretched the ellipse is: 0 is a circle; values near 1 are very elongated."),
    ("Inclination", "Tilt of the orbit relative to a reference plane (here: today's orbit)."),
    ("Delta-v", "A change of velocity. Mission planners measure how hard a manoeuvre is in delta-v."),
    ("Impulse / momentum", "Momentum = mass x velocity. Throwing a packet gives Earth the equal and opposite momentum (Newton's third law)."),
    ("Recoil", "The backward push on the thrower when mass is thrown (like a gun's kick)."),
    ("Reaction mass", "Mass thrown away to push the remaining body forward; here, pieces of the asteroid."),
    ("Exhaust / ejection velocity", "Speed of the thrown mass relative to the thrower. Higher speed = more push per kg but more energy per kg."),
    ("Rocket equation", "delta-v = v_exhaust x ln(M_start / M_end): the limit on speed change from finite reaction mass."),
    ("Escape speed", "Minimum launch speed to leave a body's gravity (11.2 km/s for Earth). Slower packets fall back."),
    ("Prograde / radial / normal (RTN frame)", "Directions relative to Earth's orbit: prograde = along the motion, radial = away from the Sun, normal = perpendicular to the orbit plane."),
    ("Kepler orbit (two-body)", "Exact motion of one body around another under gravity alone; used by the fast model between ejections."),
    ("N-body simulation", "Numerically integrating the mutual gravity of several bodies; used by the high-fidelity model."),
    ("Yoshida integrator", "A 4th-order time-stepping method that conserves energy well over long times."),
    ("Patched conic", "Treat a packet's climb out of Earth's gravity analytically, then follow it in the Sun's gravity."),
    ("Hill sphere", "Region where Earth's gravity dominates over the Sun's (~1.5 million km at 1 AU). It grows in "
     "proportion to the distance from the Sun."),
    ("Lagrange points L1 / L2", "Balance points on the Sun-Earth line at the edge of the Hill sphere: L1 toward the Sun, "
     "L2 away from it. In the gravity-well picture they are the saddles on the rim."),
    ("Gravity well", "A picture of gravitational potential as height: bodies 'roll' downhill. The Earth-POV layer "
     "shows Earth's well, the Moon's dimple and the Sun's tide tilting the rim (log-scaled)."),
    ("Solar wind / plasma", "A hot, electrically charged gas of protons and electrons streaming from the Sun at ~400 km/s. "
     "Its density falls as 1/distance^2."),
    ("Interplanetary magnetic field (IMF)", "The Sun's magnetic field dragged outward by the solar wind into a spiral "
     "(Parker spiral). About 5 nT at 1 AU."),
    ("Magnetosphere / magnetopause", "The magnetosphere is the bubble carved out of the solar wind by Earth's magnetic "
     "field; its boundary is the magnetopause (~10 Earth radii toward the Sun at 1 AU)."),
    ("Bow shock", "The standing shock wave in front of the magnetosphere where the supersonic solar wind slows down "
     "abruptly, like the bow wave of a ship."),
    ("Magnetic dipole", "The simplest magnetic field shape (a bar magnet). Earth's field is close to a dipole tilted ~10 "
     "degrees from its spin axis; its strength falls as 1/distance^3."),
    ("Radiation (Van Allen) belts", "Doughnut-shaped zones of energetic particles trapped by Earth's field: inner belt "
     "~1.2-2.5 and outer belt ~3-7 Earth radii from Earth's centre."),
    ("Mars", "The fourth planet, 1.52 AU from the Sun, about 1/9 of Earth's mass. Earth's path to 2.7 AU must cross its orbit."
     " Both planets pull on each other in the simulation."),
    ("Tide / tidal force", "The difference of a body's pull across another body; it stretches it (ocean tides). It falls as 1/distance^3, so it grows very fast during a close pass."),
    ("Planetary-safety distance", "The closest Earth may come to Mars: the largest of the tide limit, the Moon-orbit limit and the hard minimum distance."),
    ("Safety shield", "An automatic override layer in safe reinforcement learning: it forecasts the consequences of the agent's command and replaces it with a safe option when needed."),
    ("Gravitational focusing", "During a slow pass two planets pull each other closer, so the real closest distance is smaller than a straight-line forecast."),
    ("Resonance", "A simple ratio of orbital periods (e.g. 3:2) that makes repeated passes happen at the same place, so small pulls add up."),
    ("R_E (Earth radius)", "6,371 km. The distance unit of the Earth-POV view."),
    ("Osculating (coasting) orbit", "The orbit Earth would follow from its current position and velocity if all thrust stopped now."),
    ("Insolation", "Sunlight power per square metre; falls as 1/distance^2."),
    ("Reinforcement learning (RL)", "A trial-and-error method: an agent tries actions, receives rewards, and adjusts a policy to get more reward."),
    ("Policy", "The agent's decision rule: observation in, action out (here a small neural network)."),
    ("PPO", "Proximal Policy Optimization, a standard RL algorithm that changes the policy in small, safe steps."),
    ("Episode", "One complete simulated mission from the start to success, deadline or failure."),
    ("Reward / return", "Reward = score per decision; return = total score per episode. A designed scoring rule, not a physical measurement."),
    ("Discount factor", "How much future rewards count compared with immediate ones."),
    ("Deterministic evaluation", "Testing the policy without random exploration, to measure what it has really learned."),
    ("Baseline", "A simple reference strategy (no ejection, hand-designed) to judge whether RL actually helps."),
    ("Solar collector belt (Dyson swarm)", "Many separate solar collectors orbiting the Sun. Catching 30 % of all sunlight needs a belt covering 30 % of the sky seen from the Sun."),
    ("Microwave power beaming", "Sending electricity as a focused microwave beam (here 5.8 GHz) from a transmitter antenna to a receiving antenna."),
    ("Rectenna", "'Rectifying antenna': a receiving antenna that turns microwaves back into electricity. Here one sits on the tip of each arm."),
    ("Phased array", "A large flat antenna made of many small emitters whose timing steers and focuses the beam without moving parts."),
    ("Diffraction / Goubau relation", "Every beam spreads. The fraction caught by the receiver is 1 - exp(-tau^2), tau = D_t D_r / (wavelength x distance)."),
    ("Light pressure", "Sunlight pushes on what absorbs it (force = power / c). Very thin collectors would be blown outward."),
    ("Conservation error", "How much a quantity that physics says must stay constant (energy, momentum...) drifts because of numerical approximations."),
]

REASON_DOCS = {
    "success": "Orbit entered the target window (a, e and inclination within tolerance) AND (climate model on) the surface temperature was within the climate tolerance.",
    "deadline": "Mission time ran out before the target was reached.",
    "resources_exhausted": "All asteroid mass (or, with the solar collectors off, all propulsion energy) was used before reaching the target.",
    "escaped_or_unbound": "Earth was pushed onto an escape orbit (constraint violation, catastrophic).",
    "too_close_to_sun": "Earth came closer than 0.2 AU to the Sun (catastrophic).",
    "moon_lost": "The Moon was no longer bound to Earth (catastrophic).",
    "planet_unsafe_approach": "Earth came closer to Mercury, Venus or Jupiter than its planetary-safety distance (same rules as for Mars: tides on Earth and pull on the Moon's orbit), or hit it (failure). Jupiter's safe distance is ~0.35 AU.",
    "mars_unsafe_approach": "Earth came closer to Mars than the planetary-safety distance (or collided). Mars's tides or its pull on the Moon would exceed the limits set to keep people safe (failure).",
}

CHECK_DOCS = {
    "1": "Gravity-only motion follows Kepler's exact solution, and the N-body integrator agrees with it.",
    "2": "Each gravitational pull (Sun, Earth, Moon, asteroid, fragment) has the textbook magnitude G m / r^2.",
    "3": "Every force has an equal and opposite partner (Newton's third law), so internal forces cancel.",
    "4": "Throwing a packet does not create momentum or angular momentum: Earth's gain equals the packet's loss.",
    "5": "The energy put into each launch matches 1/2 x mass x speed^2 (minus the climb out of gravity), and the budget can never go negative.",
    "7": "Mass is never created or lost: Earth + asteroid + fragments + escaped mass stays constant; the asteroid cannot go below zero.",
    "8": "Reversing the throw direction reverses the effect (forward push raises the orbit, backward push lowers it).",
    "9": "Slow packets fall back, fast ones escape, and packets aimed at Earth merge, all recorded in the ledger.",
    "10": "Halving the time step reduces the error 16x, as expected for a 4th-order method (the numerics converge).",
    "11": "The same seed gives exactly the same result.",
    "12": "The fast training model and the high-fidelity N-body model agree on the orbit.",
    "13": "The 3-D view's interpolation between recorded physics states is accurate and hides no jumps.",
    "14": "The training model really simulates the Moon: its distance from Earth agrees with the N-body Moon, its pull makes Earth wobble around the Earth-Moon centre, and a launch pushes only Earth.",
    "15": "Mars is simulated in both models: the fast model's Mars and Earth agree with the N-body model, Mars's pull changes Earth's motion, and Earth pulls Mars back with an equal and opposite force (so Mars's orbit changes too).",
    "17": "The solar-collector power chain is exact (captured sunlight x efficiencies), the beam capture follows the Goubau relation, the engines never draw more than the beams delivered plus the reserve, and the collectors alone can power a mission.",
    "18": "Climate + energy ledger: the climate model reproduces today's 288 K, Earth freezes at 2.7 AU without the space mirrors and is held at the target with them; every joule of the power chain (sunlight -> panels -> microwaves -> arms -> launcher) and of the mirrors (sunlight -> Earth / space / mirror heat) is accounted for (ledger closes to round-off).",
    "19": "Collector belt: with station keeping each simulated collector stays within a few hundred km of its assigned orbit despite the beam recoil and the planets; without it the collectors drift; the belt's total pull on Earth is negligible.",
    "16": "The planetary-safety distances match the configured tide / Moon limits, the shield's forecast matches the simulation, and a shielded mission keeps Mars beyond the safe distance.",
}


ASSUMPTIONS = [
    ("Scale is hypothetical", "The default asteroid (3e22 kg, 40 % of the Moon) and energy budget (3e35 J, about 25 years "
     "of the Sun's total output) are far beyond any real technology. The 'realistic' preset shows the honest answer: infeasible."),
    ("Asteroid docked to Earth", "The asteroid is attached to Earth (same position and velocity); the launcher throws its "
     "material away piece by piece. Its gravity IS included: its mass adds to Earth's pull on the Sun, the Moon and every "
     "launched packet, and that pull weakens as the asteroid is used up. Because it is attached, its pull on Earth itself "
     "is an internal force (no net push). Its own orbit and structure are not modelled. The 3-D view draws it as a brown rock "
     "sitting on Earth's marker on the side the packets are thrown from (not to scale)."),
    ("Solar collectors", "A belt of collectors at 0.5 AU intercepts 30 % of the Sun's output and beams power to the 8 arm "
     "tips. Modelled: panel temperature (energy balance) and the resulting operating efficiency, age degradation with "
     "replacement, linear build-up of the belt, the clear lanes for Earth and Mars (or Earth's shading when switched off), "
     "the power chain, beam spreading by diffraction (Goubau) for every collector-to-Earth distance, beams blocked by the "
     "Sun, throttling to demand, manufacturing energy and mass flows, and a complete energy ledger in which every loss is a "
     "named sink (closes to round-off). The collectors' orbits are simulated in the viewer's belt simulation (station "
     "keeping vs. beam recoil and planets; it does not feed back into Earth's motion - the belt's net pull on Earth is "
     "~1e-9 of the Sun's). Within each 100-year decision interval the beam energy of that interval is available to the "
     "launches of that interval. NOT modelled: the push of the microwaves on Earth (estimated), Mercury/Venus near the belt."),
    ("Instant launches", "Each packet is launched instantly from Earth's surface; launcher hardware, heat and atmosphere are not modelled."),
    ("Non-relativistic", "Newtonian mechanics; launch speeds up to 1.7 % of light speed give errors of order 0.01 %."),
    ("Fast model (training)", "Hierarchical two-body model: the Earth-Moon centre of mass follows an exact Kepler orbit "
     "around the Sun, and the Moon follows an exact Kepler orbit around Earth(+asteroid) under their mutual gravity. Each "
     "packet pushes Earth only; the Moon follows through gravity. The packet's escape from Earth's, the asteroid's and the "
     "Moon's gravity is included in its final speed. APPROXIMATION: the Sun's tidal pull on the Moon is neglected, so the "
     "Moon's orbital phase drifts by ~20 degrees per year relative to the N-body Moon (its distance agrees within ~3 %, "
     "check 14); packets hitting the Moon and fragment trajectories after escape are only in the high-fidelity model."),
    ("High-fidelity model", "Sun, Earth, Moon, Mars and up to max_fragments representative fragments with full mutual gravity, "
     "except fragment-fragment gravity, which is neglected (fragments are tiny)."),
    ("Mars", "Mars is a full body in both models: it pulls Earth, the Moon (fast model: its tide on the Earth-Moon orbit) and "
     "the fragments, and is pulled by them, so its orbit changes. The fast model integrates the Earth-Mars pair with exact "
     "Kepler drifts plus mutual-gravity kicks (adaptive step near close passes); after 20 years it agrees with the N-body "
     "model to ~1,600 km (check 15). Mars starts from its real J2000 orbit with a random position along it. Mars's final "
     "orbit is CHAOTIC: small differences (e.g. fast vs high-fidelity model) can change the shield's choices and therefore "
     "how much Mars is moved (tested: 0.015 AU vs 0.35 AU for the same seed). Fragments that hit Mars are ledgered (high-fidelity)."),
    ("Other planets not modelled", "Venus, Jupiter, Saturn etc. are not included. Jupiter's pull on Earth at 2.7 AU is "
     "~1e-6 m/s^2 and would perturb the final orbit over thousands of years; the target orbit's long-term stability with "
     "Jupiter is NOT assessed."),
    ("Mars safety limits", "'Safe for people' is represented by physical proxies only: Mars's tide on Earth relative to the "
     "Moon's, Mars's disturbance of the Moon's orbit, and a hard distance. Effects on climate, earthquakes or debris are not "
     "modelled. The limits are user parameters, not established safety standards."),
    ("Packet paths in the 3-D view", "Computed with exact two-body (Sun-only) gravity from each packet's real release state, "
     "drawn for at most 1 year or until 60 AU. Earth's and the Moon's pull on the packet after release is approximated by "
     "the analytic escape (patched conic)."),
    ("Interpolation in the 3-D view", "Between recorded physics states (8 per year) Earth's position is drawn on a cubic "
     "curve through the recorded positions and velocities (error < 1e-4 AU, check 13). At each ejection the curve uses the "
     "velocity before and after the event separately, so the recoil is never smoothed away."),
    ("Moon distance exaggerated in the view", "The Moon is drawn 100x farther from Earth than it really is, otherwise it "
     "would sit inside Earth's marker. Its motion is the simulated one in both models. Its simulated orbit around Earth is "
     "drawn as a thin grey ellipse (same 100x scale)."),
    ("Climate model", "Two-box energy balance (surface layer 8 W yr m^-2 K^-1 + deep ocean 100 W yr m^-2 K^-1, exchange "
     "0.7 W m^-2 K^-1), ice-albedo feedback (albedo 0.30 above 280 K rising to 0.60 below 255 K, so a snowball state exists), "
     "FIXED greenhouse effect (effective emissivity 0.61, calibrated to 288 K today). Global mean only: no seasons, day/night, "
     "regions, water-vapour or carbon-cycle feedbacks, no ocean chemistry or biology. It is integrated along Earth's "
     "simulated distance from the Sun with the mirror light and the waste heat reaching Earth, and it is PART OF THE RL "
     "TASK (action a4, reward term, success condition)."),
    ("Space mirrors", "An ENABLER (E3) based on ordinary physics (reflection): mirror area x cos 45 x reflectivity x local "
     "sunlight is the most light they can send to Earth, so like sunlight it falls as 1/r^2. Modelled: that capacity, the "
     "light actually sent (agent's set-point), the light turned past Earth and the heat absorbed by the mirrors (energy "
     "ledger), their mass. Not modelled: the mirrors' orbits and pointing control, the light-pressure force on them, how the "
     "light is spread over Earth (the climate model is a global mean) and eclipses by the Moon."),
    ("Weather view: what comes from the model", "From the climate model: global-mean, equator and pole temperatures, the ice "
     "line and the share of the surface with growing temperatures (simple zonal profile, dT = 45 K equator-to-pole), the "
     "sunlight and mirror light. ILLUSTRATION: clouds, wind streaks, storms, vegetation colours, city lights; their pattern "
     "is today's general circulation, not a weather simulation."),
    ("Habitability", "The equilibrium temperature column (t_eq) is a quick no-climate indicator; the climate model above "
     "gives the surface temperature used by the RL."),
    ("Earth-POV view: what is simulated", "The close-up uses the same recorded physics states as the overview: Earth's "
     "position and velocity (hence the Sun direction, distance and orbital speed), the Moon's position, the remaining "
     "asteroid mass and every ejection (direction, speed, fate). Distances and sizes are true scale (unit = Earth radius). "
     "The asteroid's drawn size assumes a density of 2,000 kg/m^3."),
    ("Earth-POV view: ring and eight arms", "The launcher ring and the eight 'octopus' arms are an ILLUSTRATION of a mass "
     "launcher. No structure, arm dynamics or forces of the hardware are simulated (the physics assumes instant launches). "
     "Each recorded ejection is thrown by all eight arms together (one packet per event); the animation adds about 0.7 s "
     "of wall-clock delay before the streak appears."),
    ("Earth-POV view: environment layers", "Magnetic field lines (centred tilted dipole, cut at the magnetopause; tail "
     "lines schematic), magnetopause (Shue 1998, IMF Bz = 0), bow shock (1.29 x stand-off), radiation belts (schematic "
     "tori), solar wind (steady 400 km/s, 5 protons/cm^3 at 1 AU ~ 1/r^2, animated slowed down) and the gravity well "
     "(Hill-approximation effective potential, log height) are standard simplified models evaluated at Earth's simulated "
     "Sun distance. They are for understanding only and do not feed back into the simulation. Space weather, storms, "
     "atmospheric effects and changes of Earth's own field are not modelled."),
]


# Every technology this model ASSUMES but does not show to be possible. Shown in the viewer (Guide -> Enablers), printed
# at start-up and written to the report. 'gap' = how far beyond today's capability.
# The model allows exactly THREE enablers (hypothetical systems). Each is one system made of several instruments; the
# instruments' required performance is part of the enabler (calc_report.py computes every number from the code).
ENABLERS = [
    dict(id="E1", name="Earth launcher system: docked asteroid + launch ring with 8 arms + arm radiators",
         provides="Throws packets of up to 1e18 kg of asteroid material at up to 5,000 km/s, 8 times per year (all 8 arms "
                  "throw one packet together), with 50 % of the drawn energy ending up as packet kinetic energy; the arms "
                  "reject their waste heat away from Earth; the launches do not damage Earth.",
         params="asteroid_mass_kg, max_packet_mass_kg, max_ejection_speed_ms, min_ejection_speed_ms, ejections_per_year, "
                "launch_efficiency, energy_budget_j, arm_heat_to_earth_frac",
         gap="No known structure or material; each launch carries ~1e30 J (millions of Chicxulub impacts); moving a 3e22 kg "
             "body to Earth and docking it is far beyond any technology; ~1e19 m^2 of 1,000 K radiator per mission.",
         modelled="Momentum, energy, escape from Earth (+Moon) gravity, packet fates, finite reservoir, waste heat into the "
                  "climate model. NOT modelled: the hardware, launch shock, atmosphere loss and heating of Earth (problem P6).",
         instruments=[
             dict(name="Asteroid reservoir (docked to Earth)", use="Supplies the reaction mass that is thrown away. Docked on the "
                  "launch ring's axis, 1.25 R_E from Earth's centre, so all 8 arm roots are the same 1.77 R_E from it (update p36).",
                  params="asteroid_mass_kg"),
             dict(name="Launch ring + 8 launch arms (mass driver)", use="Accelerates each packet to the commanded speed in the "
                  "commanded direction; all 8 arms (each 2 R_E long, roots on a 1.25 R_E ring) throw one packet together.",
                  params="max_packet_mass_kg, max/min_ejection_speed_ms, ejections_per_year, launch_efficiency"),
             dict(name="Arm power conditioning + energy reserve", use="Turns the arm electricity into launch energy; a "
                  "reserve covers launches when the beams are short.", params="energy_budget_j"),
             dict(name="Directional arm radiators", use="Reject the launcher and rectenna waste heat to space, away from "
                  "Earth.", params="arm_heat_to_earth_frac"),
         ]),
    dict(id="E2", name="Solar power belt with microwave power beaming (collectors + transmitters + arm rectennas)",
         provides="5 million thin-film collectors at 0.5 AU intercept 30 % of the Sun's output, turn it into electricity and "
                  "beam it as 5.8 GHz microwaves (30 km phased arrays) to 1,000 km rectennas on the 8 arms; they hold their "
                  "slots by sail tilt, keep a clear sunlight lane for Earth and Mars, keep beam spill off Earth and are "
                  "built up / replaced by factories using Mercury material.",
         params="collector_fraction, collector_orbit_au, collector_count, collector_areal_density, pv_efficiency, "
                "pv_temp_coeff, collector_absorptance, collector_emissivity, pv_degradation_per_year, collector_lifetime_years, "
                "collector_recycle_frac, collector_initial_built, collector_build_years, collector_embodied_energy_jkg, "
                "microwave_efficiency, microwave_freq_ghz, transmitter_diameter_km, rectenna_diameter_km, rectenna_efficiency, "
                "earth_lane_clear, collector_sk_authority, beam_spill_to_earth_frac",
         gap="A Dyson-swarm-scale project (~1e13 x all solar panels ever built); record power beaming is ~kW over km, here "
             "~1e24 W over 0.5-3 AU with pointing to ~1e-12 rad; beam intensity at the rectennas ~2e8 x sunlight.",
         modelled="Panel temperature and efficiency, degradation and replacement, build-up, manufacturing energy and mass, "
                  "Goubau diffraction capture, Sun blocking, every loss in the energy ledger, clear lanes, the orbits of 12+12 "
                  "followed collectors with / without station keeping (beam recoil, planets, light pressure).",
         instruments=[
             dict(name="Thin-film solar collector (x 5 million)", use="Intercepts sunlight and makes electricity.",
                  params="collector_fraction, collector_orbit_au, collector_count, collector_areal_density, pv_efficiency, "
                         "pv_temp_coeff, collector_absorptance, collector_emissivity, pv_degradation_per_year, "
                         "collector_lifetime_years"),
             dict(name="Microwave phased-array transmitter (one per collector)", use="Turns the electricity into a focused "
                  "5.8 GHz beam aimed at the arms; suppresses side lobes so spill misses Earth.",
                  params="microwave_efficiency, microwave_freq_ghz, transmitter_diameter_km, beam_spill_to_earth_frac"),
             dict(name="Rectenna on each of the 8 arm tips", use="Receives the microwaves and turns them back into "
                  "electricity for the launcher.", params="rectenna_diameter_km, rectenna_efficiency"),
             dict(name="Sail-tilt station keeping (on each collector)", use="Uses part of the light-pressure push to hold "
                  "the collector in its slot (no propellant).", params="collector_sk_authority"),
             dict(name="Clear sunlight lane", use="Collectors leave a moving gap so Earth and Mars see the whole Sun.",
                  params="earth_lane_clear"),
             dict(name="Collector factories (Mercury material)", use="Build the belt up and replace aged panels.",
                  params="collector_initial_built, collector_build_years, collector_recycle_frac, "
                         "collector_embodied_energy_jkg"),
         ]),
    dict(id="E3", name="Space mirror ring around Earth (soletta)",
         provides="1.6e15 m^2 of aluminium-film mirrors (8e12 kg) in a terminator orbit 20 Earth radii out reflect extra "
                  "sunlight onto Earth: up to 1.10 x today's sunlight at 2.7 AU. The agent sets how much (mirrors not needed "
                  "are turned so their light passes Earth).",
         params="mirrors, mirror_area_m2, mirror_reflectivity, mirror_orbit_re, mirror_areal_density",
         gap="Physically possible (only reflection, no new physics, no energy needed). Demonstrated small: Znamya-2 (1993) 20 m "
             "mirror lit a 5 km spot on Earth; solar sails IKAROS (2010, ~200 m^2). Needed: ~1e13 x that area; film made from "
             "asteroid aluminium; formation flying and pointing of a million 40 km mirrors.",
         modelled="Capacity (1/r^2), light sent, light turned away, mirror heat (energy ledger), mass; the climate effect. Not "
                  "the mirrors' orbits, light pressure, pointing errors or the light pattern on Earth.",
         instruments=[
             dict(name="Mirror film panels (aluminium, 5 g/m^2)", use="Reflect sunlight; 90 % reflectivity.",
                  params="mirror_area_m2, mirror_reflectivity, mirror_areal_density"),
             dict(name="Terminator orbit at 20 Earth radii + pointing (45 degree tilt)", use="Keeps the mirrors in sunlight "
                  "and aims their light at Earth or past it (the agent's action a4).", params="mirror_orbit_re, mirrors"),
         ]),
]
assert len(ENABLERS) == 3, "the model allows exactly three enablers"

# Open and addressed problems of the model (shown in the viewer: Guide -> Major problems).
MAJOR_PROBLEMS = [
    dict(id="P1", title="Climate test integration", status="addressed",
         detail="Habitability used to be only reported. Now a two-box climate model (with ice-albedo snowball risk) runs inside the "
                "RL environment: the agent aims the space mirrors (action a4), the reward has a climate term, the observation "
                "has 4 climate numbers and success requires |T_surface - target| <= tolerance.",
         remaining="Global-mean model only (no seasons, regions, water vapour, carbon cycle, biology)."),
    dict(id="P2", title="Energy generation mapped exactly from the panels", status="addressed",
         detail="Panel temperature (energy balance, ~450 K at 0.5 AU) -> operating efficiency; age degradation with replacement; "
                "linear build-up; manufacturing energy and fresh mass (vs. Mercury's mass); Earth's shading / clear lane; waste "
                "heat at collectors, transmitters, rectennas and launcher; every joule in a ledger that closes "
                "to round-off (check 18).", remaining="Panel temperature is a single global value; no micrometeoroids or eclipses."),
    dict(id="P3", title="Energy transfer and conservation", status="addressed",
         detail="Ledger: sunlight on collectors + reserve = reflected + collector heat + manufacturing + transmitter heat + beam spill "
                "+ rectenna heat + packet kinetic energy + launcher heat; mirrors: sunlight on mirrors = light onto Earth + light "
                "turned past Earth + mirror heat. Residual shown in the "
                "viewer and checked.", remaining="The packets' kinetic energy is then followed by the mechanical checks (4-6)."),
    dict(id="P4", title="Enablers declared explicitly", status="addressed",
         detail="Exactly 3 enablers E1-E3 (launcher system, solar power belt with beaming, space mirrors), each with its instruments, what it provides, the parameters, the gap to reality and what is modelled.",
         remaining="-"),
    dict(id="P5", title="Arm waste heat (~1e24 W, 6e6 x Earth's sunlight)", status="open",
         detail="Only harmless because of the directional arm radiators of enabler E1 (1e-9 leaks to Earth). With an undirected radiator Earth would boil.",
         remaining="No physical radiator design."),
    dict(id="P6", title="Launch damage to Earth", status="open",
         detail="Each 1e18 kg packet leaves at ~2,500 km/s carrying ~3e30 J; shock, atmospheric loss and heating are not modelled.",
         remaining="Assumed as part of enabler E1 (launches do not damage Earth)."),
    dict(id="P7", title="Other planets", status="open",
         detail="Venus, Mercury (near the 0.5 AU belt) and Jupiter are not simulated; the final orbit's long-term stability with "
                "Jupiter is not assessed.", remaining="Add them to the N-body model."),
    dict(id="P8", title="Day length, seasons, tides", status="open",
         detail="Spin, obliquity and tides are not modelled; the mirrors' illumination pattern on Earth is not modelled.", remaining="-"),
    dict(id="P9", title="Climate enabler must be physically possible", status="addressed",
         detail="The hypothetical beam-powered artificial sun was replaced by space mirrors (E3): plain reflection of sunlight, no "
                "energy source needed, flown at small scale (Znamya-2). The weather view shows what the mirrors keep alive (ice "
                "line, growing area, clouds and winds) and what happens without them (snowball Earth).",
         remaining="Building 1.6e15 m^2 of mirror and keeping it pointed is far beyond today's industry."),
]


# =============================================================================================
# 12. MAIN
# =============================================================================================
def parse_args(argv=None):
    """Command-line options (see the module docstring)."""
    p = argparse.ArgumentParser(description=APP_NAME)
    p.add_argument("--preset", default="hypothetical", choices=list(PRESETS))
    p.add_argument("--smoke", action="store_true", help="short pipeline test (30 episodes, 2,000-year missions)")
    p.add_argument("--check", action="store_true", help="run all physics checks and exit")
    p.add_argument("--headless", action="store_true", help="no server/browser: checks, train, compare, report")
    p.add_argument("--no-train", action="store_true", help="start the viewer without training")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="where the PPO network is trained: auto = measure CPU and GPU, use the faster (normally the CPU)")
    p.add_argument("--no-live", action="store_true", help="start with the live training view off (training at full speed)")
    p.add_argument("--out", default=str(OUTPUTS))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Entry point: checks, then training / comparison / report, then the viewer (unless --headless)."""
    args = parse_args(argv)
    if args.check:
        res = run_physics_checks(full=True)
        return 0 if all(r["passed"] for r in res) else 1
    over = dict(kv.split("=", 1) for kv in args.set)
    if args.episodes:
        over["train_episodes"] = args.episodes
    cfg = smoke_config(args.preset) if args.smoke else make_config(args.preset)
    try:
        cfg = checked_config(cfg.preset, **{**cfg.to_dict(), **over})
    except ValueError as exc:
        print(f"[ERROR] invalid parameter: {exc}")
        return 2
    app = App(cfg, Path(args.out)); app.device_request = args.device
    if args.no_live or args.headless:
        app.train_live.clear()
    print("=" * 100)
    print(f" {APP_NAME} | preset={cfg.preset} | mission {cfg.mission_years:,.0f} y | target {cfg.target_au} AU | "
          f"{cfg.train_episodes} training episodes")
    print("=" * 100)
    app.log("[INIT] compiling physics kernels (first run can take ~30-60 s; cached afterwards) ...")
    t0 = time.time()
    app.checks = run_physics_checks(full=False, log=app.log)
    app.log(f"[INIT] physics ready ({time.time() - t0:.1f} s)")
    checks_ok = all(c["passed"] for c in app.checks)
    (app.out / "physics_checks.json").write_text(json.dumps(app.checks, indent=1))
    (app.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    (app.out / "feasibility.json").write_text(json.dumps(_clean(app.feas), indent=1))
    app.log("[ENABLERS] this model ASSUMES the following technologies (none is shown to be possible):")
    for e in ENABLERS:
        app.log(f"    {e['id']:4s} {e['name']}: {e['provides']}")
        for ins in e.get("instruments", []):
            app.log(f"           - {ins['name']}: {ins['use']}")
    app.log("[PROBLEMS] " + "; ".join(f"{m['id']} {m['title']} [{m['status']}]" for m in MAJOR_PROBLEMS))
    app.log(f"[FEASIBILITY] {app.feas['verdict']}")
    for line in app.feas["lines"]:
        app.log("    " + line)
    g = gpu_report()
    app.gpu_info = g
    app.log(f"[GPU] PyTorch {g['torch']} ({'CUDA ' + g['cuda_build'] if g['cuda_build'] else 'CPU-only build'}); "
            + (f"GPU found: {g['gpu']} (device '{args.device}': " + ("Auto measures CPU vs GPU when training starts and uses the faster one)" if args.device == "auto" else "forced)")
               if g["cuda_ok"] and args.device != "cpu" else "training network on the CPU"))
    if g["fix"]:
        app.log("[GPU] " + g["fix"])
    if g["cuda_ok"]:
        app.log("[GPU] note: the network is small (64x64) and the physics runs on the CPU, so Task Manager shows only "
                "light GPU load (look at the 'Cuda' / 'Compute' graph, not '3D').")
    if not checks_ok:
        app.log("[ERROR] physics checks failed - training is disabled until the physics is fixed.")
    if args.headless:
        if not checks_ok:
            return 1
        app.start_training()
        try:
            while app.train_thread.is_alive() or (app.compare_thread and app.compare_thread.is_alive()) \
                    or app.compare_state == "running":
                time.sleep(0.5)
        except KeyboardInterrupt:
            app.stop_train.set()
            app.train_thread.join(60)
        return 0 if app.train["state"] in ("finished", "stopped") else 1
    port = find_port(args.host, args.port)
    class _Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):     # no tracebacks for closed browser tabs
            """Ignore harmless connection-reset errors from the browser."""
            exc = sys.exc_info()[1]
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                super().handle_error(request, client_address)

    server = _Server((args.host, port), make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{port}/"
    app.log(f"[SERVER] viewer at {url}   (close this window or press Ctrl+C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    import signal

    def _on_signal(signum, frame):            # Ctrl+C, closing the console (SIGBREAK), SIGTERM
        """Shut down cleanly on Ctrl+C / SIGTERM."""
        app.shutdown.set()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), _on_signal)
            except (ValueError, OSError):
                pass
    if checks_ok and not args.no_train:
        app.start_training()
    try:
        while not app.shutdown.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    app.log("[SHUTDOWN] stopping ... (saving the model if training)")
    app.stop_train.set()
    app.stop_play.set()
    if app.train_thread and app.train_thread.is_alive():
        app.train_thread.join(120)
    for proc in app._procs.values():           # e.g. a running comparison
        if proc.is_alive():
            proc.terminate()
    server.shutdown()
    app.log("[SHUTDOWN] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
