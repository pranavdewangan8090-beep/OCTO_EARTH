"""pytest suite for the OCTO-EARTH prototype:  python -m pytest -q tests"""
import json
import re
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import octoearth_rl as R  # noqa: E402


@pytest.fixture(scope="module")
def checks():
    return R.run_physics_checks(full=True, log=lambda *a: None)


def test_all_physics_checks_pass(checks):
    failed = [(c["name"], c["detail"]) for c in checks if not c["passed"]]
    assert not failed, failed
    assert len(checks) >= 12


def test_feasibility_gate_is_honest():
    assert R.feasibility(R.make_config("hypothetical"))["feasible"]
    real = R.feasibility(R.make_config("realistic"))
    assert not real["feasible"] and "INFEASIBLE" in real["verdict"]


def test_gym_env_api():
    from gymnasium.utils.env_checker import check_env
    env = R.OctoEarthEnv(R.make_config(), "fast")
    check_env(env, skip_render_check=True)


def test_no_free_energy_and_mass_limits():
    cfg = R.make_config(energy_budget_j=1e33, asteroid_mass_kg=1e20)
    env = R.OctoEarthEnv(cfg, "fast")
    env.reset(seed=3)
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.ones(4, dtype=np.float32))
        if term or trunc:
            break
    eng = env.engine
    assert eng.E_rem >= -1e-6 * cfg.energy_budget_j
    assert eng.m_ast >= 0.0
    # Earth only moved by the energy it was allowed to spend: a stays close to 1 AU
    assert info["decision"]["a_au"] < 1.01


def test_no_ejection_baseline_stays_put():
    s = R.run_episode(R.make_config(), R.policy_none, 1001, "fast")
    assert abs(s["a_au"] - 1.0) < 1e-3 and not s["success"]


def test_heuristic_reaches_target_fast_model():
    s = R.run_episode(R.make_config(), R.policy_heuristic, 1001, "fast")
    assert s["success"], s["reason"]


def test_json_is_browser_safe():
    out = R._dumps(R._clean({"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float64(2.0)}))
    d = json.loads(out)
    assert d["a"] is None and d["b"][0] is None and d["c"] == 2.0


def test_config_overrides_and_presets():
    c = R.make_config("hypothetical", asteroid_mass_kg="1e21", train_episodes="50")
    assert c.asteroid_mass_kg == 1e21 and c.train_episodes == 50 and c.preset == "custom"
    assert R.make_config("realistic").preset == "realistic"


# ----------------------------------------------------------------------------- viewer data
def _one_decision(fid):
    env = R.OctoEarthEnv(R.make_config(randomize_phase=False), fid)
    env.reset(seed=5)
    env.step(R.policy_heuristic(env, None))
    return env


@pytest.mark.parametrize("fid", ["fast", "hifi"])
def test_recorded_states_are_continuous(fid):
    """One row per ejection event at the ejection instant; evenly spaced; recoil column equals
    |v_post - v_pre|; Earth moves no farther between rows than its orbital speed allows."""
    env = _one_decision(fid)
    s = env.last_samples
    cfg = env.cfg
    assert s.shape[1] == R.N_SAMPLE_COLS and len(s) == int(round(cfg.ejections_per_year * cfg.decision_interval_years))
    dt = np.diff(s[:, R.S_T])
    assert np.allclose(dt, 1.0 / cfg.ejections_per_year, rtol=1e-9)
    dv = np.linalg.norm(s[:, R.S_VPOST:R.S_VPOST + 3] - s[:, R.S_VPRE:R.S_VPRE + 3], axis=1) * 1000.0
    assert np.allclose(dv, s[:, R.S_DV], rtol=1e-6)
    step = np.linalg.norm(np.diff(s[:, R.S_R:R.S_R + 3], axis=0), axis=1) * R.AU
    vmax = np.linalg.norm(s[:, R.S_VPRE:R.S_VPRE + 3], axis=1).max() * 1000.0
    assert step.max() <= vmax * dt.max() * R.YEAR * 1.001          # no jumps
    assert np.all(np.diff(s[:, R.S_MAST]) <= 0)                       # asteroid only loses mass
    if fid == "hifi":
        assert len(env.engine.last_moon) == 8 * len(s)
        assert np.all(np.diff(env.engine.last_moon[:, 0]) > 0)


def test_packet_arcs_match_kepler():
    r0 = np.array([[R.AU, 0.0, 0.0]])
    v0 = np.array([[0.0, 40e3, 1e3]])                                  # bound, inclined
    offs = R.PACKET_OFFSETS_DAYS * 86400.0
    out = np.zeros((1, len(offs), 3))
    nv = np.zeros(1, dtype=np.int64)
    R.packet_arcs(r0, v0, offs, 60 * R.AU, out, nv)
    assert nv[0] == len(offs)
    for j in (5, 12, len(offs) - 1):
        r, v = r0[0].copy(), v0[0].copy()
        R.kepler_step(r, v, R.GM_SUN, offs[j])
        assert np.linalg.norm(out[0, j] - r) < 1e-6 * R.AU


def test_viewer_payload_is_json_safe():
    for fid in ("fast", "hifi"):
        env = _one_decision(fid)
        rows = R._viewer_rows(env.last_samples)
        pk = R._packet_records(env.last_samples, env.cfg, fid)
        moon = R._moon_rows(env.engine.last_moon)
        assert len(rows[0]) == R.N_VIEW_COLS and len(pk) == len(rows)
        assert all(p[4] * 3 + 5 == len(p) for p in pk)
        txt = json.dumps(dict(rows=rows, pk=pk, moon=moon), allow_nan=False)
        assert len(txt) > 1000


def test_parameter_validation_rejects_bad_values():
    assert R.validate_params(R.make_config("hypothetical").to_dict()) == []
    assert R.validate_params(R.make_config("realistic").to_dict()) == []
    bad = [dict(min_ejection_speed_ms=6e6), dict(max_ejection_speed_ms=1e8), dict(target_au=0.5),
           dict(ppo_batch_size=5000), dict(decision_interval_years=5.0), dict(max_packet_mass_kg=1e23, asteroid_mass_kg=1e22),
           dict(nonexistent_parameter=1)]
    for b in bad:
        with pytest.raises(ValueError):
            R.checked_config("hypothetical", **b)


def test_every_parameter_and_output_is_documented():
    fields = set(R.make_config().to_dict()) - {"preset"}
    assert fields == set(R.PARAM_BY_KEY)
    for d in R.PARAM_DOCS:
        for k in ("label", "unit", "category", "meaning", "why", "increase", "decrease"):
            assert d[k], (d["key"], k)
    for k in ("a", "e", "inc", "E_orb", "dv", "thrust", "impulse", "m_ast", "E_used", "reward", "return",
              "progress", "cons_E", "cons_P", "cons_L", "cons_M"):
        assert k in R.OUTPUT_BY_KEY
    assert set(R.REASON_DOCS) >= {"success", "deadline", "resources_exhausted"}


def test_fast_model_simulates_moon():
    """The training (fast) model simulates the Moon: recorded stream, real distance, bound,
    Earth's recoil changes the Earth-Moon relative velocity, and the training env gets the same
    physics whether or not Moon samples are recorded for the viewer."""
    cfg = R.make_config("hypothetical", randomize_phase=False)
    f = R.FastEngine(cfg, 3)
    smp, status = f.run_interval(2.0, 1e18, 2.5e6, [0, 1, 0])
    assert status == 0 and len(f.last_moon) == R.MOON_SUB * len(smp)
    assert np.all(np.diff(f.last_moon[:, 0]) > 0)
    d = np.linalg.norm(f.last_moon[:, 1:4], axis=1)
    assert 3.4e8 < d.min() and d.max() < 4.2e8           # real Earth-Moon distances, varying
    assert d.max() - d.min() > 1e7
    info = f.moon_info()
    assert info["modelled"] and info["bound"]
    g = R.FastEngine(cfg, 3, record_moon=False)
    g.run_interval(2.0, 1e18, 2.5e6, [0, 1, 0])
    assert np.array_equal(f.st, g.st)                     # recording does not alter the physics
    chk = {c["name"]: c for c in R.run_physics_checks(False, log=lambda m: None)}
    assert chk["14 Moon in the fast (training) model"]["passed"]


def test_space_environment_python_matches_browser_model():
    """Earth-POV environment layers: the Python reference and frontend/envmodel.js must agree, and the
    numbers must match the published reference values they are based on."""
    m = R.M_EARTH + R.M_MOON + 3e22
    e1, e27 = R.space_environment(1.0, m), R.space_environment(2.7, m)
    assert 10.5 < e1["r0_re"] < 11.3                       # typical quiet-time magnetopause ~10-11 R_E
    assert e27["r0_re"] > e1["r0_re"] and abs(e27["n_cm3"] - 5 / 2.7 ** 2) < 1e-12
    assert 4.5 < e1["imf_nt"] < 5.5 and abs(e1["insolation_wm2"] - 1361) < 1e-9
    assert abs(e1["hill_re"] * R.R_EARTH / 1e9 - 1.5) < 0.05   # Hill radius ~1.5 million km at 1 AU
    assert abs(e27["hill_re"] / e1["hill_re"] - 2.7) < 1e-9
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed: browser model not compared")
    js = Path(__file__).resolve().parents[1] / "frontend" / "envmodel.js"
    code = (f"import('{js.as_uri()}').then(M => console.log(JSON.stringify("
            f"[1.0, 1.7, 2.7].map(r => M.spaceEnv(r, {m!r})))))")
    out = subprocess.run([node, "-e", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    for r, j in zip([1.0, 1.7, 2.7], json.loads(out.stdout)):
        p = R.space_environment(r, m)
        assert set(j) == set(p)
        for k in p:
            assert math.isclose(j[k], p[k], rel_tol=1e-9), (r, k, j[k], p[k])


def test_pov_view_has_inputs_docs_and_layers(tmp_path):
    app = R.App(R.make_config(), tmp_path)
    init = app._init_view()
    v = np.array(init["earth_v_kms"])
    assert 29.0 < np.linalg.norm(v) < 31.0                   # Earth's orbital speed at year 0
    r = np.array(init["earth"])
    assert abs(float(np.dot(r, v)) / (np.linalg.norm(r) * np.linalg.norm(v))) < 0.05   # nearly circular orbit
    for k in ("env_r", "env_sw", "env_dp", "env_imf", "env_mp", "env_bs", "env_bfield", "env_hill", "env_gsun",
              "env_gmoon", "env_insol"):
        d = R.OUTPUT_BY_KEY[k]
        assert d["how"] and d["measures"] and d["kind"] in (R.K_EST, R.K_CALC, R.K_SIM)
    gl = " ".join(g[0] for g in R.GLOSSARY)
    for term in ("Magnetosphere", "Bow shock", "Solar wind", "Lagrange", "Radiation", "dipole", "Gravity well"):
        assert term.lower() in gl.lower(), term
    assum = " ".join(a[0] + a[1] for a in R.ASSUMPTIONS)
    assert "ILLUSTRATION" in assum and "do not feed back" in assum
    fe = Path(__file__).resolve().parents[1] / "frontend"
    pov = (fe / "pov.js").read_text(encoding="utf-8")
    for key in ("arms", "moon", "moonorbit", "field", "magnetopause", "belts", "wind", "gravwell", "hill", "gravvec",
                "motion", "packets", "labels"):
        assert f"key: '{key}'" in pov, key
    html = (fe / "index.html").read_text(encoding="utf-8")
    assert 'id="viewMode"' in html and 'id="povLayers"' in html and 'id="povNone"' in html
    app_js = (fe / "app.js").read_text(encoding="utf-8")
    for k in ("env_sw", "env_mp", "env_hill", "env_gsun"):
        assert f"hrow('{k}'" in app_js, k


def test_frontend_collector_view_and_group_throw_wired():
    """Static check: collector close-up module exists and is wired in; all 8 arms throw ONE packet together."""
    fe = Path(__file__).resolve().parents[1] / "frontend"
    col = (fe / "collector.js").read_text(encoding="utf-8")
    assert "export function createCollectorView" in col and "export function satelliteGeometry" in col
    assert "not to scale" in col.lower() or "NOT to scale" in col
    app_js = (fe / "app.js").read_text(encoding="utf-8")
    assert "createCollectorView" in app_js and "colOn()" in app_js and "toggleCol" in app_js
    html = (fe / "index.html").read_text(encoding="utf-8")
    assert 'value="collector"' in html and 'id="chipCol"' in html and 'id="colBox"' in html
    pov = (fe / "pov.js").read_text(encoding="utf-8")
    assert "const group = {" in pov and "group.pending" in pov and "A.pending" not in pov   # one shared throw, no per-arm queues
    launcher = (fe / "launcher.js").read_text(encoding="utf-8")
    assert "never inside Earth" in launcher and "armCurve" in pov                              # fixed-length arms arc around the planet


# ------------------------------------------------------------------ Mars
def test_mars_safety_limits_follow_parameters():
    lim = R.mars_safety_limits(R.make_config())
    assert abs(R.mars_tide_ratio(lim["d_tide_m"]) - 0.01) < 1e-9          # tide limit reproduces the configured ratio
    assert abs(R.mars_moon_perturbation(lim["d_moon_m"]) - 1e-3) < 1e-9
    assert lim["d_safe_m"] == max(lim["d_tide_m"], lim["d_moon_m"], lim["d_min_m"])
    strict = R.mars_safety_limits(R.make_config(mars_max_tide_ratio=0.001))
    assert strict["d_safe_m"] > lim["d_safe_m"]                            # stricter limit -> larger safe distance
    assert lim["caution_m"] == pytest.approx(2.0 * lim["d_safe_m"])


def test_mars_gravity_changes_both_orbits():
    from dataclasses import replace
    cfg = R.make_config()
    out = {}
    for grav in (True, False):
        e = R.FastEngine(replace(cfg, mars_gravity=grav, mars_shield=False), 3)
        r, v = e.st[0:3].copy(), e.st[3:6].copy()
        e.st[17:20] = r + 0.03 * R.AU * v / np.linalg.norm(v); e.st[20:23] = v.copy()
        e.run_interval(1.0, 0.0, 3e6, [0, 1, 0], shield=False)
        out[grav] = (e.st[3:6].copy(), e.mars_state()[1])
    assert np.linalg.norm(out[True][0] - out[False][0]) > 1.0               # Mars pulls Earth (m/s)
    assert np.linalg.norm(out[True][1] - out[False][1]) > 5.0               # Earth pulls Mars harder (mass ratio)


def test_shield_keeps_heuristic_mars_safe_and_unshielded_fails():
    from dataclasses import replace
    cfg = R.make_config()
    safe = R.mars_safety_limits(cfg)["d_safe_au"]
    shielded = [R.run_episode(cfg, R.policy_heuristic, s) for s in range(6)]
    assert all(x["mars_min_dist_au"] >= safe and x["reason"] != "mars_unsafe_approach" for x in shielded)
    raw = [R.run_episode(replace(cfg, mars_shield=False), R.policy_heuristic, s) for s in range(6)]
    assert sum(x["reason"] == "mars_unsafe_approach" for x in raw) >= 3     # the hazard is real without the shield


def test_viewer_rows_carry_mars_state():
    env = _one_decision("fast")
    rows = R._viewer_rows(env.last_samples)
    assert len(rows[0]) == R.N_VIEW_COLS
    for q in rows:                                   # recorded Earth-Mars distance = |r_mars - r_earth|
        assert abs(math.dist(q[1:4], q[22:25]) - q[28]) < 1e-5
        assert 1.3 < math.hypot(*q[22:25]) < 1.7       # Mars stays near its orbit (1.38-1.67 AU)
    assert env.observation_space.shape == (R.OBS_DIM,) and R.OBS_DIM == 31 and env.action_space.shape == (R.ACT_DIM,) and R.ACT_DIM == 5


def test_solar_collectors_power_the_launcher_without_free_energy():
    """With collectors on and a tiny reserve, the launches are paid for by beamed microwaves; drawn energy always equals
    reserve spent + beam energy used, and beam energy used never exceeds what the collectors offered."""
    cfg = R.make_config(randomize_phase=False, energy_budget_j=1e28)
    for fid in ("fast", "hifi"):
        env = R.OctoEarthEnv(cfg, fid)
        env.reset(seed=5)
        env.step(R.policy_heuristic(env, None))
        m, S = env.metrics(), env.engine.sol
        drawn = float(env.engine.ledger[R.LG_E_DRAWN])
        assert drawn > 100 * cfg.energy_budget_j
        assert abs(m["energy_from_collectors_J"] + m["energy_from_reserve_J"] - drawn) <= 1e-9 * drawn
        assert S["E_beam_used_J"] <= S["E_offered_J"] * (1 + 1e-12)
        assert 0.0 <= env.engine.E_rem <= cfg.energy_budget_j
        assert m["arm_heat_vs_earth_sunlight"] > 1.0          # flagged as unrealistic, but reported


def test_collectors_off_means_reserve_only():
    cfg = R.make_config(randomize_phase=False, energy_budget_j=1e28, solar_collectors=False)
    env = R.OctoEarthEnv(cfg, "fast")
    env.reset(seed=5)
    env.step(R.policy_heuristic(env, None))
    assert env.engine.ledger[R.LG_E_DRAWN] <= cfg.energy_budget_j * (1 + 1e-12)
    assert env.metrics()["energy_from_collectors_J"] == 0.0
    assert R.make_config("realistic").solar_collectors is False


def test_power_chain_and_viewer_power_columns():
    cfg = R.make_config(randomize_phase=False)
    pw = R.collector_power(cfg, [R.AU, 0.0, 0.0])
    op = R.collector_operating(cfg)
    chain = (R.L_SUN * cfg.collector_fraction * (1 - pw["lane_frac"]) * cfg.pv_efficiency * op["temp_factor"] * op["degr_factor"]
             * cfg.microwave_efficiency * pw["eta_capture"] * cfg.rectenna_efficiency)
    assert 0.6 < op["temp_factor"] < 0.8 and 400 < op["T_panel_K"] < 500        # hot panels at 0.5 AU lose efficiency
    half = R.collector_power(cfg, [R.AU, 0.0, 0.0], built=0.5)
    assert abs(half["P_elec_W"] - 0.5 * pw["P_elec_W"]) <= 1e-12 * pw["P_elec_W"]
    assert abs(chain - pw["P_elec_W"]) <= 1e-12 * chain
    far = R.collector_power(cfg, [3 * R.AU, 0.0, 0.0])
    assert far["eta_capture"] < pw["eta_capture"]             # beams spread more over a longer path
    env = _one_decision("fast")
    rows = R._viewer_rows(env.last_samples, env.engine.last_power, env.cfg)
    for row in rows:
        p_av, p_in, eta, p_launch = row[30:34]
        assert p_av > 0 and 0 < eta <= 1 and p_in * env.cfg.rectenna_efficiency <= p_av * (1 + 1e-3)
        assert p_launch >= p_in * env.cfg.rectenna_efficiency * (1 - 1e-3)


def test_climate_model_and_space_mirrors_in_rl():
    """The climate is part of the RL task: 288 K today, frozen at 2.7 AU without extra light, the mirror action changes
    the temperature, the reward has a climate term and success requires a habitable temperature."""
    assert abs(R.climate_equilibrium(1.0) - 288.0) < 0.05
    assert R.climate_equilibrium(1 / 2.7 ** 2) < 200
    assert "climate" in R.REWARD_KEYS
    cfg = R.make_config(randomize_phase=False)
    temps = []
    for a4 in (-1.0, 1.0):
        env = R.OctoEarthEnv(cfg, "fast")
        env.reset(seed=5)
        for _ in range(3):
            obs, r, term, trunc, info = env.step(np.array([-1, 0, 0, -1, a4], dtype=np.float32))
        temps.append(env.clim["Ts"])
        assert info["components"]["climate"] <= 0.0
    assert temps[1] > temps[0] + 20             # more mirror light -> warmer Earth
    env = R.OctoEarthEnv(cfg, "fast")
    env.reset(seed=5)
    obs, *_ = env.step(np.array([-1, 0, 0, -1], dtype=np.float32))   # 4-dim actions are still accepted (a4 = 0)
    assert obs.shape == (R.OBS_DIM,)
    s_off = R.run_episode(R.make_config(randomize_phase=False, mirrors=False), R.policy_heuristic, 5)
    assert s_off["reason"] != "success" and s_off["T_surface_K"] < 230 and not s_off["climate_ok"]
    s_on = R.run_episode(cfg, R.policy_heuristic, 5)
    assert s_on["reason"] == "success" and s_on["climate_ok"]


def test_energy_ledger_closes_and_mirrors_conserve_light():
    cfg = R.make_config(randomize_phase=False)
    s = R.run_episode(cfg, R.policy_heuristic, 5)
    L = s["energy_ledger"]
    e_in = L["sun_on_collectors"] + L["reserve_used"] + L["mirror_sun"]
    e_out = sum(L[k] for k in ("reflected", "collector_heat", "manufacturing", "transmitter_heat", "beam_spill", "rectenna_heat",
                               "packet_ke", "launcher_heat", "mirror_to_earth", "mirror_to_space", "mirror_heat"))
    assert abs(e_in - e_out) <= 1e-9 * e_in and s["energy_ledger_residual"] < 1e-9
    assert 0 < L["mirror_to_earth"] <= cfg.mirror_reflectivity * L["mirror_sun"]
    assert abs(L["mirror_heat"] - (1 - cfg.mirror_reflectivity) * L["mirror_sun"]) <= 1e-9 * L["mirror_sun"]
    # capacity falls as 1/r^2 like sunlight; default area gives about today's sunlight at 2.7 AU
    md1, md27 = R.mirror_design(cfg, 1.0), R.mirror_design(cfg, 2.7)
    assert abs(md1["cap_W"] / md27["cap_W"] - 2.7 ** 2) < 1e-9 and 1.0 < md27["cap_rel"] < 1.3
    assert L["packet_ke"] <= L["launcher_draw"] * cfg.launch_efficiency * (1 + 1e-9)
    assert L["fresh_mass_kg"] > 0 and L["heat_to_earth"] > 0
    for d in s["decisions"]:
        assert d["ledger"]["residual_rel"] < 1e-9 and 0 < d["built"] <= 1.0


def test_collector_belt_station_keeping():
    cfg = R.make_config(randomize_phase=False)
    env = R.OctoEarthEnv(cfg, "fast", record=True)
    env.reset(seed=5)
    bk = R.BeltKeeper(cfg)
    for _ in range(2):
        t_s = env.engine.t_years
        env.step(R.policy_heuristic(env, None))
        out = bk.advance(env.last_samples, env.engine.last_eled["microwaves_sent"] / (100 * R.YEAR), t_s)
    sm = bk.summary()
    assert sm["on"]["max_km"] < 2000 and sm["off"]["max_km"] > 10 * sm["on"]["max_km"]
    assert len(out["track"]["on"]) == R.BELT_OUT_N and len(out["track"]["on"][0]) == R.BELT_N_TRACK
    assert R.belt_pull_on_earth(cfg)["ratio_1au"] < 1e-6


def test_enablers_and_problems_declared():
    ids = [e["id"] for e in R.ENABLERS]
    assert len(ids) == 3 and len(set(ids)) == len(ids)          # the model allows exactly three enablers
    for e in R.ENABLERS:
        assert all(e[k] for k in ("name", "provides", "params", "gap", "modelled"))
    st = {m["id"]: m["status"] for m in R.MAJOR_PROBLEMS}
    assert st["P1"] == st["P2"] == st["P3"] == st["P4"] == "addressed"
    meaning = R.PARAM_BY_KEY["pv_efficiency"]["meaning"]
    assert "RATED" in meaning and "CHANGED MEANING" in meaning


def test_life_indicators_and_no_mirror_counterfactual():
    """Zonal profile averages to the global mean; today's 288 K gives an ice line near 70 deg; a frozen Earth has none."""
    import numpy as np
    x = np.linspace(0, 1, 200001)
    for T in (150.0, 288.0, 300.0):
        prof = T + R.LIFE_DT_EQ_POLE / 3 - R.LIFE_DT_EQ_POLE * x ** 2
        assert abs(np.trapezoid(prof, x) - T) < 1e-6
    today = R.life_indicators(288.0)
    assert 65 < today["ice_line_deg"] < 75 and today["grow_frac"] > 0.6
    frozen = R.life_indicators(152.0)
    assert frozen["ice_line_deg"] == 0.0 and frozen["grow_frac"] == 0.0 and "snowball" in frozen["state"]
    s = R.run_episode(R.make_config(randomize_phase=False), R.policy_heuristic, 5)
    d = s["decisions"][-1]
    assert d["T_no_mirror_K"] < 200 and "snowball" in d["life_no_mirror"] and d["life"]["grow_frac"] > 0.5


def test_frontend_overview_arms_mirrors_weather_view_wired():
    """Overview details (textured Earth, ring + 8 arms, collector satellites, tracked collectors + ghosts, mirror ring),
    the Ledger panel, the POV mirror layer and the weather & life view are wired up, and the viewer receives the
    declared enablers and problems. The artificial sun is gone everywhere."""
    fe = Path(__file__).resolve().parents[1] / "frontend"
    ovx = (fe / "ovextras.js").read_text(encoding="utf-8")
    for key in ("export function createOvExtras", "export function ledgerPanelHtml", "export function drawClimateChart",
                "InstancedMesh", "track", "function updateArms", "function updateMirrors", "mirror_to_earth"):
        assert key in ovx, key
    app = (fe / "app.js").read_text(encoding="utf-8")
    assert "./ovextras.js" in app and "renderLedger" in app and "r.enablers" in app
    assert "./weather.js" in app and "throwDir" in app and "weatherState" in app
    html = (fe / "index.html").read_text(encoding="utf-8")
    for key in ('data-drawer="dLedger"', 'id="dLedger"', 'id="ledgerBody"', 'id="ovTracked"', 'id="ovMirrors"', 'id="ovArms"',
                'id="climatePill"', 'value="weather"', 'id="chipWx"', 'id="wxBox"', 'id="ovZoomEarth"'):
        assert key in html, key
    assert "key: 'mirrors'" in (fe / "pov.js").read_text(encoding="utf-8")
    wx = (fe / "weather.js").read_text(encoding="utf-8")
    for key in ("export function createWeatherView", "export function lifeFromT", "windAt", "buildClouds", "noMirrors"):
        assert key in wx, key
    # the JS life profile uses the same constants as the Python one
    assert f"dT: {R.LIFE_DT_EQ_POLE:g}, T_ice: {R.LIFE_T_ICE:g}, T_grow: [{R.LIFE_T_GROW[0]:g}, {R.LIFE_T_GROW[1]:g}]" in wx
    for f in ("app.js", "ovextras.js", "pov.js", "index.html", "weather.js"):
        txt = (fe / f).read_text(encoding="utf-8").lower()
        assert "artificial sun" not in txt and not re.search(r"\blamp", txt), f
    assert len(R.ENABLERS) == 3 and all(e["gap"] and e["modelled"] and e["instruments"] for e in R.ENABLERS)
    assert {m["id"] for m in R.MAJOR_PROBLEMS} >= {"P1", "P2", "P3", "P4"}


def test_training_replay_device_and_plain_ui(tmp_path):
    """p31: training episodes are replayed with time + velocity (no frozen view), the newest episode is handed to
    the viewer on request, the training device is chosen honestly, the UI has no emojis and the collectors in the
    overview carry no red / blue colour coding."""
    cfg = R.smoke_config()
    env = R.OctoEarthEnv(cfg, "fast", record=True, record_moon=False)
    env.reset(seed=3)
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(R.ACT_DIM, dtype=np.float32))
    tr = np.array(env.last_trail)
    assert tr.shape[1] == 7 and np.all(np.diff(tr[:, 0]) > 0)            # t, x, y, z, vx, vy, vz
    v = np.linalg.norm(tr[:, 4:7], axis=1)
    assert np.all((v > 10) & (v < 40))                                    # heliocentric km/s
    app = R.App(cfg, tmp_path)
    app._on_episode(info["episode_summary"], 100, __import__("csv").writer(open(tmp_path / "x.csv", "w")), 0.0, env)
    assert app.viewer["mode"] == "training" and len(app.viewer["rows"]) == len(tr)   # first episode shown at once
    run = app.viewer["run_id"]
    app.train_latest["ep"] += 1                                            # a newer episode finished
    st = app.state_view(run, 0, 0, 0, 0, None)                             # replay not finished: no switch
    assert st["viewer"]["run_id"] == run and len(st["viewer"]["rows"]) == len(tr)
    st = app.state_view(run, len(tr), 0, 0, 0, None, trnext=True)         # replay finished: newest episode
    assert st["viewer"]["run_id"] == run + 1 and st["viewer"]["reset"]
    assert R.pick_device("cpu") == "cpu" and R.pick_device("auto") in ("cpu", "cuda")
    fe = Path(__file__).resolve().parents[1] / "frontend"
    emoji = re.compile("[\U0001F000-\U0001FFFF\u2600-\u2608\u260A-\u2714\u2716-\u2759\u275B-\u27BF]")
    for f in ("app.js", "ovextras.js", "pov.js", "index.html", "weather.js", "collector.js"):
        assert not emoji.search((fe / f).read_text(encoding="utf-8")), f
    ovx = (fe / "ovextras.js").read_text(encoding="utf-8")
    assert "0xff4040" not in ovx and "0x5ae0f0" not in ovx and "satelliteGeometry(true)" in ovx
    app_js = (fe / "app.js").read_text(encoding="utf-8")
    assert "trainEarthAt" in app_js and "trnext=1" in app_js and "// the next frame is requested FIRST, so an error in one frame can never stop the animation\n  requestAnimationFrame(frame);" in app_js


def test_worker_process_protocol_and_belt_thread(tmp_path):
    """p32: training / comparison run in a worker process that reports through a queue (log lines, status updates,
    "done"); the collector-belt simulation of a playback runs in its own thread and its results are attached to the
    decision records afterwards (the playback never waits for it); the belt kernel releases the GIL."""
    import queue as _q
    cfg = R.smoke_config()
    R.App._p32_dummy_job = lambda self: (self.log("[TEST] hello from the worker"), self._emit("train", dict(episode=7)))
    try:
        q, ev = _q.Queue(), threading.Event()
        R._worker_process("_p32_dummy_job", cfg, str(tmp_path), "cpu", dict(train={}, history=[], evals=[]), q, ev)
        msgs = []
        while not q.empty():
            msgs.append(q.get())
    finally:
        del R.App._p32_dummy_job
    assert msgs[-1] == ("done", "_p32_dummy_job")
    assert any(k == "log" and "hello from the worker" in p for k, p in msgs)
    app = R.App(cfg, tmp_path)
    for k, p in msgs[:-1]:
        app._apply_update(k, p)
    assert app.train.get("episode") == 7 and any("hello from the worker" in s for s in app.logbuf)
    assert R.belt_advance.targetoptions.get("nogil") is True
    # playback + belt thread
    app.start_playback("heuristic", "fast")
    t_end = time.time() + 240
    st = None
    while time.time() < t_end:
        st = app.state_view(app.viewer["run_id"], 0, 0, 0, 0, 1e6, ahead=1500)
        if st["viewer"]["done"] and len(app.viewer["belts"]) >= len(app.viewer["decisions"]):
            break
        time.sleep(0.5)
    v = st["viewer"]
    assert v["done"] and len(v["decisions"]) > 3
    assert v["bstart"] == 0 and len(v["belts"]) == min(50, len(v["decisions"]))
    ks = [k for k, _ in app.viewer["belts"]]
    assert ks == list(range(len(ks)))                                    # one result per decision, in order
    b = app.viewer["belts"][-1][1]
    assert b["on"]["max_km"] < b["off"]["max_km"] and "track" in b       # station keeping keeps the slot
    js = (Path(__file__).resolve().parents[1] / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "bsince=" in js and "powerPreference: 'high-performance'" in js and "Math.min(0.25" in js


def test_three_enablers_and_calculation_report():
    """p33: the model has exactly three enablers, each with its instruments; the calculation report computes every
    number with the simulation's own functions and covers every instrument."""
    assert [e["id"] for e in R.ENABLERS] == ["E1", "E2", "E3"]
    import calc_report as CR
    root = Path(__file__).resolve().parents[1]
    cfg = R.make_config()
    html = CR.build(cfg, root / "example_results")
    for e in R.ENABLERS:
        assert e["name"].split(" ")[0] in html
    for k in ("Instrument E1.1", "Instrument E1.4", "Instrument E2.1", "Instrument E2.6", "Instrument E3.1", "Instrument E3.2"):
        assert k in html, k
    assert "N/A" not in html and ">nan<" not in html.lower()
    for k in ("What happens without each instrument", "Instrument data sheets", "Measuring instruments"):
        assert k in html, k                                              # p33b: data sheet + counterfactual per instrument
    for sid in ("E1.1", "E1.2", "E1.3", "E1.4", "E2.1", "E2.2", "E2.3", "E2.4", "E2.5", "E2.6", "E3.1", "E3.2"):
        assert f'<div class="t">{sid} ' in html, sid
    fz = R.feasibility(cfg)
    assert CR.sci(fz["dv_required"], 5) in html                      # same number as the simulation's feasibility gate
    for txt in [p.read_text(encoding="utf-8") for p in (root / "frontend").glob("*.js")] + [(root / "octoearth_rl.py").read_text(encoding="utf-8")]:
        assert not re.search(r"\bE(?:[4-9]|10)\b", txt)                 # no references to a fourth (or later) enabler


def test_device_benchmark_and_live_view_switch(tmp_path):
    """p34: Auto measures the devices; the live training view can be switched off (training keeps running) and on."""
    cfg = R.make_config()
    b = R.benchmark_devices(cfg)
    assert "cpu" in b and 0 < b["cpu"] < 30
    app = R.App(cfg, tmp_path)
    assert app.train_live.is_set()
    app.set_live(False)
    assert not app.train_live.is_set() and app.viewer["mode"] == "idle"
    app.train_latest = dict(ep=5, trail=[[0, 1, 0, 0, 0, 30, 0], [1, 0, 1, 0, -30, 0, 0]], t_years=1.0, summary={},
                            decisions=[], moon_rel_km=[0, 0, 0], m_ast_left=0.0)
    app._training_view_if_wanted(True)
    assert app.viewer["mode"] == "idle"                      # off: nothing replaces the screen
    app.set_live(True)
    assert app.train_live.is_set() and app.viewer["mode"] == "training"
    st = app.state_view(0, 0, 0, 0, 0, None)
    assert st["train"]["live"] is True and st["train"]["device_request"] == "auto"


def test_other_planets_two_way_gravity_and_safety():
    """p35: Mercury, Venus, Jupiter pull Earth in both models, the models agree, and they are reported."""
    cfg = R.make_config(randomize_phase=False)
    f, h = R.FastEngine(cfg, 3), R.HifiEngine(cfg, 3)
    f.run_interval(20.0, 0.0, 3e6, [0, 1, 0])
    h.run_interval(20.0, 0.0, 3e6, [0, 1, 0])
    rf, _ = f.planets_state()
    rh, _ = h.planets_state()
    assert np.linalg.norm(rf - rh, axis=1).max() < 1e-3 * R.AU
    # switching the planets off changes Earth's path (their gravity acts) ...
    g = R.FastEngine(R.replace(cfg, other_planets=False), 3)
    g.run_interval(20.0, 0.0, 3e6, [0, 1, 0])
    assert np.linalg.norm(g.emb()[0] - f.emb()[0]) > 1e-6 * R.AU
    # ... and Earth pulls them back (their orbits are integrated, not fixed)
    lim = R.planet_safety_limits(cfg)
    assert [q["name"] for q in lim] == ["Mercury", "Venus", "Jupiter"]
    assert lim[2]["d_safe_au"] > lim[1]["d_safe_au"] > lim[0]["d_safe_au"] > 0
    env = _one_decision("fast")
    rows = R._viewer_rows(env.last_samples)
    assert len(rows[0]) == R.N_VIEW_COLS == 55
    d = np.array(rows[-1][52:55])
    assert np.all(d > 0.1)


def test_launcher_geometry_asteroid_on_axis_and_js_constants():
    """p36: asteroid on the ring axis, equidistant from all 8 roots, within reach of the 2 R_E arms; the viewer
    (frontend/launcher.js) uses the same numbers."""
    import re
    from pathlib import Path
    g = R.launcher_geometry(R.make_config())
    assert g["root_to_grab_spread_re"] < 1e-12
    assert g["root_to_grab_re"] < g["arm_length_re"] == 2.0
    assert g["root_to_asteroid_centre_re"] <= 2.0 and g["stroke_re"] > 0.1
    assert g["escape_energy_saving_rel"] < 1e-4 and g["moon_pull_error_rel"] < 1e-4
    js = (Path(__file__).resolve().parents[1] / "frontend" / "launcher.js").read_text()
    m = re.search(r"export const ARM_L = ([\d.]+), RING_R = ([\d.]+), GRAB_Z = ([\d.]+);", js)
    assert m and tuple(map(float, m.groups())) == (R.ARM_LENGTH_RE, R.RING_RADIUS_RE, R.GRAB_HEIGHT_RE)


def test_p38_failed_attempts_page_four_independent_models():
    """p38: failed.html shows FOUR independent models one at a time (own scene module each, lazily created, one
    render loop); the main viewer no longer contains the cinematic views but links to the page; assets are local,
    no runtime URLs, no emojis; the laser / hydrogen / magnet models do not use the asteroid launcher or its arms."""
    fe = Path(__file__).resolve().parents[1] / "frontend"
    rd = lambda f: (fe / f).read_text(encoding="utf-8")
    html, js = rd("failed.html"), rd("failed.js")
    for m in range(4):
        assert f'data-m="{m}"' in html
    for key in ('id="faStart"', 'id="faPause"', 'id="faReset"', 'id="faPrev"', 'id="faNext"', 'id="faInfo"', 'src="failed.js"', 'href="index.html"'):
        assert key in html, key
    for key in ("createLaserView", "createHydrogenScoopView", "createMagnetView", "createDysonView", "window.__fa", "ArrowRight", "views[i] || (views[i] = MODELS[i].make())"):
        assert key in js, key
    assert js.count("requestAnimationFrame(") == 2 and js.count("new THREE.WebGLRenderer") == 1   # one loop, one renderer
    scenes = {"laserScene.js": ("export function createLaserView", "laserClock", "E_UNBIND", "heatSpot", "Safe +2 K"),
              "hydrogenScene.js": ("export function createHydrogenScoopView", "HYD.M_NEED", "Whole solar wind", "bow shock", "fusion drive"),
              "magnetScene.js": ("export function createMagnetView", "QUENCH", "IMF reaches arms", "Ignore quench", "net force 0", "drawChart"),
              "dysonScene.js": ("export function createDysonView", "STRUCTURAL FAILURE", "DYSON.frac")}
    for f, keys in scenes.items():
        src = rd(f)
        for key in keys:
            assert key in src, (f, key)
        assert "requestAnimationFrame" not in src and "failedPhysics.js" in src, f
    for f in ("laserScene.js", "hydrogenScene.js", "magnetScene.js"):
        src = rd(f)
        assert "octoArms" not in src and "launcher.js" not in src and "createArmRig" not in src, f   # independent of the launcher
    assert "from './vendor/three" not in rd("failedPhysics.js")                                       # pure physics module
    app, idx = rd("app.js"), rd("index.html")
    for key in ("createHydrogenView", "createDysonView", "h2On()", "dyOn()"):
        assert key not in app, key
    for key in ('value="hydrogen"', 'value="dyson"', 'id="chipH2"', 'id="chipDy"'):
        assert key not in idx, key
    assert 'href="failed.html"' in idx and "Failed attempts (4 models)" in idx
    ea = rd("earthAssets.js")
    for key in ("heatDir", "heatSpot", "spotCos", "heatGlobal", "scorch", "fade", "hot"):
        assert key in ea, key
    for f in ("earth_day_4096.jpg", "earth_night_4096.jpg", "earth_clouds_2048.jpg", "earth_normal_2048.jpg", "earth_ocean_mask_2048.png"):
        assert f in ea and (fe / "assets" / "earth" / f).stat().st_size > 100_000, f
    emoji = re.compile("[\U0001F000-\U0001FFFF\u2600-\u2608\u260A-\u2714\u2716-\u2759\u275B-\u27BF]")
    for f in ("failed.html", "failed.js", "failKit.js", "failedPhysics.js", "laserScene.js", "hydrogenScene.js", "magnetScene.js",
              "dysonScene.js", "earthAssets.js", "cineKit.js", "octoArms.js"):
        txt = rd(f)
        assert not re.search(r"""['"`(]https?://""", txt), f
        assert not emoji.search(txt), f


def test_p38_failed_models_numbers_python_matches_js():
    """Every model fails quantitatively (Python), and the JavaScript physics behind the 3-D views gives the same numbers."""
    import importlib, shutil, subprocess
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    F = importlib.import_module("failed_models")
    s = F.summary()
    assert abs(s["DV_REQ"] - 11658) < 1 and abs(s["F_REQ"] - 2.206e17) / 2.206e17 < 1e-3
    # 1 laser: the required beam vaporises Earth; the safe beam moves it metres
    assert s["Tsurf_req"] > 4e4 and s["tOceans"] < 60 and s["tUnbind"] / 86400 < 50
    assert (s["a10k_safe"] - 1) * F.AU < 10
    # 2 hydrogen: never enough fuel, even the whole solar wind
    assert s["h100_frac"] < 1e-6 and s["hw_frac"] < 0.3 and s["hw_a"] < 1.25
    # 3 magnetic arms: quench limit, tiny force, even the required current falls short
    assert s["I_MAX"] < 3e8 and s["I_REQ"] > 1e17 and (s["mag_a"] - 1) * F.AU < 100
    assert s["magreq_a"] < 2.0 and s["magreq_t"] is None and s["I_ARRIVE"] > s["I_REQ"]
    # 4 Dyson: even intact arms give ~20 % of the push; Earth overheats
    assert 0.18 < s["dyson_frac"] < 0.21 and s["dyson_T"] > 2e4
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS comparison")
    out = subprocess.run([node, "--input-type=module", "-e",
                          "const m = await import(process.argv[1]); console.log(JSON.stringify(m.summary()));",
                          (root / "frontend" / "failedPhysics.js").as_uri()], capture_output=True, text=True, timeout=60)
    if out.returncode != 0 and "ERR_UNKNOWN_FILE_EXTENSION" not in out.stderr and "Cannot use import" in out.stderr:
        pytest.skip("node cannot load ES modules here")
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout.strip().splitlines()[-1])
    for key, v in s.items():
        if v is None:
            assert js[key] is None, key
        else:
            assert abs(js[key] - v) <= 1e-9 * max(1.0, abs(v)), (key, js[key], v)


def test_p38_dyson_model4_reflection_geometry_and_numbers():
    """Reflection r = d - 2(d.n)n with n = normalize(r0 - d) sends the beam to Earth; every arm (2 R_E) reaches its
    mount; the incident beam clears Earth; force / acceleration match the HUD formulas."""
    fe = Path(__file__).resolve().parents[1] / "frontend"
    dy = (fe / "dysonScene.js").read_text(encoding="utf-8")
    num = lambda k: float(re.search(rf"\b{k}: ([0-9.e+-]+)", dy).group(1))
    P, a_deg, H, M, ER, EX, CX, PH = (num(k) for k in ("P_MAX", "ALPHA_DEG", "H", "M", "ER", "EARTH_X", "CONC_X", "PHASE_DEG"))
    a = np.radians(a_deg); u = np.array([np.cos(a), np.sin(a), 0.0]); e2 = np.array([0, 0, 1.0]); e1 = np.cross(e2, u); e1 /= np.linalg.norm(e1)
    for shift in (0.0, 2.0, 6.0):                                   # Earth pushed away by up to 6 Earth radii
        C = np.array([(CX - EX) / ER - shift, 0, 0]); Rc = H * u
        d = (Rc - C) / np.linalg.norm(Rc - C); r0 = -u
        n = (r0 - d) / np.linalg.norm(r0 - d); r = d - 2 * d.dot(n) * n
        assert np.allclose(r, r0, atol=1e-12) and d.dot(n) < 0                   # hits the front, reflected to Earth
        reach = []
        for k in range(8):
            ph = np.radians(k * 45 + PH); rk = np.cos(ph) * e1 + np.sin(ph) * e2
            pk = rk - rk.dot(n) * n; pk /= np.linalg.norm(pk)
            reach.append(np.linalg.norm(Rc + M * pk - 1.30 * rk))
        assert max(reach) <= 2.0                                               # arm length 2 R_E
        t = np.clip(-C.dot(Rc - C) / np.dot(Rc - C, Rc - C), 0, 1)
        assert np.linalg.norm(C + t * (Rc - C)) > 1.1                          # incident beam passes Earth
        # forces: arms (P/c)(d - r) + absorbed (P/c) r = (P/c) d, directed away from the Sun
        assert np.allclose((d - r) + r, d) and d[0] > 0.99
    assert abs(P / 2.998e8 - 4.30e16) / 4.30e16 < 0.01 and abs(P / 2.998e8 / 5.972e24 - 7.2e-9) < 0.05e-9
