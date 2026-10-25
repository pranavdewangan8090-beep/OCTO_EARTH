"""OCTO-EARTH - calculation report generator.

Writes CALCULATIONS_REPORT.html (and CALCULATIONS_REPORT.pdf if Playwright + Chromium are installed) with every
calculation of the system: the mission requirement, the THREE enablers (E1 launcher system, E2 solar power belt with
microwave beaming, E3 space mirrors) and every instrument inside them, the climate, Mars safety, the energy ledger and
the tested results.

Every number is computed here by calling the SAME functions the simulation uses (octoearth_rl.py), so the report can
never disagree with the code. Tested-run numbers are read from example_results/ (or --results DIR).

    python calc_report.py                       # default configuration, example_results/
    python calc_report.py --results outputs     # use your own run
    python calc_report.py --set target_au=2.5   # any Config override
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import octoearth_rl as R  # noqa: E402

CHICXULUB_J = 1.0e24          # order of magnitude of the Chicxulub impact energy [J] (published estimates 1e23-6e25 J)


# ------------------------------------------------------------------------------------------------ formatting
def sci(x, n=3):
    """Number as HTML: plain for 0.001..99,999, otherwise m x 10^e."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    x = float(x)
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -3 <= e <= 4:
        d = max(0, n - 1 - e)
        return f"{x:,.{d}f}"
    m = x / 10 ** e
    if round(abs(m), n - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{n - 1}f}&times;10<sup>{e}</sup>"


def pct(x, n=1):
    return f"{100 * x:.{n}f} %"


def esc(s):
    return html.escape(str(s))


class Doc:
    """Collects the report body; numbered calculations."""

    def __init__(self):
        self.parts, self.k, self.toc, self.sec = [], 0, [], 0
        self.cur, self.refs = "MISSION", {}      # calculations are filed under the instrument being described

    def h1(self, t, anchor):
        self.sec += 1
        self.toc.append((1, f"{self.sec}. {t}", anchor))
        self.parts.append(f'<h1 id="{anchor}">{self.sec}. {esc(t)}</h1>')

    def h2(self, t, anchor=None):
        if t.startswith("Instrument E"):
            self.cur = t.split()[1]
        anchor = anchor or f"s{len(self.toc)}"
        self.toc.append((2, t, anchor))
        self.parts.append(f'<h2 id="{anchor}">{esc(t)}</h2>')

    def h3(self, t):
        self.parts.append(f"<h3>{esc(t)}</h3>")

    def p(self, t):
        self.parts.append(f"<p>{t}</p>")

    def note(self, t, kind="note"):
        self.parts.append(f'<div class="{kind}">{t}</div>')

    def calc(self, title, formula, subst, result, code=None, comment=None):
        """One calculation: formula, the numbers put in, the result, the code that does it."""
        self.k += 1
        self.refs.setdefault(self.cur, []).append(self.k)
        c = f'<div class="calc" id="c{self.k}"><div class="ct"><span class="cn">C{self.k}</span> {esc(title)}</div>'
        c += f'<div class="row"><span class="lab">Formula</span><span class="f">{formula}</span></div>'
        if subst:
            c += f'<div class="row"><span class="lab">Values</span><span>{subst}</span></div>'
        c += f'<div class="row res"><span class="lab">Result</span><span><b>{result}</b></span></div>'
        if comment:
            c += f'<div class="row"><span class="lab">Meaning</span><span>{comment}</span></div>'
        if code:
            c += f'<div class="code">code: {esc(code)}</div>'
        self.parts.append(c + "</div>")
        return self.k

    def table(self, head, rows, cls=""):
        h = f'<table class="{cls}"><tr>' + "".join(f"<th>{x}</th>" for x in head) + "</tr>"
        for r in rows:
            h += "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
        self.parts.append(h + "</table>")

    def raw(self, t):
        self.parts.append(t)


CSS = """
@page { size: A4; margin: 15mm 13mm 16mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.2pt; color: #1a1a1a; line-height: 1.38; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 16pt; color: #12324f; border-bottom: 2px solid #12324f; padding-bottom: 3px; margin-top: 26px; page-break-before: always; }
h1.first { page-break-before: avoid; }
h2 { font-size: 12.5pt; color: #1f4e79; margin: 18px 0 6px; border-bottom: 1px solid #c9d6e3; }
h3 { font-size: 11pt; color: #333; margin: 12px 0 4px; }
p { margin: 5px 0; }
.calc { border: 1px solid #c9d6e3; border-left: 4px solid #1f4e79; border-radius: 3px; margin: 7px 0; padding: 5px 8px; page-break-inside: avoid; background: #fbfcfe; }
.ct { font-weight: 600; color: #12324f; margin-bottom: 2px; }
.cn { display: inline-block; background: #1f4e79; color: #fff; border-radius: 3px; padding: 0 5px; font-size: 8.5pt; margin-right: 4px; }
.row { display: flex; gap: 8px; margin: 1px 0; }
.lab { flex: 0 0 62px; color: #667; font-size: 8.8pt; padding-top: 1px; }
.f { font-family: 'Cambria Math', Cambria, 'Times New Roman', serif; font-size: 10.8pt; }
.res b { color: #0b5e2a; }
.code { font-family: Consolas, monospace; font-size: 8pt; color: #889; text-align: right; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #1f4e79; color: #fff; text-align: left; padding: 3px 5px; font-weight: 600; }
td { border-bottom: 1px solid #dde4ec; padding: 3px 5px; vertical-align: top; }
table.kv td:first-child { width: 38%; color: #333; }
.note { background: #eef5fb; border: 1px solid #c9dff0; padding: 5px 8px; margin: 7px 0; border-radius: 3px; font-size: 9.4pt; }
.warn { background: #fff6e5; border: 1px solid #f0d39a; padding: 5px 8px; margin: 7px 0; border-radius: 3px; font-size: 9.4pt; }
.bad { background: #fdecec; border: 1px solid #f0b4b4; padding: 5px 8px; margin: 7px 0; border-radius: 3px; font-size: 9.4pt; }
.ok { color: #0b5e2a; font-weight: 600; } .no { color: #a11; font-weight: 600; }
.cover { text-align: center; padding: 30mm 0 10mm; }
.cover h0 { display: block; font-size: 26pt; color: #12324f; font-weight: 700; }
.cover .sub { font-size: 13pt; color: #1f4e79; margin-top: 6px; }
.toc { columns: 2; font-size: 9.4pt; margin-top: 8mm; }
.toc div.l2 { padding-left: 12px; color: #444; }
.ebox { border: 2px solid #1f4e79; border-radius: 5px; padding: 6px 10px; margin: 8px 0; page-break-inside: avoid; }
.ebox .t { font-weight: 700; color: #12324f; font-size: 11pt; }
.small { font-size: 8.8pt; color: #555; }
"""


# ------------------------------------------------------------------------------------------------ report
def build(cfg, results_dir: Path) -> str:
    G, AU, YEAR = R.G, R.AU, R.YEAR
    comp = None
    cp = results_dir / "comparison.json"
    if cp.exists():
        comp = json.loads(cp.read_text())
    rows = {r["policy"]: r for r in (comp or {}).get("rows", [])}
    H = rows.get("heuristic", {}).get("hifi") if rows else None      # tested hand-designed mission (high fidelity)
    d = Doc()
    fz = R.feasibility(cfg)
    des = R.collector_design(cfg)
    op = R.collector_operating(cfg)
    md1 = R.mirror_design(cfg, 1.0)
    mdt = R.mirror_design(cfg, cfg.target_au)
    ms = R.mars_safety_limits(cfg)
    u_h = R.heuristic_speed(cfg)

    # =========================================================================== 1. overview
    d.h1("What the system is and the three enablers", "overview")
    d.p(f"OCTO-EARTH moves Earth (with its Moon) from its orbit at 1 AU to a circular orbit at <b>{cfg.target_au} AU</b> "
        f"within <b>{cfg.mission_years:,.0f} simulated years</b> by throwing mass away from Earth (momentum exchange: Earth "
        f"recoils the other way). The thrown mass comes from an asteroid docked to Earth, the energy comes from sunlight "
        f"collected near the Sun and beamed to Earth as microwaves, and Earth's climate is kept livable far from the Sun by "
        f"mirrors that reflect extra sunlight onto it. A reinforcement-learning agent (PPO) chooses every "
        f"{cfg.decision_interval_years:.0f} years how hard, how fast and in which direction to throw, and how much mirror "
        f"light to send to Earth.")
    d.note("<b>Exactly three enablers.</b> An enabler is a technology the model ASSUMES can be built without showing it. "
           "The model allows only three; each one is a complete system made of the instruments listed under it. Everything "
           "else in the model is ordinary physics (gravity, momentum, energy, radiation) or ordinary control software.")
    d.table(["", "Enabler", "What it does in the model", "Instruments"],
            [[f"<b>{e['id']}</b>", esc(e["name"]), esc(e["provides"]),
              "<br>".join("&bull; " + esc(i["name"]) for i in e["instruments"])] for e in R.ENABLERS])
    d.h2("Chain of cause and effect")
    d.p("Sunlight &rarr; <b>E2</b> collectors (electricity) &rarr; <b>E2</b> transmitters (microwaves) &rarr; <b>E2</b> "
        "rectennas on the arms (electricity) &rarr; <b>E1</b> launch arms throw asteroid packets (<b>E1</b> reservoir) "
        "&rarr; Earth recoils, its orbit grows &rarr; less sunlight at Earth &rarr; <b>E3</b> mirrors add sunlight &rarr; "
        "climate stays near 288 K. Waste heat of E1 and E2 is sent away from Earth by the E1 radiators and the E2 spill "
        "control; the small share that still reaches Earth is part of the climate calculation.")
    d.h2("Key numbers at a glance")
    kv = [["Required speed change of Earth (slow spiral)", f"{sci(fz['dv_required'])} m/s"],
          ["Orbital energy Earth+Moon must gain", f"{sci(fz['dE_orbit'])} J"],
          ["Minimum launch energy (whole asteroid, slowest launch)", f"{sci(fz['energy_min_draw'])} J"],
          ["Asteroid (reaction mass)", f"{sci(cfg.asteroid_mass_kg)} kg = {sci(cfg.asteroid_mass_kg / R.M_MOON)} Moon masses"],
          ["Launch speed used by the hand-designed strategy", f"{sci(u_h / 1e3)} km/s = {pct(u_h / R.C_LIGHT, 2)} of light speed"],
          ["Sunlight intercepted by the collector belt", f"{sci(R.L_SUN * cfg.collector_fraction)} W ({pct(cfg.collector_fraction, 0)} of the Sun)"],
          ["Electricity at the arms (Earth at 1 AU / at the target)", f"{sci(fz['solar_power_min'])} W (weakest point)"],
          ["Space-mirror capacity at the target", f"{sci(mdt['cap_rel'])} &times; today's sunlight on Earth"],
          ["Analytic verdict", f"<b>{esc(fz['verdict'])}</b>"]]
    if H:
        kv += [["Tested mission (hand-designed, high fidelity)", f"{esc(H['reason'])}: a = {H['a_au']:.4f} AU, e = {H['e']:.4f} "
                f"after {H['t_years']:,.0f} y; surface {H['T_surface_K']:.1f} K"]]
    d.table(["Quantity", "Value"], kv, "kv")
    d.note("Numbers are labelled <b>C1, C2, &hellip;</b>. Each shows the formula, the values put in, the result and the "
           "function in <code>octoearth_rl.py</code> that does the same calculation in the simulation. Values marked "
           "<i>indicative</i> are NOT used by the simulation (they only help to judge an instrument).", "note")

    # =========================================================================== 2. constants
    d.h1("Physical constants and mission inputs", "inputs")
    d.table(["Constant", "Value", "Unit"], [[esc(c["name"]), sci(c["value"], 6), esc(c["unit"])] for c in R.CONSTANT_DOCS])
    d.table(["Mission input", "Value"], [
        ["Target orbit radius a<sub>t</sub>", f"{cfg.target_au} AU (tolerance &plusmn;{cfg.tol_a_au} AU, e &le; {cfg.tol_e}, i &le; {cfg.tol_inc_deg}&deg;)"],
        ["Mission time limit T", f"{cfg.mission_years:,.0f} years = {sci(cfg.mission_years * YEAR)} s"],
        ["Decision interval", f"{cfg.decision_interval_years:.0f} years ({math.ceil(cfg.mission_years / cfg.decision_interval_years)} decisions)"],
        ["Climate target", f"{cfg.climate_target_K:.0f} K &plusmn; {cfg.climate_tol_K:.0f} K"]], "kv")
    d.p("The full list of all parameters with their values, units and meaning is in the appendix.")

    # =========================================================================== 3. mission requirement
    d.h1("Mission requirement: how much push, energy and time", "mission")
    m_sys = R.M_EARTH + cfg.asteroid_mass_kg + R.M_MOON
    m_l = R.M_EARTH + cfg.asteroid_mass_kg
    mu = R.GM_SUN + G * m_sys
    a0, at = AU, cfg.target_au * AU
    v0, vt = math.sqrt(mu / a0), math.sqrt(mu / at)
    T = cfg.mission_years * YEAR
    d.calc("Mass that has to be moved", "<i>m</i><sub>sys</sub> = <i>M</i><sub>Earth</sub> + <i>m</i><sub>asteroid</sub> + <i>M</i><sub>Moon</sub>",
           f"{sci(R.M_EARTH, 5)} + {sci(cfg.asteroid_mass_kg)} + {sci(R.M_MOON, 4)} kg", f"{sci(m_sys, 5)} kg", "feasibility()")
    d.calc("Gravitational parameter of the Sun-Earth system", "&mu; = G&middot;M<sub>Sun</sub> + G&middot;m<sub>sys</sub>",
           f"{sci(R.GM_SUN, 6)} + {sci(G * m_sys, 4)} m&sup3;/s&sup2;", f"{sci(mu, 6)} m&sup3;/s&sup2;", "feasibility()")
    d.calc("Circular orbital speed today (1 AU)", "v<sub>0</sub> = &radic;(&mu; / a<sub>0</sub>)", f"&radic;({sci(mu, 4)} / {sci(a0, 5)})",
           f"{sci(v0, 6)} m/s", "feasibility()")
    d.calc(f"Circular orbital speed at the target ({cfg.target_au} AU)", "v<sub>t</sub> = &radic;(&mu; / a<sub>t</sub>)",
           f"&radic;({sci(mu, 4)} / {sci(at, 5)})", f"{sci(vt, 6)} m/s", "feasibility()")
    d.calc("Required speed change (slow outward spiral)", "&Delta;v = v<sub>0</sub> &minus; v<sub>t</sub>",
           f"{sci(v0, 6)} &minus; {sci(vt, 6)}", f"{sci(fz['dv_required'], 5)} m/s", "feasibility() dv_spiral",
           "A slow spiral (small pushes along the motion over thousands of years) needs exactly the difference of the two circular speeds.")
    a_tr = 0.5 * (a0 + at)
    d.calc("Reference: Hohmann transfer (two short burns)",
           "a<sub>tr</sub> = (a<sub>0</sub>+a<sub>t</sub>)/2; &Delta;v = |&radic;(&mu;(2/a<sub>0</sub>&minus;1/a<sub>tr</sub>)) &minus; v<sub>0</sub>| + |v<sub>t</sub> &minus; &radic;(&mu;(2/a<sub>t</sub>&minus;1/a<sub>tr</sub>))|",
           f"a<sub>tr</sub> = {sci(a_tr / AU, 4)} AU", f"{sci(fz['dv_hohmann'], 5)} m/s", "feasibility() dv_hoh",
           "Cheaper in speed change, but would need two enormous short pushes; the launcher can only push gradually, so the spiral value is used.")
    d.calc("Orbital energy Earth + Moon (+asteroid) must gain", "&Delta;E = G M<sub>Sun</sub> m<sub>sys</sub> (1/(2a<sub>0</sub>) &minus; 1/(2a<sub>t</sub>))",
           f"{sci(R.GM_SUN, 5)} &times; {sci(m_sys, 4)} &times; (1/{sci(2 * a0, 4)} &minus; 1/{sci(2 * at, 4)})", f"{sci(fz['dE_orbit'])} J", "feasibility() dE_orb")
    d.calc("Orbital angular momentum change", "&Delta;L = m<sub>sys</sub> (&radic;(&mu; a<sub>t</sub>) &minus; &radic;(&mu; a<sub>0</sub>))", "",
           f"{sci(fz['dL_orbit'])} kg m&sup2;/s", "feasibility() dL_orb")
    d.calc("Average power just for the orbital energy", "P = &Delta;E / T", f"{sci(fz['dE_orbit'])} J / {sci(T)} s",
           f"{sci(fz['dE_orbit'] / T)} W = {sci(fz['dE_orbit'] / T / R.EARTH_INTERCEPT_W)} &times; all sunlight Earth intercepts", "feasibility()")
    vesc = fz["vesc"]
    d.calc("Escape speed from Earth + docked asteroid", "v<sub>esc</sub> = &radic;(2 G (M<sub>Earth</sub>+m<sub>ast</sub>) / R<sub>Earth</sub>)",
           f"&radic;(2 &times; {sci(G, 5)} &times; {sci(m_l, 4)} / {sci(R.R_EARTH, 4)})", f"{sci(vesc, 5)} m/s", "feasibility() vesc / launch_vinf()",
           "A packet launched slower than this falls back; faster packets keep v<sub>&infin;</sub> = &radic;(u&sup2; &minus; v<sub>esc</sub>&sup2;).")
    ratio = m_sys / (m_sys - cfg.asteroid_mass_kg)
    d.calc("Rocket equation: mass ratio if the whole asteroid is thrown", "MR = m<sub>sys</sub> / (m<sub>sys</sub> &minus; m<sub>ast</sub>)",
           f"{sci(m_sys, 5)} / {sci(m_sys - cfg.asteroid_mass_kg, 5)}", f"{ratio:.6f}  (ln MR = {math.log(ratio):.6f})", "feasibility()")
    d.calc("Speed change possible at the maximum launch speed", "&Delta;v<sub>cap</sub> = v<sub>&infin;,max</sub> &middot; ln MR,  v<sub>&infin;,max</sub> = &radic;(u<sub>max</sub>&sup2; &minus; v<sub>esc</sub>&sup2;)",
           f"{sci(fz['vinf_max'], 5)} &times; {math.log(ratio):.6f}", f"{sci(fz['dv_capacity_mass'], 5)} m/s (needed {sci(fz['dv_required'], 5)} m/s)", "feasibility() dv_cap_mass")
    d.calc("Reaction mass needed at the maximum launch speed", "m = m<sub>sys</sub> (1 &minus; e<sup>&minus;&Delta;v / v<sub>&infin;,max</sub></sup>)", "",
           f"{sci(fz['mass_needed_at_umax'])} kg = {sci(fz['mass_needed_at_umax'] / R.M_MOON)} Moon masses", "feasibility() mass_needed_umax")
    vinf_req = fz["dv_required"] / math.log(ratio)
    d.calc("Slowest launch speed that works with the whole asteroid (minimum energy)", "v<sub>&infin;</sub> = &Delta;v / ln MR;  u = &radic;(v<sub>&infin;</sub>&sup2; + v<sub>esc</sub>&sup2;)",
           f"v<sub>&infin;</sub> = {sci(fz['dv_required'], 5)} / {math.log(ratio):.6f} = {sci(vinf_req, 5)} m/s", f"u = {sci(fz['u_required'], 5)} m/s", "feasibility() u_req")
    d.calc("Minimum launch energy", "E<sub>min</sub> = &frac12; m<sub>ast</sub> u&sup2; / &eta;<sub>launch</sub>",
           f"0.5 &times; {sci(cfg.asteroid_mass_kg)} &times; ({sci(fz['u_required'], 4)})&sup2; / {cfg.launch_efficiency}", f"{sci(fz['energy_min_draw'])} J "
           f"(average {sci(fz['avg_power_min'])} W over the mission)", "feasibility() e_draw_min")
    d.calc("Shortest possible transfer time (maximum packet rate and speed)", "&Delta;v per year = n &middot; m<sub>pk</sub> &middot; v<sub>&infin;,max</sub> / m<sub>sys</sub>;  t<sub>min</sub> = &Delta;v / (&Delta;v per year)",
           f"{cfg.ejections_per_year:g} &times; {sci(cfg.max_packet_mass_kg)} &times; {sci(fz['vinf_max'], 4)} / {sci(m_sys, 4)} = "
           f"{sci(cfg.ejections_per_year * cfg.max_packet_mass_kg * fz['vinf_max'] / m_sys, 4)} m/s per year", f"{fz['years_min']:,.0f} years", "feasibility() years_min")
    d.calc("Mass + time limits together", "v<sub>&infin;</sub> = max(&Delta;v/ln MR, &Delta;v m<sub>sys</sub>/(T n m<sub>pk</sub>)); m = m<sub>sys</sub>(1&minus;e<sup>&minus;&Delta;v/v<sub>&infin;</sub></sup>); E = &frac12; m u&sup2;/&eta;", "",
           f"u = {sci(fz['u_combined'], 5)} m/s, m = {sci(fz['mass_combined'])} kg, E = {sci(fz['energy_combined'])} J", "feasibility() combined")
    d.calc("Launcher peak power (maximum packet rate and speed)", "P<sub>peak</sub> = n &middot; m<sub>pk</sub> &middot; &frac12; u<sub>max</sub>&sup2; / &eta; / year",
           f"{cfg.ejections_per_year:g} &times; {sci(cfg.max_packet_mass_kg)} &times; 0.5 &times; ({sci(cfg.max_ejection_speed_ms)})&sup2; / {cfg.launch_efficiency} / {sci(YEAR, 5)}",
           f"{sci(fz['launcher_peak_power'])} W", "feasibility() p_peak")
    d.calc("Energy available over the mission", "E<sub>avail</sub> = reserve + P<sub>elec,min</sub> &middot; T",
           f"{sci(cfg.energy_budget_j)} + {sci(fz['solar_power_min'])} &times; {sci(T)}", f"{sci(fz['energy_available'])} J", "feasibility() e_avail")
    d.table(["Check", "Result", "Detail"], [[k, '<span class="ok">OK</span>' if c["ok"] else '<span class="no">FAIL</span>', esc(c["detail"])]
                                            for k, c in fz["checks"].items()])
    d.note(f"<b>Verdict:</b> {esc(fz['verdict'])}. The realistic preset (Ceres-mass asteroid, 30 km/s launcher, Earth's own "
           f"sunlight as energy) is infeasible: verdict <i>{esc(R.feasibility(R.make_config('realistic'))['verdict'])}</i>.")

    # =========================================================================== 4. E1
    e1 = R.ENABLERS[0]
    d.h1("Enabler E1 - Earth launcher system", "e1")
    d.raw(f'<div class="ebox"><div class="t">{esc(e1["name"])}</div>{esc(e1["provides"])}<br><span class="small"><b>Gap to reality:</b> '
          f'{esc(e1["gap"])}<br><b>Modelled:</b> {esc(e1["modelled"])}</span></div>')
    # --- instrument 1 reservoir
    d.h2("Instrument E1.1 - Asteroid reservoir (docked to Earth)")
    d.p("<b>Use:</b> supplies the reaction mass. Earth can only change its speed by throwing mass the other way "
        "(conservation of momentum); the reservoir is finite and is used up packet by packet.")
    d.calc("Reservoir size compared with the Moon and Earth", "m<sub>ast</sub> / M<sub>Moon</sub>,  m<sub>ast</sub> / M<sub>Earth</sub>",
           f"{sci(cfg.asteroid_mass_kg)} / {sci(R.M_MOON, 4)};  {sci(cfg.asteroid_mass_kg)} / {sci(R.M_EARTH, 4)}",
           f"{sci(cfg.asteroid_mass_kg / R.M_MOON)} Moon masses; {pct(cfg.asteroid_mass_kg / R.M_EARTH, 3)} of Earth", "Config.asteroid_mass_kg")
    d.calc("Surface gravity with the asteroid attached (indicative)", "g = G (M<sub>Earth</sub> + m<sub>ast</sub>) / R<sub>Earth</sub>&sup2;", "",
           f"{sci(G * m_l / R.R_EARTH ** 2, 5)} m/s&sup2; (today {sci(R.GM_EARTH / R.R_EARTH ** 2, 5)} m/s&sup2;, +{pct(cfg.asteroid_mass_kg / R.M_EARTH, 2)})", "indicative",
           "The model treats Earth + asteroid as one body (its gravity counts in every orbit calculation).")
    d.calc("Number of full packets in the reservoir", "N = m<sub>ast</sub> / m<sub>pk,max</sub>", f"{sci(cfg.asteroid_mass_kg)} / {sci(cfg.max_packet_mass_kg)}",
           f"{sci(cfg.asteroid_mass_kg / cfg.max_packet_mass_kg)} packets = {cfg.asteroid_mass_kg / cfg.max_packet_mass_kg / cfg.ejections_per_year:,.0f} years at the maximum rate", "")
    if H:
        d.calc("Tested mission: reservoir used", "m<sub>used</sub> / m<sub>ast</sub>", f"{sci(H['mass_used_kg'])} / {sci(cfg.asteroid_mass_kg)}",
               f"{pct(H['mass_used_frac'])} in {H['n_packets']:,} packets", "summary() mass_used_frac")
    # --- instrument 2 launcher
    d.h2("Instrument E1.2 - Launch ring with 8 arms (mass driver)")
    d.p(f"<b>Use:</b> accelerates each packet to the commanded speed u (between {sci(cfg.min_ejection_speed_ms / 1e3)} and "
        f"{sci(cfg.max_ejection_speed_ms / 1e3)} km/s) in the commanded direction; {cfg.ejections_per_year:g} launches per year. All 8 "
        f"arms throw ONE packet together (group throw), so each arm carries 1/8 of the packet and of the energy.")
    fzv = R.feasibility(cfg)
    vinf_mass = fzv["dv_required"] / math.log(m_sys / (m_sys - 0.95 * cfg.asteroid_mass_kg))
    vinf_time = fzv["dv_required"] * m_sys / (0.85 * cfg.mission_years * cfg.ejections_per_year * cfg.max_packet_mass_kg)
    d.calc("Launch speed chosen by the hand-designed strategy", "v<sub>&infin;</sub> = max(&Delta;v / ln(m<sub>sys</sub>/(m<sub>sys</sub>&minus;0.95 m<sub>ast</sub>)), &Delta;v m<sub>sys</sub> / (0.85 T n m<sub>pk</sub>)); u = &radic;(v<sub>&infin;</sub>&sup2;+v<sub>esc</sub>&sup2;)",
           f"max({sci(vinf_mass, 4)}, {sci(vinf_time, 4)}) m/s", f"u = {sci(u_h, 5)} m/s = {sci(u_h / 1e3, 4)} km/s ({pct(u_h / R.C_LIGHT, 2)} of c)",
           "heuristic_speed()", "Lowest speed that uses 95 % of the asteroid and still finishes within 85 % of the mission time.")
    M_emb = m_l + R.M_MOON
    phi_moon = R.GM_MOON / R.MOON_A
    for tag, u in (("at the strategy speed", u_h), ("at the maximum speed", cfg.max_ejection_speed_ms)):
        dm = cfg.max_packet_mass_kg
        mu_red = dm * (m_l - dm) / m_l
        Ek = 0.5 * mu_red * u * u
        Ed = Ek / cfg.launch_efficiency
        vinf = math.sqrt(u * u - 2 * G * m_l / R.R_EARTH - 2 * phi_moon)
        d.h3(f"One full packet {tag} (u = {sci(u / 1e3, 4)} km/s)")
        d.calc("Kinetic energy given to packet + Earth (two-body, reduced mass)", "&mu;<sub>red</sub> = m<sub>pk</sub>(M&minus;m<sub>pk</sub>)/M;  E<sub>kin</sub> = &frac12; &mu;<sub>red</sub> u&sup2;",
               f"&mu;<sub>red</sub> = {sci(mu_red, 6)} kg; 0.5 &times; {sci(mu_red, 4)} &times; ({sci(u, 4)})&sup2;", f"{sci(Ek)} J", "eject_core()")
        d.calc("Energy drawn from the power supply", "E<sub>draw</sub> = E<sub>kin</sub> / &eta;<sub>launch</sub>", f"{sci(Ek)} / {cfg.launch_efficiency}",
               f"{sci(Ed)} J per packet, {sci(Ed / 8)} J per arm; {sci(Ed / CHICXULUB_J)} &times; a Chicxulub impact (~10<sup>24</sup> J)", "eject_core()")
        d.calc("Packet speed after leaving Earth's and the Moon's gravity", "v<sub>&infin;</sub> = &radic;(u&sup2; &minus; 2GM/R<sub>Earth</sub> &minus; 2GM<sub>Moon</sub>/d<sub>Moon</sub>)",
               f"&radic;(({sci(u, 4)})&sup2; &minus; {sci(2 * G * m_l / R.R_EARTH, 4)} &minus; {sci(2 * phi_moon, 4)})", f"{sci(vinf, 6)} m/s", "eject_core() (fast model)")
        dv = dm * vinf / M_emb
        d.calc("Recoil: speed change of Earth + Moon per packet", "&Delta;v<sub>E</sub> = m<sub>pk</sub> v<sub>&infin;</sub> / (M<sub>Earth</sub>+m<sub>ast</sub>+M<sub>Moon</sub>)",
               f"{sci(dm)} &times; {sci(vinf, 4)} / {sci(M_emb, 5)}", f"{sci(dv)} m/s (impulse {sci(dm * vinf)} N s)", "_fast_eject() / ledger LG_IMPULSE_TOTAL")
        d.calc("Full-throttle push per year and per decision", "&Delta;v<sub>yr</sub> = n &middot; &Delta;v<sub>E</sub>", f"{cfg.ejections_per_year:g} &times; {sci(dv)}",
               f"{sci(cfg.ejections_per_year * dv)} m/s per year, {sci(cfg.ejections_per_year * dv * cfg.decision_interval_years)} m/s per {cfg.decision_interval_years:.0f}-year decision; "
               f"&Delta;v needs {fz['dv_required'] / (cfg.ejections_per_year * dv):,.0f} years of full thrust", "policy_heuristic() dv_max")
        d.calc("Average launcher power at full throttle", "P = n &middot; E<sub>draw</sub> / year", f"{cfg.ejections_per_year:g} &times; {sci(Ed)} / {sci(YEAR, 5)}",
               f"{sci(cfg.ejections_per_year * Ed / YEAR)} W ({sci(cfg.ejections_per_year * Ed / YEAR / 8)} W per arm)", "")
    d.calc("Time between launches", "&Delta;t = 1 year / n", f"365.25 d / {cfg.ejections_per_year:g}", f"{365.25 / cfg.ejections_per_year:.2f} days", "fast_interval()")
    d.calc("Mass per arm in a group throw", "m<sub>arm</sub> = m<sub>pk</sub> / 8", f"{sci(cfg.max_packet_mass_kg)} / 8", f"{sci(cfg.max_packet_mass_kg / 8)} kg", "N_ARMS = 8")
    if H:
        imp = H["impulse_Ns"]
        d.calc("Tested mission: total impulse and speed change delivered", "&Delta;v = J<sub>total</sub> / M<sub>Earth+Moon+ast</sub>", f"{sci(imp)} N s / {sci(M_emb, 5)} kg",
               f"{sci(imp / M_emb, 5)} m/s (required by the spiral estimate: {sci(fz['dv_required'], 5)} m/s)", "ledger LG_IMPULSE_TOTAL",
               "Compare with the slow-spiral estimate C5: the hand-designed strategy pushes almost purely along Earth's motion, so the delivered speed change is close to the ideal value.")
        led = H["ledger"]
        d.calc("Tested mission: packet kinetic energy and launcher draw", "&eta; = E<sub>kin</sub> / E<sub>draw</sub>", f"{sci(led['launch_ke'])} / {sci(led['energy_drawn'])}",
               f"{led['launch_ke'] / led['energy_drawn']:.4f} (= launch_efficiency {cfg.launch_efficiency})", "ledger LG_LAUNCH_KE, LG_E_DRAWN")
        d.calc("Tested mission: average launcher power", "P = E<sub>draw</sub> / t", f"{sci(led['energy_drawn'])} / ({H['t_years']:,.0f} y &times; {sci(YEAR, 5)} s)",
               f"{sci(led['energy_drawn'] / (H['t_years'] * YEAR))} W", "")
    # launcher geometry (update p36)
    lg = R.launcher_geometry(cfg)
    RE_KM = R.R_EARTH / 1e3
    d.h3("Launcher geometry: asteroid on the ring axis, arms 2 Earth radii long (update p36)")
    d.p("The 8 arm roots sit on a ring around Earth (plane perpendicular to Earth's orbit normal). The docked asteroid "
        "sits on the ring's <b>axis</b>; its Earth-facing side rests on the grab point, so every arm root is exactly the "
        "same distance from it and all 8 arms reach it equally. The docking clamps lower the asteroid as it shrinks, so the "
        "grab point never moves. Each arm is 2 Earth radii long.")
    d.calc("Distance from every arm root to the grab point", "d = &radic;(&rho;<sub>ring</sub>&sup2; + z<sub>grab</sub>&sup2;) &le; L<sub>arm</sub>",
           f"&radic;({lg['ring_radius_re']}&sup2; + {lg['grab_height_re']}&sup2;) R<sub>E</sub>; L<sub>arm</sub> = {lg['arm_length_re']} R<sub>E</sub> = {lg['arm_length_re'] * RE_KM:,.0f} km",
           f"{lg['root_to_grab_re']:.4f} R<sub>E</sub> = {lg['root_to_grab_re'] * RE_KM:,.0f} km for all 8 roots (spread {lg['root_to_grab_spread_re']:.0e}); reach to spare "
           f"{(lg['arm_length_re'] - lg['root_to_grab_re']) * RE_KM:,.0f} km", "launcher_geometry()")
    d.calc("Asteroid size and position", "r<sub>ast</sub> = (3 m<sub>ast</sub> / (4&pi;&rho;))<sup>1/3</sup>;  z<sub>centre</sub> = z<sub>grab</sub> + r<sub>ast</sub>",
           f"&rho; = {R.AST_DENSITY:.0f} kg/m&sup3; (assumed), m<sub>ast</sub> = {sci(cfg.asteroid_mass_kg)} kg",
           f"r<sub>ast</sub> = {lg['asteroid_radius_re'] * RE_KM:,.0f} km; centre {lg['asteroid_centre_re']:.3f} R<sub>E</sub> from Earth's centre, "
           f"{lg['root_to_asteroid_centre_re']:.3f} R<sub>E</sub> from every root (&le; arm length); {lg['asteroid_clearance_km']:,.0f} km above the surface", "launcher_geometry()")
    d.calc("Stroke of the joint throw (all 8 arms keep hold)", "largest s with |G + s&middot;e &minus; root<sub>k</sub>| &le; L<sub>arm</sub> for all k (worst direction e in the ring plane)",
           "bisection over directions", f"s = {lg['stroke_km']:,.0f} km ({lg['stroke_re']:.3f} R<sub>E</sub>; up to {lg['stroke_max_re']:.3f} R<sub>E</sub> between two arms); "
           f"release point {lg['release_radius_re']:.3f} R<sub>E</sub> from Earth's centre", "launcher_geometry()")
    d.calc("Acceleration the arms must give a packet over that stroke", "a = u&sup2; / (2 s);  t = 2 s / u",
           f"u = {sci(lg['u_typ_ms'] / 1e3, 4)} km/s (strategy) and {sci(lg['u_max_ms'] / 1e3, 4)} km/s (maximum); s = {lg['stroke_km']:,.0f} km",
           f"{sci(lg['accel_typ_ms2'])} m/s&sup2; ({sci(lg['accel_typ_ms2'] / 9.81)} g) for {lg['stroke_time_typ_s']:.2f} s; at the maximum speed {sci(lg['accel_max_ms2'])} m/s&sup2;",
           "launcher_geometry()", "Far beyond any material: part of enabler E1's gap.")
    d.calc("Load on the docking clamps (asteroid's weight)", "F = m<sub>ast</sub> G M<sub>Earth</sub> / z<sub>centre</sub>&sup2;",
           f"{sci(cfg.asteroid_mass_kg)} &times; {lg['g_at_dock_ms2']:.3f} m/s&sup2;", f"{sci(lg['clamp_load_n'])} N", "launcher_geometry()")
    d.calc("What the physics simplifies (and why it is allowed)", "&Delta;E/E = (GM/R)(1 &minus; R/r<sub>release</sub>) / (u&sup2;/2);  COM shift = m<sub>ast</sub> z<sub>centre</sub>/(M+m<sub>ast</sub>)",
           "escape energy from Earth's surface instead of the release point; asteroid merged with Earth",
           f"surface launch needs +{sci(lg['escape_energy_saving_rel'])} more energy (conservative); centre of mass off by {lg['com_shift_km']:.0f} km; "
           f"tidal error on the Moon {sci(lg['moon_pull_error_rel'])} of Earth's pull", "launcher_geometry(), check 21")
    d.note("<b>Not modelled</b> (assumed by E1): the arms' structure and materials, the acceleration of the packets "
           "inside the arms, and the effect of each launch on Earth itself (shock, atmosphere loss, heating). Open problem P6.", "warn")
    # --- instrument 3 power conditioning + reserve
    d.h2("Instrument E1.3 - Arm power conditioning and energy reserve")
    d.p("<b>Use:</b> the arms take electricity from the rectennas (E2) first; if a launch needs more than the beams deliver in "
        "that interval, the rest comes from a finite reserve. Unused beam energy is not sent (the beams are throttled to "
        "demand); nothing is created from nothing.")
    d.calc("Reserve size compared with the minimum mission energy", "reserve / E<sub>min</sub>", f"{sci(cfg.energy_budget_j)} / {sci(fz['energy_min_draw'])}",
           f"{cfg.energy_budget_j / fz['energy_min_draw']:.3f}", "Config.energy_budget_j")
    d.calc("Energy offered by the beams in one decision interval (Earth at 1 AU)", "E<sub>beam</sub> = P<sub>elec</sub> &middot; &Delta;t<sub>decision</sub>",
           f"{sci(R.collector_power(cfg, [AU, 0, 0])['P_elec_W'])} W &times; {cfg.decision_interval_years:.0f} y", f"{sci(R.collector_power(cfg, [AU, 0, 0])['P_elec_W'] * cfg.decision_interval_years * YEAR)} J",
           "_PowerSupply._with_supply()")
    if H:
        d.calc("Tested mission: where the launch energy came from", "E<sub>draw</sub> = E<sub>beams</sub> + E<sub>reserve</sub>",
               f"{sci(H['energy_from_collectors_J'])} + {sci(H['energy_from_reserve_J'])}", f"{sci(H['energy_used_J'])} J; beams used {pct(H['beam_use_frac'])} of what they could offer",
               "summary() energy_from_collectors_J")
    # --- instrument 4 radiators
    d.h2("Instrument E1.4 - Directional arm radiators")
    d.p(f"<b>Use:</b> the launcher loses (1&minus;&eta;) = {pct(1 - cfg.launch_efficiency, 0)} of the energy it draws as heat, and "
        f"the rectennas lose (1&minus;&eta;<sub>r</sub>) = {pct(1 - cfg.rectenna_efficiency, 0)} of the microwaves they catch. The radiators send this "
        f"heat to space, away from Earth; only the share arm_heat_to_earth_frac = {sci(cfg.arm_heat_to_earth_frac)} reaches Earth.")
    if H:
        L = H["energy_ledger"]
        q = (L["rectenna_heat"] + L["launcher_heat"])
        qm = q / (H["t_years"] * YEAR)
        d.calc("Tested mission: arm waste heat", "Q = E<sub>rectenna heat</sub> + E<sub>launcher heat</sub>", f"{sci(L['rectenna_heat'])} + {sci(L['launcher_heat'])} J",
               f"{sci(q)} J = {sci(qm)} W on average = {sci(qm / R.EARTH_INTERCEPT_W)} &times; the sunlight Earth intercepts", "settle_interval()")
        d.calc("Radiator area needed at 1,000 K (indicative, black radiator, one side)", "A = Q / (&sigma; T&sup4;)", f"{sci(qm)} / ({sci(R.SIGMA_SB, 4)} &times; 1000&sup4;)",
               f"{sci(qm / (R.SIGMA_SB * 1e12))} m&sup2; in total, {sci(qm / (R.SIGMA_SB * 1e12) / 8)} m&sup2; per arm = {sci(qm / (R.SIGMA_SB * 1e12) / 8 / (math.pi * R.R_EARTH ** 2))} &times; Earth's disc",
               "summary() radiator_area_1000K_m2")
        d.calc("Heat that still reaches Earth", "q<sub>Earth</sub> = f<sub>arm</sub>&middot;Q + f<sub>spill</sub>&middot;E<sub>spill</sub>",
               f"{sci(cfg.arm_heat_to_earth_frac)} &times; {sci(q)} + {sci(cfg.beam_spill_to_earth_frac)} &times; {sci(L['beam_spill'])}",
               f"{sci(H['heat_to_earth_J'])} J = {sci(H['heat_to_earth_J'] / (H['t_years'] * YEAR))} W = {sci(H['heat_to_earth_J'] / (H['t_years'] * YEAR) / R.EARTH_INTERCEPT_W)} &times; Earth's sunlight", "_climate_step() q_w")
        d.note(f"Without the directional radiators a geometric leak of ~8 % would put {sci(0.08 * qm)} W "
               f"({sci(0.08 * qm / R.EARTH_INTERCEPT_W)} &times; Earth's sunlight) on Earth: the oceans would boil. This is why the radiators are part of enabler E1.", "bad")

    # Mars safety (control)
    d.h2("Mars safety limits (control software, not an enabler)")
    d.cur = "MARS"
    d.p("Earth's growing orbit crosses Mars's orbit. The safety shield forecasts the Earth-Mars distance and overrides the "
        "agent (coast or full thrust) when a pass would come too close. The safe distance is the largest of three limits:")
    d.calc("Tide limit: Mars's tide on Earth &le; 1 % of the Moon's", "d<sub>tide</sub> = d<sub>Moon</sub> (GM<sub>Mars</sub> / (GM<sub>Moon</sub> &middot; k<sub>tide</sub>))<sup>1/3</sup>",
           f"{sci(R.MOON_A, 4)} &times; ({sci(R.GM_MARS, 4)} / ({sci(R.GM_MOON, 4)} &times; {cfg.mars_max_tide_ratio}))<sup>1/3</sup>", f"{sci(ms['d_tide_m'] / 1e3)} km", "mars_safety_limits()")
    d.calc("Moon-orbit limit: Mars's pull on the Moon's orbit &le; 0.1 % of Earth's", "d<sub>moon</sub> = d<sub>Moon</sub> (2 GM<sub>Mars</sub> / (GM<sub>Earth</sub> k<sub>moon</sub>))<sup>1/3</sup>",
           f"k<sub>moon</sub> = {cfg.mars_max_moon_perturbation}", f"{sci(ms['d_moon_m'] / 1e3)} km", "mars_safety_limits()")
    d.calc("Safe distance and shield caution distance", "d<sub>safe</sub> = max(d<sub>tide</sub>, d<sub>moon</sub>, d<sub>min</sub>);  d<sub>caution</sub> = k<sub>c</sub> d<sub>safe</sub>",
           f"max({sci(ms['d_tide_m'] / 1e3)}, {sci(ms['d_moon_m'] / 1e3)}, {sci(ms['d_min_m'] / 1e3)}) km; k<sub>c</sub> = {cfg.shield_caution_factor}",
           f"d<sub>safe</sub> = {sci(ms['d_safe_m'] / 1e3)} km = {sci(ms['d_safe_au'])} AU (governed by: {esc(ms['governing'])}); caution {sci(ms['caution_au'])} AU", "mars_safety_limits()")
    if H:
        d.p(f"Tested mission: closest Earth-Mars pass {H['mars_min_dist_au']:.4f} AU (&ge; {H['mars_safe_au']:.4f} AU: "
            f"{'safe' if H['mars_safe'] else 'NOT safe'}), largest tide ratio {sci(H['mars_max_tide_ratio'])}, {H['shield_overrides']} shield overrides. "
            f"Mars's own orbit changed by &Delta;a = {sci(H['mars_da_au'])} AU, &Delta;e = {sci(H['mars_de'])}.")

    # Mercury, Venus, Jupiter (update p35)
    d.h2("Mercury, Venus and Jupiter (update p35; Saturn and beyond not modelled)")
    d.p("Mercury, Venus and Jupiter start on their real (J2000) orbits and are integrated in both models: they pull on Earth, "
        "the Moon (high-fidelity model), Mars and the fragments, and Earth pulls back (two-way gravity). The same three "
        "safety rules as for Mars are applied to each planet; because the tide limits scale with GM, the safe distance "
        "scales with GM<sup>1/3</sup>. Coming closer ends the mission (failure planet_unsafe_approach). The shield's "
        "forecast still uses only Sun + Earth + Mars.")
    for q in R.planet_safety_limits(cfg):
        k = R.PLANET_NAMES.index(q["name"])
        gm = [R.GM_MERCURY, R.GM_VENUS, R.GM_JUPITER][k]
        d.calc(f"Safe distance from {q['name']}", "d<sub>safe</sub> = max(d<sub>Moon</sub>(GM/(GM<sub>Moon</sub>k<sub>tide</sub>))<sup>1/3</sup>, "
               "d<sub>Moon</sub>(2GM/(GM<sub>Earth</sub>k<sub>moon</sub>))<sup>1/3</sup>, d<sub>min</sub>)",
               f"GM = {sci(gm, 4)} m&sup3;/s&sup2;; tide {sci(q['d_tide_m'] / 1e3)} km, Moon-orbit {sci(q['d_moon_m'] / 1e3)} km, minimum {sci(q['d_min_m'] / 1e3)} km",
               f"{sci(q['d_safe_m'] / 1e3)} km = {q['d_safe_au']:.4f} AU (governed by: {esc(q['governing'])})", "planet_safety_limits()")
    # their gravity, as used in both physics models
    for k, (nm, gm) in enumerate(zip(R.PLANET_NAMES, [R.GM_MERCURY, R.GM_VENUS, R.GM_JUPITER])):
        dmin = H["planets"][k]["min_dist_au"] * AU if H and H.get("planets") else None
        a_pl = [0.387, 0.723, 5.203][k]
        d_tgt = abs(a_pl - cfg.target_au) * AU
        txt = f"at the closest approach of the tested mission ({dmin / AU:.3f} AU): {sci(gm / dmin ** 2)} m/s&sup2;; " if dmin else ""
        d.calc(f"Pull of {nm} on Earth (two-way gravity in both models)", "a = GM / d&sup2; (and Earth pulls back with G M<sub>Earth</sub> / d&sup2;)",
               f"GM = {sci(gm, 4)} m&sup3;/s&sup2;",
               txt + f"at 2.7 AU with {nm} at its nearest ({d_tgt / AU:.2f} AU): {sci(gm / d_tgt ** 2)} m/s&sup2; = {sci(gm / d_tgt ** 2 / (R.GM_SUN / (cfg.target_au * AU) ** 2))} &times; the Sun's pull there",
               "_planet_kicks() (fast) / nb_accel() (N-body)")
    if H and H.get("planets"):
        rows_pl = [[q["name"], f"{q['min_dist_au']:.3f} AU (year {q['min_dist_t']:.0f})", f"{q['safe_au']:.4f} AU",
                    "safe" if q["safe"] else "NOT safe", f"{q['da_au']:+.1e} AU"] for q in H["planets"]]
        d.table(["planet", "closest approach (tested mission, N-body)", "safe distance", "verdict", "orbit change &Delta;a"], rows_pl)
        d.p("Jupiter, 318 Earth masses, dominates the effect on Earth's final orbit; the 8:3 mean-motion resonance with "
            "Jupiter lies at about 2.705 AU, very close to the 2.7 AU target, so near the target Jupiter's periodic pull "
            "slowly changes Earth's eccentricity. Over the simulated mission this stays within the tolerance; over millions "
            "of years it is NOT checked here (long-term stability would need a dedicated study).")

    # =========================================================================== 5. E2
    e2 = R.ENABLERS[1]
    d.h1("Enabler E2 - Solar power belt with microwave power beaming", "e2")
    d.raw(f'<div class="ebox"><div class="t">{esc(e2["name"])}</div>{esc(e2["provides"])}<br><span class="small"><b>Gap to reality:</b> '
          f'{esc(e2["gap"])}<br><b>Modelled:</b> {esc(e2["modelled"])}</span></div>')
    Rb = cfg.collector_orbit_au * AU
    n_c = des["count"]
    d.h2("Instrument E2.1 - Thin-film solar collectors")
    d.p(f"<b>Use:</b> intercept sunlight and turn it into electricity. The belt is a band around the Sun at "
        f"{cfg.collector_orbit_au} AU that covers the fraction f = {cfg.collector_fraction} of a full sphere, so it intercepts "
        f"exactly that fraction of all sunlight.")
    d.calc("Belt area", "A = f &middot; 4&pi; R&sup2;", f"{cfg.collector_fraction} &times; 4&pi; &times; ({sci(Rb, 4)})&sup2;", f"{sci(des['area_m2'])} m&sup2; = {sci(des['area_m2'] / (4 * math.pi * R.R_EARTH ** 2))} &times; Earth's surface", "collector_design()")
    d.calc("Band half-width (latitude seen from the Sun)", "&beta;<sub>band</sub> = arcsin f", f"arcsin {cfg.collector_fraction}", f"{des['half_width_deg']:.2f}&deg;", "collector_design()")
    d.calc("Size of one collector", "A<sub>1</sub> = A / N;  side = &radic;A<sub>1</sub>", f"{sci(des['area_m2'])} / {sci(n_c)}",
           f"{sci(des['each_area_m2'])} m&sup2;, about {des['each_side_km']:,.0f} km &times; {des['each_side_km']:,.0f} km", "collector_design()")
    d.calc("Mass of the belt and of one collector", "m = A &middot; &sigma;<sub>areal</sub>", f"{sci(des['area_m2'])} &times; {cfg.collector_areal_density} kg/m&sup2;",
           f"{sci(des['mass_kg'])} kg ({sci(des['mass_moon_frac'])} Moon masses); one collector {sci(des['mass_kg'] / n_c)} kg", "collector_design()")
    S = op["S_wm2"]
    d.calc("Sunlight at the belt", "S = S<sub>1AU</sub> / r&sup2;", f"{R.S_1AU} / {cfg.collector_orbit_au}&sup2;", f"{sci(S, 5)} W/m&sup2;", "collector_operating()")
    d.calc("Sunlight intercepted (whole belt built)", "L<sub>cap</sub> = f &middot; L<sub>Sun</sub>", f"{cfg.collector_fraction} &times; {sci(R.L_SUN, 4)}", f"{sci(cfg.collector_fraction * R.L_SUN)} W", "collector_power()")
    d.calc("Panel temperature (energy balance, both faces radiate)", "T = [(a S &minus; &eta;(T) S) / (2 &epsilon; &sigma;)]<sup>1/4</sup>,  &eta;(T) = &eta;<sub>25&deg;C</sub>(1 &minus; k (T &minus; 298.15 K))",
           f"a = {cfg.collector_absorptance}, &epsilon; = {cfg.collector_emissivity}, &eta;<sub>25&deg;C</sub> = {cfg.pv_efficiency}, k = {cfg.pv_temp_coeff}/K (solved by iteration)",
           f"T = {op['T_panel_K']:.1f} K, &eta;(T) = {pct(op['eta_temp'], 2)} (&times;{op['temp_factor']:.3f}); idle panel (no electricity) {op['T_idle_K']:.1f} K", "collector_operating()")
    d.calc("Ageing of the fleet (panels replaced at end of life)", "f<sub>age</sub> = 1 &minus; d &middot; L<sub>life</sub> / 2", f"1 &minus; {cfg.pv_degradation_per_year} &times; {cfg.collector_lifetime_years:.0f} / 2",
           f"{op['degr_factor']:.3f}", "collector_operating()")
    d.calc("Operating solar-cell efficiency", "&eta;<sub>op</sub> = &eta;(T) &middot; f<sub>age</sub>", f"{op['eta_temp']:.4f} &times; {op['degr_factor']:.3f}", f"{pct(op['eta_op'], 2)}", "collector_operating()")
    pw1 = R.collector_power(cfg, [AU, 0, 0])
    pwt = R.collector_power(cfg, [cfg.target_au * AU, 0, 0])
    pw0 = R.collector_power(cfg, [AU, 0, 0], R.collector_built(cfg, 0.0))
    d.calc("Panel electricity (whole belt, clear lanes subtracted)", "P<sub>pv</sub> = f L<sub>Sun</sub> (1 &minus; lanes) &eta;<sub>op</sub>",
           f"{sci(cfg.collector_fraction * R.L_SUN)} &times; (1 &minus; {sci(pw1['lane_frac'])}) &times; {op['eta_op']:.4f}", f"{sci(pw1['P_pv_W'])} W; one collector {sci(pw1['P_pv_W'] / n_c)} W", "collector_power()")
    d.calc("Light-pressure ratio of a collector (push of sunlight / pull of the Sun)", "&beta; = L<sub>Sun</sub> / (4&pi; c GM<sub>Sun</sub> &sigma;<sub>areal</sub>)",
           f"{sci(R.L_SUN, 4)} / (4&pi; &times; {sci(R.C_LIGHT, 4)} &times; {sci(R.GM_SUN, 4)} &times; {cfg.collector_areal_density})", f"{sci(des['light_pressure_ratio'])} ({pct(des['light_pressure_ratio'], 2)})", "collector_design()")
    d.calc("Orbital period of a collector", "P = r<sup>3/2</sup> / &radic;(1 &minus; &beta;) years", f"{cfg.collector_orbit_au}<sup>1.5</sup> / &radic;(1 &minus; {des['light_pressure_ratio']:.4f})",
           f"{des['orbit_period_yr']:.4f} years = {des['orbit_period_yr'] * 365.25:.1f} days", "collector_design()")
    bp = des["belt_pull"]
    d.calc("Gravity of the whole belt on Earth (vs the Sun's pull)", "a = &Sigma; G m<sub>i</sub> d<sub>i</sub>/|d<sub>i</sub>|&sup3; over 6,480 belt points", "",
           f"{sci(bp['ratio_1au'])} of the Sun's pull at 1 AU, {sci(bp['ratio_target'])} at {cfg.target_au} AU (negligible)", "belt_pull_on_earth()")

    d.h2("Instrument E2.2 - Microwave phased-array transmitters")
    d.p(f"<b>Use:</b> each collector turns its electricity into a {cfg.microwave_freq_ghz} GHz beam with a "
        f"{cfg.transmitter_diameter_km:g} km phased array and points it at the arms. Efficiency {pct(cfg.microwave_efficiency, 0)}; "
        f"the beams are throttled to what the launcher needs.")
    lam = des["wavelength_m"]
    d.calc("Wavelength", "&lambda; = c / f", f"{sci(R.C_LIGHT, 6)} / {sci(cfg.microwave_freq_ghz * 1e9)}", f"{sci(lam * 100, 4)} cm", "beam_capture_efficiency()")
    d.calc("Microwave power sent (whole belt, full output)", "P<sub>mw</sub> = P<sub>pv</sub> &eta;<sub>mw</sub>", f"{sci(pw1['P_pv_W'])} &times; {cfg.microwave_efficiency}",
           f"{sci(pw1['P_mw_W'])} W; per transmitter {sci(pw1['P_mw_W'] / n_c)} W", "collector_power()")
    a_tx = math.pi * (cfg.transmitter_diameter_km * 500) ** 2
    d.calc("Intensity on a transmitter aperture at full output (indicative)", "I = (P<sub>mw</sub>/N) / (&pi; (D<sub>t</sub>/2)&sup2;)", f"{sci(pw1['P_mw_W'] / n_c)} / {sci(a_tx)} m&sup2;",
           f"{sci(pw1['P_mw_W'] / n_c / a_tx)} W/m&sup2; = {sci(pw1['P_mw_W'] / n_c / a_tx / R.S_1AU)} &times; sunlight at Earth", "indicative")
    for dd in (pw1["d_min_au"], pw1["d_max_au"]):
        tau = cfg.transmitter_diameter_km * 1e3 * cfg.rectenna_diameter_km * 1e3 / (lam * dd * AU)
        d.calc(f"Beam capture at {dd:.2f} AU (Goubau relation)", "&tau; = D<sub>t</sub> D<sub>r</sub> / (&lambda; d);  &eta;<sub>cap</sub> = 1 &minus; e<sup>&minus;&tau;&sup2;</sup>",
               f"&tau; = {sci(cfg.transmitter_diameter_km * 1e3)} &times; {sci(cfg.rectenna_diameter_km * 1e3)} / ({sci(lam, 4)} &times; {sci(dd * AU, 4)}) = {tau:.3f}",
               f"&eta;<sub>cap</sub> = {pct(1 - math.exp(-tau * tau), 2)}", "beam_capture_efficiency()")
        d.calc(f"Central beam spot diameter at {dd:.2f} AU (indicative, Airy)", "D<sub>spot</sub> &asymp; 2.44 &lambda; d / D<sub>t</sub>", f"2.44 &times; {sci(lam, 4)} &times; {sci(dd * AU, 4)} / {sci(cfg.transmitter_diameter_km * 1e3)}",
               f"{sci(2.44 * lam * dd * AU / (cfg.transmitter_diameter_km * 1e3) / 1e3)} km (rectenna {cfg.rectenna_diameter_km:g} km)", "indicative")
    d.calc("Average capture over the whole belt, Earth at 1 AU / at the target", "&eta;&#772;<sub>cap</sub> = mean over 6,480 belt points (beams within 2 R<sub>Sun</sub> of the Sun blocked)",
           f"blocked: {pct(pw1['blocked_frac'])} / {pct(pwt['blocked_frac'])} of the belt", f"{pct(pw1['eta_capture'], 2)} / {pct(pwt['eta_capture'], 2)}", "collector_power()")
    d.calc("Pointing accuracy needed (indicative)", "&delta;&theta; &asymp; (D<sub>r</sub>/2) / d<sub>max</sub>", f"{sci(cfg.rectenna_diameter_km * 500)} m / {sci(pwt['d_max_au'] * AU, 4)} m",
           f"{sci(cfg.rectenna_diameter_km * 500 / (pwt['d_max_au'] * AU))} rad", "indicative")
    if H:
        L = H["energy_ledger"]
        d.calc("Tested mission: transmitter waste heat and beam spill", "Q<sub>tx</sub> = E<sub>el</sub>(1&minus;&eta;<sub>mw</sub>);  E<sub>spill</sub> = E<sub>mw</sub>(1&minus;&eta;<sub>cap</sub>)",
               f"{sci(L['panel_electricity'])} &times; {1 - cfg.microwave_efficiency:.2f};  {sci(L['microwaves_sent'])} &minus; {sci(L['captured'])}",
               f"Q<sub>tx</sub> = {sci(L['transmitter_heat'])} J; spill {sci(L['beam_spill'])} J ({pct(L['beam_spill'] / L['microwaves_sent'], 2)}), of which {sci(cfg.beam_spill_to_earth_frac)} reaches Earth", "settle_interval()")

    d.h2("Instrument E2.3 - Rectennas on the 8 arm tips")
    d.p(f"<b>Use:</b> receive the microwaves (only the arms receive power) and turn them back into electricity with "
        f"{pct(cfg.rectenna_efficiency, 0)} efficiency. Diameter {cfg.rectenna_diameter_km:g} km each.")
    a_r = des["rectenna_area_m2"]
    d.calc("Area of one rectenna", "A<sub>r</sub> = &pi; (D<sub>r</sub>/2)&sup2;", f"&pi; &times; ({sci(cfg.rectenna_diameter_km * 500)})&sup2;", f"{sci(a_r)} m&sup2; ({sci(8 * a_r / (math.pi * R.R_EARTH ** 2))} &times; Earth's disc for all 8)", "collector_design()")
    d.calc("Microwaves arriving (full output), Earth at 1 AU / at the target", "P<sub>in</sub> = P<sub>mw</sub> &eta;&#772;<sub>cap</sub>", f"{sci(pw1['P_mw_W'])} &times; {pw1['eta_capture']:.4f};  {sci(pwt['P_mw_W'])} &times; {pwt['eta_capture']:.4f}",
           f"{sci(pw1['P_incident_W'])} W / {sci(pwt['P_incident_W'])} W", "collector_power()")
    d.calc("Electricity delivered to the arms (full output)", "P<sub>elec</sub> = P<sub>in</sub> &eta;<sub>r</sub>", f"{sci(pw1['P_incident_W'])} &times; {cfg.rectenna_efficiency}",
           f"{sci(pw1['P_elec_W'])} W at 1 AU, {sci(pwt['P_elec_W'])} W at {cfg.target_au} AU; at year 0 (only {pct(R.collector_built(cfg, 0), 0)} built) {sci(pw0['P_elec_W'])} W", "collector_power()")
    d.calc("Supply vs the launcher's peak demand", "P<sub>elec,min</sub> / P<sub>peak</sub>", f"{sci(fz['solar_power_min'])} / {sci(fz['launcher_peak_power'])}", f"{fz['solar_power_min'] / fz['launcher_peak_power']:.2f} &times;", "feasibility()")
    d.calc("Beam intensity on a rectenna at full output (Earth at 1 AU)", "I = P<sub>in</sub> / (8 A<sub>r</sub>)", f"{sci(pw1['P_incident_W'])} / (8 &times; {sci(a_r)})",
           f"{sci(des['beam_intensity_wm2_1au'])} W/m&sup2; = {sci(des['beam_intensity_wm2_1au'] / R.S_1AU)} &times; sunlight", "collector_design()")
    if H:
        bm = H["beam_mean_W"]
        d.calc("Tested mission: average and peak beam power received", "P&#772; = E<sub>captured</sub> / t", f"{sci(H['beam_energy_absorbed_J'])} J / ({H['t_years']:,.0f} y)",
               f"mean {sci(bm)} W ({sci(bm / 8)} W per arm, {sci(bm / 8 / a_r)} W/m&sup2;); peak {sci(H['beam_peak_W'])} W", "summary() beam_mean_W")
        d.calc("Tested mission: rectenna heat", "Q<sub>r</sub> = E<sub>captured</sub> (1 &minus; &eta;<sub>r</sub>)", f"{sci(L['captured'])} &times; {1 - cfg.rectenna_efficiency:.2f}",
               f"{sci(L['rectenna_heat'])} J (removed by the E1 radiators)", "settle_interval()")
        d.calc("Tested mission: push of the absorbed microwaves on Earth (indicative side effect)", "&Delta;v = E<sub>in</sub> / (c M)", "", f"{sci(H['beam_push_dv_ms'])} m/s (counted in the report, not applied)", "summary() beam_push_dv_ms")

    d.h2("Instrument E2.4 - Sail-tilt station keeping")
    d.p(f"<b>Use:</b> each collector tilts slightly so the sunlight push corrects its orbit (no propellant). It may use up to "
        f"collector_sk_authority = {cfg.collector_sk_authority} of its light push. 12 collectors are simulated with it and the "
        f"same 12 without it (RK4, half-day steps) along Earth's and Mars's simulated motion.")
    bk = R.BeltKeeper(cfg)
    g0 = R.GM_SUN / Rb ** 2
    d.calc("Sun's pull at the belt", "g = GM<sub>Sun</sub> / R&sup2;", f"{sci(R.GM_SUN, 5)} / ({sci(Rb, 4)})&sup2;", f"{sci(g0)} m/s&sup2;", "BeltKeeper")
    d.calc("Largest control acceleration", "a<sub>ctrl</sub> = k<sub>sk</sub> &beta; GM<sub>Sun</sub> / R&sup2;", f"{cfg.collector_sk_authority} &times; {des['light_pressure_ratio']:.5f} &times; {sci(g0)}",
           f"{sci(bk.auth / Rb ** 2)} m/s&sup2; = {sci(bk.auth / Rb ** 2 / g0)} of the Sun's pull", "BeltKeeper.auth")
    d.calc("Controller speed (PD gains)", "&omega;<sub>c</sub> = 4 n,  n = &radic;(GM<sub>Sun</sub>(1&minus;&beta;)/R&sup3;)", f"n = {sci(bk.n)} rad/s", f"&omega;<sub>c</sub> = {sci(bk.wc)} rad/s (time constant {1 / bk.wc / 86400:.1f} days)", "BeltKeeper.wc")
    rec_full = pw1["P_mw_W"] / n_c / (R.C_LIGHT * des["mass_kg"] / n_c)
    d.calc("Beam recoil on a collector (full output)", "a<sub>rec</sub> = (P<sub>mw</sub>/N) / (c m<sub>1</sub>)", f"{sci(pw1['P_mw_W'] / n_c)} / ({sci(R.C_LIGHT, 4)} &times; {sci(des['mass_kg'] / n_c)})",
           f"{sci(rec_full)} m/s&sup2; = {sci(rec_full / g0)} of the Sun's pull", "BeltKeeper.advance() a_rec")
    d.calc("Pull of a neighbouring collector", "a = G m<sub>1</sub> / s&sup2;", f"{sci(G, 5)} &times; {sci(des['mass_kg'] / n_c)} / ({sci(bk.spacing, 4)})&sup2;", f"{sci(bk.neighbour_ratio)} of the Sun's pull", "BeltKeeper.neighbour_ratio")
    if H and H.get("belt"):
        b = H["belt"]
        d.table(["Tested mission (" + f"{b['years']:,.0f}" + " years)", "WITH station keeping", "WITHOUT"],
                [["largest distance from the slot", f"{b['on']['max_km']:,.0f} km", f"{sci(b['off']['max_km'])} km"],
                 ["... in collector spacings (" + f"{b['spacing_km']:,.0f}" + " km)", f"{b['on']['slots']:.4f}", f"{b['off']['slots']:,.0f}"],
                 ["orbit size change |&Delta;a|", f"{b['on']['da_km']:,.0f} km", f"{b['off']['da_km']:,.0f} km"],
                 ["largest eccentricity", sci(b['on']['e_max']), sci(b['off']['e_max'])],
                 ["control effort", f"{b['on']['dv_ms_per_yr']:.1f} m/s per year", "0"]])
        d.p(f"Largest disturbances (fraction of the Sun's pull): beam recoil {sci(b['recoil_ratio_max'])}, planets {sci(b['planet_ratio'])}, "
            f"neighbours {sci(b['neighbour_ratio'])}; the controller can correct up to {sci(b['control_ratio'])}.")

    d.h2("Instrument E2.5 - Clear sunlight lane")
    d.p("<b>Use:</b> the collectors leave a moving gap on the line Sun-Earth (and Sun-Mars) so that Earth sees the whole solar disc.")
    d.calc("Share of the belt that must stay empty for Earth", "lane = &rho;&sup2; / (4 R&sup2; f),  &rho; = R<sub>Sun</sub>(r&minus;R)/r + R<sub>Earth</sub> R/r", "r = 1 AU",
           f"{sci(des['lane_frac_earth'])} (Mars: {sci(des['lane_frac_mars'])})", "lane_fraction()")
    d.calc("Power given up for the lanes", "&Delta;P = L<sub>cap</sub> &times; lanes &times; &eta;<sub>op</sub>", f"{sci(cfg.collector_fraction * R.L_SUN)} &times; {sci(pw1['lane_frac'])} &times; {op['eta_op']:.4f}",
           f"{sci(cfg.collector_fraction * R.L_SUN * pw1['lane_frac'] * op['eta_op'])} W of electricity", "collector_power()")

    d.h2("Instrument E2.6 - Collector factories (Mercury material)")
    d.p(f"<b>Use:</b> {pct(cfg.collector_initial_built, 0)} of the belt exists at year 0; factories finish it in "
        f"{cfg.collector_build_years:.0f} years and replace panels after {cfg.collector_lifetime_years:.0f} years "
        f"({pct(cfg.collector_recycle_frac, 0)} recycled). Manufacturing energy comes from the collectors' own output.")
    d.calc("Built share over time", "b(t) = b<sub>0</sub> + (1 &minus; b<sub>0</sub>) t / T<sub>build</sub> (max 1)", f"b<sub>0</sub> = {cfg.collector_initial_built}, T<sub>build</sub> = {cfg.collector_build_years:.0f} y",
           f"b(100 y) = {pct(R.collector_built(cfg, 100))}, b(200 y) = {pct(R.collector_built(cfg, 200))}, complete at {cfg.collector_build_years:.0f} y", "collector_built()")
    d.calc("Replacement mass per year (whole belt)", "&#7745; = m<sub>belt</sub> / L<sub>life</sub>", f"{sci(des['mass_kg'])} / {cfg.collector_lifetime_years:.0f}", f"{sci(des['replace_kg_per_yr'])} kg per year", "collector_design()")
    d.calc("Fresh material over the mission", "m<sub>fresh</sub> = m<sub>belt</sub> + &#7745; (1 &minus; recycle) T", f"{sci(des['mass_kg'])} + {sci(des['replace_kg_per_yr'])} &times; {1 - cfg.collector_recycle_frac:.2f} &times; {cfg.mission_years:,.0f}",
           f"{sci(des['fresh_mass_total_kg'])} kg = {pct(des['fresh_mass_mercury_frac'])} of Mercury's mass", "collector_design()")
    d.calc("Manufacturing energy over the mission", "E = e<sub>emb</sub> (m<sub>belt</sub>(1 &minus; b<sub>0</sub>) + &#7745; T)", f"e<sub>emb</sub> = {sci(cfg.collector_embodied_energy_jkg)} J/kg",
           f"{sci(des['build_energy_J'])} J = {sci(des['build_energy_J'] / (pw1['P_pv_W'] * YEAR))} years of the belt's full electricity", "collector_design()")
    if H:
        d.p(f"Tested mission: manufacturing energy charged {sci(L['manufacturing'])} J; panels built {sci(L['built_mass_kg'])} kg, "
            f"fresh material {sci(L['fresh_mass_kg'])} kg.")

    # =========================================================================== 6. E3
    e3 = R.ENABLERS[2]
    d.h1("Enabler E3 - Space mirror ring (soletta) and the climate", "e3")
    d.raw(f'<div class="ebox"><div class="t">{esc(e3["name"])}</div>{esc(e3["provides"])}<br><span class="small"><b>Gap to reality:</b> '
          f'{esc(e3["gap"])}<br><b>Modelled:</b> {esc(e3["modelled"])}</span></div>')
    d.h2("Instrument E3.1 - Mirror film panels")
    d.p(f"<b>Use:</b> reflect sunlight onto Earth. Aluminium film, {cfg.mirror_areal_density * 1000:g} g/m&sup2;, reflectivity "
        f"{pct(cfg.mirror_reflectivity, 0)}. Mirrors that are not needed are turned so their light passes Earth.")
    d.calc("Mirror mass", "m = A &sigma;<sub>areal</sub>", f"{sci(cfg.mirror_area_m2)} &times; {cfg.mirror_areal_density}", f"{sci(md1['mass_kg'])} kg", "mirror_design()")
    d.calc("Area compared with Earth's disc; equivalent single disc", "A / (&pi; R<sub>Earth</sub>&sup2;);  r = &radic;(A/&pi;)", f"{sci(cfg.mirror_area_m2)} / {sci(math.pi * R.R_EARTH ** 2)}",
           f"{cfg.mirror_area_m2 / (math.pi * R.R_EARTH ** 2):.2f} &times;; one disc of radius {md1['disc_radius_km']:,.0f} km, or {cfg.mirror_area_m2 / (math.pi * 20e3 ** 2):,.0f} mirrors of 40 km", "mirror_design()")
    d.calc("Area facing the Sun (45&deg; tilt)", "A<sub>eff</sub> = A cos 45&deg;", f"{sci(cfg.mirror_area_m2)} &times; {R.MIRROR_TILT_COS:.4f}", f"{sci(md1['area_eff_m2'])} m&sup2;", "mirror_design()")
    d.calc("Largest light the mirrors can send to Earth, Earth at 1 AU", "P<sub>cap</sub> = A<sub>eff</sub> &rho; S<sub>1AU</sub> / r&sup2;", f"{sci(md1['area_eff_m2'])} &times; {cfg.mirror_reflectivity} &times; {R.S_1AU} / 1&sup2;",
           f"{sci(md1['cap_W'])} W = {md1['cap_rel']:.3f} &times; the sunlight Earth intercepts ({sci(R.EARTH_INTERCEPT_W)} W)", "mirror_design()")
    d.calc(f"... Earth at the target ({cfg.target_au} AU)", "P<sub>cap</sub> = A<sub>eff</sub> &rho; S<sub>1AU</sub> / a<sub>t</sub>&sup2;", f"{sci(md1['area_eff_m2'])} &times; {cfg.mirror_reflectivity} &times; {R.S_1AU} / {cfg.target_au}&sup2;",
           f"{sci(mdt['cap_W'])} W = {mdt['cap_rel']:.3f} &times; today's sunlight on Earth", "mirror_design()")
    s_t = 1 / cfg.target_au ** 2
    need = (1 - s_t) * R.EARTH_INTERCEPT_W
    d.calc(f"Mirror light needed at {cfg.target_au} AU to restore today's sunlight", "P<sub>need</sub> = (1 &minus; 1/a<sub>t</sub>&sup2;) S<sub>1AU</sub> &pi; R<sub>Earth</sub>&sup2;", f"(1 &minus; {s_t:.4f}) &times; {sci(R.EARTH_INTERCEPT_W)}",
           f"{sci(need)} W = {need / mdt['cap_W']:.3f} of the mirror capacity (margin {pct(mdt['cap_W'] / need - 1)})", "climate_integrate()")
    d.calc("Heat absorbed by the mirrors", "Q = (1 &minus; &rho;) &times; sunlight on the mirrors", f"1 &minus; {cfg.mirror_reflectivity}", f"{pct(1 - cfg.mirror_reflectivity, 0)} of the sunlight they catch (radiated to space)", "settle_interval()")
    a_lp = 2 * R.S_1AU / cfg.target_au ** 2 * R.MIRROR_TILT_COS ** 2 / R.C_LIGHT / cfg.mirror_areal_density
    r_o = cfg.mirror_orbit_re * R.R_EARTH
    d.calc(f"Light pressure on a mirror at {cfg.target_au} AU (indicative, NOT simulated)", "a = 2 S cos&sup2;45&deg; / (c &sigma;<sub>areal</sub>)", f"2 &times; {sci(R.S_1AU / cfg.target_au ** 2, 4)} &times; 0.5 / ({sci(R.C_LIGHT, 4)} &times; {cfg.mirror_areal_density})",
           f"{sci(a_lp)} m/s&sup2; = {pct(a_lp / (R.GM_EARTH / r_o ** 2), 2)} of Earth's gravity at the mirror orbit", "indicative",
           "Small but not negligible: the mirrors need active orbit and attitude control (part of E3, not simulated).")
    d.h2("Instrument E3.2 - Terminator orbit and pointing")
    d.p(f"<b>Use:</b> the ring orbits Earth at {cfg.mirror_orbit_re:g} Earth radii in a plane at right angles to the sunlight, "
        f"so each mirror always sees the Sun at 90&deg; from Earth and is tilted 45&deg;. The agent's action a4 sets how much "
        f"light is aimed at Earth.")
    d.calc("Orbit radius and period", "r = k R<sub>Earth</sub>;  P = 2&pi; &radic;(r&sup3;/(G M<sub>Earth</sub>))", f"r = {cfg.mirror_orbit_re:g} &times; {sci(R.R_EARTH, 4)} = {sci(r_o, 4)} m",
           f"P = {md1['orbit_days']:.2f} days", "mirror_design()")
    d.calc("Size of the light spot on Earth from one mirror (Sun's angular size)", "D = &theta;<sub>Sun</sub> r,  &theta;<sub>Sun</sub> = 2 R<sub>Sun</sub> / (a AU)", f"&theta;<sub>Sun</sub> at {cfg.target_au} AU = {sci(2 * R.R_SUN / (cfg.target_au * AU))} rad",
           f"{mdt['spot_km']:,.0f} km at {cfg.target_au} AU ({md1['spot_km']:,.0f} km at 1 AU)", "mirror_design()")
    d.calc("Pointing accuracy to keep the spot on Earth (indicative)", "&delta;&theta; &asymp; R<sub>Earth</sub> / r", f"{sci(R.R_EARTH, 4)} / {sci(r_o, 4)}", f"{sci(R.R_EARTH / r_o)} rad = {math.degrees(R.R_EARTH / r_o):.2f}&deg; (a mirror turns by half that)", "indicative")

    d.h2("Climate model (what the mirrors must achieve)")
    d.cur = "CLIMATE"
    d.p("Two-box energy balance: a surface layer (atmosphere + ocean mixed layer) exchanging heat with the deep ocean; "
        "Earth's reflectivity rises from 0.30 to 0.60 as it freezes (snowball risk).")
    d.calc("Effective emissivity (fixed greenhouse, 288 K today)", "&epsilon;<sub>eff</sub> = S<sub>1AU</sub>(1&minus;&alpha;<sub>warm</sub>)/4 / (&sigma; 288&sup4;)", f"{R.S_1AU} &times; 0.70 / 4 / ({sci(R.SIGMA_SB, 4)} &times; 288&sup4;)",
           f"{R.CLIM_EPS:.4f}", "CLIM_EPS")
    d.calc("Surface temperature equation", "C<sub>s</sub> dT<sub>s</sub>/dt = S(1&minus;&alpha;(T<sub>s</sub>))/4 + q<sub>w</sub>/(4&pi;R&sup2;) &minus; &epsilon;&sigma;T<sub>s</sub>&sup4; &minus; &gamma;(T<sub>s</sub>&minus;T<sub>d</sub>);  C<sub>d</sub> dT<sub>d</sub>/dt = &gamma;(T<sub>s</sub>&minus;T<sub>d</sub>)",
           f"C<sub>s</sub> = {R.CLIM_C_S} W yr m<sup>&minus;2</sup>K<sup>&minus;1</sup>, C<sub>d</sub> = {R.CLIM_C_D}, &gamma; = {R.CLIM_GAMMA} W m<sup>&minus;2</sup>K<sup>&minus;1</sup>, step &le; {R.CLIM_DT_YR} y",
           f"surface response time &asymp; C<sub>s</sub>/(4&epsilon;&sigma;T&sup3; + &gamma;) = {R.CLIM_C_S / (4 * R.CLIM_EPS * R.SIGMA_SB * 288 ** 3 + R.CLIM_GAMMA):.1f} years", "climate_integrate()")
    T_no = R.climate_equilibrium(s_t)
    T_full = R.climate_equilibrium(s_t + mdt["cap_rel"])
    d.calc(f"Equilibrium at {cfg.target_au} AU WITHOUT mirrors", "solve S(1&minus;&alpha;(T))/4 = &epsilon;&sigma;T&sup4;,  S = S<sub>1AU</sub>/a<sub>t</sub>&sup2;", f"S = {sci(R.S_1AU * s_t, 4)} W/m&sup2; ({s_t:.4f} &times; today)",
           f"{T_no:.1f} K ({T_no - 273.15:.0f} &deg;C): {esc(R.life_indicators(T_no)['state'])}", "climate_equilibrium()")
    d.calc(f"Equilibrium at {cfg.target_au} AU with ALL mirror light on Earth", "S = S<sub>1AU</sub>(1/a<sub>t</sub>&sup2; + P<sub>cap</sub>/(S<sub>1AU</sub>&pi;R&sup2;))", f"{s_t:.4f} + {mdt['cap_rel']:.4f} = {s_t + mdt['cap_rel']:.4f} &times; today",
           f"{T_full:.1f} K &mdash; the agent aims less light (a set-point) to stay at {cfg.climate_target_K:.0f} K", "climate_equilibrium()")
    li = R.life_indicators(cfg.climate_target_K)
    d.calc("Life indicators at the target temperature", "T(lat) = T&#772; + &Delta;T/3 &minus; &Delta;T sin&sup2;(lat), &Delta;T = 45 K; ice where T &lt; 263 K; growth for 278-313 K", f"T&#772; = {cfg.climate_target_K:.0f} K",
           f"equator {li['T_equator_K']:.0f} K, pole {li['T_pole_K']:.0f} K, ice line {li['ice_line_deg']:.1f}&deg;, growing area {pct(li['grow_frac'])}: {esc(li['state'])}", "life_indicators()")
    if H:
        L = H["energy_ledger"]
        d.calc("Tested mission: mirror light", "sunlight on mirrors = to Earth + turned past Earth + heat", f"{sci(L['mirror_sun'])} = {sci(L['mirror_to_earth'])} + {sci(L['mirror_to_space'])} + {sci(L['mirror_heat'])} J",
               f"surface {H['T_surface_K']:.1f} K (largest deviation {H['climate_dev_max_K']:.1f} K, {H['years_outside_livable']:.0f} years outside the livable band); without mirrors it would be {H['T_no_mirror_K']:.0f} K", "settle_interval(), summary()")

    # =========================================================================== 7. ledger
    d.h1("Energy and mass ledger of the tested mission", "ledger")
    d.cur = "LEDGER"
    if H:
        L = H["energy_ledger"]
        rows_l = [["IN: sunlight on the collectors", sci(L["sun_on_collectors"])], ["IN: reserve used", sci(L["reserve_used"])],
                  ["IN: sunlight on the mirrors", sci(L["mirror_sun"])],
                  ["reflected by the panels", sci(L["reflected"])], ["collector heat", sci(L["collector_heat"])],
                  ["manufacturing (new + replaced panels)", sci(L["manufacturing"])], ["panel electricity used", sci(L["panel_electricity"])],
                  ["transmitter heat", sci(L["transmitter_heat"])], ["microwaves sent", sci(L["microwaves_sent"])], ["beam spill (missed the rectennas)", sci(L["beam_spill"])],
                  ["captured by the rectennas", sci(L["captured"])], ["rectenna heat", sci(L["rectenna_heat"])], ["arm electricity", sci(L["arm_electricity"])],
                  ["launcher draw", sci(L["launcher_draw"])], ["packet kinetic energy", sci(L["packet_ke"])], ["launcher heat", sci(L["launcher_heat"])],
                  ["mirror light onto Earth", sci(L["mirror_to_earth"])], ["mirror light turned past Earth", sci(L["mirror_to_space"])], ["mirror heat", sci(L["mirror_heat"])],
                  ["waste heat reaching Earth", sci(L["heat_to_earth"])]]
        d.table(["Flow (J)", "Hand-designed mission, high fidelity"], rows_l, "kv")
        e_in = L["sun_on_collectors"] + L["reserve_used"] + L["mirror_sun"]
        e_out = sum(L[k] for k in ("reflected", "collector_heat", "manufacturing", "transmitter_heat", "beam_spill", "rectenna_heat", "packet_ke",
                                   "launcher_heat", "mirror_to_earth", "mirror_to_space", "mirror_heat"))
        d.calc("Energy balance", "IN = reflected + collector heat + manufacturing + transmitter heat + spill + rectenna heat + packet KE + launcher heat + mirror flows",
               f"{sci(e_in, 6)} vs {sci(e_out, 6)} J", f"relative residual {sci(abs(e_in - e_out) / e_in)} (round-off level: no energy created or lost)", "settle_interval() residual_rel")
        chain = L["arm_electricity"] / L["panel_electricity"]
        d.calc("Efficiency of the beaming chain (panel electricity to arm electricity)", "&eta; = &eta;<sub>mw</sub> &eta;&#772;<sub>cap</sub> &eta;<sub>r</sub>", f"{sci(L['arm_electricity'])} / {sci(L['panel_electricity'])}",
               f"{pct(chain)}", "settle_interval()")
        d.calc("Sunlight-to-packet efficiency", "E<sub>packet KE</sub> / E<sub>sun on used panels</sub>", f"{sci(L['packet_ke'])} / ({sci(L['panel_electricity'])} / {op['eta_op']:.4f})",
               f"{pct(L['packet_ke'] / (L['panel_electricity'] / op['eta_op']), 2)}", "")
        c = H["conservation"]
        led = H["ledger"]
        d.table(["Mechanical conservation (N-body, whole mission)", "Relative residual"],
                [["energy (system vs launches, escapes, impacts)", sci(c["energy_residual_rel"])], ["momentum", sci(c["momentum_residual_rel"])],
                 ["angular momentum", sci(c["angmom_residual_rel"])], ["mass", sci(c["mass_residual_rel"])]], "kv")
        d.p(f"Mass: {sci(led['mass_ejected'])} kg ejected in {led['packets']:,} packets; {sci(led['escaped_mass'])} kg left the Solar System; "
            f"{sci(led['moon_impact_mass'])} kg hit the Moon; {sci(led['earth_impact_mass'])} kg fell back to Earth; {sci(led['mars_impact_mass'])} kg hit Mars.")
    else:
        d.p("No tested results found (run training + comparison first, or use --results example_results).")

    # =========================================================================== 8. results
    d.h1("Tested results and the control system", "results")
    d.cur = "CONTROL"
    if rows:
        def g(r, k, f=lambda v: sci(v)):
            v = (r.get("hifi") or {}).get(k)
            return f(v) if v is not None else "N/A"
        pol = [rows[k] for k in ("none", "heuristic", "ppo") if k in rows]
        d.table(["High-fidelity mission (seed 2001)"] + [esc(r["label"]) for r in pol], [
            ["result"] + [esc((r.get("hifi") or {}).get("reason", "N/A")) for r in pol],
            ["final a / e"] + [f"{r['hifi']['a_au']:.4f} AU / {r['hifi']['e']:.4f}" for r in pol],
            ["simulated years"] + [f"{r['hifi']['t_years']:,.0f}" for r in pol],
            ["asteroid used"] + [g(r, "mass_used_frac", lambda v: pct(v)) for r in pol],
            ["energy drawn"] + [g(r, "energy_used_J") + " J" for r in pol],
            ["surface temperature"] + [g(r, "T_surface_K", lambda v: f"{v:.1f} K") for r in pol],
            ["closest Mars pass"] + [g(r, "mars_min_dist_au", lambda v: f"{v:.4f} AU") for r in pol],
            ["reward (scoring only, not physics)"] + [g(r, "total_reward", lambda v: f"{v:.1f}") for r in pol]])
        d.note("The hand-designed strategy reaches the target; the trained PPO checkpoint also reaches it in this test but it "
               "is an early lucky policy (later training collapsed). Neither is claimed to be optimal.")
    d.h2("How the agent's five actions become physical commands")
    d.calc("Action decoding (each action a<sub>i</sub> in &minus;1..1)", "throttle = (a<sub>0</sub>+1)/2;  &theta; = &pi; a<sub>1</sub>;  &phi; = (&pi;/2) a<sub>2</sub>;  u = u<sub>min</sub> + (a<sub>3</sub>+1)/2 (u<sub>max</sub>&minus;u<sub>min</sub>);  S<sub>set</sub> = 1 + 0.5 a<sub>4</sub>",
           f"u<sub>min</sub> = {sci(cfg.min_ejection_speed_ms)} m/s, u<sub>max</sub> = {sci(cfg.max_ejection_speed_ms)} m/s", "packet mass = throttle &times; m<sub>pk,max</sub>; direction (sin&theta;cos&phi;, cos&theta;cos&phi;, sin&phi;) in radial/along-track/normal axes; mirror set-point 0.5-1.5 &times; today's sunlight",
           "OctoEarthEnv.decode_action()", "The agent never sets Earth's velocity; it only commands packets and mirror aim, and physics does the rest.")
    d.calc("Reward per decision (scoring only)", "r = w<sub>p</sub>&Delta;&Phi; &minus; w<sub>e</sub>(e excess) &minus; w<sub>i</sub>(i excess) &minus; w<sub>E</sub>E<sub>used</sub>/E<sub>0</sub> &minus; w<sub>m</sub>m<sub>used</sub>/m<sub>0</sub> &minus; w<sub>t</sub> &minus; w<sub>Mars</sub>&middot;pen &minus; w<sub>clim</sub>&middot;pen + terminal terms",
           f"w<sub>p</sub>={cfg.w_progress:g}, w<sub>E</sub>={cfg.w_energy:g}, w<sub>m</sub>={cfg.w_mass:g}, w<sub>t</sub>={cfg.w_time:g}, w<sub>Mars</sub>={cfg.w_mars_safety:g}, w<sub>clim</sub>={cfg.w_climate:g}, success {cfg.w_success:g}, failure {cfg.w_failure:g}",
           "a number for learning only; all physical results are reported separately", "OctoEarthEnv.step()")

    # =========================================================================== 9. instrument summary
    d.h1("Instrument summary: use and required performance", "summary")
    pwr_t = pwt["P_elec_W"]
    inst = [
        ["E1.1 Asteroid reservoir", "reaction mass", f"{sci(cfg.asteroid_mass_kg)} kg ({sci(cfg.asteroid_mass_kg / R.M_MOON)} Moon masses)"],
        ["E1.2 Launch ring + 8 arms", "throw packets", f"{sci(cfg.max_packet_mass_kg)} kg at up to {sci(cfg.max_ejection_speed_ms / 1e3)} km/s, {cfg.ejections_per_year:g}/yr, &eta; = {cfg.launch_efficiency}; peak {sci(fz['launcher_peak_power'])} W"],
        ["E1.3 Power conditioning + reserve", "launch energy", f"reserve {sci(cfg.energy_budget_j)} J; beams first"],
        ["E1.4 Directional radiators", "waste heat away from Earth", (f"~{sci(H['radiator_area_1000K_m2'])} m&sup2; at 1,000 K; leak &le; {sci(cfg.arm_heat_to_earth_frac)}" if H else f"leak &le; {sci(cfg.arm_heat_to_earth_frac)}")],
        ["E2.1 Thin-film collectors", "sunlight to electricity", f"{sci(n_c)} &times; {des['each_side_km']:,.0f} km, {sci(des['mass_kg'])} kg, &eta;<sub>op</sub> {pct(op['eta_op'], 1)} at {op['T_panel_K']:.0f} K"],
        ["E2.2 Microwave transmitters", "electricity to beam", f"{cfg.transmitter_diameter_km:g} km arrays, {cfg.microwave_freq_ghz} GHz, &eta; {pct(cfg.microwave_efficiency, 0)}, capture {pct(pwt['eta_capture'], 0)}-{pct(pw1['eta_capture'], 0)}"],
        ["E2.3 Rectennas (8)", "beam to electricity", f"{cfg.rectenna_diameter_km:g} km, &eta; {pct(cfg.rectenna_efficiency, 0)}; delivers {sci(pwr_t)}-{sci(pw1['P_elec_W'])} W"],
        ["E2.4 Sail-tilt station keeping", "hold slots", f"up to {sci(bk.auth / Rb ** 2)} m/s&sup2;" + (f"; tested {H['belt']['on']['max_km']:,.0f} km max drift" if H and H.get('belt') else "")],
        ["E2.5 Clear lane", "Earth sees the whole Sun", f"{sci(des['lane_frac_earth'])} of the belt empty"],
        ["E2.6 Factories", "build + replace", f"{sci(des['replace_kg_per_yr'])} kg/yr; fresh {pct(des['fresh_mass_mercury_frac'])} of Mercury"],
        ["E3.1 Mirror film", "reflect sunlight", f"{sci(cfg.mirror_area_m2)} m&sup2;, {sci(md1['mass_kg'])} kg, &rho; = {cfg.mirror_reflectivity}; capacity {mdt['cap_rel']:.2f} &times; at {cfg.target_au} AU"],
        ["E3.2 Terminator orbit + pointing", "aim the light", f"{cfg.mirror_orbit_re:g} R<sub>Earth</sub>, {md1['orbit_days']:.1f} d period, pointing ~{sci(R.R_EARTH / r_o)} rad"],
        ["(not an enabler) Mars safety shield", "keep Mars away", f"safe distance {sci(ms['d_safe_au'])} AU"],
        ["(not an enabler) RL agent / control", "decide every " + f"{cfg.decision_interval_years:.0f} y", "5 actions, 31 observations"]]
    d.table(["Instrument", "Use", "Required performance (this configuration)"], inst)
    d.h2("What is NOT modelled (honest list)")
    d.raw("<ul>" + "".join(f"<li><b>{esc(m['id'])} {esc(m['title'])}</b> ({esc(m['status'])}): {esc(m['detail'])} "
                         f"<i>Remaining:</i> {esc(m['remaining'])}</li>" for m in R.MAJOR_PROBLEMS if m["status"] != "addressed") + "</ul>")

    # =========================================================================== 10. what if an instrument is missing
    from dataclasses import replace as _rp
    d.h1("What happens without each instrument", "without")
    d.p("Each instrument is necessary for a reason. The calculations below show what would happen without it, or if it "
        "performed worse, using the same model functions. This is why each one is part of an enabler.")
    cf = {}
    d.cur = "E1.1"
    cf["E1.1"] = d.calc("Without the asteroid reservoir", "&Delta;v = v<sub>&infin;</sub> ln(m<sub>sys</sub> / (m<sub>sys</sub> &minus; m<sub>thrown</sub>)), m<sub>thrown</sub> = 0", "", "&Delta;v = 0: Earth cannot change its orbit (momentum is conserved; there is nothing to throw)", "feasibility()")
    d.cur = "E1.2"
    cr = _rp(cfg, max_ejection_speed_ms=3.0e4)
    cf["E1.2"] = d.calc("Launcher limited to 30 km/s (fastest electromagnetic launchers studied)", "&Delta;v<sub>cap</sub> = &radic;(u&sup2; &minus; v<sub>esc</sub>&sup2;) ln MR", f"u = 30 km/s, MR = {ratio:.6f}",
                        f"{sci(R.feasibility(cr)['dv_capacity_mass'])} m/s = {pct(R.feasibility(cr)['dv_capacity_mass'] / fz['dv_required'], 2)} of the required {sci(fz['dv_required'], 5)} m/s", "feasibility(max_ejection_speed_ms=3e4)")
    d.cur = "E1.3"
    c0 = _rp(cfg, energy_budget_j=0.0)
    cf["E1.3"] = d.calc("Without the energy reserve (energy_budget_j = 0)", "E<sub>avail</sub> = P<sub>elec,min</sub> T vs E<sub>min</sub>", f"{sci(fz['solar_power_min'])} &times; {sci(T)} vs {sci(fz['energy_min_draw'])}",
                        f"{'still enough' if R.feasibility(c0)['checks']['energy']['ok'] else 'NOT enough'}: {sci(R.feasibility(c0)['energy_available'])} J available. At year 0 (30 % of the belt built) the arms get {sci(pw0['P_elec_W'])} W vs {sci(cfg.ejections_per_year * 0.5 * R.heuristic_speed(cfg) ** 2 * cfg.max_packet_mass_kg / cfg.launch_efficiency / YEAR)} W at full throttle with the strategy speed",
                        "feasibility(energy_budget_j=0)", "In the tested mission the reserve was not touched (all launch energy came from sunlight through E2). The reserve is an assumed store of energy with no source in the model; the mission does not need it.")
    d.cur = "E1.4"
    qm_ = (H["arm_heat_mean_W"] if H else 1e24)
    F = R.S_1AU * (1 - R.CLIM_ALB_WARM) / 4 + 0.08 * qm_ / (4 * math.pi * R.R_EARTH ** 2)
    cf["E1.4"] = d.calc("Radiators that are not directional (~8 % of the heat reaches Earth)", "T = [(S(1&minus;&alpha;)/4 + q/(4&pi;R&sup2;)) / (&epsilon;&sigma;)]<sup>1/4</sup>", f"q = 0.08 &times; {sci(qm_)} W",
                        f"T &asymp; {sci((F / (R.CLIM_EPS * R.SIGMA_SB)) ** 0.25)} K: Earth's surface would be destroyed", "climate balance (indicative)")
    d.cur = "E2.1"
    cs = _rp(cfg, solar_collectors=False)
    cf["E2.1"] = d.calc("Without the collector belt", "E<sub>avail</sub> = reserve only", f"{sci(cfg.energy_budget_j)} J vs {sci(fz['energy_min_draw'])} J",
                        f"{'passes only because of the assumed reserve' if R.feasibility(cs)['checks']['energy']['ok'] else 'NOT enough energy'}; Earth's own sunlight for the whole mission is only {sci(R.EARTH_INTERCEPT_W * T)} J = {sci(R.EARTH_INTERCEPT_W * T / fz['energy_min_draw'])} of the minimum",
                        "feasibility(solar_collectors=False)", "Without E2 the launch energy has no physical source; the belt is what makes the energy come from sunlight.")
    d.cur = "E2.2"
    c1 = _rp(cfg, transmitter_diameter_km=3.0)
    cf["E2.2"] = d.calc("Transmitters 10 times smaller (3 km)", "&eta;<sub>cap</sub> = 1 &minus; e<sup>&minus;&tau;&sup2;</sup>, &tau; = D<sub>t</sub>D<sub>r</sub>/(&lambda;d)", "D<sub>t</sub> = 3 km (only the product D<sub>t</sub>D<sub>r</sub> matters, so this equals a 10 times smaller rectenna)",
                        f"capture {pct(R.collector_power(c1, [AU, 0, 0])['eta_capture'], 1)} at 1 AU, {pct(R.collector_power(c1, [cfg.target_au * AU, 0, 0])['eta_capture'], 1)} at {cfg.target_au} AU (instead of {pct(pw1['eta_capture'], 1)} / {pct(pwt['eta_capture'], 1)}); the rest is spilled", "collector_power()")
    d.cur = "E2.3"
    c2 = _rp(cfg, rectenna_diameter_km=100.0)
    cf["E2.3"] = d.calc("Rectennas 10 times smaller (100 km)", "same Goubau relation", "D<sub>r</sub> = 100 km",
                        f"capture {pct(R.collector_power(c2, [AU, 0, 0])['eta_capture'], 1)} at 1 AU, {pct(R.collector_power(c2, [cfg.target_au * AU, 0, 0])['eta_capture'], 1)} at {cfg.target_au} AU: arm electricity {sci(R.collector_power(c2, [cfg.target_au * AU, 0, 0])['P_elec_W'])} W at the target", "collector_power()")
    d.cur = "E2.4"
    if H and H.get("belt"):
        b = H["belt"]
        cf["E2.4"] = d.calc("Without station keeping (tested, same mission)", "RK4 orbits of 12 collectors without the sail-tilt control", f"{b['years']:,.0f} simulated years",
                            f"largest drift {sci(b['off']['max_km'])} km = {b['off']['slots']:,.0f} collector spacings (with it: {b['on']['max_km']:,.0f} km): the belt would scramble, beams and lanes could not be kept", "BeltKeeper (check 19)")
    d.cur = "E2.5"
    geo = 0.087
    cf["E2.5"] = d.calc("Without the clear lane", "shade = 1 when |z|/r &lt; f (Earth's line of sight to the Sun crosses the band)", f"Earth in the ecliptic: |z|/r &asymp; 0 &lt; {cfg.collector_fraction}",
                        f"shade = {R.earth_shading(_rp(cfg, earth_lane_clear=False), [AU, 0, 0]):.0f}: Earth and its mirrors get no direct sunlight; only internal heat (~0.087 W/m&sup2;) would keep the surface near {(geo / (R.CLIM_EPS * R.SIGMA_SB)) ** 0.25:.0f} K", "earth_shading()")
    d.cur = "E2.6"
    cf["E2.6"] = d.calc("Without factories (belt stays at 30 %)", "P<sub>elec</sub>(b = b<sub>0</sub>)", f"b<sub>0</sub> = {cfg.collector_initial_built}",
                        f"{sci(pw0['P_elec_W'])} W at 1 AU, {sci(R.collector_power(cfg, [cfg.target_au * AU, 0, 0], cfg.collector_initial_built)['P_elec_W'])} W at {cfg.target_au} AU (still above the tested mean demand {sci(H['beam_mean_W']) if H else 'N/A'} W, but aged panels would not be replaced: after {cfg.collector_lifetime_years:.0f} years the panels reach their end of life)", "collector_power()")
    d.cur = "E3.1"
    cf["E3.1"] = d.calc("Without mirrors at the target", "equilibrium at S = S<sub>1AU</sub>/a<sub>t</sub>&sup2;", f"{s_t:.4f} &times; today", f"{T_no:.1f} K: snowball Earth (ice to the equator, no growing area)", "climate_equilibrium()")
    d.cur = "E3.2"
    cf["E3.2"] = d.calc("Pointing error larger than Earth's angular size from the ring", "&delta;&theta; &gt; R<sub>Earth</sub>/r", f"{sci(R.R_EARTH / r_o)} rad",
                        "the mirror light misses Earth: same as having no mirrors (C" + str(cf["E3.1"]) + ")", "indicative")

    # =========================================================================== 11. data sheets
    d.h1("Instrument data sheets", "sheets")
    d.p("One sheet per instrument: what it is for, its inputs, the performance it must deliver in this configuration, the "
        "calculations behind it (links to C-numbers), what happens without it and how far it is from today's technology.")
    def refs(key):
        return ", ".join(f'<a href="#c{k}">C{k}</a>' for k in d.refs.get(key, [])) or "-"
    def pv(keys):
        def f(v):
            return sci(v) if isinstance(v, float) and (abs(v) >= 1e4 or 0 < abs(v) < 1e-3) else esc(v)
        return "; ".join(f"{k} = {f(getattr(cfg, k))}" for k in keys)
    sheets = [
        ("E1.1", "Asteroid reservoir (docked to Earth)", "E1", "Supplies all reaction mass; used up packet by packet.", ["asteroid_mass_kg"],
         f"{sci(cfg.asteroid_mass_kg)} kg ({sci(cfg.asteroid_mass_kg / R.M_MOON)} Moon masses); at least {sci(fz['mass_needed_at_umax'])} kg needed even at the maximum speed",
         "Moving and docking a body of 40 % of the Moon's mass is far beyond any technology."),
        ("E1.2", "Launch ring with 8 arms (mass driver)", "E1", "Throws packets at the commanded speed and direction; all 8 arms throw one packet together.",
         ["max_packet_mass_kg", "max_ejection_speed_ms", "min_ejection_speed_ms", "ejections_per_year", "launch_efficiency"],
         f"{sci(cfg.max_packet_mass_kg)} kg packets, {sci(u_h / 1e3)} km/s (strategy) up to {sci(cfg.max_ejection_speed_ms / 1e3)} km/s, every {365.25 / cfg.ejections_per_year:.1f} days; peak power {sci(fz['launcher_peak_power'])} W",
         "No known structure; each launch carries ~10<sup>30</sup>-10<sup>31</sup> J."),
        ("E1.3", "Arm power conditioning + energy reserve", "E1", "Turns arm electricity into launch energy; beams first, reserve second.", ["energy_budget_j"],
         f"handles up to {sci(fz['launcher_peak_power'])} W ({sci(fz['launcher_peak_power'] / 8)} W per arm); reserve {sci(cfg.energy_budget_j)} J (not used in the tested mission)",
         "Planetary-scale power electronics; the reserve has no physical source in the model."),
        ("E1.4", "Directional arm radiators", "E1", "Send launcher and rectenna waste heat to space, away from Earth.", ["arm_heat_to_earth_frac"],
         (f"reject {sci(H['arm_heat_mean_W'])} W; ~{sci(H['radiator_area_1000K_m2'])} m&sup2; at 1,000 K; leak to Earth &le; {sci(cfg.arm_heat_to_earth_frac)}" if H else "leak &le; 1e-9"),
         "No radiator design; the leak fraction 1e-9 is assumed."),
        ("E2.1", "Thin-film solar collectors (5 million)", "E2", "Intercept sunlight and make electricity.",
         ["collector_fraction", "collector_orbit_au", "collector_count", "collector_areal_density", "pv_efficiency", "collector_lifetime_years"],
         f"{sci(des['area_m2'])} m&sup2;, {sci(des['mass_kg'])} kg, {op['T_panel_K']:.0f} K panels, &eta;<sub>op</sub> {pct(op['eta_op'], 1)}, {sci(pw1['P_pv_W'])} W of electricity at full output",
         "A Dyson-swarm-scale project (~10<sup>13</sup> &times; all solar panels ever built)."),
        ("E2.2", "Microwave phased-array transmitters", "E2", "Turn electricity into focused 5.8 GHz beams aimed at the arms; keep spill off Earth.",
         ["microwave_efficiency", "microwave_freq_ghz", "transmitter_diameter_km", "beam_spill_to_earth_frac"],
         f"{cfg.transmitter_diameter_km:g} km aperture, {sci(pw1['P_mw_W'] / n_c)} W each at full output, capture {pct(pwt['eta_capture'], 1)}-{pct(pw1['eta_capture'], 1)}, pointing ~{sci(cfg.rectenna_diameter_km * 500 / (pwt['d_max_au'] * AU))} rad",
         "Record power beaming is ~kW over ~km."),
        ("E2.3", "Rectennas on the 8 arm tips", "E2", "Receive the beams and turn them into electricity (only the arms receive power).", ["rectenna_diameter_km", "rectenna_efficiency"],
         f"{cfg.rectenna_diameter_km:g} km each, {pct(cfg.rectenna_efficiency, 0)} efficient, up to {sci(des['beam_intensity_wm2_1au'])} W/m&sup2; ({sci(des['beam_intensity_wm2_1au'] / R.S_1AU)} &times; sunlight)",
         "Beam intensity ~10<sup>8</sup>-10<sup>9</sup> &times; sunlight; no known receiver survives it."),
        ("E2.4", "Sail-tilt station keeping", "E2", "Hold each collector in its slot using part of its light push (no propellant).", ["collector_sk_authority"],
         f"control up to {sci(bk.auth / Rb ** 2)} m/s&sup2;" + (f"; tested effort {H['belt']['on']['dv_ms_per_yr']:.0f} m/s per year, drift &le; {H['belt']['on']['max_km']:,.0f} km" if H and H.get("belt") else ""),
         "Solar sails demonstrated at m&sup2;-km&sup2; scale only."),
        ("E2.5", "Clear sunlight lane", "E2", "Keep a moving gap so Earth and Mars see the whole Sun.", ["earth_lane_clear"],
         f"{sci(des['lane_frac_earth'])} of the belt kept empty for Earth, {sci(des['lane_frac_mars'])} for Mars", "Coordinated motion of millions of collectors."),
        ("E2.6", "Collector factories (Mercury material)", "E2", "Build the belt up and replace aged panels.",
         ["collector_initial_built", "collector_build_years", "collector_recycle_frac", "collector_embodied_energy_jkg"],
         f"{sci(des['replace_kg_per_yr'])} kg of panels per year; fresh material {pct(des['fresh_mass_mercury_frac'])} of Mercury", "Self-replicating industry on Mercury is hypothetical."),
        ("E3.1", "Mirror film panels", "E3", "Reflect sunlight onto Earth (or past it when not needed).", ["mirror_area_m2", "mirror_reflectivity", "mirror_areal_density"],
         f"{sci(cfg.mirror_area_m2)} m&sup2;, {sci(md1['mass_kg'])} kg; must send up to {sci(need)} W at {cfg.target_au} AU (capacity {sci(mdt['cap_W'])} W)",
         "Physically possible (plain reflection); demonstrated at 20 m (Znamya-2), needed ~10<sup>13</sup> &times; that area."),
        ("E3.2", "Terminator orbit + pointing", "E3", "Keep the mirrors in sunlight at 20 Earth radii and aim the light (agent action a4).", ["mirror_orbit_re", "mirrors"],
         f"orbit period {md1['orbit_days']:.2f} days; pointing better than {sci(R.R_EARTH / r_o)} rad; light-pressure disturbance {sci(a_lp)} m/s&sup2;",
         "Formation flying of about a million 40 km mirrors."),
    ]
    for sid, name, en, use, keys, perf, gap in sheets:
        d.raw(f'<div class="ebox"><div class="t">{sid} {esc(name)} <span class="small">(part of enabler {en})</span></div>'
              f'<table class="kv"><tr><td>Use</td><td>{esc(use)}</td></tr><tr><td>Inputs</td><td><code>{pv(keys)}</code></td></tr>'
              f'<tr><td>Required performance</td><td>{perf}</td></tr><tr><td>Calculations</td><td>{refs(sid)}</td></tr>'
              f'<tr><td>Without it</td><td>see <a href="#c{cf[sid]}">C{cf[sid]}</a></td></tr>' if sid in cf else
              f'<div class="ebox"><div class="t">{sid} {esc(name)}</div><table class="kv"><tr><td>Use</td><td>{esc(use)}</td></tr>')
        d.raw(f'<tr><td>Gap to reality</td><td>{gap}</td></tr></table></div>')
    for name, use, key, gap in [
        ("Mars safety shield (control software, not an enabler)", "Forecasts the Earth-Mars distance and overrides the agent's throw when Earth would come closer than the safe distance.", "MARS", "Ordinary software; needs the Mars measurements of section 12."),
        ("RL agent PPO + heuristic baseline (control software, not an enabler)", "Chooses throttle, direction, throw speed and mirror light every 100 years from 31 observations.", "CONTROL", "Ordinary software; the tested numbers are in section 8."),
    ]:
        d.raw(f'<div class="ebox"><div class="t">{esc(name)}</div><table class="kv"><tr><td>Use</td><td>{esc(use)}</td></tr>'
              f'<tr><td>Calculations</td><td>{refs(key)}</td></tr><tr><td>Gap to reality</td><td>{esc(gap)}</td></tr></table></div>')

    # =========================================================================== 12. measurements
    d.h1("Measuring instruments the control system needs (ordinary technology, not enablers)", "measure")
    d.note("The simulation gives the agent and the safety shield <b>perfect</b> knowledge of the state. A real system needs "
           "measurements. The instruments below are existing kinds of technology (no enabler needed); the required "
           "accuracies are <i>indicative</i>, derived from the mission tolerances (1-10 % of each tolerance). They are not "
           "simulated.", "warn")
    vt_ = math.sqrt(R.GM_SUN / at)
    da = 0.01 * cfg.tol_a_au * AU
    dv_need = da * R.GM_SUN / (2 * at ** 2 * vt_)
    d.cur = "MEAS"
    c_a = d.calc("Earth's orbit size: required accuracy", "&delta;a &le; 1 % of the tolerance;  vis-viva: &delta;a = 2a&sup2;v &delta;v / &mu;  &rArr;  &delta;v = &delta;a &mu; / (2 a&sup2; v)",
                 f"&delta;a = 0.01 &times; {cfg.tol_a_au} AU = {sci(da / 1e3)} km; a = {cfg.target_au} AU, v = {sci(vt_, 5)} m/s",
                 f"position to ~{sci(da / 1e3)} km, velocity to ~{sci(dv_need)} m/s", "indicative")
    d_s = ms["d_safe_m"]
    c_m = d.calc("Earth-Mars distance: required accuracy", "&delta;d &le; 1 % of the safe distance; forecast over the shield horizon", f"0.01 &times; {sci(d_s / 1e3)} km; horizon {cfg.shield_horizon_years:g} years",
                 f"relative position to ~{sci(0.01 * d_s / 1e3)} km, relative velocity to ~{sci(0.01 * d_s / (cfg.shield_horizon_years * YEAR))} m/s", "indicative")
    c_t = d.calc("Global surface temperature: required accuracy", "&delta;T &le; 10 % of the tolerance;  &delta;S/S = &delta;T / (dT/dS)", f"0.1 &times; {cfg.climate_tol_K:g} K; dT/dS &asymp; 72 K per &times;today (heuristic_climate)",
                 f"temperature to ~{0.1 * cfg.climate_tol_K:.1f} K; sunlight (incl. mirror light) on Earth to ~{pct(0.1 * cfg.climate_tol_K / 72, 2)}", "indicative")
    c_s = d.calc("Collector positions: required accuracy", "&delta;x &le; 1 % of the tested drift with station keeping", f"0.01 &times; {H['belt']['on']['max_km'] if H and H.get('belt') else 700:,.0f} km",
                 f"~{(H['belt']['on']['max_km'] if H and H.get('belt') else 700) * 0.01:.0f} km per collector", "indicative")
    meas = [
        ["Earth's position and velocity (6 observations)", "agent + shield", "radio ranging/Doppler to spacecraft and stations (Deep Space Network type), planetary radar, star astrometry", f'<a href="#c{c_a}">C{c_a}</a>'],
        ["Orbit size, eccentricity, inclination", "agent + success test", "computed from the above (orbit determination)", f"e and i to ~{0.01 * cfg.tol_e:.4f} / {0.01 * cfg.tol_inc_deg:.2f}&deg;"],
        ["Moon distance and whether it is bound", "agent + failure test", "lunar laser ranging (cm accuracy today)", "~1 km is enough"],
        ["Asteroid mass left", "agent", "counting and weighing packets (mass flow meters on the arms)", f"~1 % of a packet = {sci(0.01 * cfg.max_packet_mass_kg)} kg"],
        ["Energy reserve / beam power", "agent + ledger", "electrical power and energy meters on the rectennas and arms", "~0.1 %"],
        ["Mars position relative to Earth (5 observations)", "agent + shield", "radio tracking of Mars orbiters/landers, radar ranging", f'<a href="#c{c_m}">C{c_m}</a>'],
        ["Surface and deep-ocean temperature, sunlight on Earth (4 observations)", "agent + climate test", "satellite radiometers, ocean floats (Argo type), solar irradiance monitors", f'<a href="#c{c_t}">C{c_t}</a>'],
        ["Beam pointing at the rectennas", "E2 transmitters", "pilot beam from each rectenna (retrodirective phased array)", f"~{sci(cfg.rectenna_diameter_km * 500 / (pwt['d_max_au'] * AU))} rad"],
        ["Collector positions in the belt", "E2 station keeping", "radio/optical ranging between neighbouring collectors + beacons", f'<a href="#c{c_s}">C{c_s}</a>'],
        ["Mirror attitude and aim", "E3 pointing", "star trackers + sun sensors on each mirror", f"better than {sci(R.R_EARTH / r_o)} rad"],
    ]
    d.table(["Quantity measured", "Used by", "Instrument (existing kind of technology)", "Required accuracy"], meas)

    # =========================================================================== appendix
    d.h1("Appendix: every parameter", "appendix")
    d.table(["Key", "Meaning", "Value", "Unit", "Category"],
            [[f"<code>{esc(p['key'])}</code>", esc(p["label"]), esc(getattr(cfg, p["key"], "")), esc(p["unit"]), esc(p["category"])] for p in R.PARAM_DOCS
             if hasattr(cfg, p["key"])])

    toc = '<div class="toc">' + "".join(f'<div class="l{lv}"><a href="#{a}">{esc(t)}</a></div>' for lv, t, a in d.toc) + "</div>"
    cover = (f'<div class="cover"><h0>OCTO-EARTH</h0><div class="sub">Calculation report: mission, the three enablers and every instrument</div>'
             f'<p class="small">Generated {time.strftime("%Y-%m-%d %H:%M")} by calc_report.py from octoearth_rl.py (preset "{esc(cfg.preset)}"); '
             f'tested results from {esc(results_dir.name)}/ {("(" + esc(comp.get("created", "")) + ")") if comp else "(none found)"}.<br>'
             f'All numbers are model results of a physically hypothetical scenario, not engineering designs.</p>{toc}</div>')
    body = "\n".join(d.parts)
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>OCTO-EARTH calculation report</title><style>{CSS}</style></head>'
            f'<body>{cover}{body}<p class="small" style="margin-top:20px">{d.k} calculations. End of report.</p></body></html>')


def to_pdf(html_path: Path, pdf_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto(html_path.resolve().as_uri())
            pg.pdf(path=str(pdf_path), format="A4", print_background=True, margin=dict(top="14mm", bottom="16mm", left="12mm", right="12mm"),
                   display_header_footer=True, header_template="<span></span>",
                   footer_template='<div style="font-size:7pt;color:#888;width:100%;text-align:center">OCTO-EARTH calculation report &middot; page '
                                   '<span class="pageNumber"></span> of <span class="totalPages"></span></div>')
            b.close()
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[calc_report] PDF not written ({exc}); open the HTML file and print it to PDF from the browser.")
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "example_results"))
    ap.add_argument("--out", default=str(ROOT / "CALCULATIONS_REPORT.html"))
    ap.add_argument("--preset", default="hypothetical", choices=list(R.PRESETS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v) if v.lower() not in ("true", "false") else v.lower() == "true"
    cfg = R.checked_config(a.preset, **over)
    out = Path(a.out)
    out.write_text(build(cfg, Path(a.results)), encoding="utf-8")
    print(f"[calc_report] wrote {out}")
    if not a.no_pdf and to_pdf(out, out.with_suffix(".pdf")):
        print(f"[calc_report] wrote {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
