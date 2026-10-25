// OCTO-EARTH "failed attempts" - the numbers behind the four independent models (update p38).
//
// Pure functions, no graphics. The same formulas are in failed_models.py (Python); a test compares both.
// Goal for every model: move Earth from a = 1 AU to a = 2.7 AU within 10,000 years (the same target as the main
// OCTO-EARTH model). A slow outward spiral needs  dv = v_circ(1 AU) - v_circ(2.7 AU)  = 11,658 m/s, i.e. an average
// push along the orbit of  F_req = M_earth * dv / T  = 2.2e17 N  for 10,000 years.
//
//   1  Laser push (radiation pressure)          F = P / c      -> the power needed would vaporise Earth
//   2  Hydrogen harvesting (fusion rocket)      rocket eq.     -> even the whole solar wind is not enough
//   3  Superconducting magnetic arms (Lorentz)  F = I L x B    -> the Sun's field is far too weak, falls with
//                                                                 distance; the arms quench; inside the
//                                                                 magnetosphere the net force is zero
//   4  Dyson beam on a reflector held by arms   F = (P/c)(d-r) -> the arms fail (and the absorbed beam overheats Earth)

export const K = {
  GM_SUN: 1.32712440018e20, AU: 1.495978707e11, M_E: 5.972e24, R_E: 6.371e6, C: 2.99792458e8, SIGMA: 5.670374419e-8,
  L_SUN: 3.828e26, S0: 1361, ALBEDO: 0.30, YEAR: 3.15576e7, T_MISSION_Y: 10000, A0: 1.0, A1: 2.7, MU0: 4e-7 * Math.PI,
  M_P: 1.67262192e-27, T_SURF: 288,
};
export const T_MISSION = K.T_MISSION_Y * K.YEAR;
export const vcirc = (aAU) => Math.sqrt(K.GM_SUN / (aAU * K.AU));
export const DV_REQ = vcirc(K.A0) - vcirc(K.A1);                    // 11,658 m/s
export const F_REQ = K.M_E * DV_REQ / T_MISSION;                     // 2.2e17 N
/** Orbit size after a slow tangential spiral that removed dv (m/s) of orbital speed (circular orbits). */
export function aAfterDv(dv, a0 = K.A0) { const x = dv / vcirc(a0); return a0 / ((1 - x) * (1 - x)); }
/** Spiral with a tangential force F(a) [N]: da/dt = 2 a^1.5 F / (M sqrt(GM)). RK4 in years; returns a(t) samples. */
export function spiral(Fof, years = K.T_MISSION_Y, nStep = 2000, a0 = K.A0) {
  const h = years * K.YEAR / nStep, f = (a) => 2 * Math.pow(a, 1.5) * Fof(a / K.AU) / (K.M_E * Math.sqrt(K.GM_SUN));
  let a = a0 * K.AU; const out = [[0, a0]]; let tReach = null;
  for (let i = 1; i <= nStep; i++) {
    const k1 = f(a), k2 = f(a + 0.5 * h * k1), k3 = f(a + 0.5 * h * k2), k4 = f(a + h * k3);
    a += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    if (tReach === null && a / K.AU >= K.A1) tReach = i * h / K.YEAR;
    out.push([i * h / K.YEAR, a / K.AU]);
  }
  return { samples: out, aEnd: a / K.AU, tReach };
}

// ------------------------------------------------------------------ 1  Laser push
export const SUN_ABSORBED = K.S0 * (1 - K.ALBEDO) * Math.PI * K.R_E ** 2;      // sunlight Earth absorbs today, 1.22e17 W
export const LASER = {
  P_REQ: F_REQ * K.C,                                                          // 6.6e25 W = 17 % of the Sun's output
  // energy needed for each stage of destruction [J] (simple estimates, see the README)
  E_ATMOSPHERE: 5.15e18 * 0.5 * 11186 ** 2,                                    // lift the whole atmosphere to escape speed
  E_OCEANS: 1.4e21 * (4186 * (373 - 288) + 2.257e6),                           // heat + boil all oceans
  E_CRUST: 2.6e22 * (1000 * (1500 - 288) + 4.0e5),                             // heat + melt the crust
  E_MANTLE: 4.0e24 * (1200 * 1500 + 4.0e5),                                    // heat + melt the mantle
  E_UNBIND: 2.49e32,                                                           // gravitational binding energy of Earth
  T_SAFE_DK: 2,                                                                // "safe": at most +2 K global warming
};
LASER.P_SAFE = SUN_ABSORBED * (((K.T_SURF + LASER.T_SAFE_DK) / K.T_SURF) ** 4 - 1);
/** Everything for a beam of power P [W] fully absorbed by Earth. */
export function laser(P) {
  const F = P / K.C, acc = F / K.M_E, dv10k = acc * T_MISSION;
  return {
    P, F, acc, dv10k, a10k: aAfterDv(Math.min(dv10k, 0.999 * vcirc(1))), fracOfNeed: F / F_REQ,
    xSunlight: P / SUN_ABSORBED,
    Tsurf: K.T_SURF * ((SUN_ABSORBED + P) / SUN_ABSORBED) ** 0.25,             // global energy balance, 288 K today
    tAtm: LASER.E_ATMOSPHERE / P, tOceans: LASER.E_OCEANS / P, tCrust: LASER.E_CRUST / P, tMantle: LASER.E_MANTLE / P, tUnbind: LASER.E_UNBIND / P,
  };
}
/** Earth's push after time t [s] of beam P: speed gained and distance moved. */
export const laserMotion = (P, t) => { const a = P / K.C / K.M_E; return { dv: a * t, x: 0.5 * a * t * t }; };

// ------------------------------------------------------------------ 2  Hydrogen harvesting
export const HYD = {
  N_CM3: 5, V_SW: 4.0e5,                     // solar wind at 1 AU: ~5 protons per cm^3 at ~400 km/s
  SUN_MASS_LOSS: 1.5e9,                      // the whole solar wind, all directions [kg/s]
  EPS_FUSION: 0.0071,                        // hydrogen -> helium releases 0.71 % of the mass as energy
};
HYD.MASS_FLUX = HYD.N_CM3 * 1e6 * K.M_P * HYD.V_SW;                           // 3.3e-15 kg / m^2 / s
HYD.VE = K.C * Math.sqrt(2 * HYD.EPS_FUSION);                                // ideal fusion exhaust 3.6e7 m/s (0.12 c)
HYD.M_NEED = K.M_E * (Math.exp(DV_REQ / HYD.VE) - 1);                        // hydrogen needed (rocket equation)
/** Scoop of radius R (Earth radii) facing the solar wind at 1 AU for 10,000 years (ideal: every proton caught,
 *  every kg fused, all energy into exhaust). */
export function hydrogen(R_RE, mdotOverride = null) {
  const area = Math.PI * (R_RE * K.R_E) ** 2, mdot = mdotOverride ?? HYD.MASS_FLUX * area, m = mdot * T_MISSION;
  const dv = HYD.VE * Math.log(1 + m / K.M_E);
  return { R_RE, area, mdot, m10k: m, frac: m / HYD.M_NEED, dv, a10k: aAfterDv(dv), windPush: mdot * HYD.V_SW, mNeed: HYD.M_NEED };
}
export const hydrogenWholeSun = () => hydrogen(0, HYD.SUN_MASS_LOSS);

// ------------------------------------------------------------------ 3  Superconducting magnetic arms
export const MAG = {
  BR0: 3.5e-9, K_SPIRAL: 2.7e-6 * K.AU / 4.0e5,   // Parker spiral: B_r = BR0 / r^2, B_phi = B_r * Omega r / v_sw
  N_ARMS: 8, L_ARM: 2 * K.R_E, B_CRIT: 45, A_CABLE: 1.0, J_C: 1e9, B_EQ: 3.12e-5,
};
MAG.L_TOT = MAG.N_ARMS * MAG.L_ARM;
/** Magnitude of the Sun's (interplanetary) magnetic field at r AU [T]. */
export const imf = (rAU) => MAG.BR0 * Math.sqrt(1 / rAU ** 4 + (MAG.K_SPIRAL / rAU) ** 2);
MAG.I_MAX = 2 * Math.PI * MAG.A_CABLE * MAG.B_CRIT / K.MU0;                 // self-field at the cable surface = B_crit
MAG.I_REQ = F_REQ / (MAG.L_TOT * imf(1));                                    // per arm, to give F_req at 1 AU
MAG.A_FOR_IREQ = K.MU0 * MAG.I_REQ / (2 * Math.PI * MAG.B_CRIT);             // cable radius that could carry it
/** Best case force [N] at r AU: all arms perpendicular to B, current switched with every field reversal, and the
 *  Sun's field reaching the arms (in reality they sit inside Earth's magnetosphere, where the net force is zero). */
export const magForce = (I, rAU) => I * MAG.L_TOT * imf(rAU);
export function magnet(I) {
  const quench = I > MAG.I_MAX, F1 = magForce(I, 1);
  const run = spiral((r) => magForce(I, r), K.T_MISSION_Y, 1500);
  const ringR = 1.25, Bmid = MAG.B_EQ / (ringR + 1) ** 3;                    // Earth's field half-way along an arm
  return { I, quench, F1, fracOfNeed: F1 / F_REQ, aEnd: run.aEnd, tReach: run.tReach, samples: run.samples, Fdipole: I * MAG.L_ARM * Bmid,
    Bself: K.MU0 * I / (2 * Math.PI * MAG.A_CABLE) };
}

/** Smallest current per arm that would reach 2.7 AU within 10,000 years (best case, quench ignored) - bisection. */
export function magCurrentToArrive() {
  let lo = MAG.I_REQ, hi = MAG.I_REQ * 100;
  for (let i = 0; i < 40; i++) { const m = Math.sqrt(lo * hi); if (spiral((r) => magForce(m, r), K.T_MISSION_Y, 600).tReach !== null) hi = m; else lo = m; }
  return hi;
}
MAG.I_ARRIVE = magCurrentToArrive();

// ------------------------------------------------------------------ 4  Dyson beam on a held reflector
export const DYSON = { P: 1.29e25 };
DYSON.F = DYSON.P / K.C;
DYSON.frac = DYSON.F / F_REQ;
DYSON.Tsurf = K.T_SURF * ((SUN_ABSORBED + DYSON.P) / SUN_ABSORBED) ** 0.25;

/** Short summary used by the test (Python must give the same numbers). */
export function summary() {
  const l = laser(LASER.P_REQ), s = laser(LASER.P_SAFE), h = hydrogen(100), hw = hydrogenWholeSun(), m = magnet(MAG.I_MAX), mi = magnet(MAG.I_REQ);
  return {
    DV_REQ, F_REQ, P_REQ: LASER.P_REQ, P_SAFE: LASER.P_SAFE, Tsurf_req: l.Tsurf, tOceans: l.tOceans, tUnbind: l.tUnbind, a10k_safe: s.a10k,
    h100_m: h.m10k, h_need: HYD.M_NEED, h100_frac: h.frac, hw_frac: hw.frac, hw_a: hw.a10k,
    I_MAX: MAG.I_MAX, I_REQ: MAG.I_REQ, mag_F: m.F1, mag_a: m.aEnd, magreq_a: mi.aEnd, magreq_t: mi.tReach, I_ARRIVE: MAG.I_ARRIVE, dyson_frac: DYSON.frac, dyson_T: DYSON.Tsurf,
  };
}
