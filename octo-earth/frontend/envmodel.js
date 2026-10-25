// OCTO-EARTH – space-environment models used by the Earth-POV view.
// These are standard, simplified empirical/analytic models evaluated at Earth's SIMULATED distance
// from the Sun. They are for display only: they do NOT feed back into the orbit simulation.
// The same formulas are implemented in Python (octoearth_rl.space_environment) and a pytest checks
// that both give identical numbers.
//
//  * Solar wind: steady average wind, speed 400 km/s (constant with distance), proton density
//    5 per cm^3 at 1 AU falling as 1/r^2. No storms, CMEs or solar cycle.
//  * Interplanetary magnetic field: Parker spiral, radial part 3.5 nT at 1 AU falling as 1/r^2,
//    azimuthal part grows with r (|B| ~ 5 nT at 1 AU).
//  * Magnetopause: Shue et al. (1998) shape r = r0 (2 / (1 + cos theta))^alpha with IMF Bz = 0.
//  * Bow shock: stand-off 1.29 x magnetopause stand-off (Farris & Russell gas-dynamic relation with
//    an assumed fast-mode Mach number of 8); shape flaring exponent 0.8 (schematic).
//  * Hill sphere: r_H = a (m / 3 M_sun)^(1/3) with a = current Sun distance.
//  * Earth's field: centred dipole, 3.12e-5 T at the magnetic equator on the surface.

export const RE_KM = 6371.0;
export const AU_KM = 1.495978707e8;
export const R_SUN_KM = 695700.0;
export const G = 6.67430e-11;
export const GM_SUN = 1.32712440041e20;       // m^3/s^2
export const M_SUN = GM_SUN / G;
export const GM_MOON = 4.9028e12;
export const B_SURFACE_T = 3.12e-5;
export const SW_N1 = 5.0;                      // cm^-3 at 1 AU
export const SW_V = 400.0;                     // km/s
export const IMF_BR1 = 3.5;                    // nT, radial component at 1 AU
export const OMEGA_SUN = 2.865e-6;             // rad/s (25.4-day sidereal rotation)
export const BS_FACTOR = 1.29;
export const BS_ALPHA = 0.8;
export const S_1AU = 1361.0;                   // W/m^2

export function spaceEnv(r_au, m_sys_kg) {
  const n = SW_N1 / (r_au * r_au);                                   // cm^-3
  const v = SW_V;                                                    // km/s
  const Dp = 1.6726e-6 * n * v * v;                                  // nPa (m_p n v^2)
  const Bz = 0.0;
  const r0 = (10.22 + 1.29 * Math.tanh(0.184 * (Bz + 8.14))) * Math.pow(Dp, -1.0 / 6.6);   // R_E
  const alpha = (0.58 - 0.007 * Bz) * (1.0 + 0.024 * Math.log(Dp));
  const br = IMF_BR1 / (r_au * r_au);
  const wind = OMEGA_SUN * r_au * AU_KM / v;                         // tan(spiral angle)
  const bimf = br * Math.sqrt(1.0 + wind * wind);                    // nT
  const hill_re = r_au * AU_KM / RE_KM * Math.cbrt(m_sys_kg / (3.0 * M_SUN));
  return {
    n_cm3: n, v_kms: v, dp_npa: Dp, r0_re: r0, alpha, rbs_re: BS_FACTOR * r0, alpha_bs: BS_ALPHA,
    imf_nt: bimf, spiral_deg: Math.atan(wind) * 180 / Math.PI, hill_re,
    insolation_wm2: S_1AU / (r_au * r_au), sun_ang_rad: R_SUN_KM / (r_au * AU_KM),
  };
}
// Shue-type boundary radius at angle theta [rad] from the Sun direction
export function shueR(r0, alpha, theta) { return r0 * Math.pow(2.0 / (1.0 + Math.cos(theta)), alpha); }
// dipole field strength [T] at radius r [R_E] and magnetic latitude lat [rad]
export function dipoleB(r, lat) { return B_SURFACE_T / (r * r * r) * Math.sqrt(1 + 3 * Math.sin(lat) ** 2); }
