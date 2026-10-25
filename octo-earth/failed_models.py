"""OCTO-EARTH - four FAILED ideas for moving Earth from 1 AU to 2.7 AU in 10,000 years (update p38).

Independent of the asteroid-launcher model in octoearth_rl.py. Same formulas as frontend/failedPhysics.js (the 3-D
views in frontend/failed.html use those); a test checks that both give the same numbers.

    python failed_models.py          prints the four verdicts with the numbers

1  Laser push (radiation pressure)          F = P / c    -> the power needed would vaporise Earth
2  Hydrogen harvesting + fusion rocket       rocket eq.   -> even the whole solar wind for 10,000 y is not enough
3  Superconducting magnetic arms (Lorentz)   F = I L B    -> Sun's field too weak, falls with distance, arms quench,
                                                             and inside the magnetosphere the net force is zero
4  Dyson beam on a reflector held by 8 arms  (P/c)(d-r)   -> the arms fail; even intact: 20 % of the force needed,
                                                             and the absorbed beam overheats Earth
"""
import math

GM_SUN, AU, M_E, R_E, C = 1.32712440018e20, 1.495978707e11, 5.972e24, 6.371e6, 2.99792458e8
S0, ALBEDO, YEAR, T_MISSION_Y, A0, A1 = 1361.0, 0.30, 3.15576e7, 10000, 1.0, 2.7
MU0, M_P, T_SURF = 4e-7 * math.pi, 1.67262192e-27, 288.0
T_MISSION = T_MISSION_Y * YEAR


def vcirc(a_au):
    return math.sqrt(GM_SUN / (a_au * AU))


DV_REQ = vcirc(A0) - vcirc(A1)                     # m/s, slow outward spiral
F_REQ = M_E * DV_REQ / T_MISSION                   # N, average push along the orbit for 10,000 years


def a_after_dv(dv, a0=A0):
    """Orbit size after a slow tangential spiral that removed dv of orbital speed (circular orbits)."""
    x = dv / vcirc(a0)
    return a0 / (1 - x) ** 2


def spiral(F_of, years=T_MISSION_Y, n_step=2000, a0=A0):
    """da/dt = 2 a^1.5 F / (M sqrt(GM)) for a tangential force F(a [AU]) [N]; RK4. Returns (a_end, t_reach)."""
    h = years * YEAR / n_step
    f = lambda a: 2 * a ** 1.5 * F_of(a / AU) / (M_E * math.sqrt(GM_SUN))
    a, t_reach = a0 * AU, None
    for i in range(1, n_step + 1):
        k1 = f(a); k2 = f(a + 0.5 * h * k1); k3 = f(a + 0.5 * h * k2); k4 = f(a + h * k3)
        a += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if t_reach is None and a / AU >= A1:
            t_reach = i * h / YEAR
    return a / AU, t_reach


# 1 ---------------------------------------------------------------- laser push
SUN_ABSORBED = S0 * (1 - ALBEDO) * math.pi * R_E ** 2
P_REQ = F_REQ * C
E_ATMOSPHERE = 5.15e18 * 0.5 * 11186 ** 2
E_OCEANS = 1.4e21 * (4186 * (373 - 288) + 2.257e6)
E_CRUST = 2.6e22 * (1000 * (1500 - 288) + 4.0e5)
E_MANTLE = 4.0e24 * (1200 * 1500 + 4.0e5)
E_UNBIND = 2.49e32
P_SAFE = SUN_ABSORBED * (((T_SURF + 2) / T_SURF) ** 4 - 1)


def laser(P):
    F = P / C
    dv = F / M_E * T_MISSION
    return dict(P=P, F=F, dv10k=dv, a10k=a_after_dv(min(dv, 0.999 * vcirc(1))),
                Tsurf=T_SURF * ((SUN_ABSORBED + P) / SUN_ABSORBED) ** 0.25,
                tAtm=E_ATMOSPHERE / P, tOceans=E_OCEANS / P, tCrust=E_CRUST / P, tMantle=E_MANTLE / P, tUnbind=E_UNBIND / P)


# 2 ---------------------------------------------------------------- hydrogen harvesting
N_CM3, V_SW, SUN_MASS_LOSS, EPS_FUSION = 5, 4.0e5, 1.5e9, 0.0071
MASS_FLUX = N_CM3 * 1e6 * M_P * V_SW
VE = C * math.sqrt(2 * EPS_FUSION)
M_NEED = M_E * (math.exp(DV_REQ / VE) - 1)


def hydrogen(R_RE, mdot=None):
    mdot = MASS_FLUX * math.pi * (R_RE * R_E) ** 2 if mdot is None else mdot
    m = mdot * T_MISSION
    dv = VE * math.log(1 + m / M_E)
    return dict(mdot=mdot, m10k=m, frac=m / M_NEED, dv=dv, a10k=a_after_dv(dv))


# 3 ---------------------------------------------------------------- superconducting magnetic arms
BR0, K_SPIRAL = 3.5e-9, 2.7e-6 * AU / 4.0e5
N_ARMS, L_ARM, B_CRIT, A_CABLE = 8, 2 * R_E, 45.0, 1.0
L_TOT = N_ARMS * L_ARM


def imf(r_au):
    return BR0 * math.sqrt(1 / r_au ** 4 + (K_SPIRAL / r_au) ** 2)


I_MAX = 2 * math.pi * A_CABLE * B_CRIT / MU0
I_REQ = F_REQ / (L_TOT * imf(1))


def current_to_arrive():
    """Smallest current per arm that reaches 2.7 AU within 10,000 years (best case, quench ignored) - bisection."""
    lo, hi = I_REQ, I_REQ * 100
    for _ in range(40):
        m = math.sqrt(lo * hi)
        if spiral(lambda r: m * L_TOT * imf(r), T_MISSION_Y, 600)[1] is not None:
            hi = m
        else:
            lo = m
    return hi


def magnet(I):
    a_end, t_reach = spiral(lambda r: I * L_TOT * imf(r), T_MISSION_Y, 1500)
    return dict(I=I, F1=I * L_TOT * imf(1), aEnd=a_end, tReach=t_reach, quench=I > I_MAX)


# 4 ---------------------------------------------------------------- Dyson beam on a held reflector
P_DYSON = 1.29e25
F_DYSON = P_DYSON / C


def summary():
    l, s, h, hw, m, mi = laser(P_REQ), laser(P_SAFE), hydrogen(100), hydrogen(0, SUN_MASS_LOSS), magnet(I_MAX), magnet(I_REQ)
    return dict(DV_REQ=DV_REQ, F_REQ=F_REQ, P_REQ=P_REQ, P_SAFE=P_SAFE, Tsurf_req=l["Tsurf"], tOceans=l["tOceans"],
                tUnbind=l["tUnbind"], a10k_safe=s["a10k"], h100_m=h["m10k"], h_need=M_NEED, h100_frac=h["frac"],
                hw_frac=hw["frac"], hw_a=hw["a10k"], I_MAX=I_MAX, I_REQ=I_REQ, mag_F=m["F1"], mag_a=m["aEnd"],
                magreq_a=mi["aEnd"], magreq_t=mi["tReach"], I_ARRIVE=current_to_arrive(), dyson_frac=F_DYSON / F_REQ,
                dyson_T=T_SURF * ((SUN_ABSORBED + P_DYSON) / SUN_ABSORBED) ** 0.25)


def main():
    s = summary()
    km = lambda a: (a - 1) * AU / 1e3
    print(f"Target: 1 -> 2.7 AU in {T_MISSION_Y:,} years: dv {s['DV_REQ']:,.0f} m/s, average push {s['F_REQ']:.3e} N\n")
    print("1 LASER PUSH (radiation pressure, F = P/c)")
    print(f"  power needed {s['P_REQ']:.3e} W = {s['P_REQ'] / 3.828e26:.1%} of the Sun's output, "
          f"{s['P_REQ'] / SUN_ABSORBED:.2e} x the sunlight Earth absorbs")
    print(f"  surface temperature at that power ~{s['Tsurf_req']:,.0f} K; energy to boil the oceans delivered in {s['tOceans']:.0f} s,"
          f" to unbind Earth in {s['tUnbind'] / 86400:.1f} days")
    print(f"  'safe' power (+2 K) {s['P_SAFE']:.2e} W moves Earth {km(s['a10k_safe']) * 1e3:.0f} m in 10,000 y -> FAILS\n")
    print("2 HYDROGEN HARVESTING (ideal fusion rocket)")
    print(f"  hydrogen needed {s['h_need']:.3e} kg; a 100 R_E scoop collects {s['h100_m']:.3e} kg in 10,000 y = {s['h100_frac']:.1e} of it")
    print(f"  the ENTIRE solar wind for 10,000 y: {s['hw_frac']:.0%} of the need -> a = {s['hw_a']:.3f} AU -> FAILS\n")
    print("3 SUPERCONDUCTING MAGNETIC ARMS (Lorentz force in the Sun's field)")
    print(f"  current needed per arm {s['I_REQ']:.2e} A; the cable quenches above {s['I_MAX']:.2e} A")
    print(f"  best possible: {s['mag_F']:.2e} N -> Earth moves {km(s['mag_a']) * 1e3:.0f} m in 10,000 y")
    print(f"  even with the impossible current: a = {s['magreq_a']:.2f} AU after 10,000 y (field falls with distance);"
          f" arriving in time needs {s['I_ARRIVE']:.2e} A per arm -> FAILS")
    print("  and in reality the arms are inside the magnetosphere: only Earth's own field acts -> internal forces, net 0\n")
    print("4 DYSON BEAM ON A REFLECTOR HELD BY 8 ARMS")
    print(f"  full beam pushes with {F_DYSON:.2e} N = {s['dyson_frac']:.0%} of the need; the arms overload and fail;"
          f" the absorbed beam would heat Earth to ~{s['dyson_T']:,.0f} K -> FAILS")


if __name__ == "__main__":
    main()
