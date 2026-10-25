# OCTO-EARTH - results report

Generated 2026-10-25 06:12:28  |  preset **hypothetical**

## Feasibility gate (analytic)

**FEASIBLE within this model (physically hypothetical scale)**

- Required slow-spiral delta-v 1 -> 2.7 AU: 11,658 m/s (Hohmann two-impulse: 10,997 m/s)
- Orbital energy change of Earth+Moon: 1.697e+33 J; angular momentum change 1.741e+40 kg m^2/s
- Average power just for the orbital energy change over 10,000 y: 5.377e+21 W  (= 3.10e+04 x all sunlight intercepted by Earth)
- Minimum launch energy with this reservoir: 1.664e+35 J -> average power 5.273e+23 W (= 1.38e-03 x total solar luminosity)
- Reaction mass needed at max speed: 1.415e+22 kg (0.193 lunar masses); available 3.000e+22 kg
- Max launch speed = 1.67% of c (non-relativistic model; error grows above ~10% c)
- Solar collectors: 30% of the Sun's output = 1.148e+26 W intercepted at 0.5 AU; electricity delivered to the arms 1.290e+25 W with Earth at 1 AU, 1.114e+25 W at 2.7 AU; launcher peak demand (max packet rate and speed) 6.338e+24 W -> the beams supply 1.76x the peak demand
- [OK ] mass: needs launch speed 2.355e+06 m/s with ALL 3.00e+22 kg; max allowed 5.000e+06 m/s (mass-limited delta-v capacity 2.475e+04 m/s vs required 1.166e+04 m/s)
- [OK ] energy: minimum energy drawn 1.664e+35 J vs available 3.817e+36 J (reserve 3.000e+35 J + collectors 1.114e+25 W x 10,000 y)
- [OK ] rate: at max packet rate and speed the transfer needs >= 1,771 years (mission 10,000 y)
- [OK ] combined: slowest speed meeting mass AND time limits 2.355e+06 m/s -> needs 3.000e+22 kg and 1.664e+35 J

## Physics checks

- not run

## Training

State finished, 1000 episodes, 93690 timesteps, 1700 s wall time (fast model).
Best model: episode 50, selection = lexicographic: success rate, then lowest terminal orbit error, then lowest energy use (deterministic, seeds 1001-1002).
Hyperparameters: {"algorithm": "PPO (Stable-Baselines3)", "net_arch": [64, 64], "n_steps": 1000, "batch_size": 200, "n_epochs": 10, "learning_rate": 0.0003, "gamma": 0.995, "gae_lambda": 0.95, "log_std_init": -0.5, "seed": 1}
Deterministic eval at selection: {"success_rate": 0.5, "error": 0.047045788568952224, "energy_frac": 0.5793407742658734, "years": 7650.0, "reward": 128.65269895435245, "a_au": 2.6576779874443943, "e": 0.011075243532827269, "score": [0.5, -0.047, -0.5793407742658734], "episode": 50, "timesteps": 3823}

## Comparison (physical metrics; reward shown separately)

| policy | fast success (3 seeds) | hi-fi result | hi-fi a [AU] | e | years | asteroid used | energy used [J] | validated | reward (hi-fi) |
|---|---|---|---|---|---|---|---|---|---|
| No ejection | 0% | deadline | 1.0000 | 0.0122 | 10,000 | 0.0% | 0.000e+00 | False | -53.3 |
| Hand-designed heuristic | 100% | success | 2.6903 | 0.0072 | 3,600 | 95.2% | 1.756e+35 | True | 214.7 |
| Best PPO model | 67% | success | 2.6854 | 0.0145 | 7,600 | 99.5% | 1.732e+35 | True | 194.8 |

## Mars safety (high-fidelity runs)

Safe distance 0.0246 AU (Earth came closer to Mars than the planetary-safety distance (or collided). Mars's tides or its pull on the Moon would exceed the limits set to keep people safe (failure).)

| policy | closest Mars pass [AU] | at year | safe | Mars tide (x Moon's) | Mars pull on Moon orbit | shield overrides | Mars a: start -> end [AU] | Mars e |
|---|---|---|---|---|---|---|---|---|
| No ejection | 0.3670 | 8,525 | True | 3.00e-06 | 7.37e-08 | 0 | 1.5237 -> 1.5237 | 0.0934 -> 0.0982 |
| Hand-designed heuristic | 0.0502 | 2,070 | True | 1.17e-03 | 2.88e-05 | 272 | 1.5237 -> 1.5577 | 0.0934 -> 0.0925 |
| Best PPO model | 0.0470 | 3,149 | True | 1.43e-03 | 3.52e-05 | 296 | 1.5237 -> 1.5114 | 0.0934 -> 0.0971 |

Mars's final orbit is chaotic (sensitive to small differences); see README, 'Mars'.

## Solar collectors and microwave power (high-fidelity runs)

Belt at 0.5 AU intercepting 30% of the Sun's output (1.148e+26 W): 2.11e+22 m^2 of collectors (+/- 17.5 deg around Earth's orbital plane), 5e+06 collectors of ~64,950 km each, ~2.1e+21 kg of material (2.9% of the Moon) at 0.1 kg/m^2; light pressure = 0.8% of the Sun's gravity on them. Chain: solar cells 25% -> microwaves 80% (5.8 GHz) -> captured by the 8 arm rectennas (Goubau, 30 km transmitters, 1000 km rectennas) -> electricity 85%.

| policy | energy from collectors [J] | from reserve [J] | microwaves absorbed [J] | mean / peak beam power [W] | capture efficiency | heat at the arms (mean) | microwave push (not applied) |
|---|---|---|---|---|---|---|---|
| No ejection | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.00e+00 / 0.00e+00 | 0.999-0.999 | 0.00e+00 W (= 0.0e+00 x the sunlight Earth intercepts) | 0 m/s |
| Hand-designed heuristic | 1.756e+35 | 0.000e+00 | 2.066e+35 | 1.82e+24 / 1.83e+24 | 0.881-0.999 | 1.05e+24 W (= 6.0e+06 x the sunlight Earth intercepts) | 114 m/s |
| Best PPO model | 1.732e+35 | 0.000e+00 | 2.037e+35 | 8.49e+23 / 1.84e+24 | 0.872-0.999 | 4.88e+23 W (= 2.8e+06 x the sunlight Earth intercepts) | 112 m/s |

The microwave push is an estimate (not applied). The heat at the arms is radiated away from Earth by assumption (enabler E1: directional arm radiators); the share arm_heat_to_earth_frac that reaches Earth is part of the climate model.

Panels: rated 25% at 25 C, operating 0.165 (panel temperature 450 K -> x0.696; age x0.950). Build-up: 30% at year 0, complete after 400 y. Replacement 4.22e+19 kg/yr (95% recycled); fresh material over the mission 2.32e+22 kg = 7.0% of Mercury. Earth's sunlight lane kept clear. Belt's pull on Earth 1e-09 of the Sun's.

## Energy ledger (high-fidelity runs, whole mission)

| policy | sunlight on collectors | reflected | collector heat | manufacturing | transmitter heat | beam spill | rectenna heat | packet kinetic energy | launcher heat | sunlight on mirrors | mirror light onto Earth | mirror light past Earth | mirror heat | reserve used | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| No ejection | 3.573e+37 | 3.573e+36 | 3.216e+37 | 4.174e+31 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 4.857e+29 | 0.000e+00 | 4.371e+29 | 4.857e+28 | 0.000e+00 | 3e-16 |
| Hand-designed heuristic | 1.254e+37 | 1.254e+36 | 1.102e+37 | 1.474e+31 | 5.299e+34 | 5.384e+33 | 3.099e+34 | 8.779e+34 | 8.779e+34 | 8.200e+28 | 1.038e+28 | 6.342e+28 | 8.200e+27 | 0.000e+00 | 6e-16 |
| Best PPO model | 2.703e+37 | 2.703e+36 | 2.407e+37 | 3.162e+31 | 5.232e+34 | 5.561e+33 | 3.056e+34 | 8.659e+34 | 8.659e+34 | 1.738e+29 | 2.033e+28 | 1.361e+29 | 1.738e+28 | 0.000e+00 | 4e-16 |

## Collector belt: do the collectors stay on their orbits? (high-fidelity runs)

| policy | max deviation WITH station keeping | control delta-v | WITHOUT: drift along the orbit | WITHOUT: distance-from-Sun swing | beam recoil / Sun's pull |
|---|---|---|---|---|---|
| No ejection | 12 km (rms 4) | 1 m/s/yr | 1,850,236 km (28.5 slots of 64,950 km) | 54,624 km | 0.0e+00 |
| Hand-designed heuristic | 730 km (rms 604) | 97 m/s/yr | 154,021,391 km (2,371.4 slots of 64,950 km) | 8,144,031 km | 1.4e-04 |
| Best PPO model | 437 km (rms 286) | 45 m/s/yr | 140,607,557 km (2,164.9 slots of 64,950 km) | 7,870,529 km | 8.3e-05 |

## Climate (high-fidelity runs; part of the RL task)

Target 288 K +- 5 K; livable band +- 15 K. Space mirrors ON (1.6e+15 m^2; at 2.7 AU they can send 1.10 x today's sunlight).

| policy | final surface T | climate ok | max deviation | years outside livable band | mirror light onto Earth | without mirrors it would settle at | ice line | growing area | waste heat reaching Earth |
|---|---|---|---|---|---|---|---|---|---|
| No ejection | 288.0 K | True | 2.1 K | 0 | 0.00e+00 J | 290 K | 71 deg | 75 % | 0.00e+00 J |
| Hand-designed heuristic | 288.4 K | True | 2.6 K | 0 | 1.04e+28 J | 157 K | 71 deg | 75 % | 1.24e+26 J |
| Best PPO model | 284.9 K | True | 4.9 K | 0 | 2.03e+28 J | 155 K | 65 deg | 70 % | 1.23e+26 J |

## Enablers (assumed technologies - NOT shown to be possible)

| id | enabler | provides | parameters | gap to reality | modelled |
|---|---|---|---|---|---|
| E1 | Earth launcher system: docked asteroid + launch ring with 8 arms + arm radiators | Throws packets of up to 1e18 kg of asteroid material at up to 5,000 km/s, 8 times per year (all 8 arms throw one packet together), with 50 % of the drawn energy ending up as packet kinetic energy; the arms reject their waste heat away from Earth; the launches do not damage Earth. | asteroid_mass_kg, max_packet_mass_kg, max_ejection_speed_ms, min_ejection_speed_ms, ejections_per_year, launch_efficiency, energy_budget_j, arm_heat_to_earth_frac | No known structure or material; each launch carries ~1e30 J (millions of Chicxulub impacts); moving a 3e22 kg body to Earth and docking it is far beyond any technology; ~1e19 m^2 of 1,000 K radiator per mission. | Momentum, energy, escape from Earth (+Moon) gravity, packet fates, finite reservoir, waste heat into the climate model. NOT modelled: the hardware, launch shock, atmosphere loss and heating of Earth (problem P6). |
| E2 | Solar power belt with microwave power beaming (collectors + transmitters + arm rectennas) | 5 million thin-film collectors at 0.5 AU intercept 30 % of the Sun's output, turn it into electricity and beam it as 5.8 GHz microwaves (30 km phased arrays) to 1,000 km rectennas on the 8 arms; they hold their slots by sail tilt, keep a clear sunlight lane for Earth and Mars, keep beam spill off Earth and are built up / replaced by factories using Mercury material. | collector_fraction, collector_orbit_au, collector_count, collector_areal_density, pv_efficiency, pv_temp_coeff, collector_absorptance, collector_emissivity, pv_degradation_per_year, collector_lifetime_years, collector_recycle_frac, collector_initial_built, collector_build_years, collector_embodied_energy_jkg, microwave_efficiency, microwave_freq_ghz, transmitter_diameter_km, rectenna_diameter_km, rectenna_efficiency, earth_lane_clear, collector_sk_authority, beam_spill_to_earth_frac | A Dyson-swarm-scale project (~1e13 x all solar panels ever built); record power beaming is ~kW over km, here ~1e24 W over 0.5-3 AU with pointing to ~1e-12 rad; beam intensity at the rectennas ~2e8 x sunlight. | Panel temperature and efficiency, degradation and replacement, build-up, manufacturing energy and mass, Goubau diffraction capture, Sun blocking, every loss in the energy ledger, clear lanes, the orbits of 12+12 followed collectors with / without station keeping (beam recoil, planets, light pressure). |
| E3 | Space mirror ring around Earth (soletta) | 1.6e15 m^2 of aluminium-film mirrors (8e12 kg) in a terminator orbit 20 Earth radii out reflect extra sunlight onto Earth: up to 1.10 x today's sunlight at 2.7 AU. The agent sets how much (mirrors not needed are turned so their light passes Earth). | mirrors, mirror_area_m2, mirror_reflectivity, mirror_orbit_re, mirror_areal_density | Physically possible (only reflection, no new physics, no energy needed). Demonstrated small: Znamya-2 (1993) 20 m mirror lit a 5 km spot on Earth; solar sails IKAROS (2010, ~200 m^2). Needed: ~1e13 x that area; film made from asteroid aluminium; formation flying and pointing of a million 40 km mirrors. | Capacity (1/r^2), light sent, light turned away, mirror heat (energy ledger), mass; the climate effect. Not the mirrors' orbits, light pressure, pointing errors or the light pattern on Earth. |

Instruments of each enabler (all numbers: CALCULATIONS_REPORT, made by calc_report.py):

* E1 - Asteroid reservoir (docked to Earth): Supplies the reaction mass that is thrown away. (asteroid_mass_kg)
* E1 - Launch ring + 8 launch arms (mass driver): Accelerates each packet to the commanded speed in the commanded direction; all 8 arms throw one packet together. (max_packet_mass_kg, max/min_ejection_speed_ms, ejections_per_year, launch_efficiency)
* E1 - Arm power conditioning + energy reserve: Turns the arm electricity into launch energy; a reserve covers launches when the beams are short. (energy_budget_j)
* E1 - Directional arm radiators: Reject the launcher and rectenna waste heat to space, away from Earth. (arm_heat_to_earth_frac)
* E2 - Thin-film solar collector (x 5 million): Intercepts sunlight and makes electricity. (collector_fraction, collector_orbit_au, collector_count, collector_areal_density, pv_efficiency, pv_temp_coeff, collector_absorptance, collector_emissivity, pv_degradation_per_year, collector_lifetime_years)
* E2 - Microwave phased-array transmitter (one per collector): Turns the electricity into a focused 5.8 GHz beam aimed at the arms; suppresses side lobes so spill misses Earth. (microwave_efficiency, microwave_freq_ghz, transmitter_diameter_km, beam_spill_to_earth_frac)
* E2 - Rectenna on each of the 8 arm tips: Receives the microwaves and turns them back into electricity for the launcher. (rectenna_diameter_km, rectenna_efficiency)
* E2 - Sail-tilt station keeping (on each collector): Uses part of the light-pressure push to hold the collector in its slot (no propellant). (collector_sk_authority)
* E2 - Clear sunlight lane: Collectors leave a moving gap so Earth and Mars see the whole Sun. (earth_lane_clear)
* E2 - Collector factories (Mercury material): Build the belt up and replace aged panels. (collector_initial_built, collector_build_years, collector_recycle_frac, collector_embodied_energy_jkg)
* E3 - Mirror film panels (aluminium, 5 g/m^2): Reflect sunlight; 90 % reflectivity. (mirror_area_m2, mirror_reflectivity, mirror_areal_density)
* E3 - Terminator orbit at 20 Earth radii + pointing (45 degree tilt): Keeps the mirrors in sunlight and aims their light at Earth or past it (the agent's action a4). (mirror_orbit_re, mirrors)

## Major problems

| id | problem | status | what was done | what remains |
|---|---|---|---|---|
| P1 | Climate test integration | addressed | Habitability used to be only reported. Now a two-box climate model (with ice-albedo snowball risk) runs inside the RL environment: the agent aims the space mirrors (action a4), the reward has a climate term, the observation has 4 climate numbers and success requires |T_surface - target| <= tolerance. | Global-mean model only (no seasons, regions, water vapour, carbon cycle, biology). |
| P2 | Energy generation mapped exactly from the panels | addressed | Panel temperature (energy balance, ~450 K at 0.5 AU) -> operating efficiency; age degradation with replacement; linear build-up; manufacturing energy and fresh mass (vs. Mercury's mass); Earth's shading / clear lane; waste heat at collectors, transmitters, rectennas and launcher; every joule in a ledger that closes to round-off (check 18). | Panel temperature is a single global value; no micrometeoroids or eclipses. |
| P3 | Energy transfer and conservation | addressed | Ledger: sunlight on collectors + reserve = reflected + collector heat + manufacturing + transmitter heat + beam spill + rectenna heat + packet kinetic energy + launcher heat; mirrors: sunlight on mirrors = light onto Earth + light turned past Earth + mirror heat. Residual shown in the viewer and checked. | The packets' kinetic energy is then followed by the mechanical checks (4-6). |
| P4 | Enablers declared explicitly | addressed | Exactly 3 enablers E1-E3 (launcher system, solar power belt with beaming, space mirrors), each with its instruments, what it provides, the parameters, the gap to reality and what is modelled. | - |
| P5 | Arm waste heat (~1e24 W, 6e6 x Earth's sunlight) | open | Only harmless because of the directional arm radiators of enabler E1 (1e-9 leaks to Earth). With an undirected radiator Earth would boil. | No physical radiator design. |
| P6 | Launch damage to Earth | open | Each 1e18 kg packet leaves at ~2,500 km/s carrying ~3e30 J; shock, atmospheric loss and heating are not modelled. | Assumed as part of enabler E1 (launches do not damage Earth). |
| P7 | Other planets | open | Venus, Mercury (near the 0.5 AU belt) and Jupiter are not simulated; the final orbit's long-term stability with Jupiter is not assessed. | Add them to the N-body model. |
| P8 | Day length, seasons, tides | open | Spin, obliquity and tides are not modelled; the mirrors' illumination pattern on Earth is not modelled. | - |
| P9 | Climate enabler must be physically possible | addressed | The hypothetical beam-powered artificial sun was replaced by space mirrors (E3): plain reflection of sunlight, no energy source needed, flown at small scale (Znamya-2). The weather view shows what the mirrors keep alive (ice line, growing area, clouds and winds) and what happens without them (snowball Earth). | Building 1.6e15 m^2 of mirror and keeping it pointed is far beyond today's industry. |

## Best model: final orbit vs target (high-fidelity run)

| criterion | value | required | result |
|---|---|---|---|
| Semi-major axis a | 2.6854 AU | 2.7 +/- 0.05 | PASS |
| Eccentricity e | 0.014497  | <= 0.05 | PASS |
| Inclination | 0.87198 deg | <= 5.0 | PASS |
| Arrival time | 7600 yr | <= 10000 | PASS |
| Moon still bound | True  | yes | PASS |
| Closest approach to Mars (planetary safety) | 0.046964 AU | >= 0.0246 AU | PASS |
| Stays in the target window for 200 yr without thrust (largest deviation of a from target) | 0.014571 AU | <= 0.05 AU, e <= 0.05, Moon bound | PASS |
| Surface temperature at the end (climate model, space mirrors aimed by the agent) | 284.86 K | 288 +/- 5 | PASS |
| Years outside the livable band (target +/- 15 K) during the mission | 0 yr | 0 (reported) | PASS |

## Best model: decisions (high-fidelity run)

Classes: helped 76

| years | throttle | packets | packet kg | speed km/s | prograde cos | dv m/s | da AU | e | effect |
|---|---|---|---|---|---|---|---|---|---|
| 0-100 | 0.47 | 800 | 3.74e+20 | 2254 | 1.00 | 138.7 | +0.0093 | 0.0166 | helped |
| 100-200 | 0.48 | 800 | 3.82e+20 | 2364 | 1.00 | 148.8 | +0.0102 | 0.0164 | helped |
| 200-300 | 0.50 | 800 | 3.98e+20 | 2425 | 0.99 | 158.9 | +0.0109 | 0.0163 | helped |
| 300-400 | 0.52 | 800 | 4.19e+20 | 2554 | 0.98 | 176.1 | +0.0122 | 0.0161 | helped |
| 400-500 | 0.47 | 800 | 3.77e+20 | 2295 | 1.00 | 142.3 | +0.0102 | 0.0158 | helped |
| 500-600 | 0.48 | 800 | 3.81e+20 | 2332 | 0.99 | 146.3 | +0.0106 | 0.0157 | helped |
| 600-700 | 0.47 | 800 | 3.76e+20 | 2325 | 1.00 | 143.8 | +0.0106 | 0.0156 | helped |
| 700-800 | 0.50 | 800 | 4.02e+20 | 2426 | 0.98 | 160.6 | +0.0119 | 0.0154 | helped |
| 800-900 | 0.51 | 800 | 4.1e+20 | 2458 | 0.97 | 166.1 | +0.0123 | 0.0151 | helped |
| 900-1000 | 0.52 | 800 | 4.18e+20 | 2528 | 0.98 | 173.9 | +0.0132 | 0.0150 | helped |
| 1000-1100 | 0.49 | 800 | 3.9e+20 | 2355 | 0.99 | 151.4 | +0.0119 | 0.0147 | helped |
| 1100-1200 | 0.46 | 800 | 3.7e+20 | 2291 | 0.99 | 139.8 | +0.0112 | 0.0146 | helped |
| 1200-1300 | 0.51 | 800 | 4.12e+20 | 2505 | 0.98 | 169.9 | +0.0136 | 0.0143 | helped |
| 1300-1400 | 0.54 | 800 | 4.33e+20 | 2604 | 0.94 | 185.8 | +0.0146 | 0.0139 | helped |
| 1400-1500 | 0.48 | 800 | 3.85e+20 | 2344 | 1.00 | 148.7 | +0.0126 | 0.0139 | helped |
| 1500-1600 | 0.46 | 800 | 3.69e+20 | 2269 | 0.99 | 137.9 | +0.0118 | 0.0136 | helped |
| 1600-1700 | 0.46 | 800 | 3.67e+20 | 2267 | 0.99 | 137.0 | +0.0119 | 0.0135 | helped |
| 1700-1800 | 0.50 | 800 | 4e+20 | 2425 | 0.98 | 159.9 | +0.0139 | 0.0133 | helped |
| 1800-1900 | 0.52 | 800 | 4.17e+20 | 2562 | 0.98 | 176.0 | +0.0155 | 0.0131 | helped |
| 1900-2000 | 0.51 | 800 | 4.08e+20 | 2476 | 0.99 | 166.4 | +0.0152 | 0.0129 | helped |
| 2000-2100 | 0.46 | 800 | 3.67e+20 | 2262 | 0.99 | 136.8 | +0.0126 | 0.0127 | helped |
| 2100-2200 | 0.48 | 800 | 3.82e+20 | 2312 | 1.00 | 145.5 | +0.0138 | 0.0125 | helped |
| 2200-2300 | 0.50 | 800 | 3.96e+20 | 2467 | 0.99 | 161.0 | +0.0157 | 0.0121 | helped |
| 2300-2400 | 0.52 | 800 | 4.13e+20 | 2520 | 0.98 | 171.7 | +0.0166 | 0.0119 | helped |
| 2400-2500 | 0.47 | 800 | 3.76e+20 | 2312 | 0.99 | 143.1 | +0.0145 | 0.0116 | helped |
| 2500-2600 | 0.48 | 776 | 3.7e+20 | 2354 | 1.00 | 143.5 | +0.0148 | 0.0107 | helped |
| 2600-2700 | 0.49 | 800 | 3.89e+20 | 2357 | 0.99 | 151.2 | +0.0156 | 0.0106 | helped |
| 2700-2800 | 0.50 | 800 | 4.17e+20 | 2440 | 0.99 | 167.8 | +0.0187 | 0.0098 | helped |
| 2800-2900 | 0.52 | 800 | 4.15e+20 | 2577 | 0.98 | 176.3 | +0.0188 | 0.0098 | helped |
| 2900-3000 | 0.53 | 728 | 3.83e+20 | 2570 | 0.97 | 162.2 | +0.0164 | 0.0095 | helped |
| 3000-3100 | 0.50 | 776 | 4.35e+20 | 2482 | 0.99 | 178.0 | +0.0098 | 0.0098 | helped |
| 3100-3200 | 0.47 | 800 | 3.84e+20 | 2296 | 0.99 | 145.2 | +0.0163 | 0.0098 | helped |
| 3200-3300 | 0.50 | 800 | 3.97e+20 | 2444 | 1.00 | 159.9 | +0.0191 | 0.0094 | helped |
| 3300-3400 | 0.51 | 800 | 4.11e+20 | 2520 | 0.99 | 171.0 | +0.0202 | 0.0095 | helped |
| 3400-3500 | 0.46 | 800 | 3.65e+20 | 2260 | 0.98 | 136.0 | +0.0161 | 0.0096 | helped |
| 3500-3600 | 0.48 | 800 | 3.84e+20 | 2368 | 1.00 | 149.9 | +0.0181 | 0.0097 | helped |
| 3600-3700 | 0.48 | 800 | 3.83e+20 | 2320 | 1.00 | 146.6 | +0.0179 | 0.0102 | helped |
| 3700-3800 | 0.47 | 800 | 3.74e+20 | 2254 | 0.99 | 139.0 | +0.0167 | 0.0102 | helped |
| 3800-3900 | 0.49 | 800 | 3.92e+20 | 2373 | 0.99 | 153.5 | +0.0201 | 0.0107 | helped |
| 3900-4000 | 0.49 | 800 | 3.94e+20 | 2383 | 0.99 | 155.0 | +0.0204 | 0.0109 | helped |
| 4000-4100 | 0.49 | 800 | 3.96e+20 | 2393 | 0.99 | 156.2 | +0.0209 | 0.0108 | helped |
| 4100-4200 | 0.50 | 800 | 4e+20 | 2416 | 0.98 | 159.3 | +0.0218 | 0.0109 | helped |
| 4200-4300 | 0.49 | 800 | 4.01e+20 | 2342 | 1.00 | 154.9 | +0.0220 | 0.0108 | helped |
| 4300-4400 | 0.45 | 800 | 3.62e+20 | 2214 | 0.98 | 132.3 | +0.0198 | 0.0107 | helped |
| 4400-4500 | 0.45 | 800 | 3.6e+20 | 2232 | 0.98 | 132.6 | +0.0182 | 0.0106 | helped |
| 4500-4600 | 0.49 | 800 | 3.89e+20 | 2402 | 1.00 | 154.4 | +0.0226 | 0.0100 | helped |
| 4600-4700 | 0.46 | 800 | 3.69e+20 | 2241 | 0.99 | 136.4 | +0.0199 | 0.0096 | helped |
| 4700-4800 | 0.55 | 800 | 4.36e+20 | 2617 | 0.95 | 188.6 | +0.0276 | 0.0092 | helped |
| 4800-4900 | 0.51 | 800 | 4.08e+20 | 2546 | 0.99 | 171.7 | +0.0267 | 0.0091 | helped |
| 4900-5000 | 0.45 | 800 | 3.63e+20 | 2256 | 0.98 | 135.1 | +0.0214 | 0.0091 | helped |
| 5000-5100 | 0.45 | 800 | 3.63e+20 | 2218 | 0.98 | 133.0 | +0.0215 | 0.0085 | helped |
| 5100-5200 | 0.46 | 800 | 3.64e+20 | 2239 | 0.98 | 134.7 | +0.0220 | 0.0084 | helped |
| 5200-5300 | 0.54 | 800 | 4.33e+20 | 2615 | 0.96 | 187.0 | +0.0304 | 0.0077 | helped |
| 5300-5400 | 0.48 | 800 | 3.87e+20 | 2356 | 1.00 | 150.5 | +0.0261 | 0.0074 | helped |
| 5400-5500 | 0.54 | 800 | 4.28e+20 | 2566 | 0.95 | 181.5 | +0.0305 | 0.0068 | helped |
| 5500-5600 | 0.52 | 800 | 4.13e+20 | 2531 | 0.99 | 172.7 | +0.0311 | 0.0065 | helped |
| 5600-5700 | 0.49 | 800 | 3.91e+20 | 2413 | 1.00 | 155.8 | +0.0291 | 0.0066 | helped |
| 5700-5800 | 0.51 | 800 | 4.08e+20 | 2476 | 0.99 | 166.9 | +0.0312 | 0.0060 | helped |
| 5800-5900 | 0.51 | 800 | 4.09e+20 | 2456 | 0.98 | 166.1 | +0.0317 | 0.0058 | helped |
| 5900-6000 | 0.50 | 800 | 3.96e+20 | 2460 | 1.00 | 161.0 | +0.0325 | 0.0043 | helped |
| 6000-6100 | 0.49 | 800 | 3.9e+20 | 2360 | 1.00 | 151.9 | +0.0310 | 0.0043 | helped |
| 6100-6200 | 0.48 | 800 | 3.87e+20 | 2333 | 1.00 | 149.1 | +0.0307 | 0.0044 | helped |
| 6200-6300 | 0.45 | 800 | 3.63e+20 | 2214 | 0.98 | 132.9 | +0.0275 | 0.0033 | helped |
| 6300-6400 | 0.53 | 800 | 4.23e+20 | 2570 | 0.98 | 179.7 | +0.0382 | 0.0030 | helped |
| 6400-6500 | 0.51 | 800 | 4.08e+20 | 2524 | 0.99 | 170.3 | +0.0378 | 0.0026 | helped |
| 6500-6600 | 0.51 | 800 | 4.07e+20 | 2545 | 1.00 | 171.1 | +0.0394 | 0.0027 | helped |
| 6600-6700 | 0.53 | 800 | 4.25e+20 | 2581 | 0.97 | 181.1 | +0.0412 | 0.0027 | helped |
| 6700-6800 | 0.45 | 800 | 3.56e+20 | 2226 | 0.97 | 131.2 | +0.0307 | 0.0027 | helped |
| 6800-6900 | 0.46 | 800 | 3.66e+20 | 2240 | 0.98 | 135.7 | +0.0340 | 0.0016 | helped |
| 6900-7000 | 0.49 | 800 | 3.92e+20 | 2430 | 1.00 | 157.5 | +0.0381 | 0.0020 | helped |
| 7000-7100 | 0.49 | 800 | 3.89e+20 | 2360 | 1.00 | 151.9 | +0.0392 | 0.0009 | helped |
| 7100-7200 | 0.46 | 800 | 3.69e+20 | 2275 | 0.99 | 138.7 | +0.0360 | 0.0093 | helped |
| 7200-7300 | 0.50 | 800 | 4.03e+20 | 2500 | 1.00 | 166.6 | +0.0446 | 0.0152 | helped |
| 7300-7400 | 0.47 | 800 | 3.73e+20 | 2331 | 0.99 | 143.9 | +0.0401 | 0.0140 | helped |
| 7400-7500 | 0.54 | 800 | 4.33e+20 | 2603 | 0.96 | 186.5 | +0.0503 | 0.0138 | helped |
| 7500-7600 | 0.48 | 800 | 3.83e+20 | 2369 | 1.00 | 150.1 | +0.0435 | 0.0145 | helped |

## Training episode outcomes (stochastic exploration)

- deadline: 559 (Mission time ran out before the target was reached.)
- resources_exhausted: 441 (All asteroid mass (or, with the solar collectors off, all propulsion energy) was used before reaching the target.)

> This is the best policy found in this training run, judged on a few deterministic test episodes. It is not proven to be globally optimal; other strategies (e.g. the heuristic) can be better on some measures.
> Reward is only the scoring rule used to train the agent. Mission success is judged separately from physics: final a, e and inclination, a ballistic stability check, and the Moon staying bound.

## Why the best model behaves as it does

- RESULT: the best PPO model reached the target in the high-fidelity N-body check: a = 2.6854 AU, e = 0.0145 after 7,600 simulated years, and the orbit stayed within tolerance for 200 ballistic years with the Moon bound.
- Direction: the impulse-weighted recoil direction on Earth has along-track (prograde) cosine 0.98, |radial| 0.13, |out-of-plane| 0.08; 0.0% of the impulse was retrograde. Along-track thrust raises orbital energy fastest (Gauss: da/dt = 2 a^2 v f_T / mu, with f_T the along-track acceleration); radial/normal components mostly waste reaction mass.
- Launch speed: mass-weighted mean 2.415e+06 m/s vs 2.355e+06 m/s, the minimum-energy speed that uses the whole asteroid (heuristic uses 2.480e+06 m/s). Momentum per joule falls as 2/u, so a lower speed saves energy but costs more mass.
- Most effective decisions (largest da per 100-y interval, fast model): t=7100 y: da=+0.0484 AU (prograde cos 0.96, u=2.68e+06 m/s, throttle 0.54); t=6600 y: da=+0.0405 AU (prograde cos 0.98, u=2.58e+06 m/s, throttle 0.53); t=7000 y: da=+0.0396 AU (prograde cos 1.00, u=2.43e+06 m/s, throttle 0.49). Least effective: t=100 y: da=+0.0093 AU (prograde cos 1.00, u=2.25e+06 m/s, throttle 0.47). At similar throttle, gains per interval grow with a, because at constant thrust acceleration da/dt ~ a^1.5.
- Timing: ejected in 75 of 75 decision intervals (first 100 y, last 7500 y). It keeps thrusting until the orbit enters the target window, where the episode ends.
- Resources (high-fidelity run): PPO used 99.5% of the asteroid and 57.7% of the energy budget; heuristic 95.2% / 58.5%; no-ejection baseline stays at a = 1.0000 AU.
- Versus the hand-designed heuristic: PPO arrives slower (7,600 vs 3,600 y) and uses less energy.
- This is the best model found in the tested training run, not a proven global optimum.

## Conservation (high-fidelity runs)

- No ejection: energy residual -3.0e-12 (rel. to |E0|), momentum 5.5e-11, angular momentum 8.9e-14, mass 0.0e+00; fragment merges 0
- Hand-designed heuristic: energy residual 3.6e-08 (rel. to |E0|), momentum 1.9e-11, angular momentum 1.5e-13, mass 1.4e-16; fragment merges 0
- Best PPO model: energy residual 2.6e-08 (rel. to |E0|), momentum 3.6e-11, angular momentum 1.0e-13, mass 0.0e+00; fragment merges 0

## Limitations

See README.md - simplified prototype, not a validated engineering or climate model.
