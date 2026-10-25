# OCTO-EARTH — RL code walkthrough

This guide explains the complete reinforcement-learning (RL) part of OCTO-EARTH: where each piece
lives in `octoearth_rl.py`, what it uses, and how the pieces connect. Line numbers refer to the
current file (3,885 lines). Every example number was produced by actually running the code.

---

## 1. The idea in one picture

```
            ┌───────────────────────── PPO agent (Stable-Baselines3 + PyTorch) ─────────────────────────┐
            │  policy network  31 numbers in  →  64 tanh  →  64 tanh  →  5 numbers out (action mean)    │
            │  value  network  31 numbers in  →  64 tanh  →  64 tanh  →  1 number out (expected return) │
            └───────────────▲────────────────────────────────────────────┬──────────────────────────────┘
                observation │ (31 numbers)                                │ action (5 numbers in [-1, 1])
                    reward  │ (1 number)                                  ▼
            ┌───────────────┴──────────── OctoEarthEnv (Gymnasium) ───────────────────────────────────────┐
            │ decode_action → packet mass, launch speed, thrust direction, mirror-light set-point         │
            │ engine.run_interval(100 years) → 800 packet launches + exact orbital motion (Numba)         │
            │ climate model + energy ledger → orbit elements → reward parts → termination → observation   │
            └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **One RL step = one decision = 100 simulated years.** The mission is at most 10,000 years, so an
  episode has at most **100 steps**.
* The agent never touches Earth's velocity. It only chooses *how to throw asteroid material*. The
  physics engine computes the recoil (momentum conservation) and charges the energy budget.

---

## 2. What it uses (libraries)

| library | version tested | used for | where |
|---|---|---|---|
| **numpy** | 2.3 | state vectors, observations, ledgers | everywhere |
| **numba** (`@njit`) | 0.66 | compiles the physics kernels to machine code, so one 100-year step takes milliseconds | section 5, lines 1005–1875 |
| **gymnasium** | 1.4 | the standard RL environment API (`gym.Env`, `spaces.Box`, `reset`, `step`) | section 7, lines 2080–2392 |
| **stable-baselines3** | 2.9 | the PPO algorithm, `MlpPolicy`, `DummyVecEnv`, `BaseCallback`, `model.save/load/predict` | `_train_worker` line 3415, `load_ppo` 3536, `model_policy` 2744 |
| **torch** (CPU) | 2.x | the neural networks and gradient descent inside SB3 | via SB3 (`torch.set_num_threads(1)`) |
| stdlib: `multiprocessing`, `threading`, `csv`, `json`, `http.server` | — | training / comparison worker process (p32), playback and belt threads, `training_log.csv`, results JSON, the viewer API | sections 10–11 |

`_dependency_guard()` (line 41) stops with a clear message if one of these is missing;
`RUN_OCTOEARTH.bat` installs them from `requirements.txt` into `.venv`.

---

## 3. Map of the RL-relevant code

| # | what | where (line) |
|---|---|---|
| 1 | `Config` dataclass: every tunable number incl. RL hyperparameters, reward weights and the Mars-safety limits | 146 |
| 2 | `PRESETS` (hypothetical / realistic) | 208 |
| 3 | `feasibility()`: analytic gate run *before* training (can the target be reached at all?) | 818 |
| 3b | `mars_safety_limits()`: turns the "people must not die" parameters into a safe distance (tide on Earth, Moon's orbit, hard minimum) and the shield's caution distance | 930 |
| 4 | `initial_conditions()`: random start phase of Earth, Moon and Mars per episode seed | 984 |
| 5 | physics kernels used by every step: `kepler_step` 1242, `rtn_to_inertial` 1298, `eject_core` 1308, `_pair_kick` 1350, `mars_forecast` 1366, `shield_decide` 1401, `_fast_eject` 1441, `_mars_kick` 1512, `fast_interval` 1584 | 1005–1875 |
| 6 | `FastEngine` (training physics) / `HifiEngine` (N-body validation) / `make_engine` | 1891 / 1972 / 2075 |
| 7 | **`OctoEarthEnv`**: spaces 2105, `reset` 2383, `_potential` 2424, `_obs` 2439, `_mars_obs` 2472, `decode_action` 2487, `in_target` 2500, **`step` 2506**, `_mars_record` 2601, `error_norm` 2642 | 2087–2389 |
| 8 | `gym.register("OctoEarth-v0")` | 2389 |
| 9 | baselines: `policy_none` 2704, `heuristic_speed` 2709, `policy_heuristic` 2722; wrapper `model_policy` 2744 | 2395–2440 |
| 10 | `run_episode` (plays one full mission with any policy) 2442, `validate_orbit` (2774-year coast check) 2465 | 2442–2480 |
| 11 | `App.start_training` 3393, `_eval_policy_fast` 3405, **`_train_worker` 3415**, `_on_episode` 3493 | 3019–3153 |
| 12 | comparison: `start_comparison` 3528, `load_ppo` 3536, `_compare_worker` 3543 | 3154–3240 |
| 13 | explanation and report: `failure_counts` 3811, `best_panel` 3819, `explain` 3914, `make_report` 3987 | 3434–3672 |
| 14 | CLI and start-up: `parse_args` 4180, `main` 4196 | 3778– |

---

## 4. Configuration (`Config`, line 146)

All RL settings are plain dataclass fields. You can override any of them from the command line with
`--set key=value`, or in the viewer's parameter panel.

| field | default | meaning |
|---|---|---|
| `train_episodes` | 1000 | training stops after this many finished episodes |
| `decision_interval_years` | 100 | simulated years per RL step (→ ≤ 100 steps per episode) |
| `mission_years` | 10,000 | deadline |
| `ejections_per_year` | 8 | packet launches per year (→ 800 per step) |
| `ppo_n_steps` | 1000 | steps collected before each PPO update (≈ 12 episodes) |
| `ppo_batch_size` | 200 | minibatch size for gradient steps |
| `ppo_epochs` | 10 | passes over each 1,000-step batch |
| `ppo_learning_rate` | 3e-4 | Adam step size |
| `ppo_gamma` | 0.995 | discount factor (horizon ≈ 1/(1−γ) = 200 steps, longer than an episode) |
| `eval_every_episodes` | 50 | deterministic test every 50 episodes |
| `seed` | 1 | PPO / environment random seed |
| `w_progress … w_failure` | see §6.4 | reward weights (scoring only, not physics) |
| `tol_a_au`, `tol_e`, `tol_inc_deg` | 0.05 AU, 0.05, 5° | success window around 2.7 AU |
| `mars_gravity` | on | Mars pulls Earth, Moon and fragments and is pulled back |
| `other_planets` | on | (p35) Mercury, Venus, Jupiter pull Earth, Moon, Mars, fragments and are pulled back (`_planet_kicks` in the fast model, N-body bodies 4–6 in the high-fidelity model); same safety rules as Mars, failure `planet_unsafe_approach`; Saturn and beyond not modelled |
| `mars_max_tide_ratio`, `mars_max_moon_perturbation`, `mars_min_distance_km` | 1 %, 0.1 %, 2 million km | the three limits that define the safe Earth–Mars distance (0.0246 AU by default) |
| `mars_shield`, `shield_caution_factor`, `shield_horizon_years` | on, 2, 10 y | the safety shield (see §6.6) |

`max_steps` (line 215) = ceil(mission_years / decision_interval_years) = 100.

---

## 5. Physics under one step (what `step()` calls)

The agent trains on the **fast model** (`FastEngine`, line 2148). It is exact between launches:

1. **`fast_interval`** (1372, Numba) loops over the 800 launch events of the interval. For each
   event it:
   * moves the Earth–Moon barycentre half an interval along its exact Kepler orbit around the
     Sun, and the Moon along its exact Kepler orbit around Earth(+asteroid) (`kepler_step`, 1242,
     universal-variable solver);
   * launches one packet (`_fast_eject`, 1441);
   * moves **Mars** along its own exact Kepler orbit and applies the **mutual Earth–Mars gravity
     kick** (`_mars_kick` 1512 / `_pair_kick` 1350; equal and opposite momentum, Mars's tide on the
     Earth–Moon orbit included; the step is subdivided during close passes);
   * once per simulated year, if the shield is on, lets `shield_decide` (1401) replace the
     agent's command by coasting / full thrust / braking when `mars_forecast` (1366) predicts a
     pass inside the caution distance;
   * moves the next half interval;
   * stops early with `status = 1` (escaped or unbound), `status = 2` (closer than 0.2 AU to the
     Sun) or `status = 3` (closer to Mars than the hard minimum).
2. **`_fast_eject`** (1229) turns the chosen direction from the orbit frame into space coordinates
   (`rtn_to_inertial`, 1298). The packet flies opposite to the thrust direction (`ehat`). Earth
   recoils by `−(dm/M)·v∞·ehat`, the Moon keeps its velocity, and the new barycentre follows from
   momentum conservation. Everything is written to the **ledger** (mass, momentum, energy,
   packet fate).
3. **`eject_core`** (1096) holds the launch physics shared by both models:
   * caps the packet mass at what is left;
   * kinetic energy ½·μ·u² (μ = reduced mass); energy drawn = that ÷ efficiency;
   * if the energy budget is short, the packet is made **smaller**; energy is never created;
   * packets slower than escape speed (Earth + asteroid + Moon) **fall back**: energy is spent and
     there is no net push.

4. **Power supply (`_PowerSupply`, `_with_supply`)** wraps every `run_interval` of both engines. With
   the solar collectors on, `collector_power(cfg, r)` (section 3a) gives the electric power the arm
   rectennas can deliver at Earth's distance: L☉ × collector fraction × solar cells × microwaves ×
   capture (Goubau formula, averaged over the belt) × rectenna. `P_el × interval` is added to the energy
   available for the interval; launches use it first, the reserve only for a shortfall, and unused
   beam energy is not stored. The ledger `self.sol` records beam energy used / offered, reserve used,
   microwave energy absorbed and the peak beam power; `last_power` feeds the viewer (columns 30–33).
   With the defaults the beams supply ~2.7× the launcher's peak demand, so for the agent **energy
   is no longer the limiting resource**: mass, speed, rate, time and Mars safety are. The beams' push on
   Earth and the arm heating are reported as estimates (`_power_metrics`), not simulated.

The reward's energy penalty still counts **all** energy drawn (reserve + beams) relative to the reserve
E0, so wasting launch energy is still discouraged even though the beams make it available. The observation's
velocity-capacity feature uses the reserve plus beam energy for the rest of the mission (`_energy_available`).

That is why the agent cannot cheat: its only lever is launching mass. Every joule and kilogram is
accounted for, and the physics checks verify conservation to round-off.

---

## 6. The environment, `OctoEarthEnv` (line 2358)

### 6.1 Spaces (line 2376)

```python
self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)   # OBS_DIM = 31
self.action_space      = spaces.Box(-1.0, 1.0,  shape=(ACT_DIM,), dtype=np.float32)   # ACT_DIM = 5
```

### 6.2 `reset()` (line 2383)

* Picks a seed. During training this comes from the environment's random generator, so **every
  episode starts at a different orbital phase of Earth, of the Moon and of Mars**
  (`initial_conditions`, 1196).
* Builds a new engine with a full asteroid (3e22 kg) and a full energy budget (3e35 J).
* Stores the target orbital energy `eps_t = −μ / (2·2.7 AU)` and the starting energy `eps0`.
* Sets the progress potential `phi = _potential()` (≈ −1 at the start).

### 6.3 Observation, `_obs()`: 31 numbers, all scaled to roughly −1…+5

| idx | name | formula | example at year 0 (seed 1001) |
|---|---|---|---|
| 0–2 | position | r / (2.7 AU) | 0.287, −0.242, 0 |
| 3–5 | velocity | v / 30 km/s | 0.624, 0.755, 0 |
| 6 | orbit size | a / 2.7 AU | 0.370 |
| 7 | eccentricity | e / 0.05 | 0.334 |
| 8 | inclination | inc / 5° | 0 |
| 9 | **energy gap** | (ε − ε_target) / \|ε0 − ε_target\| | −1.0 (goal: 0) |
| 10 | Moon distance | d / 384,400 km | 0.945 |
| 11 | Moon bound | 1 or 0 | 1 |
| 12 | asteroid left | m / m0 | 1.0 |
| 13 | energy left (reserve) | E / E0 (with collectors on it stays ≈ 1 because the beams pay for the launches) | 1.0 |
| 14 | time left | 1 − t / 10,000 y | 1.0 |
| 15 | mass capacity | log10(Δv possible from the remaining mass ÷ Δv still needed) / 3 | 0.109 (> 0 = enough mass) |
| 16 | energy capacity | log10(Δv possible from the remaining energy ÷ Δv still needed) / 3 | 0.042 |
| 17–21 | previous action | the last 5 action numbers | 0, 0, 0, 0, 0 |
| 22–24 | Mars direction | (r_Mars − r_Earth) / 1 AU, clipped to ±5 | −0.345, 2.121, 0.020 |
| 25 | Mars distance now | ½ · log10(d / safe distance), clipped to ±1.5 (0 = exactly at the limit) | 0.971 (≈ 88 × the safe distance) |
| 26 | Mars forecast | the same for the closest distance in the next 10 y if Earth coasts (`_mars_obs`, 2472, uses `mars_forecast`) | 0.592 (closest in 10 y ≈ 15 × safe) |
| 27 | surface temperature error | (T_s − 288 K) / 10 K, clipped ±5 (`_climate_obs`) | 0 |
| 28 | deep-ocean temperature error | (T_d − 288 K) / 10 K | 0 |
| 29 | sunlight share | sunlight at Earth ÷ today's (≤ 2) | ≈ 1 |
| 30 | mirror capacity (p30; was waste heat) | min(5, capacity of the space mirrors at Earth's distance ÷ today's sunlight) / 5 | ≈ 1 at 1 AU (capacity ~8×), 0.22 at 2.7 AU (1.10×) |

Items 15–16 tell the agent whether it is running out of resources *relative to what is still
needed*, which is exactly what it must budget. NaN or infinite values are replaced and everything is
clipped to ±10.

### 6.4 Action, `decode_action()` (line 2487)

| action number | turned into | formula |
|---|---|---|
| a0 | **throttle** → packet mass | throttle = (a0+1)/2; dm = throttle × 1e18 kg (below 2 % → no ejection) |
| a1 | in-plane thrust angle θ | θ = a1·π, measured from prograde toward radial-outward |
| a2 | out-of-plane angle φ | φ = a2·π/2 |
| a3 | launch speed u | u = 20 km/s + (a3+1)/2 × (5,000 − 20) km/s |
| a4 | **mirror-light set-point** (climate) | wanted insolation = (1 + 0.5·a4) × today's sunlight; the space mirrors add what sunlight lacks, up to their capacity at Earth's distance (`mirror_design`); the rest of their light is turned past Earth (4-number actions still work: a4 = 0) |

The thrust direction in the orbit frame (R, T, N) is `(sin θ cos φ, cos θ cos φ, sin φ)`. It is held
fixed relative to the orbit for the 100-year interval, and packets fly the opposite way. SB3 clips
the network output to [−1, 1], and `decode_action` clips again.

Example: the heuristic's first action `[1, 0, 0, −0.012]` decodes to throttle 1.0 (1e18 kg per
packet), straight prograde push (θ = 0), u = 2,480 km/s. The trained PPO model's first action
on the same seed, `[−0.525, 0.104, 0.107, 0.356]`, decodes to throttle 0.24, θ = 18.8°, φ = 9.6°,
u = 3,400 km/s: mostly prograde with some outward and out-of-plane tilt, faster packets, about a
quarter of the mass per packet.

### 6.5 `step()` (line 2506): one decision

```text
decode action → engine.run_interval(100 y, dm, u, d_rtn) → _climate_step (two-box climate along the
recorded distances, mirror light up to the set-point (no electricity), energy ledger) → new orbit elements → reward parts
→ termination check → record the decision → return (obs, reward, terminated, truncated=False, info)
```

**Reward components** (each one is logged separately in `info["components"]`):

| component | formula | weight | purpose |
|---|---|---|---|
| progress | w · (φ_new − φ_old), with φ = −min(3, \|ε − ε_t\| / \|ε0 − ε_t\|) | 100 | **potential-based shaping**: the sum over an episode equals w·(φ_end − φ_start), so it cannot be farmed; +100 in total for going from 1 AU to exactly the target energy |
| eccentricity | −w · min(5, max(0, e − 0.05)/0.05) × years/100 | 1 | keep the orbit round |
| inclination | −w · min(5, max(0, inc − 5°)/5°) × years/100 | 1 | keep the orbit in plane |
| energy_use | −w · E_used_this_step / E0 | 10 | −10 if the whole budget is spent |
| mass_use | −w · dm_used_this_step / m0 | 5 | −5 if the whole asteroid is thrown |
| time | −w per step | 0.02 | slight push to finish sooner |
| climate | −w · (years/100) · mean over the interval of min(3, max(0, \|T_s − 288 K\| − tol/2) / tol), tol = 5 K | 2 | keep Earth habitable – the agent can act on it with a4 |
| mars_safety | −w · clip((caution − d_min) / (caution − safe), 0, 1), d_min = closest Earth–Mars distance in this step | 20 | warns before the hard limit; 0 while Earth stays outside the caution distance (0.049 AU) |
| success | +w · (1 + 0.5 × average fraction of mass and energy left) | 100 | big terminal bonus, larger if resources are left |
| terminal_error | −w · error_norm (distance from the target a and e) | 50 | for running out of time or resources |
| failure | −w | 100 | escaped, fell toward the Sun, came closer to Mars than the safe distance, or lost the Moon |

Example (first heuristic step, real output): progress +3.463, energy −0.164, mass −0.133,
time −0.020 → **reward +3.145**. In that step 800 packets (8e20 kg) were thrown at 2,480 km/s,
Earth gained 326.5 m/s, and a went 1.0000 → 1.0223 AU.

**Termination reasons** (checked in this order):

`escaped_or_unbound` (status 1) · `too_close_to_sun` (status 2) · `mars_unsafe_approach` (status 3, or closest
pass < safe distance) · `moon_lost` · **`success`**
(a within 2.7 ± 0.05 AU, e ≤ 0.05, inc ≤ 5° **and** surface temperature within 288 ± 5 K) · `deadline` · `resources_exhausted`.

`truncated` is always `False`: the deadline is a real end of the task, and the time left is part of
the observation.

**`info`** contains `reason`, `metrics` (physical values only), `components` (reward parts) and
`decision` (a full record: Δv, impulse, energy, packets, fall-backs, Moon distance …). On the last
step it also contains `episode_summary`, which the training callback reads. Mars values (closest
pass, shield overrides, Mars's orbit and its change) are in `metrics`.

### 6.6 The safety shield (planetary safety)

Earth must cross Mars's orbit on the way out. Unprotected, 10 of 12 heuristic missions came closer than
the safe distance, and a last-minute dodge is physically impossible (a full push moves Earth only
~900 km per month). The shield (`shield_decide`, 1401) therefore checks **once per simulated year**,
while the orbits overlap, the closest Earth–Mars distance forecast for the next 10 years
(`mars_forecast`, 1366: Sun gravity plus the mutual Earth–Mars pull) for four options: the agent's
command, coasting, full thrust and full braking. The agent's command is kept if its forecast stays outside the
caution distance (2 × safe); otherwise the cheapest safe option replaces it for that year. The shield
cannot create energy or momentum: it only changes how much asteroid is thrown, within the agent's own
limits. Overrides are counted (`shield_overrides`) and shown, so you can see how much of the safety came
from the agent and how much from the shield. With the shield on, 12 of 12 tested heuristic missions
stayed safe. The agent still sees the Mars observations and the `mars_safety` penalty, so it can learn
to need the shield less.

---

### 6.7 Climate, energy ledger and collector belt (added for the climate update)

* `_climate_step(samples, t0, years, dec, led0)` integrates the two-box climate (`climate_rhs`) along the
  recorded Earth–Sun distances of the interval (`climate_integrate`). Insolation = sunlight × (lane clear ? 1 : 0)
  + mirror light + waste heat reaching Earth. Mirror light = min(set-point − sunlight − waste heat, capacity), where
  capacity = area × cos 45° × reflectivity × sunlight flux at Earth's distance (falls as 1/r²). Mirrors use no
  electricity. It returns the climate penalty and a per-sample track [t, T_s, sunlight, mirror W] (viewer), and
  stores `T_no_mirror_K` (equilibrium with the same sunlight and no mirrors), `life` and `life_no_mirror`
  (`life_indicators`: ice line, ice-free surface, growing zone, state).
* The same function closes the **energy ledger** for the interval (`dec["ledger"]`): sunlight on the
  collectors + reserve used = reflected + panel heat + manufacturing + transmitter heat + spill + rectenna heat +
  packet kinetic energy + launcher heat; and sunlight on the mirrors = mirror light onto Earth + light turned past
  Earth + mirror heat. `dec["ledger"]["residual_rel"]` is
  ~1e-16. Cumulative totals are in `metrics["energy_ledger"]`.
* `BeltKeeper` (viewer playback and comparison only – not inside training) follows 12 collectors with and
  without station keeping along the recorded Earth/Mars motion (Encke + RK4, 0.5 d). Results are in
  `rec.belt` (`track.on/off` = [dr km, dθ rad, dz km] per snapshot and collector) and `belt.summary()`.
* `ENABLERS` (exactly three since p33: E1 launcher system, E2 solar power belt with beaming, E3 space-mirror ring; each lists its instruments) and `MAJOR_PROBLEMS` (P1–P9) are plain lists in APPENDIX A at the end of the file; they are printed
  at start, sent to the viewer in `/api/results` and written to `report.md`.

Note: the line numbers in this guide come from an earlier version; use your editor's search for the names.
Since p30 the file starts with a *HOW TO READ THIS FILE* map, every function has a one-sentence docstring, and the
documentation tables (PARAM_DOCS, OUTPUT_DOCS, GLOSSARY, ENABLERS, MAJOR_PROBLEMS, ...) are in APPENDIX A at the end.

## 7. Baselines (lines 2704–2750)

* **`policy_none`**: never ejects. Proves Earth stays at 1 AU (test
  `test_no_ejection_baseline_stays_put`).
* **`policy_heuristic`**: hand-designed. Pushes prograde with a fixed speed from
  `heuristic_speed` (the slowest speed for which 95 % of the asteroid supplies the needed Δv and that
  still finishes within 85 % of the mission at full packet rate). The throttle is proportional to the Δv still needed;
  it retro-thrusts if it overshoots and stops once a is within a quarter of the tolerance (±0.0125 AU). It is the yardstick PPO
  must beat. Its 5th number comes from `heuristic_climate(env)`: set-point 1 + 0.8·(288 − T_s)/72 +
  0.4·(288 − T_d)/72 (more mirror light the colder the surface and the deep ocean).
  `policy_none` keeps a4 = −1 (no mirror light; Earth stays at 1 AU anyway).
* **`model_policy(model)`**: wraps a trained SB3 model as `policy(env, obs)` using
  `model.predict(obs, deterministic=True)` (no exploration noise).

`run_episode` (2751) plays a full mission for **any** of these, on either model, and optionally
runs `validate_orbit` (2774): 200 years of coasting in the N-body model that must keep
|a − 2.7| ≤ 0.05, e ≤ 0.05 and the Moon bound.

---

## 8. Training, `App._train_worker` (line 3415)

It runs in its own worker process since update p32 (`App._start_job` -> `_worker_process`, which builds an `App` in the child and calls `_train_worker`; status, log lines and episodes come back through a queue and are merged by `_apply_update`; a thread is the fallback), started by `start_training`. Before p32 it was a thread of the viewer process, which made the viewer lag. That happens when you press
**Start training** in the viewer (POST `/api/train`) or run `--headless`.

```python
env = OctoEarthEnv(cfg, "fast", record=True, record_moon=False)
vec = DummyVecEnv([lambda: env])                      # 1 environment, SB3's vector wrapper
model = PPO("MlpPolicy", vec, n_steps=1000, batch_size=200, n_epochs=10,
            learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, ent_coef=0.0,
            seed=cfg.seed, device=pick_device("auto"),   # "cuda" if an NVIDIA GPU is visible, else "cpu"
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5))
model.learn(total_timesteps=1000*100*1.05 + 1000, callback=CB())
```

`pick_device` (p31) returns "cuda" when PyTorch sees an NVIDIA GPU and `--device` is `auto` or `cuda`; only the
network uses it, the numba physics always runs on the CPU, so the speed-up is small. During training the
callback also stores a reference to the newest finished episode (`train_latest`); the viewer fetches it when it
has finished replaying the previous one (`/api/state?...&trnext=1`).

**The network SB3 builds from this** (printed from the saved `best_model.zip`):

```
policy_net: Linear(26→64) Tanh Linear(64→64) Tanh → action_net Linear(64→4)   (mean of a Gaussian)
value_net : Linear(26→64) Tanh Linear(64→64) Tanh → value_net  Linear(64→1)
+ 4 learnable log-standard-deviations (start at −0.5 → σ ≈ 0.61)      total 12,105 parameters
```

**What PPO does with it, on each update:**

1. **Collect** 1,000 steps (≈ 12 missions) with *random* actions sampled from the Gaussian
   policy. This is exploration.
2. **Compute advantages** with GAE (γ = 0.995, λ = 0.95): how much better each action was than
   the value network expected.
3. **Optimize** for 10 epochs over minibatches of 200 (50 Adam gradient steps) using the
   **clipped surrogate objective** (clip range 0.2, the SB3 default). The policy only moves a
   little per update. The value loss trains the critic.
4. Repeat. The tested run did 83,559 steps, about 83 updates, in 663 s on one CPU thread.

**The callback `CB`** (inside `_train_worker`):

* `_on_rollout_start`: copies SB3's losses (policy-gradient loss, value loss, entropy,
  approx_kl, explained variance) into the log and the viewer ("PPO internals").
* `_on_step`: when an episode ends it calls `_on_episode` (3493). That writes one row to
  `training_log.csv`, updates the charts and sometimes sends the trajectory to the 3-D view.
* Every 50 episodes it runs a **deterministic evaluation** (`_eval_policy_fast`, 3405) on fixed
  seeds 1001 and 1002 and builds a **lexicographic score**:
  `(success rate, −orbit error, −energy used)`. If the score beats the best so far, it saves
  **`best_model.zip`** and `best_model_info.json` (episode, eval, hyperparameters, selection rule).
* It returns `False` (stop) after `train_episodes` episodes or when you press Stop.

After training it saves `last_model.zip` and automatically starts the comparison.

**Files written by training:** `training_log.csv`, `evals.json`, `best_model.zip`,
`best_model_info.json`, `last_model.zip`.

---

## 9. Evaluation and comparison, `_compare_worker` (line 3543)

For each policy (none, heuristic, PPO loaded with `load_ppo` → `PPO.load(best_model.zip)`), it:

1. runs the **fast model on 3 seeds** (2001–2003), which gives the success rate;
2. runs the **high-fidelity N-body model** (Sun, Earth, Moon, Mars and explicit fragments) on seed 2001,
   then the **200-year stability check**;
3. counts a mission as successful only if the N-body run succeeds *and* the stability check
   passes.

It then builds:

* `explain()` (3914): plain-language reasons from the recorded numbers (thrust direction,
  speed vs. the energy-optimal speed, resource use …);
* `best_panel()` (3819): the decision-by-decision table with helped / mixed / hurt / coast labels,
  the criteria table and failure counts;
* `make_report()` (3987): `report.md`. The results go to `comparison.json`.

---

## 10. How it all starts, `main()` (line 4196)

```
parse args → Config (preset + --set overrides, validated) → App(cfg, out_dir)
→ physics self-checks (training disabled if any fails) → feasibility gate printed
→ --headless: start_training(), wait for training + comparison, exit
→ otherwise: HTTP server for the viewer (training starts when you press the button,
  or skip it with --no-train)
```

Useful commands (run in the `octo-earth` folder):

```bash
python octoearth_rl.py --headless                        # full 1,000-episode training + comparison, no browser
python octoearth_rl.py --smoke --headless                # 30-episode pipeline test (~1 min)
python octoearth_rl.py --episodes 3000                   # longer training
python octoearth_rl.py --set w_time=0.1 w_energy=20      # change reward weights
python octoearth_rl.py --set ppo_learning_rate=1e-4 ppo_gamma=0.999
python -m pytest -q tests                                # includes check_env on OctoEarthEnv
```

Using the environment and the trained model in your own script:

```python
import octoearth_rl as R
from stable_baselines3 import PPO

env = R.OctoEarthEnv(R.make_config(), fidelity="fast")      # or "hifi"
model = PPO.load("example_results/best_model.zip", device="cpu")
obs, _ = env.reset(seed=2001)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    print(env.render(), info["components"])
print(env.summary()["reason"])
```

---

> **Viewer-only changes do not touch the RL.** The 8-arm joint throw, the stand-by beams and the solar
> collector close-up (`frontend/collector.js`) are drawing code. They read the recorded physics rows and
> never feed anything back, so observations, actions, rewards and the trained model are unchanged
> (one packet per ejection event, as before).

## 11. Results of the tested run (`example_results/`) and honest caveats

* 1,000 episodes, 89,773 steps, 877 s (with collectors, climate and the space mirrors). `best_model.zip` =
  **episode 50**, deterministic test success 100 % (a = 2.667 AU, e = 0.0085, 6,950 years). Tests from episode
  100 on failed; from ~400 PPO collapsed to "barely push" (a ≈ 1.01 AU at 1,000, 4 % of the energy) because
  missing the deadline costs less reward than a failed attempt. The early success is because near-zero actions
  already mean "push prograde at ~2,500 km/s, set-point ≈ 1×". So this run does **not** show that PPO learned a
  better strategy – it shows that the best checkpoint is kept.
* N-body run, seed 2001: PPO **succeeds and passes** the 200-year stability check (a = 2.6870 AU,
  e = 0.0085), keeps Earth habitable with the mirrors (final 286.8 K, max deviation 4.7 K, 0 years outside
  ±15 K; mirror light onto Earth 1.89e28 J; ice line 68°, growing zone 73 %; without mirrors 156 K) and stays
  Mars-safe (closest 0.0413 AU) – but with **560 shield overrides** (heuristic 176). The heuristic is faster
  (3,600 vs 7,100 y), uses about the same mass and energy, and controls the temperature more tightly (max
  deviation 2.6 K).
* Training episodes mostly "fail" (571 out of reaction mass, 429 deadline) because they
  explore randomly. The deterministic tests are what count.
* **Not a global optimum**: the result is the best of the tested checkpoints, judged on only a few
  seeds. The reward is a training score, not a physical quantity, and changing the weights changes
  the strategy.
* Training uses the fast model. It neglects the Sun's tide on the Moon (about 22°/yr phase drift)
  and fragment trajectories after escape, and its Mars outcome can differ strongly from the N-body
  model (chaotic); the N-body comparison checks the policy anyway.
