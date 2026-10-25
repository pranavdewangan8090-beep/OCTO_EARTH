# OCTO-EARTH prototype

A PPO agent ejects asteroid mass from Earth so the recoil moves Earth from 1 AU to 2.7 AU. A 3-D browser viewer shows the results.

## Run

```
pip install -r requirements.txt
python octoearth_rl.py
```

## Demo Simulation:
https://drive.google.com/file/d/1nDeFmHMzqYLWCHHH2DhLang99N0Wt2jm/view?usp=sharing

The viewer opens at `http://127.0.0.1:8800/` (or the next free port). Training starts automatically; press Ctrl+C to stop (the model is still saved).

## The left-hand tabs

Click an icon on the left rail to open a drawer. Only one is open at a time; close it with the X or Esc.

| Tab | What it is for | What you see |
|---|---|---|
| **Mission** | Run things | Start/stop training, choose what is shown on screen: *Live training*, *Best PPO model*, *Best hand-coded* (heuristic), or *No ejection*. Also the training device (Auto/CPU/GPU), progress, speed report and overview layer switches (Mercury/Venus/Jupiter, their orbits, "whole system"). |
| **View** | Change what you look at | Switch between Overview and Earth POV, camera options ("camera follows Earth", look buttons), layer check-boxes, which screen panels to show. |
| **Params** | Change settings | Every parameter with its value, unit and an (i) tooltip explaining what it means and what raising or lowering it does. Change a value, click **Apply**, then start training from **Mission**. Invalid values are rejected. |
| **Results** | See how well it did | Training curves, comparison of PPO vs the baselines (no ejection, heuristic), the best model's final orbit vs each target (pass/fail), a decision-by-decision table, Mars-safety table, and resource counts. Training reward is shown separately from physical results. |

The rail also has **Guide** (explains every parameter), **Ledger** (climate, energy ledger, collector belt, enablers) and **Log** (messages).

### Reading the main screen

- **Bottom bar:** play / pause / reset, simulated time, speed slider (1 day to 500 years per real second), and view chips.
- **Telemetry (top left):** orbit size, eccentricity, asteroid left, distance to Mars, shield state. Click **more** for everything.
- **Overview symbols:** solid line = past path, dashed = orbit Earth would follow if thrust stopped now, green circle = target. Packets: orange leaves the Solar System, yellow stays bound to the Sun, red X fell back to Earth. Green arrow on Earth = latest recoil.
- **Keys:** Space play/pause, R reset, V switch view, L legend, Esc close panel.

## Different views (POVs)

| View | How to open | What it shows |
|---|---|---|
| **Overview** | Default; **View** tab | Solar system: Sun, Earth, Moon, Mars, asteroid, paths and target. Not to scale. |
| **Earth POV** | Key **V**, the POV chip, or **View** > Earth POV | True-scale close-up from Earth: Moon, asteroid, launcher ring and 8 arms, magnetic field, solar wind, Hill sphere, Mars. Turn layers on or off with the check-boxes. |
| **Power beams** | Key **B** or the **Beams** chip | Microwave beams arriving at the 8 arm tips; brightness follows the recorded power. |
| **Collector** | Key **C** or the **Collector** chip | Close-up of one solar-collector satellite. |
| **Weather & life** | Key **W** or the **Weather** chip | Earth's climate under the space mirrors; **show without** shows the frozen Earth with no mirrors. |

## Failed attempts

Click **Failed attempts (4 models)** in the viewer header (opens in a new tab), or go to `http://127.0.0.1:8800/failed.html`.

Four other ideas, each with its own 3-D model and numbers showing why it fails:

1. Laser push
2. Hydrogen harvesting
3. Superconducting magnetic arms
4. Dyson beam on a reflector held by the arms

Switch with tabs **1-4**, **Prev / Next**, or the number and arrow keys. **Start / Pause / Reset** are in the bottom bar (Space, R). The right panel shows the numbers; hover a row for its formula. Drag to orbit, wheel to zoom.

To print the same verdicts without graphics: `python failed_models.py`
