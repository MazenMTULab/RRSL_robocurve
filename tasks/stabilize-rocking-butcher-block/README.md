# Stabilize a wobbling cutting board while keeping a mug upright

Files for the `yam_arena` embodiment (Isaac Lab-Arena, bimanual YAM, `station` scene). An oiled walnut butcher-block cutting board (`18.0 × 24.0 × 2.8 cm`, `0.95 kg`) stands on three `20 mm` rubber bumper feet and one bare `12 mm` brass mounting stud (`8 mm` shorter, fixed at the front-left `FL` corner). An open stoneware mug (`0.42 kg`) stands on top of the board above that corner, pressing the bare stud down onto the tabletop (`0 mm` gap at rest, `3.8°` tilt). A spare `4.4 × 3.6 × 0.8 cm` black EPDM rubber pad lies flat at `(12.5, 0.0) cm`. Setup, full scoring rules, and fidelity notes are in `task.json`.

## Run a trial

Source the simulator environment (`source /data/IsaacLab-Arena/yam.env`), then run the scripted demo (`yam_scripted`) on the fixed front-left corner (the default):

```bash
BOARD_CORNER=FL inspect-robots run \
  --embodiment yam_arena \
  --policy yam_scripted \
  -E task=tasks/stabilize-rocking-butcher-block/task.py \
  -E object_jitter=0 \
  -E control_hz=20 \
  -E cam_width=320 \
  -E cam_height=176 \
  --instruction "Fit the spare black pad under the short foot so the cutting board sits level. Keep the mug on the board within 4.5 cm of its starting spot on the board and within 25 degrees of upright. Let go when finished." \
  --scorer task_score \
  --max-steps 2400 \
  --store-frames \
  --no-prompt
```

For the Dream-1 baseline (requires `DREAM_API_KEY` configured in the environment or credentials file):

```bash
BOARD_CORNER=FL inspect-robots run \
  --embodiment yam_arena \
  --policy dream \
  -E task=tasks/stabilize-rocking-butcher-block/task.py \
  -E object_jitter=0 \
  -E control_hz=20 \
  -E cam_width=320 \
  -E cam_height=176 \
  --instruction "Fit the spare black pad under the short foot so the cutting board sits level. Keep the mug on the board within 4.5 cm of its starting spot on the board and within 25 degrees of upright. Let go when finished." \
  --scorer task_score \
  --max-steps 2400 \
  --store-frames \
  --no-prompt
```

Media upload (once Dream-1 baseline is recorded):

```bash
python tools/upload_media.py tasks/stabilize-rocking-butcher-block --video model_run.mp4 --demo demo.mp4 --screenshot setup.png
```

Render logged videos with `inspect-robots video <log.json>`, and run the offline unit tests (`FL`, `FR`, and the fixed default) without Isaac Sim:

```bash
python -B tasks/stabilize-rocking-butcher-block/test_task.py
BOARD_CORNER=FL python -B -m pytest -q -p no:cacheprovider tasks/stabilize-rocking-butcher-block/test_task.py
BOARD_CORNER=FR python -B -m pytest -q -p no:cacheprovider tasks/stabilize-rocking-butcher-block/test_task.py
```

For the push-without-lifting control, use the same scripted command and add
`DEMO_NO_LIFT=1` before `BOARD_CORNER=FL`. Use separate log directories for the
two trials. Leave `DEMO_NO_LIFT` unset for the two-arm demo.

## Runtime and seeds

The fixed layout for evaluation and media is `BOARD_CORNER=FL`. The FR
layout is an optional mirror check.

* **`yam_scripted` demo (`FL`):** `417` control steps (`20.85 s` simulated at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded` (`1.0` held for `1.0 s` / `20` steps).
* **Prior `yam_scripted` demo (`FR`, before the scoring update):** `416` control steps (`20.80 s` simulated at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded`.
* **One-arm negative control (`DEMO_NO_LIFT=1`, `FL`):** `500` control steps (`25.0 s`) -> `task_score: 0.25` (pushing the pad without lifting the corner hits the weighted brass stud at `~3.4 cm` offset; the board never levels).
* **Run limit:** `2,400` control steps (`2 minutes` at `20 Hz`).
* **Seed:** The submission uses the fixed front-left corner and `object_jitter=0`. `BOARD_CORNER=FR` enables an optional mirror check. Environment toggles for testing: `DEMO_NO_LIFT=1` (idle lifting arm control), `BOARD_TRACE=1` (periodic step telemetry to stderr).

## Scoring summary

| Score | Condition |
|---|---|
| `1.0` | Spare pad flat on table (`<= 10°`, bottom `<= 4 mm` up) within `1.2 cm` of bare stud axis; bare stud bottom at `5–11 mm` and within 3 mm of the pad’s top face; board flat (`<= 0.8°`); mug upright (`<= 5°`) on its board spot; all objects still (`< 5 mm/s`, `< 5°/s`) with no gripper touching (`<= 2 mm`) for `1.0 s` |
| `0.75` | Seated pad (`<= 1.2 cm`, `5–11 mm` stud height) + board flat (`<= 1.2°`) + upright mug (`<= 5°`), before release or `1.0 s` settle |
| `0.50` | Pad flat within `2.5 cm` of bare foot, or bare foot bottom lifted `>= 6 mm` above tabletop while pad is within `6.0 cm` |
| `0.25` | Pad moved `>= 2.5 cm` toward bare corner (or within `9.0 cm` of bare foot) |
| `0.0` (latched) | Mug tilts `> 25°` or leaves its board spot (`> 4.5 cm` / below board), or board tilts `> 18°` or drifts `> 6.0 cm` horizontally for 2 consecutive steps (`0.1 s`) |

## Real-world build notes

* **Board:** Commercial `18 × 24 × 2.8 cm` solid walnut bar cutting board (`~0.85 kg`) fitted with three `32 mm dia × 20 mm` screw-on SBR rubber bumper feet (inset `1.6 cm` from corners) and one `22 mm dia × 12 mm` brass collar spacer on a corner mounting screw (creating the `8 mm` step down).
* **Pad & Mug:** Standard `4.4 × 3.6 × 0.8 cm` flat black EPDM corner pad (`25 g` with `14 × 14 mm` steel washer top weight/contrast square) and standard `7.0 cm dia × 9.5 cm` stoneware mug filled to `0.42 kg` (`2 cm` below rim).

## Optional physical comparison

Recreate the FL setup on a level tabletop. Use water in the mug at the
documented fill level. Weigh the filled mug before and after each trial and
inspect the board and tabletop for droplets. Record the scale resolution and
any visible spill. Check the board with a level and verify the pad is bearing
load by testing whether it can move sideways with light pressure after both
hands release. Repeat with a push-only attempt from the same starting layout.
These optional checks compare physical support and the behavior of real water with the simulation.

## Files

* `task.py`: Scene geometry, fixed default layout, `score()`, `success()`, and symmetric bimanual `demo()`.
* `task.json`: Benchmark task metadata and specification.
* `README.md`: Run commands, verified runtimes, seed controls, scoring table, and hardware specs.
* `test_task.py`: Offline unit tests covering tiers, boundary conditions, latched hazards, reset, fixed layout, and demo plans.
* `yamkit.py`, `guardkit.py`, `flowkit.py`: Self-contained prop builder (`yamkit`), trimmed gripper-contact/latch guards (`guardkit`), and ceramic-on-wood friction material (`flowkit`).
