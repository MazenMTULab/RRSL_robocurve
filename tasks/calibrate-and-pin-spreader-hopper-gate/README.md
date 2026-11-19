# Calibrate and Pin a Rotary Spreader Hopper Gate at Setting 4

Bimanual tabletop task for `yam_arena`: one YAM arm slides and holds a spring-returned horizontal spreader hopper metering gate (`PrismaticJoint` with `k = 70.0 N/m`, `damping = 4.0 N*s/m` pulling toward `0.0 cm` closed) open to Setting 4 (`4.0 cm`) without exceeding the Setting 5 (`> 5.2 cm`) spill limit, while the opposite arm picks a locking stop pin (`10 x 10 x 44 mm` tang, `28 x 28 x 8 mm` stop collar) from a staging stand and seats it vertically through the `20 x 20 mm` Setting 4 clevis bore so the gate stays pinned at `4.0 +/- 0.4 cm` after both grippers release and hold clear for `1.0 s` (`20` steps at `20 Hz`).

## Files

- `task.py`: 1:1 rotary spreader hopper gate assembly (`18.0 x 14.0 x 10.0 cm`), `pin_stand`, `stop_pin`, `score`/`success` logic with latched over-opening/dropped-pin hazards, and symmetric `demo()` (`SIDE=L|R`, plus `DEMO_NO_PIN=1` and `DEMO_OVER_OPEN=1` negative controls).
- `yamkit.py`, `mechkit.py`, `guardkit.py`: Self-contained prop/articulation builder (`yamkit`), return-spring law (`mechkit.Spring`), and gripper-contact/latch guards (`guardkit`).
- `test_task.py`: Offline geometry, clearance, scorer/hazard, negative-control, and 20 Hz `ArmPlan` IK tests (`< 0.5 mm`, `< 0.5 deg`).
- `task.json`: Task specification and scoring rubric.

## Commands

Run offline unit tests (and 20 Hz `ArmPlan` IK verification when `yam_arena` is installed):

```bash
python -B -m pytest -q -p no:cacheprovider tasks/calibrate-and-pin-spreader-hopper-gate/test_task.py
```

Run the scripted demonstration in Isaac Lab-Arena (`SIDE=L` or `SIDE=R`):

```bash
SIDE=L inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/calibrate-and-pin-spreader-hopper-gate/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide and hold the spring-loaded spreader hopper gate open to Setting 4 (4.0 cm) without over-opening, insert and seat the locking stop pin through the Setting 4 clevis slot, and release both hands clear so the gate stays pinned at 4.0 +/- 0.4 cm." \
  --scorer task_score --max-steps 400 --no-prompt --log-dir ~/yam-eval/spreader-L
```

Run negative-control checks (both score `0.0`):

```bash
# 1. Early release without inserting stop pin -> return spring snaps gate back to 0.0 cm (0.0)
SIDE=L DEMO_NO_PIN=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/calibrate-and-pin-spreader-hopper-gate/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide and hold the spring-loaded spreader hopper gate open to Setting 4 (4.0 cm) without over-opening, insert and seat the locking stop pin through the Setting 4 clevis slot, and release both hands clear so the gate stays pinned at 4.0 +/- 0.4 cm." \
  --scorer task_score --max-steps 300 --no-prompt --log-dir ~/yam-eval/spreader-nopin

# 2. Over-opening gate past Setting 5 (> 5.2 cm) -> latched spill hazard (0.0)
SIDE=L DEMO_OVER_OPEN=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/calibrate-and-pin-spreader-hopper-gate/task.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide and hold the spring-loaded spreader hopper gate open to Setting 4 (4.0 cm) without over-opening, insert and seat the locking stop pin through the Setting 4 clevis slot, and release both hands clear so the gate stays pinned at 4.0 +/- 0.4 cm." \
  --scorer task_score --max-steps 400 --no-prompt --log-dir ~/yam-eval/spreader-overopen
```

## Runtime and seeds

- **`yam_scripted` (`SIDE=L`):** `256` control steps (`12.80 s` at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded` (`gate=4.01 cm`, `hold=20`).
- **`yam_scripted` (`SIDE=R`):** `256` control steps (`12.80 s` at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded` (`gate=4.00 cm`, `hold=20`).
- **Negative control (`DEMO_NO_PIN=1`):** `260` steps -> `task_score: 0.0` (spring snaps gate to `0.00 cm` by step `130`).
- **Negative control (`DEMO_OVER_OPEN=1`):** `370` steps -> `task_score: 0.0` (latched over-opening spill hazard at step `75`).
- **Run limit:** `2,400` control steps (`2 minutes` at `20 Hz`).
- **Determinism / toggles:** Deterministic starting layout (`object_jitter=0`). Environment toggles: `SIDE=L|R`, `DEMO_NO_PIN=1`, `DEMO_OVER_OPEN=1`, `SPREADER_TRACE=1`.
