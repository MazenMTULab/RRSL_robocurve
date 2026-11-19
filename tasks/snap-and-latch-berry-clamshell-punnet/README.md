# Hold and Latch a Spring-Open Berry Clamshell Punnet

Bimanual tabletop task for `yam_arena` (`station` scene): a 1:1 clear PET hinged berry clamshell punnet (`16.0 x 12.0 x 6.5 cm` closed, `525 g` packed with eight ripe strawberries) rests on the station table with its rear-hinged lid sprung open (`~24.4 deg`) and a front sliding snap catch parked open (`+/-3.2 cm`, randomized `HAND_SIDE=L|R`). One YAM arm must gently compress and hold the lid's structural side shoulder flush onto the base rim (`<= 3.0 deg`) without slamming or reaching into the berries, while the opposite arm pinches and slides the front snap catch cleanly home (`0.0 +/- 0.4 cm`) so the punnet remains latched shut after both grippers release and hold clear for `1.0 s` (`20` steps at `20 Hz`).

## Files

- `task.py`: Punnet geometry, strawberry meshes, sprung lid & sliding snap catch mechanism, `score()`/`success()` with latched slam/crush/disturbance hazards, and symmetric `demo()` (`HAND_SIDE=L|R`, plus `DEMO_NO_LATCH=1`, `DEMO_LATCH_BEFORE_CLOSE=1`, and `DEMO_CRUSH_BERRIES=1` negative controls).
- `mechkit.py`: Coupled joint laws (`SprungLid` hinge return spring + latch-coupled limits, `SnapCatch` center detent + premature-latch jam guard).
- `yamkit.py`, `guardkit.py`: Self-contained prop/articulation USD builder and gripper-contact / latch helpers.
- `test_task.py`: Offline unit tests (geometry clearances, mechanism interlock laws, scoring tiers, latched hazards, negative controls, and 20 Hz `ArmPlan` IK checks).
- `task.json`: Benchmark task specification and scoring rubric.

## Commands

Run offline unit tests (and 20 Hz `ArmPlan` IK checks when `yam_arena` is installed):

```bash
python -B -m pytest -q -p no:cacheprovider tasks/snap-and-latch-berry-clamshell-punnet/test_task.py
```

Run the scripted bimanual demonstration (`HAND_SIDE=L` or `HAND_SIDE=R`):

```bash
HAND_SIDE=L inspect-robots run \
  --embodiment yam_arena \
  --policy yam_scripted \
  -E task=tasks/snap-and-latch-berry-clamshell-punnet/task.py \
  -E object_jitter=0 \
  -E control_hz=20 \
  -E cam_width=320 \
  -E cam_height=176 \
  --instruction "Press and hold the spring-open berry clamshell lid flush down onto the base rim with one hand without slamming or crushing the strawberries, slide the front snap catch cleanly home with the other hand to latch it shut, then release both hands clear." \
  --scorer task_score \
  --max-steps 500 \
  --no-prompt
```

Run the negative-control checks (all score `0.0`):

```bash
# 1. Release lid without sliding snap catch -> lid springs back open to 24.4 deg -> 0.0
HAND_SIDE=L DEMO_NO_LATCH=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/snap-and-latch-berry-clamshell-punnet/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Close and latch the berry punnet." \
  --scorer task_score --max-steps 240 --no-prompt

# 2. Slide snap catch before closing lid -> catch jams under lid, blocking closure at 6.5 deg -> 0.0
HAND_SIDE=R DEMO_LATCH_BEFORE_CLOSE=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/snap-and-latch-berry-clamshell-punnet/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Close and latch the berry punnet." \
  --scorer task_score --max-steps 260 --no-prompt

# 3. Plunge gripper onto open berry mound -> latches slam/crush hazard -> 0.0
HAND_SIDE=L DEMO_CRUSH_BERRIES=1 inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=tasks/snap-and-latch-berry-clamshell-punnet/task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Close and latch the berry punnet." \
  --scorer task_score --max-steps 200 --no-prompt
```

## Runtime and seeds

- **`yam_scripted` (`HAND_SIDE=L`):** `197` steps (`9.85 s` simulated at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded`.
- **`yam_scripted` (`HAND_SIDE=R`):** `202` steps (`10.10 s` simulated at `20 Hz`) -> `task_score: 1.0`, `outcome: succeeded`.
- **Negative control (`DEMO_NO_LATCH=1`):** `240` steps -> `task_score: 0.0` (lid springs back to `24.4 deg`).
- **Negative control (`DEMO_LATCH_BEFORE_CLOSE=1`):** `260` steps -> `task_score: 0.0` (catch jams under open lid, physically blocking closure at `6.5 deg`).
- **Negative control (`DEMO_CRUSH_BERRIES=1`):** `200` steps -> `task_score: 0.0` (latched `slam` hazard).
- **Run limit:** `2,400` control steps (`2 minutes` at `20 Hz`).
- **Determinism / toggles:** Deterministic starting layout with `-E object_jitter=0`. Verification flags: `HAND_SIDE=L|R`, `DEMO_NO_LATCH=1`, `DEMO_LATCH_BEFORE_CLOSE=1`, `DEMO_CRUSH_BERRIES=1`, `PUNNET_TRACE=1`.

## Scoring summary

| Score | Condition |
|---|---|
| `1.0` | Held `1.0 s` (`20` steps) with zero hazards: lid latched closed (`<= 3.0 deg`, `|vel| <= 0.10 rad/s`), catch locked at center (`<= 4.0 mm`, `|vel| <= 0.010 m/s`), punnet flat and still (`tilt <= 4.0 deg`, `|low_z - TOP| <= 5 mm`, `< 5 mm/s`, `< 5 deg/s`), both grippers clear (`> 2 mm`) |
| `0.75` | Lid flush (`<= 3.0 deg`) and snap catch locked at center (`<= 4.0 mm`) prior to `1.0 s` hands-clear settle |
| `0.50` | Lid flush (`<= 3.0 deg`) and snap catch within `+/-12 mm` of center without jamming |
| `0.25` | Lid held flush (`<= 3.0 deg`) with catch unengaged (reverts to `0.0` if released and lid springs open) |
| `0.0` (latched) | Lid slammed (`<= 15.0 deg`, `< -2.20 rad/s`), gripper enters open berry cavity (`lid > 5.0 deg`), or punnet disturbed (`> 12.0 deg` tilt, `> 15.0 deg` yaw, `> 4.5 cm` XY drift, or `< TOP - 1.5 cm`) |
