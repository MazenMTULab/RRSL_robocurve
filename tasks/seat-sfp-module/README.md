# Seat an SFP module in a network card's left port

**Status: draft. Not yet run in Isaac Sim: no Dream-1 run, demo video, screenshot or media ID.**

A network card leans back 45 degrees in a slanted cradle, turned 8 degrees from square to the robot, its two SFP ports
facing up and toward the arms. An SFP module lies in a slanted holder (the same 45 degrees) beside it, tip down. The robot
draws the module out along its axis, turns it to match the card, and pushes it along the port's slanted axis all the way
into the left port. The ports are 16.0 x 11.0 mm inside and the module is 13.75 mm wide at the
body, so the margin is about 1.1 mm a side in the width direction. Credit rises with insertion depth; the module must be
near-vertical and turned to match the card, and the right port scores 0.

## Files
- `seat_sfp_module.py`: the task file (scene, `score()`, `success()`, `distance()`, scripted `demo()`).
- `sfp_seat_scoring.py`: the scorer as pure functions (poses in, score out). `test_sfp_seat_scoring.py` tests it
  (`python -m pytest`, needs numpy and pytest).
- `geometry.json`, `gen_geometry.py`: box tables from the AIC SDFs in `assets/aic` (Apache-2.0), and the script that builds them.
- `partkit.py`: rigid parts built from boxes (the module), shared with the `din-rail-mount` task; `scenekit.py`: fixed boxes in any orientation (the slanted card and holder), shared with `lock-before-loading`.
- `check_offline.py`: the geometry and demo checks below (needs the `yam-isaac-lab-arena` Python environment, no GPU).

## Run a trial
On a machine set up with the official simulator (the Robocurve Brev launchable), with a Dream API key in
`~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="Take the SFP module out of its holder and push it, tip first, all the way into the network card's left SFP port (the one nearer the left arm)."
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=seat_sfp_module.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sfp-dream
# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=seat_sfp_module.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/sfp-demo
```

## Scoring
| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | left port, tip depth >= 42 mm, within 3 degrees of vertical and of the card's turn, at rest 1 s | at most 13 mm of the module (its latch end) above the port mouth; a phone level on it |
| 0.5 | left port, depth >= 20 mm, within 10 degrees of vertical and of the card's turn | about 35 mm above the mouth or less |
| 0.25 | left port, depth >= 5 mm, same angles as 0.5 | a ruler at the mouth |
| 0 | anything else: right port, off the table, tip more than 1.5 mm off the port's centre line at the mouth (4.5 mm once inside) | |

## Checked offline (no GPU)
- The task file loads through `yam_arena.task_file.load` (33 objects, one of them free).
- With the module centred, its boxes clear the port's boxes by at least 1.05 mm from 0 to 42 mm depth (box-to-box separation test). The first colliding sideways shift is 1.1 mm.
- The module rests in its holder with every gap at least 0.37 mm, and starts 157 mm from the card.
- The scripted demo (draw out along the axis, lift, carry while turning, approach and push in along the port axis) plans within the joint limits, worst IK error 0.1 mm, and the module stays at least 35 mm from the card until the push-in.
- 18 scorer tests pass, for an upright and a 45-degree-leaned card.

## Not yet checked (needs Isaac Sim)
Whether the gripper holds the module's latch end (at 45 degrees) without the pads touching the card or the holder; whether the slanted holder holds the module still;
whether the two shared helper files (`partkit.py` and `scenekit.py`) work together, which no existing task shows; whether the module slides in under
contact; Dream-1's result; run time per trial.
