# Seat an SFP module in a network card's left port

**Status: draft. Not yet run in Isaac Sim: no Dream-1 run, demo video, screenshot or media ID.**

A network card stands upright in a cradle, turned 8 degrees from square to the robot, its two SFP ports opening upward.
An SFP module stands tip-down in a small holder beside it. The robot lifts the module out, turns it to match the card
and pushes it all the way into the left port. The ports are 16.0 x 11.0 mm inside and the module is 13.75 mm wide at the
body, so the margin is about 1.1 mm a side in the width direction. Credit rises with insertion depth; the module must be
near-vertical and turned to match the card, and the right port scores 0.

## Files
- `seat_sfp_module.py`: the task file (scene, `score()`, `success()`, `distance()`, scripted `demo()`).
- `sfp_seat_scoring.py`: the scorer as pure functions (poses in, score out). `test_sfp_seat_scoring.py` tests it
  (`python -m pytest`, needs numpy and pytest).
- `geometry.json`, `gen_geometry.py`: box tables from the AIC SDFs in `assets/aic` (Apache-2.0), and the script that builds them.
- `partkit.py`: rigid parts built from boxes (the module), shared with the `din-rail-mount` task.

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
- The task file loads through `yam_arena.task_file.load` (32 objects, one of them free).
- The module's boxes pass through the port's boxes from 0 to 46 mm depth with no interpenetration; the first colliding
  sideways shift is 1.2 mm.
- The scripted demo plans within the joint limits, worst IK error 1.0 mm, and carries the module's tip 12.9 mm above the card's top edge.
- 7 scorer tests pass.

## Not yet checked (needs Isaac Sim)
Whether the gripper holds the module's latch end without the pads touching the card; whether the module slides in under
contact; Dream-1's result; run time per trial.
