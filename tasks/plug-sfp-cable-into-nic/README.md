# Plug a fibre patch cable's SFP end into a network card

**Status: draft. Scripted demo reaches 1.0 in Isaac Sim (3 of 3 runs, 1437 steps each). No Dream-1 run, media ID or review yet.**

A full SFP-to-SC fibre patch cable lies on the table. Its SC plug (blue) rests in a fixed tray; its SFP plug (gold) lies in
a slanted holder, tip down at 45 degrees. A network card leans back 45 degrees in a cradle, square to the robot, its two SFP
ports facing up and toward the arms. The robot lifts the SFP plug out along its axis, carries it with the cable trailing,
and pushes it along the port's slanted axis all the way into the left port. The port is 14.0 x 9.0 mm and the plug body
13.75 x 8.6 mm (the AIC models' own sizes), so the clearance is about 0.13 mm a side. Credit rises with depth; the plug must
be aligned with the port axis, and the right port scores 0.

## Files
- `plug_sfp_cable_into_nic.py`: the task file (scene, `score()`, `success()`, `distance()`, and the closed-loop `demo_script`).
- `sfp_cable_scoring.py`: the scorer as pure functions. `test_sfp_cable_scoring.py` tests it (`python -m pytest`).
- `fibrekit.py`: the cable (a chain of capsule segments on joints, with a plug at each end), built from `scenekit.py` boxes.
- `cabledemo.py`: runs a task's `demo_script` (closed loop on the plug's measured pose) and writes videos, `setup.png`, `result.json`.
- `geometry.json`, `gen_geometry.py`: box tables from the AIC SDFs (Apache-2.0) and the script that builds them.

## Run a trial
On a machine set up with the official simulator, with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="Take the cable's gold SFP plug out of its slanted holder and push it, tip first, all the way into the network card's left SFP port (the one nearer the left arm). The blue SC plug stays in its tray."
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=plug_sfp_cable_into_nic.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ./dream_run
# the scripted demo (about 6 minutes of wall time; writes videos, setup.png and result.json)
python cabledemo.py plug_sfp_cable_into_nic.py ./demo_run --max-steps 2400
```
The demo reads the plug's true pose from the simulator, so it is a demonstration that full score is reachable, not a policy
that sees only the cameras. `object_jitter=0`: one fixed layout, no seeds.

## Scoring
| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | left port, tip depth >= 40 mm, within 3 degrees of the port axis, at rest 1 s | at most ~7 mm of plug body outside the port; a phone level on it |
| 0.5 | left port, depth >= 20 mm, within 10 degrees | about 27 mm of body outside or less |
| 0.25 | left port, depth >= 5 mm, within 10 degrees | a ruler at the mouth |
| 0 | anything else: right port, off the table, tip more than 1.5 mm off the centre line at the mouth (4.5 mm once inside) | |

## Known simplifications
A 2 mm flared lead-in is added at each port mouth. The cable is a soft 12 g chain with no twist stiffness. No latch spring or
click. Plugs are boxes. The plug stops at about 41.5 mm where its shoulder meets the port.
