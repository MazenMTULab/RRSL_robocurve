# Plug a fibre patch cable's SFP end into a network card

**Status: draft. Scripted demo reaches 1.0 in Isaac Sim (3 of 3 runs, 1068 steps each, both arms). Dream-1 run: 0.0 (step limit reached). No media ID or review yet.**

A full SFP-to-SC fibre patch cable lies on the table. Its SC plug (blue) rests in a fixed tray; its SFP plug (gold) lies in
a slanted holder, tip down at 45 degrees. A network card stands upright on a foot, its two SFP ports facing straight up. The
right arm picks up the cable 28 cm behind the plug and holds its slack clear; the left arm lifts the SFP plug out along its
axis, tilts its hand until the plug stands upright, carries it over the card, and pushes it straight down into the target
(nearer) port. The port is 14.0 x 9.0 mm and the plug body
13.75 x 8.6 mm (the AIC models' own sizes), so the clearance is about 0.13 mm a side. Credit rises with depth; the plug must
be aligned with the port axis, and the other port scores 0.

## Terms, for readers new to fibre patching
- **NIC (network interface card):** the plug-in card that connects a server to the network. Real NICs and switches have
  several SFP ports side by side along one edge; here the card stands upright with two of them facing up.
- **SFP port (cage):** the small rectangular metal socket on the card (14.0 x 9.0 mm here, 48.9 mm deep) that takes an SFP
  module. The module is keyed: it fits one way only (tip first, right side up) and has to be pushed in square to the port.
- **SFP plug / module:** the 47.3 x 13.75 x 8.6 mm gold body on the cable's one end. It carries a latch that clicks into the
  cage; in simulation the plug simply stops where its shoulder meets the port (about 41.5 mm).
- **SC plug:** the larger blue square connector on the cable's other end. It stays in its tray; only the SFP end is plugged in.
- **Patch cable:** the thin flexible cable between the two plugs. It is the reason the task needs two arms.
- **Tolerance / clearance:** the gap between plug and port: 0.13 mm a side in width, 0.2 mm in height (a sheet of paper is
  about 0.1 mm thick).

## Why this task matters: coordinated assembly of wired connectors at high tolerance
Plugging a cable into a port looks simple, but a wired connector is the hard case for assembly robots, and a fibre SFP plug
is the hardest common one. Three things combine here, and none can be dropped without making the task easy:

1. **The fit is very tight.** The plug body (13.75 x 8.6 mm) goes into a 14.0 x 9.0 mm port: about 0.13 mm a side, and a
   roll of the plug about its own axis of about 1 degree already jams it. A robot cannot land this open loop. It must
   measure the plug's pose and correct it while pushing, because the plug sits in the gripper, not rigidly at the arm's tip.
2. **The part is not free-standing, it is a cable.** The plug is welded to a 12 g cable that drags on it, twists it and
   springs back. Holding the plug alone is not enough: something has to manage the cable.
3. **The plug has to be re-oriented first.** It starts slanted at 45 degrees and only fits upright and tip first. The robot
   has to pull it out along its axis, turn it in the air, and carry it over the card without letting it tilt back.

That is why this needs **coordinated, two-arm assembly**: one arm holds the cable and keeps its slack clear (so the cable
does not twist or pull the plug off line), while the other arm does the precise work of lifting, turning and inserting the
plug. A single arm that grasps the plug alone is dragged around by the cable and loses the 1-degree roll budget; a robot that
ignores the cable fails. The same split (a precise insertion arm plus a supporting cable-handling arm) is how technicians
patch fibre by hand: one hand steadies the cable, the other seats the connector.

**Practical use.** Patching, re-patching and swapping transceivers in data centres, telecom cabinets and test labs is
repetitive, done in cramped racks, and costly when a connector is not fully seated (a dirty or half-seated fibre end shows
up as intermittent loss that is hard to trace). The same skill (grasp, re-orient, align under feedback, insert while a
cable pulls back) carries over to any wired connector with a tight fit: board-to-board and FFC connectors in electronics
assembly, wiring harnesses in cars and aircraft, charging plugs, and cable tester or patch-panel work. A policy that
solves this one has to learn the part of assembly that stops most robots: contact-rich alignment at sub-millimetre
tolerance with a deformable part attached.

**How it differs from other connector tasks.** The neighbouring tasks (`plug-both-ends-into-tester`,
`reseat-half-inserted-plug`) use jacks and ports with millimetres of play and a plug that starts already aligned or
partly inserted. Here the clearance is 0.13 mm, the plug starts at 45 degrees outside the port, it must be turned
upright in the air, and a second arm is part of the solution.

## For reviewers: the five criteria
1. **Importance.** Fibre patching and transceiver swaps are routine data-centre and telecom work (O*NET 49-2011.00,
   15-1231.00, 49-2022.00). It differs from the connector tasks already on the board by its 0.13 mm clearance, the in-air
   re-orientation from a 45-degree start, and the second arm that handles the cable (see "How it differs" above).
2. **Construct validity.** The plug and port are the AIC models' own collision boxes, unmodified, so the fit is the real
   one. The only addition is a 2 mm lead-in, which real cages also have. Holder, tray and card are fixed, so they cannot be
   moved or broken for a score. The other port scores 0. Dropping the plug from above or pushing it in sideways cannot
   satisfy the 40 mm depth, 3 degree and 1.5 mm centre-line tests together. The scorer was checked on hand-built poses
   (wrong port, upside down, horizontal, off the table, resting on the rim, 39.9 mm deep, 2.9 degree lean, NaN); the unit
   tests cover the tiers, the wrong port, the lean and centre-line limits, and a fall.
3. **Scoring.** All conditions are numbers: depth (a ruler, or a mark 40 mm from the tip), angle (a phone level), centre
   line (1.5 mm, by eye on the port mouth), at rest 1 s. Partial credit at 5 mm (0.25) and 20 mm (0.5). Every outcome is
   covered: anything else scores 0.
4. **Clarity.** `task.json` `setup` lists every object with type, size, material and position; one fixed layout; the
   screenshot shows it; `reset_notes` says how to reset. No seeds (`object_jitter=0`).
5. **Feasibility.** The scripted demo reaches 1.0 within the 2,400-step limit (1,068 steps, about 53 s at 20 Hz) in
   repeated runs. It reads the plug's true pose from the simulator, so it shows that full score is reachable, not that a
   camera-only policy can do it. A real trial needs only a card, a cable and a holder, and is safe: no force beyond a plug push.

## Files
- `plug_sfp_cable_into_nic.py`: the task file (scene, `score()`, `success()`, `distance()`, and the closed-loop `demo_script`).
- `sfp_cable_scoring.py`: the scorer as pure functions. `test_sfp_cable_scoring.py` tests it (`python -m pytest`).
- `fibrekit.py`: the cable (a chain of capsule segments on joints, with a plug at each end), built from `scenekit.py` boxes.
- `cabledemo.py`: runs a task's `demo_script` (closed loop on the plug's measured pose) and writes videos, `setup.png`, `result.json`.
- `demo_result.json`: the scorer's result for the scripted demo run (score 1.0, success, 1,068 steps).
- `geometry.json`, `gen_geometry.py`: box tables from the AIC SDFs (Apache-2.0) and the script that builds them.

## Run a trial
On a machine set up with the official simulator, with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="Take the cable's gold SFP plug out of its slanted holder, stand it upright, and push it, tip first, all the way down into the network card's port nearer the robot (the card stands upright and its ports face up). The cable's blue SC plug stays in its tray. Both arms may be used."
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
| 1.0 | target port, tip depth >= 40 mm, within 3 degrees of the port axis, at rest 1 s | at most ~7 mm of plug body outside the port; a phone level on it |
| 0.5 | target port, depth >= 20 mm, within 10 degrees | about 27 mm of body outside or less |
| 0.25 | target port, depth >= 5 mm, within 10 degrees | a ruler at the mouth |
| 0 | anything else: other port, off the table, tip more than 1.5 mm off the port's centre line | |

## Known simplifications
A 2 mm flared lead-in is added at each port mouth. The cable is a soft 12 g chain with no twist stiffness. No latch spring or
click. Plugs are boxes. The plug stops at about 41.5 mm where its shoulder meets the port.
