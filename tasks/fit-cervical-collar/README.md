# Fit a two-piece neck collar on a lying patient while keeping the head from rolling

A task file for the `yam_arena` embodiment (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1) on the station scene:
[fit_cervical_collar.py](fit_cervical_collar.py). The file holds the scene, the scorer and the scripted
demonstration. [parts.py](parts.py) is the articulated-parts helper shared by other station tasks; it simulates the
head's roll on a hinge and the collar halves as multi-shape loose parts.

## Run a trial

Dream-1 (from the repository root):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/fit-cervical-collar/fit_cervical_collar.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the two-piece neck collar on the training manikin: slide the back half under its neck, then set the front half on the front of its neck, keeping its head straight the whole time." \
  --scorer task_score --max-steps 2400 --store-frames --no-prompt --log-dir ~/yam-eval/collar-dream
```

Scripted demo: the same command with `--policy yam_scripted`. It reaches 1.0 and the episode ends on success.

The run prints `task_score` at the end; `success()` is a score of 1.0.

## The head

The head rolls about its own long axis on a hinge through the middle of the skull. Its weight is modelled as a
weak spring (0.045 N m/rad) pulling it toward lying on its left side (0.9 rad), against damping (0.7 N m s/rad):
about 0.04 N m near straight, half a newton sideways at the top of the head. Unheld, it passes 10 degrees about 3 s
after the start and keeps rolling. The skull is a cylinder, so a push on it points through the roll axis and only
friction resists; the flat face and forehead give a hand leverage, so a light press on the side of the forehead that
rises (the patient's right) holds the head. The scripted demo's right arm does this, holding it about 3 degrees over.

## Scoring

| Score | Condition |
| --- | --- |
| 0.5 | The back half lies flat (tilted no more than 10 degrees, lowest point within 1 cm of the table) under the neck, its centre within 1.5 cm of the neck's centre both along and across the neck. |
| 1.0 | In addition, the front half rests on the front of the neck (its shell within 8 mm of the neck's front surface), level within 10 degrees, its centre within 1.5 cm of the neck's centre both ways. |
| Frozen | If the head ever rolls more than 10 degrees either way from straight, the score stays at its value at that moment for the rest of the trial. |
| 0 | Otherwise. The front half earns nothing without the back half. |

The score is read from the current state every step, so a half knocked out of place loses its credit. The task
module stays loaded across trials, so the scorer clears its record only on a reset, which it recognises by the back
half jumping back to its start within one control step.

**Real-world scoring:** film from above and from the side with a ruler on the table. Positions and the 1 cm and
8 mm heights are read against the ruler; tilts and the head's roll with a protractor or a phone level app on the
forehead and on each collar half, or from the overhead video (10 degrees of roll moves the nose about 1.5 cm
sideways).

## Checks

Run in the simulator with this task file (`object_jitter=0`):

| Run | Result |
| --- | --- |
| Scripted demo (right arm on the forehead throughout) | 1.0 in 406 steps; head held 2.5-4 degrees over; back half within 2 mm of the neck's centre, front half within 6 mm |
| Same, but the right arm never touches the head | 0: the head passes 10 degrees at about 3 s, before the back half is in |
| Back half only | 0.5 |
| Front half only, no back half | 0 (the front half itself rests correctly on the neck) |
| Right arm lets go of the head once the back half is in | 0.5: the head rolls past 10 degrees before the front half is on |

`test_scoring.py` checks the rules offline on synthetic states (`python test_scoring.py`): each stage, a back half
off-centre, lifted onto the neck or tilted, a front half without a back half, a front half caught on the neck,
head rolls both ways before and after each stage, a roll of just under 10 degrees, a reset between trials and a back
half carried out and back without a reset.

## Scene and reset

Positions are from the midpoint between the two arm bases: x forward, y to the left, on the station table
(top 74.75 cm). The robot, table and cameras are the standard YAM station.

| Object | Position (cm) | Size and material |
| --- | --- | --- |
| Head (hinged to the manikin) | x 2-22, y -8 to 8 | Skin-toned vinyl. Skull a cylinder 16 cm across, 20 cm crown to chin, 2 mm off the table; a flat face and forehead 14 x 12 cm, 16.5 cm above the table; nose 4.5 x 2.8 x 2 cm; closed eyes. Rolls about its long axis toward +y (the patient's left) unless held. |
| Neck (fixed) | x 22-31, y -5.5 to 5.5 | Same vinyl, 11 cm wide; its front 11.5 cm above the table, a 3 cm gap under it. |
| Upper chest (fixed) | x 31-58, y -19 to 19 | Same vinyl, 14 cm high. |
| Collar back half (free) | centre (26.5, 27), lying flat, raised end at +y | Blue closed-cell foam on a rigid core: plate 24 x 6 x 1.2 cm with one end raised 4.5 cm (1.2 cm thick), 50 g. |
| Collar front half (free) | centre (10, 25), standing on its flanges, chin rest toward the robot | White polyethylene shell 14.6 x 7 x 1 cm with two side flanges 3.5 cm deep, 1 cm thick and 12.6 cm apart inside; blue foam chin rest 6 x 1 x 2 cm along the head-side edge; 60 g. |

Face up with the head toward the robot, the patient's right is the robot's right (-y).

Reset: put both collar halves back in their start positions, the head straight and both arms in the ready pose. The
head starts to roll as soon as it is let go.

## Runtime and seeds

- Demo: 406 steps (20.3 s simulated); the episode ends on success. About 1.5 minutes on an L40S with frames stored.
- Dream-1: 2,400 steps (2 minutes simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Approximations

The manikin is rigid blocks; the head only rolls (no nodding or turning), and its weight is a weak spring toward
lying on its side. The collar is two rigid pieces after a two-piece adult collar: the back half does not bend
around the neck, the front half is flat-topped, and the hook-and-loop straps that join the halves are not modelled,
so fitting means placing both halves correctly. The neck front is 11.5 cm above the table (a slim adult) so a
straight-down grip can carry the front half over it.

In the top camera's view the right arm, holding the forehead, hides much of the neck; the left wrist camera shows the
collar going on.
