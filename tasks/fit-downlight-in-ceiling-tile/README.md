# Fit a spring-clip mini downlight into its hole in a ceiling tile

**Instruction:** "Fit the downlight into the hole in the ceiling tile so that its rim sits flush on the tile and its spring clips hold it from behind."

A recessed downlight is held in the ceiling only by its own clips: two spring-steel torsion clips on opposite sides
of the body. At rest the clips spring out wider than the hole. The maker's instruction is to press both clips back
flat against the body and push the light into the hole. Once the clips' pivots pass the back of the board, they spring
out again and clamp the board between themselves and the rim. Electricians and lighting installers fit these by the
dozen. The clips cannot be pushed in sprung out (they bear on the tile's face and stop the light), and from the room
side a light that is in the hole but whose clips have not opened behind the board looks almost the same as a fitted one.

Here a suspended-ceiling tile has been lifted out of its grid and laid face up across two battens on the bench, and a
36 mm hole has been cut in it. The mini downlight stands rim up on the tile beside the hole. The robot has to:

1. press both clips back flat against the body and keep them there;
2. carry the light over the hole and start it in, clips first, until the clips' ends are inside the hole (the hole's
   wall then keeps them flat);
3. push the light home until the clips spring out behind the tile and the rim sits flush.

A light dropped onto the hole with its clips out rests on them and scores 0. A light pushed only part way, with its
clips still flat inside the hole, scores 0.25.

The task file is for the `yam_arena` embodiment on the station scene:
[fit_downlight_in_ceiling_tile.py](fit_downlight_in_ceiling_tile.py) holds the scene, the scorer and the
closed-loop scripted demonstration. The helper kits are unchanged from accepted tasks: [yam_parts.py](yam_parts.py),
[yam_joints.py](yam_joints.py) (the light with its two hinged, sprung clips) and [yam_mesh.py](yam_mesh.py) (the
tile's round hole). The release and stillness code follows the accepted pulse-oximeter task, and
[cabledemo.py](cabledemo.py) (from the accepted cable tasks) runs closed-loop demos that read object states.

## Run a trial

Dream-1 (from the repository root):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/fit-downlight-in-ceiling-tile/fit_downlight_in_ceiling_tile.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the downlight into the hole in the ceiling tile so that its rim sits flush on the tile and its spring clips hold it from behind." \
  --scorer task_score --max-steps 1600 --store-frames --no-prompt --log-dir ~/yam-eval/downlight-dream
```

Scripted demo (closed loop):

```bash
cd tasks/fit-downlight-in-ceiling-tile
python cabledemo.py fit_downlight_in_ceiling_tile.py ~/yam-eval/downlight-demo --max-steps 1600
```

It reaches 1.0 in 524 steps, and the episode ends on success (`success()` is a score of 1.0). Runtime per trial: about
1.5 minutes with cameras on one L40S. Object jitter is zero, so no seed is needed.

## Scoring

| Score | Condition |
| --- | --- |
| 0 (sticky, whole trial) | The light is knocked off the table (below the table top). |
| 0.25 | In the hole: the light's axis within 4 mm of the hole's centre, tilted no more than 20 degrees, and both clips' pivots at least 3 mm below the tile's face (the rim no more than 17 mm proud). |
| 0.5 | In the hole, with at least one clip sprung out behind the tile: its pivot below the tile's back face and its arm at least 60 degrees out from the body. |
| 1.0 | Both clips sprung out behind the tile, and the rim flush (its back face within 1.5 mm of the tile's face) and within 3 degrees of square to the tile. |
| 0 | Anything else, including the light resting on the hole with its clips still out on the tile's face, or not over the hole. |

Every credit counts only once every gripper is at least 30 mm from the light and the light has stayed still (under
0.5 mm per 50 ms step) for 1 s. While the light is held, the score is 0.

**Real-world scoring:** a 1.5 mm feeler gauge (or a coin) and a look behind the tile. After the robot lets go:
check whether the gauge slides under the rim at any point (if it does, the rim is not flush), then lift the tile and
look at the back to see whether each clip is sprung out against it or still flat inside the hole. For 0.25, the
light is in the hole with its clips hidden inside it: the clips' pivots are 20 mm behind the rim, so the rim stands
between 5 and 17 mm proud of the tile (a ruler).

## Checks

Run in the simulator with this task file (`object_jitter=0`):

| Run | Result |
| --- | --- |
| Scripted demo (twice, identical) | 1.0 in 524 steps: rim flush (0.0 mm), square (0.0 degrees), both clips sprung out to 101 degrees against the tile's back |
| Nothing done | 0 |
| Clips pressed flat, but let go 5 cm above the hole, then the rim pressed | 0: the clips spring out in the air, the light lands on the tile's face and tips over (77 degrees) |
| Clips pressed flat, let go 2 cm above the hole, then the rim pressed | 0: the clips spring out and the light falls over beside the hole |
| Pushed part way and let go | 0.25: in the hole with both clips still flat inside it (rim 14.5 mm proud, tilted 4.6 degrees) |
| Seated by the demo, but the gripper stays pressing on the rim | 0: never released |

`test_scoring.py` (run where yam_arena is installed) checks the rules on synthetic layouts:
- the start;
- seated with both clips out, and the same 1 mm proud;
- resting on the hole on its open clips;
- pushed part way with the clips still flat;
- one clip out with the rim proud, and both clips out with the rim proud;
- tilted 5 degrees;
- 6 mm off the hole;
- on the tile beside the hole;
- knocked off the table;
- still held.

## Scene and reset

Positions are from the midpoint between the two arm bases: x forward, y to the left, on the station table (top
74.75 cm). The robot, table and cameras are the standard YAM station.

| Object | Position (cm) | Size and material |
| --- | --- | --- |
| Two battens (fixed) | along y under the tile's edges, x 10.0 to 12.5 and 65.5 to 68.0 | Softwood, 5 cm tall, 2.5 cm wide, 60 cm long |
| Ceiling tile (fixed) | face up on the battens, x 9 to 69, y -38 to 22, face at z 81.25; hole centred at (30.0, -8.0) | Mineral fibre, white, 60 x 60 x 1.5 cm, 1.6 kg; 36 mm round hole |
| Mini LED downlight (free, 49 g) | standing on its back end on the tile at (23.0, -17.0), rim up, clips toward +y and -y | White rim 40 mm across and 3 mm thick; aluminium body 32 mm across, 52 mm long (55 mm overall); two spring-steel torsion clips pivoting 20 mm behind the rim, arms 24 x 6 x 1 mm, springing out to 140 degrees from flat (0.015 N m/rad; 1.7 N at the arm's end holds one flat) |

The light's rim, cut-out and height are those of the KSR Mimas mini LED baffle downlight (40 mm rim, 36 mm cut-out,
55 mm). Its body diameter, clip size and clip spring are typical values; the maker does not publish them.
Friction (static/dynamic): tile 0.6/0.5, light 0.4/0.35, clips 0.35/0.3.
**Reset:** press both clips flat from behind the tile, push the light back out, and stand it rim up on its mark.
Nothing is consumed.
**Safety:** a 49 g light with no wiring connected, on a bench; nothing sharp, hot or heavy.

## Fidelity

- The light is a rigid rim and body with two hinged clip arms on torsion springs. The spring rate is a typical value
  for mini-downlight clips, not a measured one.
- The round hole is a 32-sided polygon.
- The tile is fixed and rigid. A real mineral-fibre tile lies loose on its supports, and its cut edge crumbles a little.
- The tile lies face up on the bench, as when lights are fitted to a tile lifted out of its grid. Overhead, the same
  clips would also have to hold the light up against gravity. Here gravity helps the rim sit flush, so flush alone
  is not credited: both clips must be sprung out behind the tile.
- No wiring.
- The 80 mm patch around the hole is built from separate mesh pieces, so in the videos it renders a slightly
  different shade from the rest of the tile; it is the same tile.
