# Cleft-graft an apple stub: set a scion at each end of the split, cambium to cambium, then pull the wedge

**Instruction:** "Cleft-graft the two scions onto the split apple stub, one at each end of the cleft with its cambium against the stub's cambium, then remove the grafting tool."

Cleft grafting is the standard way to top-work an apple or pear tree onto a new variety, and the way many
nursery trees are made. The stock is cut off flat and split across the middle. The wedge of a cleft-grafting tool
holds the split open. A dormant scion, its base cut to a long wedge, is set at each end of the split. Then the tool
comes out, and the springing wood clamps the scions.

A scion only takes where its cambium touches the stock's cambium. The cambium is the thin green growing layer
just under the bark. The stock's bark is thicker than the scion's, so the extension guides' rule is to set each
scion with its bark slightly *inside* the stock's bark, not flush with it:
- set flush, the two cambium layers miss;
- pushed in toward the middle, they miss the other way.

Here the bark is 2.5 mm on the 38 mm stub and 1 mm on the 9 mm scions. The cambium lines meet when the scion's bark
is 1.5 mm inside the stub's. The scorer accepts 0.5 to 2.5 mm, which puts the cambium lines within 1 mm of each other.

What the robot has to do:
1. **Pick each scion out of its stand** and carry it over its own end of the open cleft, without turning it: the
   wedge's cut faces have to face the cleft's walls.
2. **Lower it into the cleft at the right radial position,** to within about 1 mm, until its top of cut is about
   level with the stub's cut face. The left arm sets the left scion; the right arm sets the right one.
3. **Pull the tool's wedge out of the middle of the cleft** after both scions are in, so the two halves spring
   shut on them.

The order matters physically. If the tool comes out first, the cleft snaps shut and no scion can be pushed into it:
in the simulator the scions topple off the closed stub.

Files:
- [set_scions_in_a_cleft_graft.py](set_scions_in_a_cleft_graft.py): the scene, the scorer and the closed-loop
  scripted demo.
- The helper kits, unchanged from accepted tasks:
  - [yam_parts.py](yam_parts.py);
  - [yam_joints.py](yam_joints.py), for the two sprung halves of the stub;
  - [yam_mesh.py](yam_mesh.py), for the D-shaped halves, the scions' wedges and the tool's prong.
- [cabledemo.py](cabledemo.py), from the accepted cable tasks: it runs closed-loop demos that read object states.
- The release and stillness code follows the accepted pulse-oximeter task.

## Run a trial

Dream-1 (from the repository root):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/set-scions-in-a-cleft-graft/set_scions_in_a_cleft_graft.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cleft-graft the two scions onto the split apple stub, one at each end of the cleft with its cambium against the stub's cambium, then remove the grafting tool." \
  --scorer task_score --max-steps 1800 --store-frames --no-prompt --log-dir ~/yam-eval/graft-dream
```

Scripted demo (closed loop):

```bash
cd tasks/set-scions-in-a-cleft-graft
python cabledemo.py set_scions_in_a_cleft_graft.py ~/yam-eval/graft-demo --max-steps 1800
```

It reaches 1.0 in 883 steps, and the episode ends on success (`success()` is a score of 1.0).
Runtime per trial: about 2 minutes with cameras on one L40S. Object jitter is zero, so no seed is needed.

## Scoring

Measured on each scion where its axis crosses the stub's cut face.

**A scion is set right when all of these hold:**
- **In the cleft:** its axis is within 4 mm of the cleft's middle plane, and its wedge is at least 20 mm down.
- **At its own end:** the left scion at the left end of the cleft, the right scion at the right end.
- **Bark inset:** its bark is 0.5 to 2.5 mm inside the stub's bark, so the cambium lines are within 1 mm.
- **Height:** its top of cut is within 5 mm of the stub's cut face.
- **Upright:** it leans no more than 15 degrees.

| Score per scion set right | When |
| --- | --- |
| 0.25 | the tool's prong is still in the cleft |
| 0.5 | the prong is out (its tip at least 2 mm above the cut face) and the cleft has closed on the scions |

The total is the sum over the two scions, from 0 to 1.0. Credit counts only while every gripper is at least 30 mm
from both scions and both scions have been still (under 0.5 mm per 50 ms step) for 1 s. While a scion is held,
the score is 0.

**Real-world scoring** needs calipers (or a 0.5 mm and a 2.5 mm feeler gauge), a ruler and a look:
- **Inset:** after the robot lets go, measure at each end of the cleft how far the scion's bark sits inside the
  stub's bark.
- **Height:** check the top of the scion's cut against the stub's cut face, within 5 mm.
- **Upright:** check that the scion stands within 15 degrees of upright, using a protractor or a phone level.
- **Tool:** check whether the tool is out of the cleft.

## Checks

Run in the simulator with this task file (`object_jitter=0`):

| Run | Result |
| --- | --- |
| Scripted demo | 1.0 in 883 steps: bark insets 1.3 mm (left) and 1.8 mm (right), tops of cut 1.4 and 1.7 mm below the cut face, upright within 2.1 degrees |
| Scripted demo, repeated (twice more, including the rendered run) | 1.0 in 883 steps both times, identical |
| Nothing done | 0 |
| Both scions set with their bark flush with the stub's bark | 0: insets 0.1 and -0.1 mm |
| Both scions set in toward the middle (against the prong) | 0: the left one at 3.2 mm inset; the right one is dragged out of line when the prong is pulled past it |
| Tool pulled out first, then the scions tried | 0: the cleft snaps shut, and both scions topple off the stub |
| Only the left scion set, tool pulled | 0.5 |
| Both scions set, tool left in the cleft | 0.5 |
| Left arm never lets go of its scion (right arm carries on) | 0: never released; the held scion is pulled 36 degrees out of line |
| Each scion let go 20 mm above its spot, aligned | 1.0: the open cleft guides an accurately aligned scion down to the same position (insets 2.4 and 1.4 mm). This is a correct graft, not a shortcut: the radial position still has to be right to within 1 mm. |

`test_scoring.py` (run where yam_arena is installed) checks the rules on synthetic layouts:
- the start;
- both scions set right, with the tool out, lifted clear, or still in;
- only one scion;
- the edges of the inset band; flush; pushed 3 mm in; toward the middle;
- one scion right and one flush;
- the top of cut 4 mm down, sunk 7 mm, and 6 mm proud;
- leaning 6 and 20 degrees;
- beside the cleft;
- knocked onto the bench;
- still held.

## Scene and reset

Positions are from the midpoint between the two arm bases: x forward, y to the left, on the station table (top
74.75 cm). The robot, table and cameras are the standard YAM station.

| Object | Position (cm) | Size and material |
| --- | --- | --- |
| Bench vise (fixed) | centred at (22.0, 0.0) | Steel, bed 11 x 7 x 0.6 cm, jaws 1.4 x 7 x 2 cm |
| Apple rootstock stub (fixed below the cleft) | upright in the vise at (22.0, 0.0), cut face 6.5 cm above the table | 38 mm across, 2.5 mm bark; split across the middle (the cleft runs along y), 40 mm deep. Each half above the root springs shut about it at 2.0 N m/rad, preloaded 3 degrees past closed. |
| Cleft-grafting tool (free, 110 g) | prong driven 21 mm into the middle of the cleft, holding it 10 mm open; bar along x toward the robot | Steel bar 10 x 10 mm, 105 mm long, 14-24 mm above the cut face; wedge prong 12 mm wide, 12 mm thick at the top tapering to 1.4 mm over 25 mm (12 degrees per face) |
| Scion stand (fixed) | sockets at (17.5, 11.5) and (17.5, -11.5) | Wood; two 11 mm square sockets, 18 mm deep |
| Two dormant apple scions (free, 6 g each) | standing wedge down in the sockets, cut faces facing front and back | 9 mm across, 85 mm long, 1 mm bark; bottom 32 mm cut to a two-sided wedge, 9 mm at the top of the cut to 1.8 mm at its edge |

Friction (static/dynamic): wood 0.5/0.45, tool 0.35/0.3.

**Reset:** open the cleft with the tool's prong, lift both scions out, stand them back in their sockets with their
cut faces front and back, and leave the prong driven in to hold the cleft 10 mm open. Nothing is consumed in the
simulation. A real-world trial needs a freshly split stub and fresh scions, or a reusable trainer: a split hardwood
dowel on a spring hinge, with dowel scions.

**Safety:** no blade is used during the trial. The stub is already split and the scions are already cut. The tool's
blade edge should be taped or blunted for robot trials.

## Fidelity

- **Dimensions.** They follow extension-service guidance on cleft grafting: stock 1-4 in across, scions 1/4-3/8 in
  across with a 1-1.5 in wedge, and the scion set slightly in from the stock's bark because the stock's bark is
  thicker. Sources: OSU PNW 496, "Propagating Plants by Grafting and Budding"; Penn State Extension, "Cleft
  Grafting".
- **Cleft depth.** The guides split the stock 2-3 in deep. This cleft is 40 mm (1.6 in), and the stub stands only
  65 mm above the bench, so that the scions can be lowered into the cleft within the arms' reach.
- **The cleft's spring is far weaker than real wood.** Each half is a rigid piece hinged at the cleft's root with
  2.0 N m/rad, which gives about 8 N on each scion's edge. A real split 38 mm apple stem pinches much harder, and
  grafters lever or tap the tool out against it. The order of the steps and the geometry of the graft are real; the
  force needed to withdraw the wedge is not.
- **The halves are rigid and hinged,** where real wood bends along the whole split.
- **The cambium is not modelled as tissue.** It is scored as geometry: the scion's bark line against the stub's.
  Real grafters also tilt the scion slightly so the cambium lines cross. Here the scion has to be upright within
  15 degrees, and its cambium must lie within 1 mm of the stub's at the cut face.
- **The wedge cut is idealised:** a smooth two-sided taper whose angle matches the closed cleft, so its faces bear
  along their length.
- **The scions start upright in a stand.** The YAM's grippers can only turn about the vertical, so they cannot
  stand up a stick lying on the bench.
- **The tool has already been driven in.** A grafter splits the stock with the tool's blade and drives the wedge in
  with a mallet; this task starts after that.
- No grafting wax or tape is applied.
