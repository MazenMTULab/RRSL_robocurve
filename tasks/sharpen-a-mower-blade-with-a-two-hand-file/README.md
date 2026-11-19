# Sharpen a clamped mower blade's edge with a two-handed file, cutting only on the push stroke along the bevel

A lawn mower blade is clamped flat on a wooden block, its cutting end toward the arms. The edge's bevel has been colored
with marker ink, so every spot the file cuts shows bright steel. A 10-inch mill file lies on the table in front.
"Sharpen the mower blade's cutting edge with the file."

A hand file needs both hands: one at the handle pushing, the other near the tip steadying the file and pressing it onto
the work. It cuts only on the push stroke, toward the tip, and is lifted for the return, since dragging it back across
the edge dulls its teeth. It must ride the edge's 30 degree bevel: held flatter or steeper it grinds a new, wrong bevel.
The job is the whole 6 cm of edge filed bright. The instruction says none of this.

## Run

From the repo root, with Isaac Lab-Arena installed (the Robocurve Brev launchable, `source /data/IsaacLab-Arena/yam.env`):

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=tasks/sharpen-a-mower-blade-with-a-two-hand-file/task.py \
      -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Sharpen the mower blade's cutting edge with the file." \
      --scorer task_score --max-steps 1200
    python -B tasks/sharpen-a-mower-blade-with-a-two-hand-file/test_task.py    # offline scorer checks, prints "all pass"
    python -B -m pytest -q tasks/sharpen-a-mower-blade-with-a-two-hand-file/test_task.py   # the same checks under pytest

Use `--policy dream --max-steps 1200` for a Dream-1 trial. The scene is deterministic (no seeds, `object_jitter=0`).
The scripted demo is 30.2 s (604 steps): six push strokes, two over each 2 cm of edge, with the file tilted off the
edge for every return. About 3 minutes of wall time per trial on an L40S.

Debug and checks:
- `FILE_DEBUG=1` prints the file's pose, its angle to the bevel, its speed along its length, which 1 cm lengths it is
  on, the passes per length, the drag count and both grippers every step.
- `DEMO_DRAG=1` runs a negative control: every return stroke drags the file back along the edge. It files all six
  lengths but scores 0.5 (6 drags).

Files: `task.py` (scene, stroke tracker, scorer, demo), `test_task.py`, and the kit it imports, copied here: `yamkit.py`
(props, colliders, gripper states).

## Spec

Positions in cm from the midpoint between the two arm bases: x forward, y to the robot's left, table top z 0.

| Object | What it is | Where |
|---|---|---|
| Wooden block | 22 x 12 x 11 cm | center x 40, y -6, fixed |
| Mower blade | 4 mm steel, 5.2 cm wide; cutting edge along x on its +y side, 30 degree bevel facing up and toward +y | flat on the block along x from x 28 (runs on past the block); edge at y 0, 11.4 cm up |
| Marker ink | blue ink on the bevel with 1 cm tick marks | x 28.5 to 34.5 (6 cm of edge) |
| C-clamp | red, across the blade | x 49, holding the blade to the block |
| Mill file | 10-inch (25 cm), 2.5 cm wide, 6 mm thick, on a 10 cm wooden handle 2.6 cm across; 0.30 kg | on the table along y at x 17, blade's middle at y 3, tip toward +y |

The blade, block and clamp are one fixed body. The file is loose.

## Scoring

The edge is scored in six 1 cm lengths. Everything is read from the final state, which must hold 1 s
(`SCORE_OVER = "end"`).

- A cutting pass over a length: the file's face on it (within 2.5 mm, and within 10 degrees of the 30 degree bevel)
  while the file moves toward its tip faster than 1 cm/s. A length is filed after 2 passes.
- A drag: the file sliding 1 cm or more back toward its handle while on the edge.

| Outcome | Score |
|---|---|
| All 6 lengths filed | 1.0 |
| 3 to 5 lengths filed | 0.5 |
| 1 or 2 lengths filed | 0.25 |
| None filed | 0 |
| 3 or more drags (cap) | at most 0.5 |

Possible scores: 0, 0.25, 0.5, 1.0. Where the arms and the file end up does not count.

Why each rule, and how a real trial checks it:
- Passes: two light passes take the marker ink off a 1 cm length and show bright steel. Real trial: look at the ink
  and count the bright 1 cm lengths against the tick marks.
- The bevel angle: a file held flatter or steeper cuts a new bevel and the ink stays on the old one, so off-angle
  strokes do not count. Real trial: the ink, again.
- Drags: dragging the file back across the edge is the classic beginner's mistake, dulling the file's teeth. A drag
  must travel 1 cm, so a file settling a few mm on the edge before a push does not count. Real trial: the video.
- Coverage: a blade sharpened only in one spot still tears the grass elsewhere along its edge.

## Real-world build

- A used rotary mower blade (any 18 to 22 inch, 4 mm steel, 30 degree factory bevel), clamped flat on a wooden block
  with a C-clamp, cutting end overhanging toward the robot. Color the bevel with a blue permanent marker and add 1 cm
  ticks beside it.
- A 10-inch mill bastard file with a wooden handle (about 0.3 kg), on the table in front.
- Trials file only lightly (two passes per length), so one blade lasts many trials. Re-ink between trials.

## How it differs from nearby board tasks

- Sharpen a kitchen knife on a whetstone at 15 to 20 degrees: the knife moves over a fixed stone, one-handed, with
  both sides of the edge. Here a two-handed tool works a fixed blade, cuts only on the push stroke, and must cover the
  whole edge.
- Screed a paver bed with a board on two rails; ream the burr off a copper tube: tool use without a cutting direction
  or a bevel to follow.

## Sources

- BLS Occupational Employment and Wage Statistics, May 2022: Landscaping and Groundskeeping Workers (37-3011),
  914,230 employed.
- This Old House, "How to Sharpen Lawn Mower Blades": landscape contractor Roger Cook's method is a mill bastard file
  in two gloved hands, pushed along the factory bevel and lifted off the blade on the backstroke, because filing on
  the backstroke dulls the file.
- Empire Abrasives, "How to Sharpen Lawn Mower Blades": most rotary mower blades have a factory bevel of about
  30 degrees, which sharpening should follow, not change. The file is held firmly in both hands and pushed along the
  edge.

## License

MIT, as the repository. All meshes are built in code; no third-party assets.
