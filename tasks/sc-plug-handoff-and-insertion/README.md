# Hand off and insert a duplex fibre connector

At a fibre service bench, the free end of a transceiver-terminated lead is handed between arms and seated in a keyed duplex coupler while the attached cable remains constrained by the fixed transceiver. The right arm picks up the cable-bound duplex SC connector and presents it to the left arm. The left arm secures it before the right arm releases and retreats; the left arm then levels and inserts the connector, releases it, and retreats. This focuses on transfer and retention of the cable's SC end, distinct from seating the SFP end in a network-card cage. The SFP end remains seated in the network card.

## Run and score

From the repository root, with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=tasks/sc-plug-handoff-and-insertion/plug_sc_duplex_into_adapter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --seed 0 --scorer task_score --max-steps 2400 --store-frames --log-dir task_runs/sc_plug_dream --no-prompt --instruction "The right arm passes the cable-bound blue duplex SC connector to the left arm. The left arm secures it before the right arm releases and retreats. Insert the connector into the adapter port, release it, and retreat."
```

The scripted feasibility demo is generated with:

```bash
python3 tasks/sc-plug-handoff-and-insertion/cabledemo.py tasks/sc-plug-handoff-and-insertion/plug_sc_duplex_into_adapter.py demo_out --max-steps 1800
```

The run limit is 2,400 control steps at 20 Hz. Layout is fixed; object jitter is zero.

## Scoring

The measured handoff state gates all insertion credit. Both TCPs must be within 40 mm of the connector simultaneously while both grippers are closed; after the donor opens and retreats beyond 40 mm, the receiver must remain closed and within 40 mm, then move at least 8 mm while the plug follows within 4 mm. Insertion credit is awarded after both grippers open and both TCPs retreat beyond 40 mm from the connector. The released connector scores 0.25 at 2 mm, 0.5 at 6 mm, and 1.0 at 11 mm tip depth past the mouth. The tip must stay within 1 mm of the port centreline, roll within 5 degrees, and tilt within 10 degrees for partial credit or 3 degrees for full credit. Episode success also requires connector speed below 0.03 m/s continuously for 20 seconds.

## Demonstrated result

The recorded scripted run scored 1.0 in 1,467 steps and held the connector seated after release and retreat for 22 seconds; success requires a continuous 20-second hold. The Dream-1 evaluation was stopped at 425 steps after repeated expired-session responses, so it has no final score. The scripted video, setup screenshot and any Dream run video belong in the task-board media upload, not in this repository. The recorded scripted score predates the current dual-grip, receiver-retention, and final-release scorer gates and must be reconfirmed by a simulator run before submission.
