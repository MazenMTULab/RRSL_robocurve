# Hand off and insert a duplex fibre connector

The right arm picks up the cable-bound duplex SC connector and presents it to the left arm. The left arm secures the connector; the right arm releases and retreats. The left arm then levels the connector, aligns it with the adapter, inserts it, releases it, and retreats. The SFP end remains seated in the network card.

## Run and score

From the repository root, with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena -E task=tasks/sc-plug-handoff-and-insertion/plug_sc_duplex_into_adapter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --seed 0 --scorer task_score --max-steps 2400 --instruction "Pass the blue duplex SC connector from the right arm to the left arm. The left arm must secure the connector before the right arm releases it. Then insert the connector into the adapter port, release it, and retreat."
```

The scripted feasibility demo is generated with:

```bash
python3 tasks/sc-plug-handoff-and-insertion/cabledemo.py tasks/sc-plug-handoff-and-insertion/plug_sc_duplex_into_adapter.py demo_out --max-steps 1800
```

The run limit is 2,400 control steps at 20 Hz. Layout is fixed; object jitter is zero.

## Scoring

The measured handoff state gates all insertion credit. Both grippers must close near the connector; after the donor opens and retreats, the receiver must remain closed and move at least 8 mm while the plug follows within 4 mm. Insertion scores 0.25 at 2 mm, 0.5 at 6 mm, and 1.0 at 11 mm tip depth past the mouth. The tip must stay within 1 mm of the port centreline, roll within 5 degrees, and tilt within 10 degrees for partial credit or 3 degrees for full credit. The task holds full score and connector speed below 0.03 m/s for 20 seconds to register success.

## Demonstrated result

The local scripted run scored 1.0 in 1,467 steps. It held the connector seated after release and retreat for 22 seconds; the task's success condition requires a continuous 20-second hold. `demo.mp4` and `setup.png` show the recorded feasibility run. The Dream-1 attempt stalled before the first action, so it produced no score or model video.
