# Hand off and insert a duplex fibre connector

One arm picks up the cable-bound duplex SC connector and presents it to the other arm. The receiving arm secures the connector; the first arm releases. The receiving arm then levels the connector, aligns it with the adapter, inserts it, releases it, and retreats. The SFP end remains seated in the network card.

## Run and score

From the repository root, with `yam-isaac-lab-arena` installed:

```bash
inspect-robots run --embodiment yam_arena -E task=tasks/plug-sc-duplex-into-adapter/plug_sc_duplex_into_adapter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 2400 --instruction "Pass the blue SC connector from one arm to the other. The receiving arm must secure it before the first arm releases. Insert it fully into the adapter, let go, and retreat."
```

The scripted feasibility demo is generated with:

```bash
python tasks/plug-sc-duplex-into-adapter/cabledemo.py tasks/plug-sc-duplex-into-adapter/plug_sc_duplex_into_adapter.py demo_out --max-steps 1800
```

The run limit is 2,400 control steps at 20 Hz. Layout is fixed; object jitter is zero.

## Scoring

The scorer evaluates measured connector pose. Score 0.25 at 2 mm insertion, 0.5 at 6 mm, and 1.0 at 11 mm, subject to the alignment and roll limits in `sc_scoring.py`. Success requires full score and speed below 0.03 m/s for 20 seconds. The scripted demo also checks transfer retention: after the donor releases, a 10 mm receiver lift must move the connector by the commanded amount. The current pose-based scorer does not independently verify a handoff event.

## Demonstrated result

The local scripted run scored 1.0 in 1,467 steps. The connector remained seated after gripper release and retreat during a 20-second hold. `demo.mp4` and `setup.png` show the recorded feasibility run. Dream-1 was attempted once, but the hosted session repeatedly expired before the first action; no Dream score is available.
