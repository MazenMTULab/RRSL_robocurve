# Seat an SFP module in a hand-held network card

**Status: draft. The scene file (`sfp_nic_bimanual_seat.py`), the runs and the media are not done yet.**

One arm holds a PCIe network card in the air; the other holds an SFP module and pushes it straight into the card's
left SFP port. Neither arm can do it alone: the card has to be lifted and kept steady while the module is aligned in
both position and angle to within a few millimetres and degrees. Credit rises with insertion depth.

## Files
- `gen_geometry.py` builds `geometry.json` (box tables) from the AIC SDFs in `assets/aic`.
- `sfp_nic_scoring.py` is the scorer: poses in, score out, no simulator imports. `test_sfp_nic_scoring.py` tests it
  (`python -m pytest`, needs numpy and pytest).
- `task.json` is the submission text.

## Run a trial
TODO once the scene exists (inspect-robots commands for Dream-1 and for the scripted demo, as in `pin-fit`).

## Scoring
Reference frame is the card's. Tip depth = how far the module's tip is past the cage mouth along the port axis.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | left port, tip depth >= 42 mm, within 3 degrees of the port axis and 5 degrees of level, held 1 s, card >= 2 cm above the table | the module's visible length beyond the cage mouth is at most about 13 mm (55 - 42); a phone level on its top |
| 0.5 | left port, depth >= 20 mm, within 10 degrees of the axis and 15 degrees of level | visible length at most about 35 mm |
| 0.25 | left port, depth >= 5 mm, same angles as 0.5 | a ruler at the mouth |
| 0 | anything else, including the right port, a card resting on the table, or anything off the table | |

The card must be held up because otherwise a single arm could press the module into a card lying on the table.
Inserting into the right port scores 0.

## Runtime and seeds
TODO after the first runs.
