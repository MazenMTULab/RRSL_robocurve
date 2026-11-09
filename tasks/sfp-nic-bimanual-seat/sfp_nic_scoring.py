"""Pure scoring for 'seat an SFP module in a hand-held NIC card'. No simulator imports: poses in, score out.

Frames: the NIC card's frame is the AIC NIC model's (x across the card, y along it, z up from the PCB); the two SFP
cages open toward -y at y = CAGE_OPEN_Y, so a module is inserted by moving it toward +y, tip first. The SFP module's
frame is the AIC module's: its tip (the end that goes into the cage) is its -y end, so a correctly oriented module has
its -y axis along the card's +y and its z up along the card's z. Quaternions are xyzw.
"""

import json
import math
from pathlib import Path

import numpy as np

GEOM = json.loads((Path(__file__).parent / "geometry.json").read_text())
CAGE_OPEN_Y = GEOM["nic"]["cage_open_y"]
BAYS = GEOM["nic"]["bays"]
CAGE_FLOOR_Z, CAGE_ROOF_Z = GEOM["nic"]["cage_floor_z"], GEOM["nic"]["cage_roof_z"]
BAY_CENTRE_Z = (CAGE_FLOOR_Z + CAGE_ROOF_Z) / 2
TIP_Y = GEOM["module"]["body_y"][0]  # tip face in the module's frame

TARGET_BAY = 0  # the bay at the smaller x (the card's left bay when its connector end faces away)

FULL_DEPTH = 0.042  # m of tip travel past the cage mouth (the cage is 48.9 mm deep)
HALF_DEPTH = 0.020
QUARTER_DEPTH = 0.005
FULL_TILT_DEG, FULL_ROLL_DEG = 3.0, 5.0
HALF_TILT_DEG, HALF_ROLL_DEG = 10.0, 15.0
LATERAL_M = 0.0015  # tip centre off the bay's centreline, at the mouth
LIFT_MIN = 0.02  # the card's lowest point must be this high above the table, i.e. held in the air
OFF_TABLE_M = 0.02


def rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _angle_deg(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def measure(module_pos, module_quat, nic_pos, nic_quat) -> dict:
    """Module pose relative to the card: tip depth past the cage mouth (m), tilt about the insertion axis (deg),
    roll (deg), the tip's offset (m) from each bay's centreline, and which bay it is lined up with."""
    rn, rm = rotation(nic_quat), rotation(module_quat)
    rel = rn.T @ rm
    tip = rn.T @ (np.asarray(module_pos, float) - np.asarray(nic_pos, float)) + rel @ np.array([0.0, TIP_Y, 0.0])
    axis = rel @ np.array([0.0, -1.0, 0.0])  # where the tip points, in the card's frame
    up = rel @ np.array([0.0, 0.0, 1.0])
    tilt = _angle_deg(axis, np.array([0.0, 1.0, 0.0]))
    roll = _angle_deg(up, np.array([0.0, 0.0, 1.0]))
    offsets = [math.hypot(tip[0] - (b["x_min"] + b["x_max"]) / 2, tip[2] - BAY_CENTRE_Z) for b in BAYS]
    return {"depth": float(tip[1] - CAGE_OPEN_Y), "tilt": tilt, "roll": roll, "offsets": offsets, "bay": int(np.argmin(offsets))}


def score(module_pos, module_quat, nic_pos, nic_quat, nic_lowest_z=None, table_z=0.0, module_z=None, off_table=False) -> float:
    """0 to 1. `nic_lowest_z`: lowest point of the card (world z); if given, the card must be held LIFT_MIN above `table_z`."""
    if off_table or (module_z is not None and module_z < table_z - OFF_TABLE_M):
        return 0.0
    if nic_lowest_z is not None and nic_lowest_z < table_z + LIFT_MIN:
        return 0.0  # a card resting on the table can be shoved onto a module with one arm
    m = measure(module_pos, module_quat, nic_pos, nic_quat)
    if m["bay"] != TARGET_BAY or m["offsets"][TARGET_BAY] > LATERAL_M * (1.0 if m["depth"] < QUARTER_DEPTH else 3.0):
        return 0.0
    d, t, r = m["depth"], m["tilt"], m["roll"]
    if d >= FULL_DEPTH and t <= FULL_TILT_DEG and r <= FULL_ROLL_DEG:
        return 1.0
    if d >= HALF_DEPTH and t <= HALF_TILT_DEG and r <= HALF_ROLL_DEG:
        return 0.5
    if d >= QUARTER_DEPTH and t <= HALF_TILT_DEG and r <= HALF_ROLL_DEG:
        return 0.25
    return 0.0
