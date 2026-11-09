"""Pure scoring for 'seat an SFP module in a network card's SFP port'. Poses in, score out; no simulator imports.

World frame: x forward from the arms, y left, z up. The card stands upright with its two SFP cages opening upward and its
PCB facing the arms, turned CARD_YAW about the vertical. The module is the AIC SFP module standing tip-down: its frame
(the part frame) is axis-aligned with the world when it stands straight, its long axis is the frame's z and its tip is
the lowest point. Quaternions are xyzw.
"""

import json
import math
from pathlib import Path

import numpy as np

GEOM = json.loads((Path(__file__).parent / "geometry.json").read_text())
NIC, MOD = GEOM["nic"], GEOM["module"]

MOUTH_Z = -NIC["cage_open_y"]  # cage mouth, above the card origin (card y runs downward)
BACK_Z = -NIC["cage_back_y"]
# bay centre lines in the card's upright frame (x toward the arms, y left): card x runs to -y, card z to +x
BAY_Y = [-(b["x_min"] + b["x_max"]) / 2 for b in NIC["bays"]]  # [left bay, right bay]
BAY_X = (NIC["cage_floor_z"] + NIC["cage_roof_z"]) / 2
TIP_Z = MOD["body_y"][0]  # tip face below the module's origin (negative)

TARGET_BAY = 0  # the left bay (larger y)

FULL_DEPTH, HALF_DEPTH, QUARTER_DEPTH = 0.042, 0.020, 0.005  # m of tip travel below the cage mouth
FULL_TILT_DEG, FULL_YAW_DEG = 3.0, 3.0
PART_TILT_DEG, PART_YAW_DEG = 10.0, 10.0
LATERAL_M = 0.0015  # tip centre off the bay's centre line at the mouth; three times that once it is in


def rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def measure(mod_pos, mod_quat, card_pos, card_yaw) -> dict:
    """Depth of the module's tip below the cage mouth (m), its lean from vertical and its turn from the card's
    orientation (deg), and each bay's tip offset (m)."""
    r = rotation(mod_quat)
    tip = np.asarray(mod_pos, float) + r @ np.array([0.0, 0.0, TIP_Z])
    local = rz(-card_yaw) @ (tip - np.asarray(card_pos, float))  # tip in the card's upright frame
    axis = r @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(axis[2])))))
    x_axis = r @ np.array([1.0, 0.0, 0.0])
    turn = math.degrees(abs((math.atan2(x_axis[1], x_axis[0]) - card_yaw + math.pi) % (2 * math.pi) - math.pi))
    offsets = [math.hypot(local[0] - BAY_X, local[1] - y) for y in BAY_Y]
    return {"depth": float(MOUTH_Z - local[2]), "tilt": tilt, "turn": turn, "offsets": offsets, "bay": int(np.argmin(offsets))}


def score(mod_pos, mod_quat, card_pos, card_yaw, module_z=None, table_z=0.0) -> float:
    if module_z is not None and module_z < table_z - 0.02:
        return 0.0  # fell off the table
    m = measure(mod_pos, mod_quat, card_pos, card_yaw)
    if m["bay"] != TARGET_BAY:
        return 0.0
    d, t, y = m["depth"], m["tilt"], m["turn"]
    if m["offsets"][TARGET_BAY] > LATERAL_M * (1.0 if d < QUARTER_DEPTH else 3.0):
        return 0.0
    if d >= FULL_DEPTH and t <= FULL_TILT_DEG and y <= FULL_YAW_DEG:
        return 1.0
    if d >= HALF_DEPTH and t <= PART_TILT_DEG and y <= PART_YAW_DEG:
        return 0.5
    if d >= QUARTER_DEPTH and t <= PART_TILT_DEG and y <= PART_YAW_DEG:
        return 0.25
    return 0.0
