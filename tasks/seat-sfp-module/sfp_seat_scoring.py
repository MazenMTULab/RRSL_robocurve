"""Pure scoring for 'seat an SFP module in a network card's SFP port'. Poses in, score out; no simulator imports.

World frame: x forward from the arms, y left, z up. The card's upright frame has its SFP cages opening along +z, its PCB
normal along +x and its bays side by side along y. The card is placed with that frame leaning back by CARD_TILT (its
+z pointing up and toward the arms, so the ports face up and toward the robot) and turned CARD_YAW about the vertical:
CARD_R = Rz(CARD_YAW) Ry(-CARD_TILT). The SFP module's own frame has its long axis along z and its tip (the end that goes
in) at -z; seated, the module's frame equals the card's. A module body may be posed in a different frame (the holder
leans it over): ``part_to_module`` turns the body's frame into the module's. Quaternions are xyzw.
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
CARD_TILT = math.radians(45.0)
CARD_YAW = math.radians(8.0)

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


def quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (xyzw) of a rotation matrix."""
    t = float(np.trace(r))
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        q = ((r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k, (r[1, 0] - r[0, 1]) / k, 0.25 * k)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        k = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k_, k_]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * k
        v[j] = (r[j, i] + r[i, j]) / k
        v[k_] = (r[k_, i] + r[i, k_]) / k
        q = (v[0], v[1], v[2], (r[k_, j] - r[j, k_]) / k)
    return tuple(float(x) for x in q)


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def card_rotation(yaw: float = CARD_YAW, tilt: float = CARD_TILT) -> np.ndarray:
    """The card's upright frame in the world: leaned back by ``tilt`` (cages face up and toward the arms), then turned."""
    return rz(yaw) @ ry(-tilt)


def measure(mod_pos, mod_quat, card_pos, card_r, part_to_module=None) -> dict:
    """Depth of the module's tip below the cage mouth along the port axis (m), its lean from that axis and its turn
    about it (deg), and each bay's tip offset (m)."""
    r = rotation(mod_quat) if part_to_module is None else rotation(mod_quat) @ part_to_module
    tip = np.asarray(mod_pos, float) + r @ np.array([0.0, 0.0, TIP_Z])
    local = card_r.T @ (tip - np.asarray(card_pos, float))  # tip in the card's upright frame
    rel = card_r.T @ r
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rel[2, 2])))))
    turn = math.degrees(abs(math.atan2(rel[1, 0], rel[0, 0])))
    offsets = [math.hypot(local[0] - BAY_X, local[1] - y) for y in BAY_Y]
    return {"depth": float(MOUTH_Z - local[2]), "tilt": tilt, "turn": turn, "offsets": offsets, "bay": int(np.argmin(offsets))}


def score(mod_pos, mod_quat, card_pos, card_r, part_to_module=None, module_z=None, table_z=0.0) -> float:
    if module_z is not None and module_z < table_z - 0.02:
        return 0.0  # fell off the table
    m = measure(mod_pos, mod_quat, card_pos, card_r, part_to_module)
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
