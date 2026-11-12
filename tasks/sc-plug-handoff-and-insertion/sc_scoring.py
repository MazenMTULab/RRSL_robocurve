"""Pure scoring for 'plug the SC end of a cable into a duplex SC adapter port'. Poses in, score out; no simulator imports.

World frame: x forward from the arms, y left, z up. The adapter's port frame (the AIC SC Port model's own) has its channel along y,
its two bays side by side along x and its keyed plate on top (+z). The port is turned PORT_YAW about the vertical (0: its mouth,
the +y face, faces the left arm; the plug goes in along the world -y). The plug's frame is the cable's start plug frame: its origin
is the middle of the plug's rear body, its x axis points into the cable (away from the plug's tip), its z axis is up when the plug is
the right way up. Seated, the plug's rear shoulder is against the port's mouth and its tip is about 13.4 mm inside.
"""

import json
import math
from pathlib import Path

import numpy as np

GEOM = json.loads((Path(__file__).parent / "geometry.json").read_text())
PLUG, PORT = GEOM["sc_plug"], GEOM["sc_port"]

PORT_YAW = 0.0  # port y (its mouth side, +y) -> world +y: the mouth faces the left arm
MOUTH_Y = max(b["pos"][1] + b["size"][1] / 2 for b in PORT if b["size"][1] > 0.02)  # the port's +y face: its mouth (13.7 mm)
_REAR = next(b for b in PLUG if b["name"].startswith("cube002"))  # the plug's rear body: 12.55 mm long, 25 mm wide, 10.1 mm tall
REAR_CENTRE_X = _REAR["pos"][0]  # in the AIC plug model's frame
TIP_X = max(b["pos"][0] + abs(b["rot"][0][0]) * b["size"][0] / 2 + abs(b["rot"][0][1]) * b["size"][1] / 2 for b in PLUG)  # the tip face
TIP_OFFSET = TIP_X - REAR_CENTRE_X  # from the rear body's centre to the tip face along the plug's axis (19.7 mm)
SHOULDER_X = _REAR["pos"][0] + _REAR["size"][0] / 2  # the rear body's front face
SEATED_DEPTH = TIP_X - SHOULDER_X  # tip depth when the shoulder meets the mouth (13.4 mm)

FULL_DEPTH, HALF_DEPTH, QUARTER_DEPTH = 0.0110, 0.0060, 0.002  # m of tip travel past the mouth. Seated, the plug rests about 11.9 mm in the simulation (13.4 mm by the collider sizes)
FULL_TILT_DEG, PART_TILT_DEG = 3.0, 10.0
LATERAL_M = 0.001  # tip centre off the port's centre line
ROLL_DEG = 5.0  # the plug's top face against the port's keyed plate


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


R_PORT = rz(PORT_YAW)


def plug_tip(centre, quat):
    """Tip face centre, unit axis toward the tip and unit 'up' vector of the cable's start plug, from its centre and orientation
    (the plug's x axis points into the cable, so the tip is the other way)."""
    r = rotation(quat)
    return np.asarray(centre, float) - r[:, 0] * TIP_OFFSET, -r[:, 0], r[:, 2]


def measure(tip, axis, up, port_pos) -> dict:
    """Depth of the plug's tip past the port's mouth (m), the angle between the plug's axis and the port's (deg), the tip's offset
    from the port's centre line (m) and the plug's roll (deg, its top face against the world's up)."""
    into = R_PORT @ np.array([0.0, -1.0, 0.0])  # the way into the port: world -y
    mouth = np.asarray(port_pos, float) + R_PORT @ np.array([0.0, MOUTH_Y, 0.0])
    rel = np.asarray(tip, float) - mouth
    depth = float(rel @ into)
    local = R_PORT.T @ rel  # in the port frame: x across the bays, z up
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ into)))))
    roll = math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(up, float) @ np.array([0.0, 0.0, 1.0]) / np.linalg.norm(up))))))
    return {"depth": depth, "tilt": tilt, "offset": float(math.hypot(local[0], local[2])), "roll": roll}


def score(tip, axis, up, port_pos, plug_z=None, table_z=0.0) -> float:
    if plug_z is not None and plug_z < table_z - 0.02:
        return 0.0  # fell off the table
    vals = (*np.asarray(tip, float), *np.asarray(axis, float), *np.asarray(up, float), *np.asarray(port_pos, float))
    if not all(math.isfinite(v) for v in vals):
        return 0.0
    m = measure(tip, axis, up, port_pos)
    d, t = m["depth"], m["tilt"]
    if m["offset"] > LATERAL_M or m["roll"] > ROLL_DEG:
        return 0.0
    if d >= FULL_DEPTH and t <= FULL_TILT_DEG:
        return 1.0
    if d >= HALF_DEPTH and t <= PART_TILT_DEG:
        return 0.5
    if d >= QUARTER_DEPTH and t <= PART_TILT_DEG:
        return 0.25
    return 0.0
