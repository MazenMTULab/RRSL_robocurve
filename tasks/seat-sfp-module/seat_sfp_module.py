"""On the station table: a PCIe network card leans back in a slanted cradle in front of the left arm, its flat face toward
the arms and turned 8 degrees (counterclockwise from above) from square to them, its two SFP ports at its upper edge
facing up and toward the arms at 45 degrees. An SFP module (14 x 12 mm, 55 mm long) lies in a slanted holder to the left
of the card, also at 45 degrees, tip down and latch end up. The robot must lift the module out and seat it, tip first, in
the card's left port (the one at the larger y).
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sfp_seat_scoring as sc
from partkit import part
from scenekit import Shape
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

R_CARD = sc.card_rotation()  # the card's upright frame in the world: leaned back 45 degrees, turned 8
R_HOLD = sc.ry(-sc.CARD_TILT)  # the holder's (and the resting module's) frame: leaned the same way, not turned
CLEARANCE = 0.0010  # WHY: real ports leave ~0.1 mm a side, which jams in simulation; widened to 1.0 mm
CARD_Z_CUT = -0.032  # WHY: only the card's upper part is modelled (the rest sits inside the cradle)
BASE_T = 0.006  # the flat plates the slanted cradle and holder stand on
SKIN, METAL, MODULE_COLOR, STEEL = (0.10, 0.45, 0.20), (0.72, 0.74, 0.77), (0.85, 0.70, 0.20), (0.35, 0.36, 0.38)
CAGE = "10099100-011lfc001"
MOUTH_XY = (0.30, 0.08)  # where the middle of the two port mouths is, from above
HOLDER_XY = (0.22, 0.30)  # where the module's origin is, from above
GEOM = sc.GEOM
_M_CARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # card model frame -> its upright frame
_M_MOD = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # module model frame -> its frame (tip down)


def _upright(centre, size, m):
    return tuple((m @ np.array(centre)).tolist()), (size[2], size[0], size[1])


def _card_boxes():
    """(centre, size, colour) in the card's upright frame, ports widened by CLEARANCE, plus the cradle."""
    centres = [(b["x_min"] + b["x_max"]) / 2 for b in sc.NIC["bays"]]
    out = []
    for b in sc.NIC["boxes"]:
        cx, cy, cz = b["pos"]
        sx, sy, sz = b["size"]
        color = SKIN
        if b["name"].startswith(CAGE):
            color = METAL
            near = min(centres, key=lambda c: abs(c - cx))
            if sx < 0.002 and sz > 0.01:  # a port's side wall
                cx += CLEARANCE if cx > near else -CLEARANCE
            elif sy > 0.04 and sx > 0.01 and sz < 0.0015:  # a port's floor or roof
                cz += -CLEARANCE if cz < 0.005 else CLEARANCE
        if b["name"] in ("plane001_collider_box", "plane001_collider_box.001"):  # the bracket's slot: opened by CLEARANCE each way
            cz += -CLEARANCE if cz < sc.BAY_X else CLEARANCE
        c, s = _upright((cx, cy, cz), (sx, sy, sz), _M_CARD)
        lo, hi = c[2] - s[2] / 2, c[2] + s[2] / 2
        if hi <= CARD_Z_CUT:
            continue
        if lo < CARD_Z_CUT:  # clip the box to the modelled part of the card
            c, s = (c[0], c[1], (CARD_Z_CUT + hi) / 2), (s[0], s[1], hi - CARD_Z_CUT)
        out.append((c, s, color))
    out.append(((-0.0046, 0.0, -0.040 + 0.0225), (0.006, 0.12, 0.045), STEEL))  # back plate behind the PCB
    out.append(((0.0, 0.0, CARD_Z_CUT - 0.004), (0.060, 0.110, 0.008), STEEL))  # foot under the cut edge
    return out


def _holder_boxes():
    """(centre, size, colour) in the holder's frame (the module's frame when it rests): a plate under the module's body,
    two side rails and an end stop under its tip."""
    tip = sc.TIP_Z
    return [
        ((-0.0066, 0.0, -0.0075), (0.004, 0.025, 0.045), STEEL),  # plate: inner face at x = -4.6 mm, under the body
        ((0.0015, 0.0100, -0.0075), (0.012, 0.003, 0.040), STEEL),  # rails: 17 mm apart, module is 13.75 mm wide
        ((0.0015, -0.0100, -0.0075), (0.012, 0.003, 0.040), STEEL),
        ((0.0, 0.0, tip - 0.0005 - 0.0015), (0.016, 0.017, 0.003), STEEL),  # stop, 0.5 mm under the tip
    ]


def _module_boxes():
    keep = ("body_collider", "head_collider", "latch_collider")
    return [(*_upright(b["pos"], b["size"], _M_MOD), MODULE_COLOR) for b in sc.MOD["boxes"] if b["name"].startswith(keep)]


def _lowest(boxes, r):
    """Lowest corner height of oriented boxes whose frame is `r` (relative to the frame's origin)."""
    return min(float((r @ (np.array(c) + np.array([dx, dy, dz]) * np.array(s) / 2))[2])
               for c, s, _ in boxes for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1))


_CARD = _card_boxes()
_HOLD = _holder_boxes()
_mouth = R_CARD @ np.array([sc.BAY_X, sum(sc.BAY_Y) / 2, sc.MOUTH_Z])
CARD_POS = np.array([MOUTH_XY[0] - _mouth[0], MOUTH_XY[1] - _mouth[1], TOP + BASE_T - _lowest(_CARD, R_CARD)])
HOLD_POS = np.array([*HOLDER_XY, TOP + BASE_T - _lowest(_HOLD, R_HOLD)])
_Q_CARD, _Q_HOLD = sc.quat_xyzw(R_CARD), sc.quat_xyzw(R_HOLD)


def _shapes(prefix, boxes, origin, r, quat):
    return [Shape(f"{prefix}_{i:02d}", pos=tuple((origin + r @ np.array(c)).tolist()), size=tuple(s), color=col, rot_xyzw=quat)
            for i, (c, s, col) in enumerate(boxes)]


def _wxyz(r):
    x, y, z, w = sc.quat_xyzw(r)
    return (w, x, y, z)


# the module's body is posed in the holder's frame, so it starts leaning; the scorer undoes that with R_HOLD
MODULE = part("module", pos=tuple(HOLD_POS.tolist()), mass=0.02,
              boxes=tuple((tuple((R_HOLD @ np.array(c)).tolist()), s, col, _wxyz(R_HOLD)) for c, s, col in _module_boxes()))
OBJECTS = (
    _shapes("card", _CARD, CARD_POS, R_CARD, _Q_CARD)
    + _shapes("holder", _HOLD, HOLD_POS, R_HOLD, _Q_HOLD)
    + [Box("card_base", pos=(MOUTH_XY[0] - 0.02, MOUTH_XY[1], TOP + BASE_T / 2), size=(0.14, 0.15, BASE_T), color=STEEL),
       Box("holder_base", pos=(HOLDER_XY[0], HOLDER_XY[1], TOP + BASE_T / 2), size=(0.07, 0.06, BASE_T), color=STEEL),
       MODULE]
)


def score(objects):
    m = objects["module"]
    if os.environ.get("SFP_DEBUG"):  # development only: one line per step with the module's pose in the card's frame
        with open(os.environ["SFP_DEBUG"], "a") as f:
            r = sc.rotation(m.quat_xyzw) @ R_HOLD
            tip = R_CARD.T @ (m.pos + r @ np.array([0.0, 0.0, sc.TIP_Z]) - CARD_POS)
            rel = R_CARD.T @ r  # module axes in the card's frame
            pitch = math.degrees(math.atan2(rel[0, 2], rel[2, 2]))  # lean about the card's y axis (toward/away from the PCB)
            roll = math.degrees(math.atan2(-rel[1, 2], rel[2, 2]))  # lean about its x axis (sideways, along the bays)
            yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
            f.write(f"tip(x,y)mm=({(tip[0]-sc.BAY_X)*1000:.1f},{(tip[1]-sc.BAY_Y[0])*1000:.1f}) depth={(sc.MOUTH_Z-tip[2])*1000:.1f} "
                    f"pitch={pitch:.1f} roll={roll:.1f} yaw={yaw:.1f} z={m.pos[2]:.3f} speed={float(np.linalg.norm(m.lin_vel)):.3f}\n")
    return sc.score(m.pos, m.quat_xyzw, CARD_POS, R_CARD, part_to_module=R_HOLD, module_z=float(m.pos[2]), table_z=TOP)


def success(objects):
    return score(objects) >= 1.0 and objects["module"].resting()


def distance(objects):
    m = objects["module"]
    return max(0.0, sc.FULL_DEPTH - sc.measure(m.pos, m.quat_xyzw, CARD_POS, R_CARD, R_HOLD)["depth"])


# -- the scripted demo --------------------------------------------------------------------------------------------
HEAD_Z = 0.0276  # the latch end's middle, along the module's axis from its origin
GRIP_UP = 0.010  # the TCP rides this far above the grip point so the pads' tips (13 mm below it) close on the head
DEMO_DEPTH = 0.044
LIFT_MARGIN = 0.01  # carry height above the higher of the two draw-out points
Z_UP = np.array([0.0, 0.0, 1.0])
# WHY: measured in simulation (SFP_DEBUG logs): the held module's tip reached the port mouth 3.4 mm off in the card's x and 3.0 mm
# off in y (the grasp and the arm's sag), where the port allows about 1 mm. The demo shifts its target by the opposite.
DEMO_CORRECTION = np.array([0.0034, -0.0030, 0.0])  # in the card's frame (m)


def _tcp(origin, r):
    return origin + r @ np.array([0.0, 0.0, HEAD_Z]) + Z_UP * GRIP_UP


def demo():
    axis_start, axis_card = R_HOLD @ Z_UP, R_CARD @ Z_UP
    pick = _tcp(HOLD_POS, R_HOLD)
    tip = CARD_POS + R_CARD @ np.array([sc.BAY_X, sc.BAY_Y[sc.TARGET_BAY], sc.MOUTH_Z - DEMO_DEPTH])
    place = _tcp(tip - R_CARD @ np.array([0.0, 0.0, sc.TIP_Z]), R_CARD) + R_CARD @ DEMO_CORRECTION
    out_start, out_card = pick + axis_start * 0.065, place + axis_card * 0.080
    top_z = max(out_start[2], out_card[2]) + LIFT_MARGIN
    lift = np.array([out_start[0], out_start[1], top_z])
    pre = np.array([out_card[0], out_card[1], top_z])
    yaw = sc.CARD_YAW
    wp = Waypoint
    return {"left": [
        wp(1.5, tuple(pick + axis_start * 0.08), FINGER_OPEN, "to the module", 0.0),
        wp(1.5, tuple(pick), FINGER_OPEN, "descend", 0.0),
        wp(0.8, tuple(pick), FINGER_CLOSED, "close", 0.0),
        wp(1.5, tuple(out_start), FINGER_CLOSED, "draw out along its axis", 0.0),
        wp(1.0, tuple(lift), FINGER_CLOSED, "lift", 0.0),
        wp(2.0, tuple(pre), FINGER_CLOSED, "carry, turning to the card", yaw),
        wp(1.0, tuple(out_card), FINGER_CLOSED, "down to the port axis", yaw),
        wp(2.0, tuple(place), FINGER_CLOSED, "push in along the port axis", yaw),
        wp(0.6, tuple(place), FINGER_OPEN, "release", yaw),
        wp(1.0, tuple(out_card), FINGER_OPEN, "retreat along the axis", yaw),
    ]}
