"""On the station table: a PCIe network card stands upright in a cradle in front of the left arm, its PCB facing the arms
and turned 8 degrees (counterclockwise from above) from square to them. Its two SFP cages (14 x 9 mm openings, 49 mm
deep, 23 mm apart) open upward at its top edge. An SFP module (14 x 12 mm, 55 mm long) stands tip-down in a small
holder to the left of the card, its latch end up. The robot must lift the module out and seat it, tip first, in the
card's left cage (the one at the larger y).
"""

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sfp_seat_scoring as sc
from partkit import part
from yam_arena.demo import pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
GEOM = sc.GEOM

CARD_YAW = math.radians(8.0)
CARD_Z_CUT = -0.032  # WHY: only the card's upper part is modelled (the rest sits inside the cradle), so the arm can reach over it
CARD_POS = (0.30, 0.08, TOP + 0.040)  # card origin: the cage mouth is 117 mm above the table, the card's top edge 127 mm
CLEARANCE = 0.0010  # WHY: real bays leave ~0.1 mm a side, which jams in simulation; widened to 1.0 mm
SKIN, METAL, MODULE_COLOR, STEEL = (0.10, 0.45, 0.20), (0.72, 0.74, 0.77), (0.85, 0.70, 0.20), (0.35, 0.36, 0.38)
CAGE = "10099100-011lfc001"

# model frames -> the upright, world-aligned frames (card: PCB normal toward the arms, cages up; module: tip down)
_M_CARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # card model (x, y, z) -> upright (x, y, z)
_M_MOD = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # module model -> upright (tip down)


def _upright(centre, size, m):
    return tuple((m @ np.array(centre)).tolist()), tuple(float(v) for v in (size[2], size[0], size[1]))


def _card_boxes():
    """(centre, size, colour) in the card's upright frame, bays widened by CLEARANCE."""
    centres = [(b["x_min"] + b["x_max"]) / 2 for b in sc.NIC["bays"]]
    out = []
    for b in sc.NIC["boxes"]:
        cx, cy, cz = b["pos"]
        sx, sy, sz = b["size"]
        color = SKIN
        if b["name"].startswith(CAGE):
            color = METAL
            near = min(centres, key=lambda c: abs(c - cx))
            if sx < 0.002 and sz > 0.01:  # a bay's side wall
                cx += CLEARANCE if cx > near else -CLEARANCE
            elif sy > 0.04 and sx > 0.01 and sz < 0.0015:  # a bay's floor or roof
                cz += -CLEARANCE if cz < 0.005 else CLEARANCE
        c, s = _upright((cx, cy, cz), (sx, sy, sz), _M_CARD)
        lo, hi = c[2] - s[2] / 2, c[2] + s[2] / 2
        if hi <= CARD_Z_CUT:
            continue
        if lo < CARD_Z_CUT:  # clip the box to the modelled part of the card
            c, s = (c[0], c[1], (CARD_Z_CUT + hi) / 2), (s[0], s[1], hi - CARD_Z_CUT)
        out.append((c, s, color))
    # the cradle: a foot under the card's cut edge and a back plate behind its PCB
    out.append(((-0.0046, 0.0, -0.040 + 0.0225), (0.006, 0.12, 0.045), STEEL))
    out.append(((0.0, 0.0, CARD_Z_CUT - 0.004), (0.060, 0.110, 0.008), STEEL))
    return out


def _module_boxes():
    keep = ("body_collider", "head_collider", "latch_collider")
    out = []
    for b in sc.MOD["boxes"]:
        if b["name"].startswith(keep):
            c, s = _upright(b["pos"], b["size"], _M_MOD)
            out.append((c, s, MODULE_COLOR))
    return tuple(out)


def _world_boxes():
    rz = sc.rz(CARD_YAW)
    objs = []
    for i, (c, s, color) in enumerate(_card_boxes()):
        pos = np.array(CARD_POS) + rz @ np.array(c)
        objs.append(Box(f"card_{i:02d}", pos=tuple(pos.tolist()), size=s, color=color, yaw=CARD_YAW))
    return objs


# the module's holder: a floor and four walls around its tip, left of the card
HOLDER_XY = (0.24, 0.30)
FLOOR_T, WALL_T, WALL_H = 0.004, 0.004, 0.020
IN_X, IN_Y = 0.0122 + 0.003, 0.014 + 0.003  # inner cavity: module's 12.2 x 14 mm plus 1.5 mm a side


def _holder():
    hx, hy = HOLDER_XY
    z0 = TOP
    ox, oy = IN_X / 2 + WALL_T / 2, IN_Y / 2 + WALL_T / 2
    return [
        Box("holder_floor", pos=(hx, hy, z0 + FLOOR_T / 2), size=(IN_X + 2 * WALL_T, IN_Y + 2 * WALL_T, FLOOR_T), color=STEEL),
        Box("holder_n", pos=(hx + ox, hy, z0 + FLOOR_T + WALL_H / 2), size=(WALL_T, IN_Y + 2 * WALL_T, WALL_H), color=STEEL),
        Box("holder_s", pos=(hx - ox, hy, z0 + FLOOR_T + WALL_H / 2), size=(WALL_T, IN_Y + 2 * WALL_T, WALL_H), color=STEEL),
        Box("holder_l", pos=(hx, hy + oy, z0 + FLOOR_T + WALL_H / 2), size=(IN_X, WALL_T, WALL_H), color=STEEL),
        Box("holder_r", pos=(hx, hy - oy, z0 + FLOOR_T + WALL_H / 2), size=(IN_X, WALL_T, WALL_H), color=STEEL),
    ]


MODULE_START_Z = TOP + FLOOR_T - sc.TIP_Z + 0.0005  # origin height with the tip on the holder's floor
MODULE = part("module", pos=(*HOLDER_XY, MODULE_START_Z), mass=0.02, boxes=_module_boxes())
OBJECTS = _holder() + _world_boxes() + [MODULE]


def score(objects):
    m = objects["module"]
    return sc.score(m.pos, m.quat_xyzw, CARD_POS, CARD_YAW, module_z=float(m.pos[2]), table_z=TOP)


def success(objects):
    return score(objects) >= 1.0 and objects["module"].resting()


def distance(objects):
    m = objects["module"]
    return max(0.0, sc.FULL_DEPTH - sc.measure(m.pos, m.quat_xyzw, CARD_POS, CARD_YAW)["depth"])


# -- the scripted demo --------------------------------------------------------------------------------------------
HOLD_ABOVE_TIP = 0.0375  # TCP height above the module's origin when the pads grip its latch end
DEMO_DEPTH = 0.044


def demo():
    pick = (*HOLDER_XY, MODULE_START_Z + HOLD_ABOVE_TIP)
    bay = sc.rz(CARD_YAW) @ np.array([sc.BAY_X, sc.BAY_Y[sc.TARGET_BAY], 0.0])
    tip_z = CARD_POS[2] + sc.MOUTH_Z - DEMO_DEPTH
    place = (CARD_POS[0] + bay[0], CARD_POS[1] + bay[1], tip_z - sc.TIP_Z + HOLD_ABOVE_TIP)
    return {"left": pick_place_waypoints(pick, place, travel_z=TOP + 0.20, place_yaw=CARD_YAW)}
