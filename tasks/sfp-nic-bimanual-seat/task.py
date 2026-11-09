"""On the station table: a PCIe network card lies flat in front of the left arm, its SFP cages (two bays, 14 x 9 mm, 49 mm
deep) opening toward the arms, and an SFP module lies flat in front of the right arm, its tip toward the far side. The
robot must lift the card, hold it in the air, and push the module tip-first into the card's left bay (the bay nearer the
robot's left side when the cages face the arms is the one at the smaller card x; see README).
"""

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sfp_nic_scoring as sc
from partkit import part
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
GEOM = json.loads((pathlib.Path(__file__).parent / "geometry.json").read_text())

CLEARANCE = 0.0006  # WHY: the real bays leave ~0.1 mm per side, which jams in simulation; widened to 0.6 mm
SKIN = (0.10, 0.45, 0.20)  # the card's green
METAL = (0.72, 0.74, 0.77)
MODULE_COLOR = (0.85, 0.70, 0.20)


def _card_boxes():
    out = []
    bays = GEOM["nic"]["bays"]
    centres = [(b["x_min"] + b["x_max"]) / 2 for b in bays]
    for b in GEOM["nic"]["boxes"]:
        cx, cy, cz = b["pos"]
        sx, sy, sz = b["size"]
        if b["name"].startswith("10099100-011lfc001"):
            near = min(centres, key=lambda c: abs(c - cx))
            if sx < 0.002 and sz > 0.01:  # a bay's side wall: move it away from the bay's centre line
                cx += CLEARANCE if cx > near else -CLEARANCE
            elif sy > 0.04 and sx > 0.01 and sz < 0.0015:  # floor or roof of a bay
                cz += -CLEARANCE if cz < 0.005 else CLEARANCE
            out.append(((cx, cy, cz), (sx, sy, sz), METAL))
        else:
            out.append(((cx, cy, cz), (sx, sy, sz), SKIN))
    return tuple(out)


def _module_boxes():
    keep = ("body_collider", "head_collider", "latch_collider")
    return tuple((tuple(b["pos"]), tuple(b["size"]), MODULE_COLOR) for b in GEOM["module"]["boxes"] if b["name"].startswith(keep))


_CARD = _card_boxes()
_MOD = _module_boxes()
_card_lo = min(c[2] - s[2] / 2 for c, s, _ in _CARD)
_mod_lo = min(c[2] - s[2] / 2 for c, s, _ in _MOD)

# card flat on the table, insertion axis (card +y) pointing away from the arms (world +x); module flat, tip (its -y) away
CARD = part("card", pos=(0.28, 0.16, TOP - _card_lo + 0.001), yaw=-math.pi / 2, mass=0.15, boxes=_CARD)
MODULE = part("module", pos=(0.28, -0.16, TOP - _mod_lo + 0.001), yaw=math.pi / 2, mass=0.02, boxes=_MOD)
OBJECTS = [CARD, MODULE]


def _lowest_z(card) -> float:
    pts = [card.to_world((cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2))
           for (cx, cy, cz), (sx, sy, sz), _ in _CARD for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    return float(min(p[2] for p in pts))


def score(objects):
    card, mod = objects["card"], objects["module"]
    return sc.score(mod.pos, mod.quat_xyzw, card.pos, card.quat_xyzw, nic_lowest_z=_lowest_z(card), table_z=TOP,
                    module_z=float(mod.pos[2]))


def success(objects):
    return score(objects) >= 1.0
