"""Offline checks of score() and success() on synthetic layouts (no GPU, no Isaac Sim):

    python test_scoring.py        (in an environment with yam_arena installed)

Offline there is no step clock and no arm poses, so release and stillness are assumed unless TEST_GRIPPERS is set.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("yam_parts", None)
import yam_parts  # noqa: E402
from yam_arena import ik  # noqa: E402
from yam_arena.task_file import load  # noqa: E402

TASK = load(Path(__file__).with_name("set_scions_in_a_cleft_graft.py"))
T = TASK.module


def state(part, origin, R=np.eye(3)):
    origin = np.asarray(origin, float)
    low, high = part.aabb(origin, R)
    return yam_parts.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(R), float), np.zeros(3),
                               np.zeros(3), low, high, origin)


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


SCION_PARTS = {p.name: p for p in T.OBJECTS if p.name in T.SCIONS}


def scion(name, inset=0.0015, top=-0.0015, x=None, tilt=0.0, at=None):
    """A scion in the cleft at its own end: bark `inset` inside the stub's bark, top of cut `top` from the cut face."""
    side = 1.0 if name == "scion_left" else -1.0
    if at is not None:
        return state(SCION_PARTS[name], at)
    y = side * (T.STOCK_R - inset - T.SCION_R)
    origin = T.STOCK + [T.GAP_MID_X if x is None else x, y, top]
    return state(SCION_PARTS[name], origin, rot_x(tilt))


def tool(lift=0.0, at=None):
    return state(T.TOOL, np.asarray(T.TOOL_START if at is None else at, float) + [0, 0, lift])


START = {"stock": None, "stand": None, "tool": tool(),
         "scion_left": state(SCION_PARTS["scion_left"], SCION_PARTS["scion_left"].pos),
         "scion_right": state(SCION_PARTS["scion_right"], SCION_PARTS["scion_right"].pos)}


def layout(left, right, t):
    return {"stock": None, "stand": None, "tool": t, "scion_left": left, "scion_right": right}


L, R = "scion_left", "scion_right"
OUT = tool(at=(0.13, -0.17, T.TOP + 0.03))
CASES = [
    ("start: scions in their stand, tool in the cleft", START, 0.0),
    ("both scions set right, tool out", layout(scion(L), scion(R), OUT), 1.0),
    ("both set right, tool lifted 5 mm clear", layout(scion(L), scion(R), tool(lift=T.PRONG_L + 0.005)), 1.0),
    ("both set right, tool still in the cleft", layout(scion(L), scion(R), tool()), 0.5),
    ("only the left scion set, tool out", layout(scion(L), START[R], OUT), 0.5),
    ("only the left scion set, tool in", layout(scion(L), START[R], tool()), 0.25),
    ("inset 0.6 mm (just inside the band)", layout(scion(L, inset=0.0006), scion(R, inset=0.0024), OUT), 1.0),
    ("bark flush with the stub's bark (inset 0): cambium 1.5 mm outside",
     layout(scion(L, inset=0.0), scion(R, inset=0.0), OUT), 0.0),
    ("pushed 3 mm in (cambium 1.5 mm inside)", layout(scion(L, inset=0.003), scion(R, inset=0.003), OUT), 0.0),
    ("set towards the middle of the cleft", layout(scion(L, inset=0.0045), scion(R, inset=0.0045), OUT), 0.0),
    ("left right, right one flush", layout(scion(L), scion(R, inset=0.0), OUT), 0.5),
    ("top of cut 4 mm down", layout(scion(L, top=-0.004), scion(R, top=-0.004), OUT), 1.0),
    ("sunk 7 mm down", layout(scion(L, top=-0.007), scion(R, top=-0.007), OUT), 0.0),
    ("standing 6 mm proud", layout(scion(L, top=0.006), scion(R, top=0.006), OUT), 0.0),
    ("leaning 20 degrees", layout(scion(L, tilt=20.0), scion(R, tilt=-20.0), OUT), 0.0),
    ("leaning 6 degrees", layout(scion(L, tilt=6.0), scion(R, tilt=-6.0), OUT), 1.0),
    ("not in the cleft: beside the stub on the vise side", layout(scion(L, x=-0.012), scion(R, x=-0.012), OUT), 0.0),
    ("scions knocked onto the bench", layout(scion(L, at=(0.3, 0.2, T.TOP + 0.01)),
                                             scion(R, at=(0.3, -0.2, T.TOP + 0.01)), OUT), 0.0),
]


def test_cases():
    for name, objs, want in CASES:
        got = T.score(objs)
        assert abs(got - want) < 1e-9, f"{name}: score {got}, expected {want}"
        assert T.success(objs) == (want == 1.0)


def test_held_scion_scores_nothing():
    objs = layout(scion(L), scion(R), OUT)
    held = np.asarray(objs[L].origin) + [0, 0, 0.04]
    T.TEST_GRIPPERS = {"left": (held, 0.0), "right": (np.array([0.1, -0.3, 1.1]), 1.0)}
    try:
        assert T.score(objs) == 0.0
    finally:
        T.TEST_GRIPPERS = None


def test_start_geometry():
    # The prong's sloped face touches the far half's top edge; the gap at the cut face fits a 9 mm scion.
    assert 0.0095 < 2 * T._EX < 0.0105
    assert T.TOOL_H0 - T.PRONG_L < -0.015          # prong 15+ mm down in the cleft
    assert T.tool_out(START["tool"]) is False


def test_demo_defined():
    assert callable(T.demo_script)


if __name__ == "__main__":
    test_cases()
    test_held_scion_scores_nothing()
    test_start_geometry()
    test_demo_defined()
    for name, objs, want in CASES:
        print(f"{T.score(objs):.2f}  {name}")
    print("all passed")
