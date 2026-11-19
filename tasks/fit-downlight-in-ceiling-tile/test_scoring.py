"""Offline checks of score() and success() on synthetic layouts (no GPU, no Isaac Sim):

    python test_scoring.py        (in an environment with yam_arena installed, as for the accepted oximeter task)

Offline there is no step clock and no arm poses, so release and stillness are assumed unless TEST_GRIPPERS is set.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yam_arena.task_file import load  # noqa: E402

TASK = load(Path(__file__).with_name("fit_downlight_in_ceiling_tile.py"))
T = TASK.module


def light(rim_above=0.0, left=102.0, right=102.0, shift=(0.0, 0.0), tilt_deg=0.0, at=None):
    """The light with its rim's back face `rim_above` over the tile's face at the hole, clip angles in degrees
    (0 pressed flat along the body, 140 sprung fully out)."""
    origin = np.asarray(T.HOLE if at is None else at, float) + [shift[0], shift[1], rim_above]
    a = math.radians(tilt_deg)
    R = np.array([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    q = {"clip_left": math.radians(left), "clip_right": -math.radians(right)}
    return T.LIGHT.offline_state(origin, 0.0, q, R=R)


def layout(lt):
    return {"tile": None, "light": lt}


START = layout(T.LIGHT.offline_state(np.asarray(T.LIGHT_START), 0.0, {}))
# Pressed flat, the clips' pivots (20 mm behind the rim) pass the tile's back face (15 mm) when the rim is < 5 mm up.
CASES = [
    ("start: standing on the tile beside the hole", START, 0.0),
    ("seated flush, both clips sprung out against the tile's back", layout(light()), 1.0),
    ("seated, rim 1 mm proud, clips out", layout(light(0.001)), 1.0),
    ("dropped onto the hole without pressing the clips back: resting on them",
     layout(light(0.0182, 140.0, 140.0)), 0.0),
    ("pushed part way, clips still pressed flat inside the hole (rim 12 mm up)", layout(light(0.012, 3.0, 3.0)), 0.25),
    ("rim 3 mm proud, one clip out behind the tile, the other still flat in the hole",
     layout(light(0.003, 100.0, 4.0)), 0.5),
    ("rim 3 mm proud, both clips out (not flush)", layout(light(0.003, 100.0, 100.0)), 0.5),
    ("seated but tilted 5 degrees, clips out", layout(light(0.0, 102.0, 102.0, tilt_deg=5.0)), 0.5),
    ("off the hole by 6 mm", layout(light(0.0, shift=(0.006, 0.0))), 0.0),
    ("clips out but lying on the tile's face beside the hole", layout(light(0.0, at=T.HOLE + [0.08, 0.0, 0.0])), 0.0),
    ("knocked off the table", layout(light(at=(0.3, -0.2, T.TOP - 0.05))), 0.0),
]


def test_cases():
    for name, objs, want in CASES:
        got = T.score(objs)
        assert abs(got - want) < 1e-9, f"{name}: score {got}, expected {want}"
        assert T.success(objs) == (want == 1.0)


def test_held_light_scores_nothing():
    objs = layout(light())
    T.TEST_GRIPPERS = {"right": (np.asarray(T.HOLE) + [0, 0, 0.01], 0.0), "left": (np.array([0.3, 0.3, 1.1]), 1.0)}
    try:
        assert T.score(objs) == 0.0
    finally:
        T.TEST_GRIPPERS = None


def test_demo_defined():
    assert callable(T.demo_script)


if __name__ == "__main__":
    test_cases()
    test_held_light_scores_nothing()
    test_demo_defined()
    for name, objs, want in CASES:
        print(f"{T.score(objs):.2f}  {name}")
    print("all passed")
