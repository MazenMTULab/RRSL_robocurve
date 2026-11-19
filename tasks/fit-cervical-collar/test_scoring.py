"""Offline checks of score() and success() on synthetic states: python test_scoring.py (or pytest).

States come from the parts helper's own kinematics (Part.state). yam_arena is stubbed when it is not installed.
"""

import importlib.util
import math
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _load():
    try:
        import yam_arena.task_file  # noqa: F401
    except ImportError:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Shape:
            name: str
            pos: tuple
            color: tuple = (0.7, 0.7, 0.7)
            mass: float = None
            yaw: float = 0.0

            def _check(self):
                pass

        @dataclass(frozen=True)
        class Box(_Shape):
            size: tuple = (0.04, 0.04, 0.04)

        @dataclass(frozen=True)
        class ObjectState:
            pos: np.ndarray
            quat_xyzw: np.ndarray
            lin_vel: np.ndarray
            ang_vel: np.ndarray
            low: np.ndarray
            high: np.ndarray

        c = types.ModuleType("yam_arena.constants")
        c.FINGER_CLOSED, c.FINGER_OPEN, c.PAD_TIP_PAST_TCP = 0.0, -0.04695, 0.013
        d = types.ModuleType("yam_arena.demo")
        d.Waypoint = lambda *a: a
        t = types.ModuleType("yam_arena.task_file")
        t.TABLE_TOP, t.Box, t.ObjectState, t.GREY = {"station": 0.7475}, Box, ObjectState, (0.7, 0.7, 0.7)
        root = types.ModuleType("yam_arena")
        root.task_file = t
        sys.modules.update({"yam_arena": root, "yam_arena.constants": c, "yam_arena.demo": d,
                            "yam_arena.task_file": t})
    spec = importlib.util.spec_from_file_location("collar_task", HERE / "fit_cervical_collar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load()

BACK_FITTED = (T.NECK_X, 0.0, T.TOP + T.PLATE_T / 2)
FRONT_FITTED = (T.NECK_X + 0.006, 0.004, T.FRONT_SEATED_Z)   # where the scripted demo leaves it


def _tilted(deg):
    """quat_xyzw for a tilt of ``deg`` about the x axis."""
    h = math.radians(deg) / 2
    return np.array([math.sin(h), 0.0, 0.0, math.cos(h)])


def scene(roll_deg=0.0, back=None, front=None, back_tilt=0.0):
    return {
        T.MANIKIN.name: T.MANIKIN.state(joints={"head": math.radians(roll_deg)}),
        T.BACK.name: T.BACK.state(pos=back or T.BACK_START, quat_xyzw=_tilted(back_tilt)),
        T.FRONT.name: T.FRONT.state(pos=front or T.FRONT_START),
    }


def run(frames):
    """Score a trial from its start; returns (last score, last success)."""
    T._fresh()
    s = None
    for f in [scene()] + frames:
        s = T.score(f)
    return s, T.success(frames[-1] if frames else scene())


def test_start_is_zero():
    assert run([scene()] * 5) == (0.0, False)


def test_back_then_front():
    s, ok = run([scene(2.5), scene(3.0, back=BACK_FITTED)])
    assert (s, ok) == (0.5, False)
    s, ok = run([scene(2.5), scene(3.0, back=BACK_FITTED), scene(3.0, back=BACK_FITTED, front=FRONT_FITTED)])
    assert (s, ok) == (1.0, True)


def test_front_without_back_earns_nothing():
    assert run([scene(2.0, front=FRONT_FITTED)])[0] == 0.0


def test_back_off_centre_or_lifted_or_tilted():
    for back in [(T.NECK_X, 0.02, BACK_FITTED[2]), (T.NECK_X + 0.02, 0.0, BACK_FITTED[2]),
                 (T.NECK_X, 0.0, T.TOP + T.NECK_TOP + 0.01)]:   # laid on top of the neck instead of under it
        assert run([scene(back=back)])[0] == 0.0, back
    assert run([scene(back=BACK_FITTED, back_tilt=15)])[0] == 0.0
    assert run([scene(back=(T.NECK_X + 0.014, -0.014, BACK_FITTED[2]))])[0] == 0.5   # just inside 1.5 cm


def test_front_not_resting_on_the_neck():
    high = (FRONT_FITTED[0], FRONT_FITTED[1], T.FRONT_SEATED_Z + 0.012)   # a flange caught on the neck
    assert run([scene(back=BACK_FITTED, front=high)])[0] == 0.5
    off = (FRONT_FITTED[0], 0.02, FRONT_FITTED[2])
    assert run([scene(back=BACK_FITTED, front=off)])[0] == 0.5


def test_head_rolled_before_anything_scores_zero():
    frames = [scene(5), scene(11), scene(30, back=BACK_FITTED), scene(30, back=BACK_FITTED, front=FRONT_FITTED)]
    assert run(frames) == (0.0, False)


def test_head_rolled_either_way_freezes_the_score():
    frames = [scene(3, back=BACK_FITTED), scene(-11, back=BACK_FITTED),
              scene(0, back=BACK_FITTED, front=FRONT_FITTED)]
    assert run(frames) == (0.5, False)


def test_up_to_ten_degrees_is_allowed():
    frames = [scene(9.9, back=BACK_FITTED), scene(-9.9, back=BACK_FITTED, front=FRONT_FITTED)]
    assert run(frames) == (1.0, True)


def test_head_rolling_after_the_collar_is_on_keeps_full_score():
    frames = [scene(2, back=BACK_FITTED, front=FRONT_FITTED), scene(20, back=BACK_FITTED, front=FRONT_FITTED)]
    assert run(frames) == (1.0, True)


def test_reset_between_trials():
    T._fresh()
    T.score(scene())
    T.score(scene(15))                       # trial 1: head rolled, frozen at 0
    T.score(scene(15, back=BACK_FITTED))
    assert T.score(scene(15, back=BACK_FITTED, front=FRONT_FITTED)) == 0.0
    T.score(scene(0))                        # reset: back half jumps back to its start
    T.score(scene(2, back=BACK_FITTED))
    assert T.score(scene(2, back=BACK_FITTED, front=FRONT_FITTED)) == 1.0


def test_back_carried_back_is_not_a_reset():
    T._fresh()
    T.score(scene())
    T.score(scene(15))                       # frozen at 0
    ys = list(np.arange(0.27, 0.0, -0.01)) + list(np.arange(0.0, 0.27, 0.01))
    for y in ys:                             # under the neck and back out again, 1 cm a step: never a jump
        T.score(scene(15, back=(T.NECK_X, float(y), T.BACK_START[2])))
    T.score(scene(15, back=T.BACK_START))
    assert T.score(scene(15, back=BACK_FITTED)) == 0.0


def test_demo_builds():
    d = T.demo()
    assert set(d) == {"left", "right"}
    left_s = sum(w.duration for w in d["left"])
    assert 25 < left_s < 40, left_s


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} passed")
