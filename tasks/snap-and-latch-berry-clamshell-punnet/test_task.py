"""Unit tests for snap-and-latch-berry-clamshell-punnet (geometry, laws, scorer, hazards, and 20 Hz ArmPlan IK)."""

from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent


def _install_stubs():
    """Install lightweight pure-Python stubs when yam_arena is absent (local Windows unit run)."""
    if "yam_arena" in sys.modules:
        return
    ya = types.ModuleType("yam_arena")
    yc = types.ModuleType("yam_arena.constants")
    yd = types.ModuleType("yam_arena.demo")
    yt = types.ModuleType("yam_arena.task_file")

    yc.FINGER_OPEN = -0.04695
    yc.FINGER_CLOSED = 0.0
    yc.PAD_TIP_PAST_TCP = 0.013
    yc.TCP_OFFSET = (0.0, 0.0, 0.134)

    class Waypoint:
        def __init__(self, duration, pos, finger, label="", yaw=0.0):
            self.duration = duration
            self.pos = tuple(pos)
            self.finger = finger
            self.label = label
            self.yaw = yaw

    class _StubArmPlan:
        pass

    yd.Waypoint = Waypoint
    yd.ArmPlan = _StubArmPlan
    yd._IS_STUB = True
    yd.grip_rotation = staticmethod(lambda yaw=0.0: np.eye(3))
    yd.ik = types.SimpleNamespace(smoothstep=lambda u: u, slerp_matrix=lambda a, b, u: a)

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Box:
        name: str = "box"
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        color: tuple[float, float, float] = (0.5, 0.5, 0.5)
        mass: float | None = None
        yaw: float = 0.0
        size: tuple[float, float, float] = (0.04, 0.04, 0.04)

        def _check(self):
            pass

        @property
        def free(self):
            return self.mass is not None

    @dataclass(frozen=True)
    class ObjectState:
        pos: np.ndarray
        quat_xyzw: np.ndarray
        lin_vel: np.ndarray
        ang_vel: np.ndarray
        low: np.ndarray
        high: np.ndarray

        def resting(self, max_speed=0.01, max_spin=0.5):
            return bool(np.linalg.norm(self.lin_vel) < max_speed and np.linalg.norm(self.ang_vel) < max_spin)

    yt.TABLE_TOP = {"station": 0.7475, "tabletop": 0.75}
    yt.Box = Box
    yt.ObjectState = ObjectState

    sys.modules["yam_arena"] = ya
    sys.modules["yam_arena.constants"] = yc
    sys.modules["yam_arena.demo"] = yd
    sys.modules["yam_arena.task_file"] = yt


try:
    import yam_arena.demo  # noqa: F401
except ImportError:
    _install_stubs()
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from yam_arena.task_file import ObjectState  # noqa: E402


def _load_task(side: str = "L", env_overrides: dict | None = None):
    keys = ("HAND_SIDE", "DEMO_NO_LATCH", "DEMO_LATCH_BEFORE_CLOSE", "DEMO_CRUSH_BERRIES")
    old = {k: os.environ.get(k) for k in keys}
    os.environ["HAND_SIDE"] = side
    for k in keys[1:]:
        os.environ.pop(k, None)
    if env_overrides:
        os.environ.update(env_overrides)
    try:
        sys.modules.pop("task", None)
        import task  # noqa: E402

        return importlib.reload(task)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _state(pos, tilt_deg=0.0, yaw_deg=0.0, vel=(0, 0, 0), ang_vel=(0, 0, 0), low_z=None, size=(0.12, 0.16, 0.065)):
    pos = np.asarray(pos, float)
    rx = math.radians(tilt_deg) / 2.0
    rz = math.radians(yaw_deg) / 2.0
    qx = np.array([math.sin(rx), 0.0, 0.0, math.cos(rx)], float)
    qz = np.array([0.0, 0.0, math.sin(rz), math.cos(rz)], float)
    # qz * qx
    ax, ay, az, aw = qz
    bx, by, bz, bw = qx
    q = np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], float)
    half = np.asarray(size, float) / 2.0
    lz = pos[2] if low_z is None else float(low_z)
    low = np.array([pos[0] - half[0], pos[1] - half[1], lz], float)
    high = np.array([pos[0] + half[0], pos[1] + half[1], lz + size[2]], float)
    return ObjectState(pos, q, np.asarray(vel, float), np.asarray(ang_vel, float), low, high)


def _fresh_trial(t):
    import yamkit as kit

    kit._EPISODE["count"] += 1
    kit._EPISODE["last"] = 0
    kit._LAW_MEM.clear()
    kit._OPENING.clear()
    kit._JOINTS.clear()
    t.HAZARD.run, t.HAZARD.on, t.HAZARD._key, t.HAZARD._objs = 0, False, None, None
    t._T = t._Tracker()


def _step_state(
    t,
    lid_deg: float,
    catch_y: float,
    *,
    lid_vel: float = 0.0,
    catch_vel: float = 0.0,
    punnet_pos=None,
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    pos_l=(0.0, 0.30, 0.96),
    pos_r=(0.0, -0.30, 0.96),
    open_l: float = 1.0,
    open_r: float = 1.0,
):
    import yamkit as kit

    kit._EPISODE["last"] = (kit._EPISODE["last"] or 0) + 1
    lid_q = math.radians(lid_deg)
    catch_lk = next(l for l in t.PUNNET.links if l.name == "catch")
    lid_lk = next(l for l in t.PUNNET.links if l.name == "lid")
    J = {"catch": catch_y, "lid": lid_q}
    catch_lk.law.drive("punnet/catch", catch_y, catch_vel, J, kit._law_mem("punnet/catch", catch_lk))
    lid_lk.law.drive("punnet/lid", lid_q, lid_vel, J, kit._law_mem("punnet/lid", lid_lk))

    kit._JOINTS["punnet/catch"] = kit.JointState(catch_y, catch_vel, False)
    kit._JOINTS["punnet/lid"] = kit.JointState(lid_q, lid_vel, False)
    kit._OPENING["left_gripper"] = open_l
    kit._OPENING["right_gripper"] = open_r

    p_pos = t.PUNNET_POS if punnet_pos is None else punnet_pos
    c_pos = (
        p_pos[0] + t.CATCH_ANCHOR[0],
        p_pos[1] + catch_y,
        p_pos[2] + t.CATCH_ANCHOR[2],
    )
    l_pos = (
        p_pos[0] + t.HINGE_LOCAL[0],
        p_pos[1],
        p_pos[2] + t.HINGE_LOCAL[2],
    )
    ry = lid_q / 2.0
    l_quat = np.array([0.0, math.sin(ry), 0.0, math.cos(ry)], float)

    objs = {
        "punnet": _state(p_pos, tilt_deg=tilt_deg, yaw_deg=yaw_deg, low_z=p_pos[2]),
        "punnet/catch": _state(c_pos, low_z=c_pos[2] - 0.005, size=(0.03, 0.024, 0.024)),
        "punnet/lid": ObjectState(
            np.asarray(l_pos, float),
            l_quat,
            np.zeros(3),
            np.array([0.0, lid_vel, 0.0]),
            np.array([p_pos[0] - 0.065, p_pos[1] - 0.08, l_pos[2]]),
            np.array([p_pos[0] + 0.06, p_pos[1] + 0.08, l_pos[2] + 0.022]),
        ),
        "left_gripper": ObjectState(
            np.asarray(pos_l, float),
            np.array([0.70710678, -0.70710678, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.asarray(pos_l, float) - 0.013,
            np.asarray(pos_l, float) + 0.013,
        ),
        "right_gripper": ObjectState(
            np.asarray(pos_r, float),
            np.array([0.70710678, -0.70710678, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.asarray(pos_r, float) - 0.013,
            np.asarray(pos_r, float) + 0.013,
        ),
    }
    return objs


def test_geometry_and_clearances():
    for side in ("L", "R"):
        t = _load_task(side)
        # 1:1 punnet dimensions: 16.0 x 12.0 x 6.5 cm
        assert math.isclose(t.FLANGE_Y, 0.160, abs_tol=1e-6)
        assert math.isclose(t.FLANGE_X, 0.120, abs_tol=1e-6)
        assert math.isclose(t.CLOSED_H, 0.065, abs_tol=1e-6)
        assert 22.0 <= t.LID_START_DEG <= 25.0
        # Press shoulder clears central berry dome
        assert abs(t.PRESS_Y_LID) - 0.010 > t.LID_DOME_Y
        # Catch tab sits clear above tabletop and forward of front wall
        tab_world_x = t.PUNNET_X + t.CATCH_ANCHOR[0] + t.CATCH_TAB_X_LOCAL
        front_wall_x = t.PUNNET_X - t.FLANGE_X / 2.0
        assert tab_world_x < front_wall_x - 0.020
        assert t.TOP + t.CATCH_ANCHOR[2] - 0.015 > t.TOP + 0.020
        # Symmetric start side for catch
        expected_sign = 1.0 if side == "L" else -1.0
        assert math.isclose(t.CATCH_START_Y, -expected_sign * 0.032, abs_tol=1e-6)


def test_mechanism_laws_and_interlock():
    import yamkit as kit

    t = _load_task("L")
    _fresh_trial(t)

    # 1. At start (24 deg, catch=-32mm), lid springs open & catch is unlocked
    _step_state(t, 24.0, -0.032)
    catch_mem = kit.law_state("punnet/catch")
    lid_mem = kit.law_state("punnet/lid")
    assert not catch_mem["locked"]
    assert not lid_mem["latched"]
    assert lid_mem["limits"][0] == 0.0

    # 2. Premature catch slide while lid is open (> 4 deg) jams under lid & blocks closure
    _step_state(t, 22.0, 0.0)
    catch_mem = kit.law_state("punnet/catch")
    lid_mem = kit.law_state("punnet/lid")
    assert catch_mem["jammed_under_lid"]
    assert not catch_mem["locked"]
    assert lid_mem["blocked"]
    assert lid_mem["limits"][0] >= math.radians(6.0)

    # 3. Pulling catch back out clears jam
    _step_state(t, 22.0, -0.025)
    assert not kit.law_state("punnet/catch")["jammed_under_lid"]
    assert not kit.law_state("punnet/lid")["blocked"]

    # 4. Closing lid flush (<= 3 deg) then sliding catch to 0.0 locks both
    _step_state(t, 1.0, 0.0)
    assert kit.law_state("punnet/catch")["locked"]
    assert kit.law_state("punnet/lid")["latched"]
    assert kit.law_state("punnet/lid")["limits"][1] <= math.radians(3.1)


def test_score_tiers_and_latched_hazards():
    t = _load_task("L")
    _fresh_trial(t)

    # Start -> 0.0
    assert t.score(_step_state(t, 24.0, -0.032)) == 0.0

    # Held flush (<= 3 deg), catch still open -> 0.25
    assert t.score(_step_state(t, 1.5, -0.032)) == 0.25

    # If released without latching, lid springs back open -> drops back to 0.0!
    assert t.score(_step_state(t, 24.0, -0.032)) == 0.0

    # Held flush + catch engaging (<= 12 mm) -> 0.50
    _step_state(t, 1.5, -0.020)
    assert t.score(_step_state(t, 1.5, -0.010)) == 0.50

    # Held flush + catch locked (<= 4 mm) while gripper still touching shoulder -> 0.75
    pad = t._press_pad_world(0.0, t.SIDE_SIGN * 0.062)
    for _ in range(t.HOLD_STEPS + 2):
        sc = t.score(_step_state(t, 1.0, 0.0, pos_l=(pad[0], pad[1], pad[2] + 0.013), open_l=0.0))
        assert sc == 0.75

    # Grippers clear: 19 steps -> 0.75, 20th step -> 1.0
    for _ in range(t.HOLD_STEPS - 1):
        assert t.score(_step_state(t, 1.0, 0.0)) == 0.75
    assert t.score(_step_state(t, 1.0, 0.0)) == 1.0
    assert t.success(_step_state(t, 1.0, 0.0))

    # Slam hazard permanently zeroes score
    _fresh_trial(t)
    assert t.score(_step_state(t, 8.0, -0.032, lid_vel=-2.8)) == 0.0
    assert t.HAZARD.on and t._T.reason == "slam"
    for _ in range(t.HOLD_STEPS + 2):
        assert t.score(_step_state(t, 1.0, 0.0)) == 0.0

    # Berry crush hazard (gripper inside open berry cavity) permanently zeroes score
    _fresh_trial(t)
    assert t.score(_step_state(t, 24.0, -0.032, pos_l=(t.PUNNET_X, 0.0, t.TOP + 0.025))) == 0.0
    assert t.HAZARD.on and t._T.reason == "berry_crush"

    # Punnet tip hazard permanently zeroes score
    _fresh_trial(t)
    assert t.score(_step_state(t, 24.0, -0.032, tilt_deg=15.0)) == 0.0
    assert t.HAZARD.on and t._T.reason == "punnet_disturbed"


def test_symmetric_demo_ik_and_negative_controls():
    from yam_arena import demo as ydemo

    for side in ("L", "R"):
        t = _load_task(side)
        plan = t.demo()
        assert set(plan) == {"left", "right"}
        hold = "left" if side == "L" else "right"
        latch = "right" if side == "L" else "left"
        assert any(w.label == "press_flush" for w in plan[hold])
        assert any(w.label == "slide_catch_home" for w in plan[latch])

        if not getattr(ydemo, "_IS_STUB", False):
            from yam_arena import station

            for arm in ("left", "right"):
                ap = ydemo.ArmPlan(list(plan[arm]), station.BASES[arm])
                for step_t in np.linspace(0.0, ap.duration, int(round(ap.duration * 20)) + 1):
                    ap.action7(step_t)
                assert ap.worst_ik_mm < 0.5, f"{side} {arm} worst_ik_mm={ap.worst_ik_mm:.3f}"
                assert ap.worst_ik_deg < 0.5, f"{side} {arm} worst_ik_deg={ap.worst_ik_deg:.3f}"

    # Verify negative-control env flags produce their distinct trajectories
    for flag, side, arm, expected_label in (
        ("DEMO_NO_LATCH", "L", "left", "hold_no_latch"),
        ("DEMO_LATCH_BEFORE_CLOSE", "R", "left", "slide_catch_early"),
        ("DEMO_CRUSH_BERRIES", "L", "left", "crush_berries"),
    ):
        t_nc = _load_task(side, {flag: "1"})
        os.environ[flag] = "1"
        try:
            p_nc = t_nc.demo()
            assert any(w.label == expected_label for w in p_nc[arm])
        finally:
            os.environ.pop(flag, None)


if __name__ == "__main__":
    test_geometry_and_clearances()
    test_mechanism_laws_and_interlock()
    test_score_tiers_and_latched_hazards()
    test_symmetric_demo_ik_and_negative_controls()
    print("All snap-and-latch-berry-clamshell-punnet unit tests passed!")
