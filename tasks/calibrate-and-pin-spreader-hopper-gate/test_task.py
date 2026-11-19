"""Unit tests for calibrate-and-pin-spreader-hopper-gate (geometry, scorer, spring/pin logic, hazards, and 20 Hz IK)."""

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
    yd.READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]
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

import yamkit as kit  # noqa: E402
from yam_arena.task_file import ObjectState  # noqa: E402


def _load_task(side: str = "L", env_overrides: dict | None = None):
    old = {k: os.environ.get(k) for k in ("SIDE", "DEMO_NO_PIN", "DEMO_OVER_OPEN")}
    os.environ["SIDE"] = side
    os.environ.pop("DEMO_NO_PIN", None)
    os.environ.pop("DEMO_OVER_OPEN", None)
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


def _state(pos, tilt_deg=0.0, vel=(0, 0, 0), ang_vel=(0, 0, 0), low_z=None, size=(0.028, 0.026, 0.086)):
    pos = np.asarray(pos, float)
    rad = math.radians(tilt_deg)
    q = np.array([math.sin(rad / 2), 0.0, 0.0, math.cos(rad / 2)], float)
    half = np.asarray(size, float) / 2
    lz = pos[2] - 0.044 if low_z is None else float(low_z)
    low = np.array([pos[0] - half[0], pos[1] - half[1], lz], float)
    high = np.array([pos[0] + half[0], pos[1] + half[1], lz + size[2]], float)
    return ObjectState(pos, q, np.asarray(vel, float), np.asarray(ang_vel, float), low, high)


def _trial_objects(t, gate_open_m: float = 0.0, gate_vel: float = 0.0, pin_pos=None, pin_tilt: float = 0.0,
                   pin_vel=(0, 0, 0), pos_l=(0.0, 0.30, 0.96), pos_r=(0.0, -0.30, 0.96)):
    q = t.SIDE_SIGN * gate_open_m
    kit._JOINTS["hopper_gate/gate"] = kit.JointState(float(q), float(t.SIDE_SIGN * gate_vel), False)
    if pin_pos is None:
        pin_pos = t.PIN_START_POS
    return {
        "hopper_gate": _state((t.GATE_X, t.GATE_Y, t.TOP), low_z=t.TOP,
                              size=(t.HOUSING_X, t.HOUSING_Y, t.HOUSING_Z)),
        "hopper_gate/gate": _state((t.GATE_X, t.GATE_Y + q, t.TOP), low_z=t.TOP + 0.014,
                                   size=(0.068, 0.210, 0.060)),
        "stop_pin": _state(pin_pos, tilt_deg=pin_tilt, vel=pin_vel),
        "left_gripper": _state(pos_l, low_z=pos_l[2] - 0.02, size=(0.04, 0.04, 0.04)),
        "right_gripper": _state(pos_r, low_z=pos_r[2] - 0.02, size=(0.04, 0.04, 0.04)),
    }


def _reset_trial(t):
    kit._EPISODE["count"] += 1
    kit._EPISODE["last"] = 0
    t.HAZARD.run, t.HAZARD.on, t.HAZARD._key, t.HAZARD._objs = 0, False, None, None
    t._T = t._Tracker()


def _step(t, **kw):
    kit._EPISODE["last"] = (kit._EPISODE["last"] or 0) + 1
    objs = _trial_objects(t, **kw)
    t.success(objs)
    return t.score(objs)


def test_geometry_clearances():
    for side in ("L", "R"):
        t = _load_task(side)
        s = t.SIDE_SIGN
        # Main assembly matches 18 x 14 x 10 cm envelope
        assert math.isclose(t.HOUSING_Y, 0.180, abs_tol=1e-6)
        assert math.isclose(t.HOUSING_X, 0.140, abs_tol=1e-6)
        assert math.isclose(t.HOUSING_Z, 0.100, abs_tol=1e-6)
        # Spring returns toward 0.0 with positive stiffness
        gate_link = t.HOPPER_GATE.links[0]
        assert gate_link.kind == "prismatic" and gate_link.axis == "Y"
        assert gate_link.law.rest == 0.0 and gate_link.law.k > 0.0
        # Bore clearance around 12 mm pin shaft (18 mm bore -> 3 mm clearance per side)
        assert t.BORE_W - t.PIN_SHAFT_W >= 0.005
        # Pin collar overlaps clevis bore by >= 4 mm per side
        assert (t.PIN_COLLAR_W - t.BORE_W) / 2.0 >= 0.004
        # At gate_open=0.0, leading stop block covers the entire bore; at 0.040 it seats against pin; at 0.046 bore is clear
        lead_edge_0 = -s * 0.055
        pin_plus_face = t.BORE_Y_LOCAL - s * (t.BORE_W / 2.0 - t.PIN_SHAFT_W)  # pin pushed to -s bore wall -> +s face at BORE_Y_LOCAL
        assert abs((lead_edge_0 + s * 0.040) - pin_plus_face) < 1e-6
        assert s * ((lead_edge_0 + s * 0.046) - t.BORE_Y_LOCAL) >= 0.0059
        # Pin stand sits outside housing Y footprint
        assert abs(t.STAND_Y) > t.HOUSING_Y / 2.0 + 0.06


def test_score_tiers_and_hazards():
    t = _load_task("L")
    _reset_trial(t)

    # Start (0.0 cm closed, pin on stand) -> 0.0
    assert _step(t, gate_open_m=0.0) == 0.0

    # Gate pulled open >= 3.2 cm -> 0.25
    assert _step(t, gate_open_m=0.036) == 0.25

    # Gate held near Setting 4 and pin lifted off stand -> 0.50
    assert _step(t, gate_open_m=0.044, pin_pos=(t.STAND_X, t.STAND_Y, t.TOP + 0.12)) == 0.50

    # Gate at 4.0 cm and pin seated in clevis bore while gripper still touching -> capped at 0.75
    bore_pin = (t.BORE_X, t.BORE_Y, t.PIN_SEAT_Z)
    for _ in range(t.HOLD_STEPS + 2):
        sc_touch = _step(
            t,
            gate_open_m=0.040,
            pin_pos=bore_pin,
            pos_l=(t.HANDLE_X, t.HANDLE_Y0 + 0.040, t.TOP + t.HANDLE_Z_LOCAL),
        )
        assert sc_touch == 0.75

    # Both grippers clear for 19 steps -> still 0.75; 20th step -> 1.0
    for _ in range(t.HOLD_STEPS - 1):
        assert _step(t, gate_open_m=0.040, pin_pos=bore_pin) == 0.75
    assert _step(t, gate_open_m=0.040, pin_pos=bore_pin) == 1.0
    assert t.success(_trial_objects(t, gate_open_m=0.040, pin_pos=bore_pin))

    # Over-opening past 5.2 cm for 2 steps latches 0.0 permanently
    _reset_trial(t)
    _step(t, gate_open_m=0.056)
    assert _step(t, gate_open_m=0.056) == 0.0
    assert t.HAZARD.on
    for _ in range(t.HOLD_STEPS + 2):
        assert _step(t, gate_open_m=0.040, pin_pos=bore_pin) == 0.0

    # Dropped pin off tabletop latches 0.0 permanently
    _reset_trial(t)
    _step(t, gate_open_m=0.040, pin_pos=(t.BORE_X, t.BORE_Y, t.TOP - 0.05))
    assert _step(t, gate_open_m=0.040, pin_pos=(t.BORE_X, t.BORE_Y, t.TOP - 0.05)) == 0.0
    assert t.HAZARD.on


def test_symmetric_demo_and_ik():
    for side in ("L", "R"):
        t = _load_task(side)
        plan = t.demo()
        assert set(plan) == {"left", "right"}
        gate_arm = "left" if side == "L" else "right"
        pin_arm = "right" if side == "L" else "left"
        assert any(w.label == "pull_to_setting4" for w in plan[gate_arm])
        assert any(w.label == "seat_pin" for w in plan[pin_arm])

        from yam_arena import demo as ydemo  # noqa: E402

        if not getattr(ydemo, "_IS_STUB", False):
            from yam_arena import station  # noqa: E402

            for arm in ("left", "right"):
                ap = ydemo.ArmPlan(list(plan[arm]), station.BASES[arm])
                for step_t in np.linspace(0.0, ap.duration, int(round(ap.duration * 20)) + 1):
                    ap.action7(step_t)
                assert ap.worst_ik_mm < 0.5, f"{side} {arm} worst_ik_mm={ap.worst_ik_mm:.3f}"
                assert ap.worst_ik_deg < 0.5, f"{side} {arm} worst_ik_deg={ap.worst_ik_deg:.3f}"


def test_negative_control_flags():
    t_np = _load_task("L")
    os.environ["DEMO_NO_PIN"] = "1"
    try:
        p_np = t_np.demo()
        assert any(w.label == "release_early" for w in p_np["left"])
        assert all(w.label == "stay_home" for w in p_np["right"])
    finally:
        os.environ.pop("DEMO_NO_PIN", None)

    t_oo = _load_task("R")
    os.environ["DEMO_OVER_OPEN"] = "1"
    try:
        p_oo = t_oo.demo()
        assert any(w.label == "over_open_gate" for w in p_oo["right"])
    finally:
        os.environ.pop("DEMO_OVER_OPEN", None)


if __name__ == "__main__":
    test_geometry_clearances()
    test_score_tiers_and_hazards()
    test_symmetric_demo_and_ik()
    test_negative_control_flags()
    print("All calibrate-and-pin-spreader-hopper-gate unit tests passed!")
