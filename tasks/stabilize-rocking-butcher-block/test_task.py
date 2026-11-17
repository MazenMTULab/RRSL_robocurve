"""Offline tests of the stabilize-rocking-butcher-block scorer, hazards, boundaries, the fixed default layout, and demo plans.

Run directly or via pytest with yam_arena on PYTHONPATH:
    python3 tasks/stabilize-rocking-butcher-block/test_task.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from yam_arena.urdf import quat_xyzw_to_matrix
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).parent

if os.environ.get("BOARD_CORNER", "").upper() not in ("FL", "FR") and __name__ == "__main__":
    rc = 0
    for c in ("FL", "FR"):
        print(f"== BOARD_CORNER={c} ==", flush=True)
        rc |= subprocess.call([sys.executable, __file__], env={**os.environ, "BOARD_CORNER": c})
    os.environ.pop("BOARD_CORNER", None)
    assert load(HERE / "task.py").module.MISSING_CORNER == "FL"
    print("ok   default layout uses the front-left corner")
    raise SystemExit(rc)

M = load(HERE / "task.py").module
import yamkit as kit  # noqa: E402

IDENT = np.array([0.0, 0.0, 0.0, 1.0])
DOWN_X = np.array([1.0, 0.0, 0.0, 0.0])
AWAY = {"left_gripper": (0.15, 0.28, M.TOP + 0.30), "right_gripper": (0.15, -0.28, M.TOP + 0.30)}


def _axis_tilt(axis, degrees):
    half_angle = np.deg2rad(degrees) / 2
    quat = np.array([0.0, 0.0, 0.0, np.cos(half_angle)])
    quat["xyz".index(axis)] = np.sin(half_angle)
    return quat


def _state_box(pos, quat, low_local, high_local, vel=(0, 0, 0)):
    p = np.asarray(pos, float)
    R = quat_xyzw_to_matrix(quat)
    corners = np.array([
        [x, y, z]
        for x in (low_local[0], high_local[0])
        for y in (low_local[1], high_local[1])
        for z in (low_local[2], high_local[2])
    ])
    w = p + corners @ R.T
    return ObjectState(
        pos=p,
        quat_xyzw=np.asarray(quat, float),
        lin_vel=np.asarray(vel, float),
        ang_vel=np.zeros(3),
        low=w.min(0),
        high=w.max(0),
    )


def _cup_world_from_board(board_pos, board_quat, local_xy=None, local_z_off=0.0):
    lx, ly = M.CUP_BOARD_LOCAL_XY if local_xy is None else local_xy
    R = quat_xyzw_to_matrix(board_quat)
    p = np.asarray(board_pos, float) + R @ np.array([lx, ly, M.FOOT_H + M.BOARD_Z + M.CUP_H / 2.0 + local_z_off])
    return p, np.asarray(board_quat, float)


class Trial:
    def __init__(self):
        kit._EPISODE["count"] += 1
        kit._EPISODE["last"] = None
        self.board_pos = np.array(M.BOARD_POS0, float)
        self.board_quat = np.array(M.BOARD_TILT_XYZW, float)
        self.pad_pos = np.array(M.PAD_START, float)
        self.pad_quat = IDENT.copy()
        self.cup_pos = np.array(M.CUP_POS0, float)
        self.cup_quat = np.array(M.BOARD_TILT_XYZW, float)
        self.vel = {"cutting_board": (0, 0, 0), "rubber_pad": (0, 0, 0), "cup": (0, 0, 0)}
        self.grip = {g: (AWAY[g], DOWN_X, 1.0) for g in M.GRIPPERS}
        self.s = 0.0
        self.step()

    def build_objects(self):
        o = {
            "cutting_board": _state_box(
                self.board_pos,
                self.board_quat,
                (-M.BOARD_X / 2, -M.BOARD_Y / 2, 0.0),
                (M.BOARD_X / 2, M.BOARD_Y / 2, M.FOOT_H + M.BOARD_Z),
                self.vel["cutting_board"],
            ),
            "rubber_pad": _state_box(
                self.pad_pos,
                self.pad_quat,
                (-M.PAD_SIZE[0] / 2, -M.PAD_SIZE[1] / 2, -M.PAD_SIZE[2] / 2),
                (M.PAD_SIZE[0] / 2, M.PAD_SIZE[1] / 2, M.PAD_SIZE[2] / 2),
                self.vel["rubber_pad"],
            ),
            "cup": _state_box(
                self.cup_pos,
                self.cup_quat,
                (-M.CUP_R, -M.CUP_R, -M.CUP_H / 2),
                (M.CUP_R, M.CUP_R, M.CUP_H / 2),
                self.vel["cup"],
            ),
        }
        for g, (p, q, op) in self.grip.items():
            p = np.asarray(p, float)
            o[g] = ObjectState(
                pos=p, quat_xyzw=np.asarray(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                low=p - 0.013, high=p + 0.013,
            )
            kit._OPENING[g] = op
        return o

    def step(self, n=1):
        for i in range(n):
            kit._EPISODE["last"] = (kit._EPISODE["last"] or 0) + 1
            o = self.build_objects()
            M.success(o)
            self.s = M.score(o)
        return self.s

    def set_level_solved(self):
        self.board_pos = np.array(M.BOARD_POS0, float)
        self.board_quat = IDENT.copy()
        fx, fy, _ = M.MISS_FOOT_LOCAL
        self.pad_pos = np.array([M.X_B + fx, M.Y_B + fy, M.TOP + M.PAD_H / 2.0], float)
        self.pad_quat = IDENT.copy()
        self.cup_pos, self.cup_quat = _cup_world_from_board(self.board_pos, self.board_quat)


def test_start_is_zero():
    t = Trial()
    assert t.s == 0.0
    assert not M.success(t.build_objects())


def test_partial_credit_progression_and_hold_gate():
    t = Trial()
    fx, fy, _ = M.MISS_FOOT_LOCAL
    foot_xy = np.array([M.X_B + fx, M.Y_B + fy])

    # Move pad 4 cm toward corner -> 0.25
    t.pad_pos = np.array([foot_xy[0] - 0.06, foot_xy[1] - M.SIDE_SIGN * 0.04, M.TOP + M.PAD_H / 2.0])
    assert t.step() == 0.25

    # Put pad within 2.0 cm (still outside 1.2 cm seat tolerance) -> 0.50
    t.pad_pos = np.array([foot_xy[0] - 0.018, foot_xy[1], M.TOP + M.PAD_H / 2.0])
    assert t.step() == 0.50

    # Level board & seat pad squarely, but gripper still touching side rim -> capped at 0.75
    t.set_level_solved()
    t.grip["left_gripper"] = ((M.X_B, M.Y_B + M.SIDE_SIGN * (M.BOARD_Y / 2.0 + 0.005), M.TOP + 0.03), DOWN_X, 0.5)
    assert t.step(25) == 0.75

    # Release gripper: 19 steps still 0.75; 20th step reaches 1.0
    t.grip["left_gripper"] = (AWAY["left_gripper"], DOWN_X, 1.0)
    assert t.step(19) == 0.75
    assert t.step(1) == 1.0
    assert M.success(t.build_objects())


def test_pad_offset_and_tilt_boundaries():
    t = Trial()
    t.set_level_solved()
    # Pad shifted 1.5 cm (> 1.2 cm PAD_SEAT_TOL) even with level board -> 0.50
    t.pad_pos[0] -= 0.015
    assert t.step(25) == 0.50

    # Pad seated cleanly, but board still tilted 1.5 deg (> 1.2 deg) -> 0.50
    t = Trial()
    t.set_level_solved()
    t.board_quat = _axis_tilt("x", 1.5)
    assert t.step(25) == 0.50

    # Pad on edge (tilted 25 deg) -> 0.25 max
    t = Trial()
    t.set_level_solved()
    t.pad_quat = _axis_tilt("y", 25.0)
    assert t.step(25) < 0.75


def test_hazards_latch_permanently_until_reset():
    # 1. Cup tipped past 25 deg
    t = Trial()
    t.cup_quat = _axis_tilt("x", 30.0)
    t.step(2)
    t.set_level_solved()
    assert t.step(25) == 0.0

    # 2. Cup removed / slid off board (> 4.5 cm)
    t = Trial()
    t.cup_pos = np.array([0.15, 0.20, M.TOP + M.CUP_H / 2.0])
    t.cup_quat = IDENT.copy()
    t.step(2)
    t.set_level_solved()
    assert t.step(25) == 0.0

    # 3. Board tipped past 18 deg
    t = Trial()
    t.board_quat = _axis_tilt("x", 22.0)
    t.step(2)
    t.set_level_solved()
    assert t.step(25) == 0.0

    # 4. New episode clears latched hazard and allows full score
    t = Trial()
    t.set_level_solved()
    assert t.step(20) == 1.0


def test_missing_support_and_wrong_way_motion_do_not_get_credit():
    trial = Trial()
    trial.pad_pos[0] -= 0.04
    assert trial.step() == 0.0
    trial.set_level_solved()
    trial.pad_pos[2] -= 0.005
    assert trial.step(25) < 0.75
    assert not M.success(trial.build_objects())


def test_demo_plans_valid():
    plans = M.demo()
    assert set(plans) == {"left", "right"}
    assert all(len(plans[s]) >= 8 for s in ("left", "right"))


if __name__ == "__main__":
    tests = [
        test_start_is_zero,
        test_partial_credit_progression_and_hold_gate,
        test_pad_offset_and_tilt_boundaries,
        test_hazards_latch_permanently_until_reset,
        test_missing_support_and_wrong_way_motion_do_not_get_credit,
        test_demo_plans_valid,
    ]
    for fn in tests:
        fn()
        print(f"ok   {fn.__name__}")
