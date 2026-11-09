import math

import numpy as np

import sfp_seat_scoring as s

CARD_POS, CARD_YAW = np.array([0.30, 0.08, 0.80]), math.radians(8.0)


def quat_z(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def quat_axis(axis, deg):
    a = math.radians(deg) / 2
    v = np.array(axis, float) / np.linalg.norm(axis)
    return (*(v * math.sin(a)), math.cos(a))


def module_at(depth, bay=0, dy=0.0, dx=0.0, q=None):
    """Module origin whose tip is `depth` m below the cage mouth in `bay`, with orientation `q` (default: turned with the card)."""
    q = quat_z(CARD_YAW) if q is None else q
    tip_local = np.array([s.BAY_X + dx, s.BAY_Y[bay] + dy, s.MOUTH_Z - depth])
    tip = CARD_POS + s.rz(CARD_YAW) @ tip_local
    return tuple(tip - s.rotation(q) @ np.array([0.0, 0.0, s.TIP_Z])), q


def run(depth, **kw):
    pos, q = module_at(depth, **kw)
    return s.score(pos, q, CARD_POS, CARD_YAW, module_z=pos[2], table_z=0.7475)


def test_tiers():
    assert run(0.044) == 1.0
    assert run(s.FULL_DEPTH) == 1.0
    assert run(0.030) == 0.5
    assert run(0.010) == 0.25
    assert run(0.002) == 0.0


def test_right_bay_scores_zero():
    assert run(0.044, bay=1) == 0.0


def test_lean_limit():
    q = mul(quat_axis((1, 0, 0), 2), quat_z(CARD_YAW))
    assert run(0.044, q=q) == 1.0
    assert run(0.044, q=mul(quat_axis((1, 0, 0), 6), quat_z(CARD_YAW))) == 0.5
    assert run(0.044, q=mul(quat_axis((1, 0, 0), 20), quat_z(CARD_YAW))) == 0.0


def test_turn_must_match_the_card():
    assert run(0.044, q=quat_z(0.0)) == 0.5  # square to the table, 8 degrees off the card: partial credit only
    assert run(0.044, q=quat_z(CARD_YAW + math.radians(2))) == 1.0
    assert run(0.044, q=quat_z(CARD_YAW + math.radians(30))) == 0.0
    assert run(0.044, q=quat_z(CARD_YAW + math.pi)) == 0.0  # turned around


def test_upside_down_scores_zero():
    assert run(0.044, q=mul(quat_axis((1, 0, 0), 180), quat_z(CARD_YAW))) == 0.0


def test_lateral_offset():
    assert run(0.044, dy=0.002) == 1.0  # inside three times 1.5 mm once it is in
    assert run(0.044, dy=0.006) == 0.0
    assert run(0.002, dy=0.003) == 0.0


def test_start_state_scores_zero_and_fall_scores_zero():
    assert s.score((0.24, 0.30, 0.7746), (0, 0, 0, 1), CARD_POS, CARD_YAW, module_z=0.7746, table_z=0.7475) == 0.0
    pos, q = module_at(0.044)
    assert s.score(pos, q, CARD_POS, CARD_YAW, module_z=0.5, table_z=0.7475) == 0.0
