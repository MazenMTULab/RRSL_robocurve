import math

import numpy as np
import pytest

import sfp_seat_scoring as s

CARD_POS = np.array([0.30, 0.08, 0.80])
TABLE = 0.7475


@pytest.fixture(params=[0.0, math.radians(45.0)], ids=["upright", "leaned45"])
def card_r(request):
    return s.card_rotation(s.CARD_YAW, request.param)


def axis_angle(axis, deg):
    a = np.array(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def run(card_r, depth, bay=0, dy=0.0, dx=0.0, extra=None, part_to_module=None):
    """Score of a module whose tip is `depth` below the mouth in `bay`, rotated by `extra` (in the card's frame) from seated."""
    r = card_r @ (np.eye(3) if extra is None else extra)
    tip = CARD_POS + card_r @ np.array([s.BAY_X + dx, s.BAY_Y[bay] + dy, s.MOUTH_Z - depth])
    pos = tip - r @ np.array([0.0, 0.0, s.TIP_Z])
    q_module = s.quat_xyzw(r if part_to_module is None else r @ part_to_module.T)
    return s.score(pos, q_module, CARD_POS, card_r, part_to_module=part_to_module, module_z=pos[2], table_z=TABLE)


def test_quaternion_roundtrip():
    for r in (np.eye(3), s.card_rotation(), axis_angle((1, 2, 3), 170), axis_angle((0, 0, 1), 179)):
        assert np.allclose(s.rotation(s.quat_xyzw(r)), r, atol=1e-9)


def test_tiers(card_r):
    assert run(card_r, 0.044) == 1.0
    assert run(card_r, s.FULL_DEPTH + 1e-6) == 1.0  # exactly 42 mm is a float coin-flip
    assert run(card_r, 0.030) == 0.5
    assert run(card_r, 0.010) == 0.25
    assert run(card_r, 0.002) == 0.0


def test_right_bay_scores_zero(card_r):
    assert run(card_r, 0.044, bay=1) == 0.0


def test_lean_limit(card_r):
    assert run(card_r, 0.044, extra=axis_angle((0, 1, 0), 2)) == 1.0
    assert run(card_r, 0.044, extra=axis_angle((0, 1, 0), 6)) == 0.5
    assert run(card_r, 0.044, extra=axis_angle((1, 0, 0), 6)) == 0.5
    assert run(card_r, 0.044, extra=axis_angle((0, 1, 0), 20)) == 0.0


def test_turn_must_match_the_card(card_r):
    assert run(card_r, 0.044, extra=axis_angle((0, 0, 1), 2)) == 1.0
    assert run(card_r, 0.044, extra=axis_angle((0, 0, 1), 6)) == 0.5
    assert run(card_r, 0.044, extra=axis_angle((0, 0, 1), 30)) == 0.0
    assert run(card_r, 0.044, extra=axis_angle((0, 0, 1), 180)) == 0.0  # turned around


def test_upside_down_scores_zero(card_r):
    assert run(card_r, 0.044, extra=axis_angle((1, 0, 0), 180)) == 0.0


def test_lateral_offset(card_r):
    assert run(card_r, 0.044, dy=0.002) == 1.0  # inside three times 1.5 mm once it is in
    assert run(card_r, 0.044, dy=0.006) == 0.0
    assert run(card_r, 0.002, dy=0.003) == 0.0


def test_untilted_module_in_a_leaned_card_scores_zero():
    leaned = s.card_rotation(s.CARD_YAW, math.radians(45.0))
    assert run(leaned, 0.044, extra=axis_angle((0, 1, 0), 45)) == 0.0  # still upright: 45 degrees off the port axis


def test_body_posed_in_a_holder_frame(card_r):
    # the module's body frame is pitched 45 degrees from its own frame; the scorer undoes it
    off = axis_angle((0, 1, 0), -45)
    assert run(card_r, 0.044, part_to_module=off) == 1.0
    assert run(card_r, 0.030, part_to_module=off) == 0.5


def test_fall_scores_zero(card_r):
    r = card_r
    pos = CARD_POS + r @ np.array([s.BAY_X, s.BAY_Y[0], s.MOUTH_Z - 0.044]) - r @ np.array([0, 0, s.TIP_Z])
    assert s.score(pos, s.quat_xyzw(r), CARD_POS, card_r, module_z=0.5, table_z=TABLE) == 0.0
