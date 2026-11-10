import math

import numpy as np
import pytest

import sfp_cable_scoring as s

CARD_POS = np.array([0.30, 0.08, 0.80])


@pytest.fixture(params=[0.0, math.radians(45.0)], ids=["upright", "leaned45"])
def card_r(request):
    return s.card_rotation(s.CARD_YAW, request.param)


def tilted(card_r, deg, axis=(0, 1, 0)):
    a = np.array(axis, float)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return card_r @ (np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k)


def run(card_r, depth, bay=0, dy=0.0, dx=0.0, tilt_deg=0.0):
    tip = CARD_POS + card_r @ np.array([s.BAY_X + dx, s.BAY_Y[bay] + dy, s.MOUTH_Z - depth])
    axis = tilted(card_r, tilt_deg) @ np.array([0.0, 0.0, -1.0])
    return s.score(tip, axis, CARD_POS, card_r, plug_z=tip[2], table_z=0.7475)


def test_tiers(card_r):
    assert run(card_r, 0.042) == 1.0
    assert run(card_r, s.FULL_DEPTH + 1e-6) == 1.0
    assert run(card_r, 0.030) == 0.5
    assert run(card_r, 0.010) == 0.25
    assert run(card_r, 0.002) == 0.0


def test_right_bay_scores_zero(card_r):
    assert run(card_r, 0.046, bay=1) == 0.0


def test_lean_limits(card_r):
    assert run(card_r, 0.046, tilt_deg=2) == 1.0
    assert run(card_r, 0.046, tilt_deg=6) == 0.5
    assert run(card_r, 0.046, tilt_deg=6, ) == 0.5
    assert run(card_r, 0.046, tilt_deg=20) == 0.0
    assert run(card_r, 0.046, tilt_deg=180) == 0.0  # pointing out of the port


def test_lateral_offset(card_r):
    assert run(card_r, 0.046, dy=0.002) == 1.0
    assert run(card_r, 0.046, dy=0.006) == 0.0
    assert run(card_r, 0.002, dy=0.003) == 0.0


def test_plug_tip_from_centre_and_first_segment():
    seg0 = np.array([0.0, 0.0, 0.0])
    centre = np.array([0.0, 0.0, -0.03])  # the plug hangs 30 mm below the first segment
    tip, axis = s.plug_tip(centre, seg0)
    assert np.allclose(axis, [0, 0, -1])
    assert np.allclose(tip, [0, 0, -0.03 - s.PLUG_LEN / 2])


def test_fall_scores_zero(card_r):
    tip = CARD_POS + card_r @ np.array([s.BAY_X, s.BAY_Y[0], s.MOUTH_Z - 0.046])
    assert s.score(tip, card_r @ np.array([0, 0, -1.0]), CARD_POS, card_r, plug_z=0.5, table_z=0.7475) == 0.0
