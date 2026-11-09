import math

import numpy as np
import pytest

import sfp_nic_scoring as s

NIC_POS, NIC_Q = (0.0, 0.0, 0.10), (0.0, 0.0, 0.0, 1.0)
YAW180 = (0.0, 0.0, 1.0, 0.0)  # tip (module -y) points along the card's +y


def quat_about(axis, deg):
    a = math.radians(deg) / 2
    v = np.array(axis, float) / np.linalg.norm(axis)
    return (*(v * math.sin(a)), math.cos(a))


def mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def module_at(depth, bay=0, dx=0.0, dz=0.0, q=YAW180):
    """Module position (card frame = world offset) whose tip is `depth` m past the mouth in the given bay."""
    b = s.BAYS[bay]
    tip = np.array([(b["x_min"] + b["x_max"]) / 2 + dx, s.CAGE_OPEN_Y + depth, s.BAY_CENTRE_Z + dz])
    rel = s.rotation(q) @ np.array([0.0, s.TIP_Y, 0.0])
    return tuple(np.array(NIC_POS) + tip - rel)


def run(depth, **kw):
    q = kw.pop("q", YAW180)
    lowest = kw.pop("nic_lowest_z", 0.10)
    return s.score(module_at(depth, q=q, **kw), q, NIC_POS, NIC_Q, nic_lowest_z=lowest, table_z=0.0)


def test_full_seat():
    assert run(0.045) == 1.0
    assert run(s.FULL_DEPTH) == 1.0


def test_partial_tiers():
    assert run(0.030) == 0.5
    assert run(0.010) == 0.25
    assert run(0.002) == 0.0  # not past the mouth


def test_wrong_bay_scores_zero():
    assert run(0.045, bay=1) == 0.0


def test_tilt_and_roll_limits():
    assert run(0.045, q=mul(quat_about((0, 0, 1), 2), YAW180)) == 1.0
    assert run(0.045, q=mul(quat_about((0, 0, 1), 6), YAW180)) == 0.5  # 6 deg yaw: too tilted for full
    assert run(0.045, q=mul(quat_about((0, 1, 0), 10), YAW180)) == 0.5  # 10 deg roll
    assert run(0.045, q=mul(quat_about((0, 0, 1), 20), YAW180)) == 0.0


def test_upside_down_or_backwards_scores_zero():
    assert run(0.045, q=mul(quat_about((0, 1, 0), 180), YAW180)) == 0.0  # flipped
    assert run(0.045, q=(0, 0, 0, 1)) == 0.0  # tip pointing out of the cage


def test_lateral_offset():
    assert run(0.045, dx=0.001) == 1.0
    assert run(0.045, dx=0.006) == 0.0
    assert run(0.010, dx=0.003) == 0.25  # outside bay tolerance at the mouth would fail; 3 mm < 3 x 1.5 mm


def test_card_on_table_scores_zero():
    assert run(0.045, nic_lowest_z=0.005) == 0.0
    assert run(0.045, nic_lowest_z=0.03) == 1.0


def test_off_table():
    assert s.score(module_at(0.045), YAW180, NIC_POS, NIC_Q, nic_lowest_z=0.1, off_table=True) == 0.0


def test_card_pose_is_respected():
    q = quat_about((0, 0, 1), 30)
    rn = s.rotation(q)
    pos = np.array([0.2, 0.1, 0.12])
    tip = np.array([(s.BAYS[0]["x_min"] + s.BAYS[0]["x_max"]) / 2, s.CAGE_OPEN_Y + 0.045, s.BAY_CENTRE_Z])
    qm = mul(q, YAW180)
    mpos = pos + rn @ tip - s.rotation(qm) @ np.array([0, s.TIP_Y, 0])
    assert s.score(mpos, qm, pos, q, nic_lowest_z=0.1) == 1.0
