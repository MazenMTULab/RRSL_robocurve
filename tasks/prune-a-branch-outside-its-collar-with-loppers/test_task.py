"""Offline checks of the pruning scorer: loppers poses and jaw openings stepped through the cut tracker and scorer,
near misses included, plus the demo's plan. No simulator.

    python -B tasks/prune-a-branch-outside-its-collar-with-loppers/test_task.py      (prints "all pass")
"""

import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("prune_task", os.path.join(HERE, "task.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
kit = m.kit
from yam_arena.task_file import ObjectState  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def quat(yaw, roll=0.0):
    """yaw about z, then roll about the loppers' own x (tilting the jaws' plane)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    # q = qz(yaw) * qx(roll), xyzw
    return np.array([cy * sr, sy * sr, sy * cr, cy * cr])


def st(pos, q):
    p = np.array(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.asarray(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=p - 0.01, high=p + 0.01)


GRIP = {"left_gripper": (0.2, 0.15, m.TOP + 0.1), "right_gripper": (0.2, -0.15, m.TOP + 0.1)}


def new_trial():
    kit._EPISODE["count"] += 1


def pivot_for(r=0.045, a=0.22, yaw=m.YAW0, z=None):
    off = m._rotz(yaw) @ np.array([r * math.cos(m.HOOK_ANG + a), r * math.sin(m.HOOK_ANG + a), 0.0])
    return (m.BRANCH[0] - off[0], m.BRANCH[1] - off[1], z)


def step(pivot, opening_deg, yaw=m.YAW0, roll=0.0):
    q = m.Q_SHUT + math.radians(opening_deg)
    kit._JOINTS["loppers/blade_arm"] = kit.JointState(q, 0.0, False)
    kit._JOINTS["log/branch"] = kit.JointState(0.0, 0.0, False)
    objs = {"loppers": st(pivot, quat(yaw, roll)), "log": st((m.LOG_X, 0, m.TOP), (0, 0, 0, 1)),
            "log/branch": st((m.LOG_X, 0, m.TOP + m.COLLAR_TOP), (0, 0, 0, 1)),
            **{k: st(v, (1, 0, 0, 0)) for k, v in GRIP.items()}}
    return m.score(objs), m.success(objs)


def close_at(stub_cm, r=0.045, a=0.22, yaw=m.YAW0, roll=0.0, open0=35.0, open1=-3.0):
    new_trial()
    x, y, _ = pivot_for(r, a, yaw)
    z = m.TOP + m.COLLAR_TOP + stub_cm / 100
    out = None
    for k in range(12):
        out = step((x, y, z), open0 + (open1 - open0) * k / 11, yaw, roll)
    for _ in range(5):
        out = step((x - 0.05, y, z), open1, yaw, roll)
    return out


s, ok = close_at(0.7)
check(s == 1.0 and ok, f"cut 7 mm above the collar -> 1.0 (got {s})")
check(abs(m.stub(m._CUT['objects']) - 0.007) < 1e-6, "the stub is read from the jaws' plane at the branch")
for cm, want in ((0.1, 1.0), (1.4, 1.0), (1.6, 0.5), (2.9, 0.5), (3.1, 0.25), (8.0, 0.25), (-0.3, 0.0), (-1.0, 0.0)):
    s, _ = close_at(cm)
    check(s == want, f"stub {cm} cm -> {want} (got {s})")

# no cut: the jaws closed with the branch outside the throat, or never closed
s, _ = close_at(0.7, r=0.10)
check(s == 0.0, "branch beyond the jaw tips (10 cm out) -> no cut, 0")
s, _ = close_at(0.7, r=0.008)
check(s == 0.0, "branch jammed against the pivot (0.8 cm) -> no cut, 0")
s, _ = close_at(0.7, a=-0.6)
check(s == 0.0, "branch outside the hook -> no cut, 0")
s, _ = close_at(0.7, open1=10.0)
check(s == 0.0, "jaws closed only to 10 degrees -> no cut, 0")
s, _ = close_at(0.7, roll=math.radians(40))
check(s == 0.0, "jaws' plane 40 degrees off square -> the blade skids, no cut")
s, _ = close_at(0.7, roll=math.radians(20))
h = m.stub(m._CUT["objects"])
check(h is not None and s > 0, f"jaws' plane 20 degrees off square -> still a cut, read where the plane meets the branch "
      f"(stub {h * 100:.1f} cm, score {s})")

# the first cut counts: a high cut, then a second cut at the collar
new_trial()
x, y, _ = pivot_for()
for z_cm, rng in ((6.0, (35, -3)), (0.7, (-3, 35)), (0.7, (35, -3))):
    for k in range(12):
        step((x, y, m.TOP + m.COLLAR_TOP + z_cm / 100), rng[0] + (rng[1] - rng[0]) * k / 11)
s, _ = step((x, y, m.TOP + 0.3), -3)
check(s == 0.25, "a 6 cm stub cut first, then a cut at the collar -> the first cut counts (0.25)")

# the loppers never brought to the branch
new_trial()
for k in range(12):
    s, _ = step((m.P0[0], m.P0[1], m.TOP + 0.01), 35 - 38 * k / 11)
check(s == 0.0, "closing the loppers on the table -> 0")

d = m.demo()
check(abs(sum(w.duration for w in d["left"]) - sum(w.duration for w in d["right"])) < 1e-9, "both plans same length")
total = sum(w.duration for w in d["right"])
check(total * 20 <= 1200, f"demo plan {total:.1f} s = {total * 20:.0f} steps, under the 1,200-step run limit")
print("all pass" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
