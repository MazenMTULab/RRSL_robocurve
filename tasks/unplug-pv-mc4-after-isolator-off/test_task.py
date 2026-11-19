"""Offline checks of the MC4 scorer: knob angles, connector withdrawals, lead draws and gripper pinches stepped
through the latch law, the tracker and the scorer, near misses included, plus the demo's plan. No simulator.

    python -B tasks/unplug-pv-mc4-after-isolator-off/test_task.py      (prints "all pass")
"""

import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("mc4_task", os.path.join(HERE, "task.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
kit = m.kit
from yam_arena.task_file import ObjectState  # noqa: E402

fails = 0
CAB = m.OBJECTS[1].links[1]


def check(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


DOWN = np.array([1.0, 0.0, 0.0, 0.0])
DOWN_X = np.array([0.7071068, 0.7071068, 0.0, 0.0])  # pointing down, fingers closing along x
PARK = {"left_gripper": (0.2, 0.27, m.TOP + 0.2), "right_gripper": (0.2, -0.27, m.TOP + 0.2)}


def st(pos, q=(0, 0, 0, 1.0)):
    p = np.array(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.array(q, float), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p - 0.01,
                       high=p + 0.01)


def new_trial():
    kit._EPISODE["count"] += 1


def step(knob=m.ON, w=0.0, lead=0.0, right=None, left=None, r_open=1.0, l_open=1.0):
    kit._JOINTS["isolator/knob"] = kit.JointState(knob, 0.0, False)
    kit._JOINTS["string/cable_half"] = kit.JointState(-w, 0.0, False)
    kit._JOINTS["string/module_half"] = kit.JointState(-lead, 0.0, False)
    g = dict(PARK)
    if right:
        g["right_gripper"] = right
    if left:
        g["left_gripper"] = left
    objs = {"isolator": st((*m.ISO, m.TOP)), "isolator/knob": st((*m.ISO, m.TOP + m.ISO_BOX[2])),
            "string": st((m.RAIL_X, m.JOINT_Y, m.TOP)),
            "string/module_half": st((m.RAIL_X, m.JOINT_Y + m.MOD_L / 2 - lead, m.TOP + m.AXIS_Z)),
            "string/cable_half": st((m.RAIL_X, m.JOINT_Y - m.CAB_L / 2 - w - lead, m.TOP + m.AXIS_Z)),
            **{k: st(v, DOWN_X) for k, v in g.items()}}
    kit._OPENING["right_gripper"], kit._OPENING["left_gripper"] = r_open, l_open
    return m.score(objs), m.success(objs)


def on_cab(w=0.0, lead=0.0):
    return (m.RAIL_X, m.JOINT_Y - 0.048 - w - lead, m.TOP + m.GRIP_Z)


def on_mod():
    return (m.RAIL_X, m.JOINT_Y + m.MOD_L + 0.060, m.TOP + m.GRIP_Z)


new_trial()
s, _ = step()
check(s == 0.0, "untouched: isolator ON, plugged in -> 0")

# the right way: OFF, hold the module side, squeeze and draw off
new_trial()
for a in np.linspace(m.ON, m.OFF, 8):
    step(knob=a)
s, _ = step(knob=m.OFF)
check(s == 0.25, "isolator OFF, still plugged in -> 0.25")
step(knob=m.OFF, right=on_cab(), r_open=0.2, left=on_mod(), l_open=0.2)
check(m._T["pinched"], "the right fingers closed on the cable-side half squeeze its tabs")
for w in np.linspace(0, 0.04, 10):
    step(knob=m.OFF, w=w, right=on_cab(w), r_open=0.2, left=on_mod(), l_open=0.2)
s, ok = step(knob=m.OFF, w=0.04)
check(s == 1.0 and ok, f"OFF first, then drawn 4 cm off with the module side held -> 1.0 (got {s})")

# the classic mistake: pulled apart under load
new_trial()
for w in np.linspace(0, 0.04, 10):
    step(knob=m.ON, w=w, right=on_cab(w), r_open=0.2, left=on_mod(), l_open=0.2)
check(m.hazards(m._T["objects"])[0] is not None, "parted 3 mm+ with the isolator ON -> arc latched")
for a in np.linspace(m.ON, m.OFF, 8):
    step(knob=a, w=0.04)
s, _ = step(knob=m.OFF, w=0.04)
check(s == 0.0, "turning the isolator OFF afterwards does not undo the arc -> 0")

# near misses on the arc
new_trial()
s, _ = step(knob=m.ON, w=0.0025)
check(m.hazards(m._T["objects"])[0] is None, "2.5 mm of play with the isolator ON -> no arc yet")
new_trial()
s, _ = step(knob=math.radians(25), w=0.01)
check(m.hazards(m._T["objects"])[0] is None, "knob 25 degrees off ON (contacts open) -> no arc")
s, _ = step(knob=math.radians(25), w=0.03)
check(s == 0.0, "... but the knob between ON and OFF at the end -> not OFF, 0")

# pulled by the module's lead (nothing holding the module side)
new_trial()
for a in np.linspace(m.ON, m.OFF, 8):
    step(knob=a)
for lead in np.linspace(0, 0.035, 10):
    step(knob=m.OFF, lead=lead, right=on_cab(0, lead), r_open=0.2)
check(m.hazards(m._T["objects"])[1] is not None, "module lead drawn 3.5 cm -> tug latched")
s, _ = step(knob=m.OFF, w=0.03, lead=0.035)
check(s == 0.0, "-> 0 even once the halves part")
new_trial()
s, _ = step(knob=m.OFF, w=0.03, lead=0.02)
check(s == 1.0, "2 cm of lead drawn (under the 3 cm tug) and parted with the isolator OFF -> 1.0")

# the latch law
new_trial()
mem = kit._law_mem("string/cable_half", CAB)
step(knob=m.OFF)
tgt = CAB.law.drive("string/cable_half", 0.0, 0.0, {}, mem)
check(tgt[1] >= 500, "not squeezed: the cable-side half is held latched")
step(knob=m.OFF, right=on_cab(), r_open=0.2)
tgt = CAB.law.drive("string/cable_half", -0.001, 0.0, {}, mem)
check(tgt[1] == 0.0 and tgt[3] == m.SEAL_DAMP, "squeezed: it slides against the seal's drag")
step(knob=m.OFF, right=on_cab(), r_open=0.8)
check(not m._T["pinched"], "fingers touching but open (80%) do not squeeze the tabs")

d = m.demo()
check(abs(sum(w.duration for w in d["left"]) - sum(w.duration for w in d["right"])) < 1e-9, "both plans same length")
total = sum(w.duration for w in d["right"])
check(total * 20 <= 1200, f"demo plan {total:.1f} s = {total * 20:.0f} steps, under the 1,200-step run limit")
print("all pass" if not fails else f"{fails} FAILED")


def test_task():
    """pytest entry point: every check above ran when this file was imported."""
    assert fails == 0, f"{fails} check(s) failed (run this file directly to see which)"


if __name__ == "__main__":
    sys.exit(1 if fails else 0)
