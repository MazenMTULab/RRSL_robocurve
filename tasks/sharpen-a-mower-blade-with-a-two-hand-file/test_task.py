"""Offline checks of the blade-sharpening scorer: file poses and velocities stepped through the pass counter and the
scorer, near misses included, plus the demo's plan. No simulator.

    python -B tasks/sharpen-a-mower-blade-with-a-two-hand-file/test_task.py      (prints "all pass")
"""

import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("file_task", os.path.join(HERE, "task.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
kit = m.kit
from yam_arena.task_file import ObjectState  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def quat_from_R(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])


def file_R(tilt=m.BEVEL):
    """The file along y (handle at -y), its face tilted about x by `tilt` (down-slope toward +y)."""
    xa = np.array([0.0, math.cos(tilt), -math.sin(tilt)])
    za = np.array([0.0, math.sin(tilt), math.cos(tilt)])
    ya = np.cross(za, xa)
    return np.stack([xa, ya, za], axis=1)


def st(pos, q, v=np.zeros(3)):
    p = np.array(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.asarray(q, float), lin_vel=np.asarray(v, float), ang_vel=np.zeros(3),
                       low=p - 0.01, high=p + 0.01)


G = {"left_gripper": (0.2, 0.2, m.TOP + 0.2), "right_gripper": (0.2, -0.2, m.TOP + 0.2)}


def new_trial():
    kit._EPISODE["count"] += 1


def step(x, s, speed, tilt=m.BEVEL, gap=0.0):
    """The file on the edge at x, the edge `s` along the file from its origin, moving `speed` toward its tip."""
    R = file_R(tilt)
    n = R[:, 2]
    e = np.array([x, m.EDGE_Y, m.TOP + m.EDGE_Z])
    o = e - R[:, 0] * s + n * (m.FILE_T / 2 + gap)
    objs = {"file": st(o, quat_from_R(R), R[:, 0] * speed),
            "blade_on_block": st((*m.BLOCK, m.TOP), (0, 0, 0, 1)),
            **{k: st(v, (1, 0, 0, 0)) for k, v in G.items()}}
    return m.score(objs), m.success(objs)


def stroke(x, tilt=m.BEVEL, gap=0.0, speed=0.05, back="lift"):
    out = None
    for s in np.linspace(-0.02, -0.08, 8):
        out = step(x, s, speed, tilt, gap)
    for s in np.linspace(-0.08, -0.02, 5):
        out = step(x, s, -speed, tilt, gap if back == "drag" else 0.02)
    return out


new_trial()
s, _ = step(0.30, 0.0, 0.0, gap=0.05)
check(s == 0.0, "untouched -> 0")

new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, ok = stroke(x)
check(s == 1.0 and ok, f"six push strokes over the whole edge on the bevel, lifted for the returns -> 1.0 (got {s})")
check(m._T["passes"] == [2] * 6, f"two passes on every 1 cm length {m._T['passes']}")
check(m.drags(m._T["objects"]) == 0, "no drags")

new_trial()
for x in (0.293, 0.293, 0.315, 0.315):
    s, _ = stroke(x)
check(s == 0.5, "four lengths filed -> 0.5")
new_trial()
for x in (0.293, 0.293):
    s, _ = stroke(x)
check(s == 0.25, "two lengths filed -> 0.25")
new_trial()
stroke(0.293)
s, _ = stroke(0.315)
check(s == 0.0, "one pass on each length (none filed twice) -> 0")

# dragging back across the edge
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, _ = stroke(x, back="drag")
check(m.drags(m._T["objects"]) == 6 and s == 0.5, f"every return dragged back (6 drags) -> capped at 0.5 (got {s})")
new_trial()
for k, x in enumerate((0.293, 0.293, 0.315, 0.315, 0.337, 0.337)):
    s, _ = stroke(x, back="drag" if k < 2 else "lift")
check(s == 1.0, "two drags only -> still 1.0")
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    stroke(x)
    for s_ in np.linspace(-0.020, -0.014, 4):  # settling back a few mm on the edge before the next push
        s, _ = step(x, s_, -0.03)
check(m.drags(m._T["objects"]) == 0 and s == 1.0, f"a few mm of settling back on the edge is not a drag (got {s})")

# the wrong angle
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, _ = stroke(x, tilt=math.radians(5))
check(s == 0.0, "file held almost flat (5 degrees) -> grinds the wrong bevel, nothing counts")
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, _ = stroke(x, tilt=math.radians(38))
check(s == 1.0, "38 degrees (within 10 of the bevel) -> counts")
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, _ = stroke(x, gap=0.006)
check(s == 0.0, "file 6 mm off the edge (air strokes) -> nothing counts")

# slow creeping does not count as a stroke
new_trial()
for x in (0.293, 0.293, 0.315, 0.315, 0.337, 0.337):
    s, _ = stroke(x, speed=0.005)
check(s == 0.0, "the file crept along at 5 mm/s -> no cutting strokes")

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
