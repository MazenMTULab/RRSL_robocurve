"""Sharpening a lawn mower blade's cutting edge with a hand file. The blade is clamped flat on a wooden block with a
C-clamp, its cutting end toward the arms; the edge's bevel has been colored with a marker, so every spot the file has
cut shows bright steel. A 10-inch mill file lies on the table in front.

A file needs both hands: one at the handle pushing, one near the tip steadying and pressing it onto the work. It cuts
only on the push stroke, toward the tip; it is lifted for the return, since dragging it back across the edge dulls its
teeth. It must ride the edge's 30 degree bevel: held flatter or steeper it grinds a new, wrong bevel. The job is the
whole 6 cm of the edge filed bright.

Kits: yamkit (the clamped blade and block, the loose file, gripper states).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import yamkit as kit  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
INSTRUCTION = "Sharpen the mower blade's cutting edge with the file."

# --- Scene, world coordinates (m): x forward from the arm bases, y to the robot's left, z up from the table top.
# The blade lies flat along x on a wooden block, its cutting end toward the arms; the cutting edge runs along x on the
# blade's +y side, its 30 degree bevel facing up and toward +y.
BLOCK = (0.400, -0.060)  # the wooden block's center
BLOCK_SIZE = (0.220, 0.120, 0.110)  # 22 x 12 x 11 cm
BLADE_T = 0.004  # the blade: 4 mm steel, 5.2 cm wide
BLADE_W = 0.052
BLADE_X = (0.280, 0.720)  # its visible length (the far part runs on past the block)
EDGE_Y = 0.000  # the cutting edge's line
EDGE_Z = BLOCK_SIZE[2] + BLADE_T  # the edge's top corner, above the table
EDGE_X = (0.285, 0.345)  # the 6 cm of cutting edge to sharpen, marked with marker ink
BINS = 6  # scored in 1 cm lengths
BEVEL = math.radians(30.0)
BEVEL_N = np.array([0.0, math.sin(BEVEL), math.cos(BEVEL)])  # the bevel's outward normal

# The file: a 10-inch (25 cm) mill file, 2.5 cm wide, 6 mm thick, on a 10 cm wooden handle. In its own frame x runs
# from the handle toward the tip, z is the face normal. Its origin is the file blade's middle.
FILE_L, FILE_W, FILE_T = 0.250, 0.025, 0.006
HANDLE_L, HANDLE_D = 0.100, 0.026
FILE0 = (0.170, 0.030)  # lying flat on the table, along y, tip toward +y (the robot's left)

# --- Scoring.
CONTACT = 0.0025  # the file's face within 2.5 mm of the edge counts as on it
ANGLE_TOL = math.radians(10.0)  # and its face within 10 degrees of the bevel
MOVE = 0.01  # m/s along the file's length: a stroke
PASSES = 2  # cutting passes a 1 cm length needs to show bright steel
DRAG_CAP = 3  # three or more drags back across the edge cap the score at 0.5
DRAG_MIN = 0.010  # a drag is the file sliding 1 cm or more back along its length while on the edge
GAP_STEPS = 4  # a stroke that leaves a length for under 0.2 s (contact chatter) is still the same stroke

WOOD = kit.Look((0.66, 0.50, 0.32), 0.8)
STEEL = kit.Look((0.40, 0.42, 0.44), 0.4, metallic=0.8)
INK = kit.Look((0.10, 0.25, 0.75), 0.6)
CLAMP = kit.Look((0.80, 0.20, 0.10), 0.5, metallic=0.4)
FILE_STEEL = kit.Look((0.30, 0.31, 0.33), 0.7, metallic=0.6)
HANDLE = kit.Look((0.55, 0.35, 0.18), 0.6)


def _blade_on_block():
    """The block, the blade on it and the C-clamp holding it (all fixed); marker ink along the 6 cm of edge."""
    o = np.array([BLOCK[0], BLOCK[1], 0.0])
    bx = (BLADE_X[0] + BLADE_X[1]) / 2 - o[0]
    by = EDGE_Y - BLADE_W / 2 - o[1]
    m = [kit.box((0, 0, BLOCK_SIZE[2] / 2), BLOCK_SIZE, WOOD),
         kit.box((bx, by, BLOCK_SIZE[2] + BLADE_T / 2), (BLADE_X[1] - BLADE_X[0], BLADE_W, BLADE_T), STEEL),
         kit.box(((EDGE_X[0] + EDGE_X[1]) / 2 - o[0], EDGE_Y - 0.004 - o[1], EDGE_Z + 0.0002),
                 (EDGE_X[1] - EDGE_X[0], 0.008, 0.0004), INK)]
    for k in range(BINS + 1):  # 1 cm tick marks beside the inked edge
        x = EDGE_X[0] + k * (EDGE_X[1] - EDGE_X[0]) / BINS - o[0]
        m.append(kit.box((x, EDGE_Y - 0.012 - o[1], EDGE_Z + 0.0002), (0.001, 0.006, 0.0004), INK))
    cx = 0.090  # the C-clamp across the blade on the block's far half
    m += [kit.box((cx, by, EDGE_Z + 0.012), (0.030, 0.090, 0.016), CLAMP),
          kit.box((cx, by - 0.050, BLOCK_SIZE[2] / 2), (0.030, 0.012, BLOCK_SIZE[2] + 0.030), CLAMP)]
    c = [kit.Collider((0, 0, BLOCK_SIZE[2] / 2), size=BLOCK_SIZE),
         kit.Collider((bx, by, BLOCK_SIZE[2] + BLADE_T / 2), size=(BLADE_X[1] - BLADE_X[0], BLADE_W, BLADE_T)),
         kit.Collider((cx, by, EDGE_Z + 0.012), size=(0.030, 0.090, 0.016))]
    return kit.Prop("blade_on_block", pos=(*BLOCK, TOP), size=BLOCK_SIZE, meshes=tuple(m), colliders=tuple(c))


def _file():
    """The mill file (loose): its blade along local x, the handle on its -x end."""
    hx = -FILE_L / 2 - HANDLE_L / 2
    m = [kit.box((0, 0, 0), (FILE_L, FILE_W, FILE_T), FILE_STEEL),
         kit.box((FILE_L / 2 - 0.01, 0, 0), (0.02, FILE_W * 0.7, FILE_T * 0.8), FILE_STEEL),
         kit.cylinder((hx, 0, 0), HANDLE_D / 2, HANDLE_L, HANDLE, axis="x")]
    c = [kit.Collider((0, 0, 0), size=(FILE_L, FILE_W, FILE_T)),
         kit.Collider((hx, 0, 0), size=(HANDLE_L, HANDLE_D, HANDLE_D))]
    return kit.Prop("file", pos=(*FILE0, TOP + HANDLE_D / 2), yaw=math.pi / 2, size=(FILE_L + HANDLE_L, HANDLE_D, HANDLE_D),
                    mass=0.30, meshes=tuple(m), colliders=tuple(c))


OBJECTS = [_blade_on_block(), _file()]


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


BIN_X = [EDGE_X[0] + (k + 0.5) * (EDGE_X[1] - EDGE_X[0]) / BINS for k in range(BINS)]


def file_on_edge(objects):
    """Which 1 cm lengths of the edge the file's face is on now (within 2.5 mm, inside the face's width and the cut
    length, the face within 10 degrees of the bevel), and the file's speed along its length (+: toward its tip)."""
    f = objects["file"]
    R = _rot(f.quat_xyzw)
    p0 = np.asarray(f.pos, float)
    n = R[:, 2]
    if np.dot(n, BEVEL_N) < 0:
        n = -n  # either face of the file cuts
    ang = math.acos(min(1.0, float(np.dot(n, BEVEL_N))))
    v_ax = float(np.dot(np.asarray(f.lin_vel, float), R[:, 0]))
    on = []
    if ang <= ANGLE_TOL:
        for k, x in enumerate(BIN_X):
            e = np.array([x, EDGE_Y, TOP + EDGE_Z])
            loc = R.T @ (e - p0)
            d = float(np.dot(n, e - p0)) + FILE_T / 2  # the edge's distance off the face (0 = touching)
            if abs(d) <= CONTACT and abs(loc[1]) <= FILE_W / 2 and -FILE_L / 2 <= loc[0] <= FILE_L / 2 - 0.02:
                on.append(k)
    return on, ang, v_ax


_T = {"episode": None, "objects": None, "step": 0, "passes": None, "drags": 0, "last_cut": None, "last_drag": -99,
      "drag_s0": None, "drag_counted": False}


def _track(objects):
    if _T["episode"] != kit.episode():
        _T.update(episode=kit.episode(), objects=None, step=0, passes=[0] * BINS, drags=0, last_cut=[-99] * BINS,
                  last_drag=-99, drag_s0=None, drag_counted=False)
    if objects is _T["objects"]:
        return
    _T["objects"] = objects
    _T["step"] += 1
    on, ang, v = file_on_edge(objects)
    n = _T["step"]
    if v > MOVE:
        for k in on:
            if n - _T["last_cut"][k] > GAP_STEPS:
                _T["passes"][k] += 1  # a new cutting pass over this length
            _T["last_cut"][k] = n
    f = objects["file"]
    s_ax = float(np.dot(np.asarray(f.pos, float), _rot(f.quat_xyzw)[:, 0]))  # the file's place along its length
    if on and v < -MOVE:
        if _T["drag_s0"] is None or n - _T["last_drag"] > GAP_STEPS:
            _T.update(drag_s0=s_ax, drag_counted=False)  # a backward slide on the edge starts here
        _T["last_drag"] = n
        if not _T["drag_counted"] and _T["drag_s0"] - s_ax >= DRAG_MIN:
            _T["drags"] += 1  # it has slid 1 cm back across the edge: a drag
            _T["drag_counted"] = True
    elif v > MOVE:
        _T["drag_s0"] = None  # pushing forward again ends any backward slide
    if os.environ.get("FILE_DEBUG"):
        f = objects["file"]
        print(f"FILE {_T['step']} pos={np.round(f.pos, 3)} ang={math.degrees(ang):.1f} v={v:.3f} on={on} "
              f"passes={_T['passes']} drags={_T['drags']} L={np.round(objects['left_gripper'].pos, 3)} "
              f"R={np.round(objects['right_gripper'].pos, 3)}", flush=True)


def filed(objects):
    """How many of the six 1 cm lengths have had 2 or more cutting passes."""
    _track(objects)
    return sum(1 for p in _T["passes"] if p >= PASSES)


def drags(objects):
    _track(objects)
    return _T["drags"]


def _job(objects):
    n = filed(objects)
    if n == BINS:
        s = 1.0
    elif n >= BINS // 2:
        s = 0.5
    elif n >= 1:
        s = 0.25
    else:
        s = 0.0
    if _T["drags"] >= DRAG_CAP:
        s = min(s, 0.5)
    return s


def success(objects):
    """All six 1 cm lengths of the edge filed (2 or more cutting passes each, on the bevel), with at most 2 drags."""
    return _job(objects) == 1.0


def score(objects):
    """A cutting pass over a 1 cm length of the edge: the file's face on it (within 2.5 mm, within 10 degrees of the
    30 degree bevel) while the file moves toward its tip faster than 1 cm/s. A length is filed after 2 passes. 1.0
    with all 6 lengths filed; 0.5 with 3 to 5; 0.25 with 1 or 2; 0 with none. A drag (the file on the edge sliding 1 cm or
    more back toward its handle, faster than 1 cm/s) 3 or more times caps the score at 0.5. Possible scores: 0, 0.25, 0.5, 1.0."""
    return _job(objects)


# --- Demo: both arms lift the file (right on the handle, left near the tip), tilt it onto the bevel and take six
# push strokes across the edge at three places along it, lifting for each return; then set it down.
U = np.array([0.0, math.cos(BEVEL), -math.sin(BEVEL)])  # the file's length on the bevel, handle to tip (down-slope)
S_R, S_L = -FILE_L / 2 - 0.030, FILE_L / 2 - 0.035  # where each hand holds it, along the file from its origin
STROKE = 0.060


def _hands(contact_x, s_contact, press=0.002, lift=0.0, tip_up=0.0):
    """Tool points for the file lying on the bevel with the edge at `s_contact` along it from its origin, `press`
    below the touching position (pushed onto the edge) or `lift` above it; `tip_up` raises the tip hand alone (the
    file tilted up off the edge, pivoting near the handle, for the return stroke)."""
    e = np.array([contact_x, EDGE_Y, TOP + EDGE_Z])
    o = e - U * s_contact + BEVEL_N * (FILE_T / 2 - press + lift)  # the file's origin
    right = o + U * S_R
    d = (S_L - S_R) * U  # right hand to left hand along the file
    if tip_up:  # swing the tip hand up round the right hand, keeping the hands' spacing on the file
        L = float(np.linalg.norm(d))
        dz = min(-1e-3, d[2] + tip_up)
        d = np.array([0.0, math.sqrt(max(0.0, L * L - dz * dz)), dz])
    return tuple(right), tuple(right + d)


def demo():
    """Right arm on the handle, left arm on the file near its tip, fingers closing across it (along x). (DEMO_DRAG=1:
    a negative control that drags the file back across the edge on every return.)"""
    drag = bool(os.environ.get("DEMO_DRAG"))
    yaw = math.pi / 2  # fingers closing along x, across the file
    z0 = TOP + HANDLE_D / 2
    r0 = (FILE0[0], FILE0[1] + S_R, z0)
    l0 = (FILE0[0], FILE0[1] + S_L, TOP + FILE_T / 2 + 0.010)
    gz = 0.004  # tool point above an object's middle
    right = [Waypoint(2.0, (r0[0], r0[1], r0[2] + 0.07), FINGER_OPEN * 0.6, "over the handle", yaw),
             Waypoint(1.0, (r0[0], r0[1], r0[2] + gz), FINGER_OPEN * 0.6, "fingers round the handle", yaw),
             Waypoint(0.8, (r0[0], r0[1], r0[2] + gz), FINGER_CLOSED, "grip the handle", yaw)]
    left = [Waypoint(2.0, (l0[0], l0[1], l0[2] + 0.07), FINGER_OPEN * 0.5, "over the file's tip", -yaw),
            Waypoint(1.0, (l0[0], l0[1], l0[2]), FINGER_OPEN * 0.5, "fingers round the file", -yaw),
            Waypoint(0.8, (l0[0], l0[1], l0[2]), FINGER_CLOSED, "hold it near the tip", -yaw)]
    rl, ll = _hands(EDGE_X[0] + 0.008, -0.02, lift=0.015)
    right.append(Waypoint(2.4, rl, FINGER_CLOSED, "lift it onto the bevel's angle", yaw))
    left.append(Waypoint(2.4, ll, FINGER_CLOSED, "lift it onto the bevel's angle", -yaw))
    for k, x in enumerate((0.293, 0.293, 0.315, 0.315, 0.337, 0.337)):
        s0, s1 = -0.020, -0.020 - STROKE  # the edge's place along the file, start and end of the push
        for (dur, s, press, tip, lab) in ((0.6, s0, 0.0, 0.035, "back over the edge, tip raised"),
                                          (0.6, s0, 0.002, 0.0, "onto the edge"),
                                          (1.2, s1, 0.002, 0.0, "push stroke toward the tip")):
            r, l = _hands(x, s, press, 0.004 if tip else 0.0, tip)
            right.append(Waypoint(dur, r, FINGER_CLOSED, lab, yaw))
            left.append(Waypoint(dur, l, FINGER_CLOSED, lab, -yaw))
        if drag:  # drag it back along the edge
            r, l = _hands(x, s0, 0.002, 0.0)
            right.append(Waypoint(0.8, r, FINGER_CLOSED, "drag back", yaw))
            left.append(Waypoint(0.8, l, FINGER_CLOSED, "drag back", -yaw))
        else:
            r, l = _hands(x, s1, 0.0, 0.004, 0.035)
            right.append(Waypoint(0.4, r, FINGER_CLOSED, "tilt it off the edge for the return", yaw))
            left.append(Waypoint(0.4, l, FINGER_CLOSED, "tilt it off the edge for the return", -yaw))
    right += [Waypoint(2.0, (r0[0], r0[1], r0[2] + 0.06), FINGER_CLOSED, "carry it back", yaw),
              Waypoint(1.0, (r0[0], r0[1], r0[2] + gz + 0.004), FINGER_CLOSED, "set it down", yaw),
              Waypoint(0.6, (r0[0], r0[1], r0[2] + gz + 0.004), FINGER_OPEN * 0.6, "let go", yaw),
              Waypoint(0.8, (r0[0], r0[1], r0[2] + 0.07), FINGER_OPEN * 0.6, "lift clear", yaw),
              kit.Home(1.8), kit.Home(1.0, label="stay home")]
    left += [Waypoint(2.0, (l0[0], l0[1], l0[2] + 0.06), FINGER_CLOSED, "carry it back", -yaw),
             Waypoint(1.0, (l0[0], l0[1], l0[2] + 0.004), FINGER_CLOSED, "set it down", -yaw),
             Waypoint(0.6, (l0[0], l0[1], l0[2] + 0.004), FINGER_OPEN * 0.5, "let go", -yaw),
             Waypoint(0.8, (l0[0], l0[1], l0[2] + 0.07), FINGER_OPEN * 0.5, "lift clear", -yaw),
             kit.Home(1.8), kit.Home(1.0, label="stay home")]
    return {"left": left, "right": right}
