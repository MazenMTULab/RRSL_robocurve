"""Cleft grafting an apple rootstock on the bench. A 38 mm apple stub (2.5 mm bark) is held upright in a low bench
vise, its top cut flat 65 mm above the bench and already split across the middle: the cleft runs left-right (along y)
and is 40 mm deep. The cleft is wood under spring: left alone it closes. The wedge prong of a steel cleft-grafting
tool has been driven into the middle of the cleft and holds it about 10 mm open at the top; the tool's bar lies
toward the robot, about 2 cm above the stub. To the robot's front left and front right a wooden stand holds two
dormant apple scions upright (9 mm, 85 mm long, 1 mm bark), each with its bottom 32 mm already cut to a long wedge,
the wedge's cut faces facing front and back.

The job: set one scion at each end of the cleft (left end and right end), wedge down, its top of cut about level with
the stub's cut face, and pushed in from the stub's edge just far enough that the scion's cambium (the green layer
under its bark) lines up with the stub's cambium. Because the stub's bark is thicker than the scion's, the scion's
bark sits slightly inside the stub's bark (about 0.5-2.5 mm), not flush with it and not deeper in. Then pull the
tool's wedge out, so the cleft springs shut and clamps both scions. Zero object jitter; 20 Hz.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Hinge, Link  # noqa: E402
from yam_mesh import Mesh, MeshMechanism, MeshPart  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"

RELEASE_CLEAR = 0.030     # released: every gripper TCP at least 3 cm from the scion's box
STILL_S = 1.0
TEST_GRIPPERS = None


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run"):
        return cls._rc_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}
_POSE_PREV = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        joints = art.data.joint_pos
        q = getattr(joints, "torch", joints)[0]
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q)
        tcp = ik.tcp_pose(q[arm_ids], station.BASES[arm])[0]
        out[arm] = (tcp, float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0)))
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _POSE_PREV.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls are harmless."""
    _episode_memory()
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    previous_step, start = _TIMERS.get(key, (None, None))
    if not cond:
        _TIMERS[key] = (step, None)
        return 0.0
    if start is None or step not in (previous_step, previous_step + 1) or t < start:
        start = t
    _TIMERS[key] = (step, start)
    return float(t - start)


def pose_quiet(key, points, tol=0.0005):
    """Still: every given point moved under 0.5 mm since the previous control step (pose differences, because
    contact chatter makes the simulator's reported velocities unreliable). Offline it is always true."""
    if _RUN["t"] is None:
        return True
    _episode_memory()
    step = _RUN["step"]
    pts = np.asarray(points, float)
    prev = _POSE_PREV.get(key)
    if prev is not None and prev[0] == step:
        return prev[2]
    quiet = bool(prev is not None and prev[0] == step - 1 and np.all(np.abs(pts - prev[1]) < tol))
    _POSE_PREV[key] = (step, pts, quiet)
    return quiet


def _convex(name, points, color, proxy):
    """A convex mesh from its corner points (outward-facing hull triangles)."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(points, float)
    pts = pts[ConvexHull(pts).vertices]          # drop points inside the hull (the kit rejects unused vertices)
    hull = ConvexHull(pts)
    centre = pts.mean(0)
    faces = []
    for tri in hull.simplices:
        a, b, c = pts[tri]
        if np.dot(np.cross(b - a, c - a), a - centre) < 0:
            tri = tri[::-1]
        faces.append(tuple(int(i) for i in tri))
    return Mesh(name, pts, np.array(faces, int), collision="convex", color=color, proxy=proxy)


# ------------------------------------------------------------------ layout (metres, world frame)
TOP = TABLE_TOP[SCENE]
BARK, WOOD, STEEL, GREEN = (0.42, 0.30, 0.20), (0.86, 0.78, 0.58), (0.50, 0.52, 0.55), (0.40, 0.36, 0.22)
VISE = (0.23, 0.24, 0.27)

# Rootstock (part frame: origin at the centre of the stub's cut face; axes as world). The cleft is the plane x = 0.
# Below the cleft's root the stub is held by the vise; above it the two halves (near: x < 0, far: x > 0) each spring
# about the root, as the two sides of a split stem do.
STOCK = np.array([0.220, 0.000, TOP + 0.065])
STOCK_R = 0.019                 # 38 mm (1.5 in) stub
STOCK_BARK = 0.0025             # stub bark 2.5 mm: its cambium is 16.5 mm from the axis
STOCK_H = 0.065                 # cut face 65 mm above the bench
CLEFT_D = 0.040                 # split 40 mm deep
SPLIT_K = 2.0                   # N m/rad per half about the cleft's root (see README: scaled down from real wood)
SPLIT_REST = 3.0                # degrees past closed: each half is preloaded shut against its stop
SPLIT_MASS = 0.020
HALVES = ("near_half", "far_half")


def _half(name, a0, a1, color):
    pts = []
    for a in np.linspace(math.radians(a0), math.radians(a1), 25):
        for z in (-CLEFT_D, 0.0):
            pts.append((STOCK_R * math.cos(a), STOCK_R * math.sin(a), z))
    lo = np.min(pts, 0)
    hi = np.max(pts, 0)
    return _convex(name, pts, color, proxy=(box(tuple((lo + hi) / 2), tuple(hi - lo)),))


_VJ = 0.020                     # vise jaws 20 mm tall
STOCK_BASE = [
    cyl((0.0, 0.0, -(CLEFT_D + (STOCK_H - CLEFT_D) / 2)), STOCK_R, STOCK_H - CLEFT_D, color=BARK),
    box((0.0, 0.0, -STOCK_H + 0.003), (0.110, 0.070, 0.006), VISE),                       # vise bed
    box((STOCK_R + 0.007, 0.0, -STOCK_H + 0.006 + _VJ / 2), (0.014, 0.070, _VJ), VISE),  # far jaw
    box((-STOCK_R - 0.007, 0.0, -STOCK_H + 0.006 + _VJ / 2), (0.014, 0.070, _VJ), VISE),  # near jaw
]

# Cleft-grafting tool: a steel bar with a wedge prong bent down. Tool frame: origin at the top of the prong, on the
# cleft plane; axes as world; the bar runs toward the robot (-x).
T_TOP = 0.012                   # prong 12 mm thick at its top...
PRONG_L = 0.025                 # ...25 mm long...
PRONG_SLOPE = math.tan(math.radians(12.0))   # ...each face sloped 12 degrees: tip 1.4 mm
T_TIP = T_TOP - 2 * PRONG_L * PRONG_SLOPE
PRONG_W = 0.012
TOOL_MASS = 0.110
BAR_Z = (0.014, 0.024)          # bar 10 mm deep, 10 mm wide
BAR_X = (-0.075, 0.030)
GRIP_X = -0.040                 # where the demo grips the bar


def _tool_prong():
    pts = [(x, y, z) for y in (-PRONG_W / 2, PRONG_W / 2)
           for (x, z) in ((-T_TOP / 2, 0.0), (T_TOP / 2, 0.0), (-T_TIP / 2, -PRONG_L), (T_TIP / 2, -PRONG_L))]
    return _convex("prong", pts, STEEL, proxy=(box((0.0, 0.0, -PRONG_L / 2), (T_TOP, PRONG_W, PRONG_L)),))


TOOL_SHAPES = [
    box((0.0, 0.0, BAR_Z[0] / 2), (T_TOP, PRONG_W, BAR_Z[0]), STEEL),                       # riser
    box(((BAR_X[0] + BAR_X[1]) / 2, 0.0, sum(BAR_Z) / 2), (BAR_X[1] - BAR_X[0], 0.010, BAR_Z[1] - BAR_Z[0]), STEEL),
]

# Initial opening: each half rests on one face of the prong. A half's inner top edge, turned PHI0 about the root,
# sits at (CLEFT_D sin, CLEFT_D cos - CLEFT_D); the prong's height is chosen so its faces pass through those edges.
PHI0 = 7.2                      # degrees per half: about 10 mm open at the cut face
_EX = CLEFT_D * math.sin(math.radians(PHI0))
_EZ = CLEFT_D * math.cos(math.radians(PHI0)) - CLEFT_D
TOOL_H0 = _EZ + (T_TOP / 2 - _EX) / PRONG_SLOPE + 0.0002
TOOL_START = (float(STOCK[0]), float(STOCK[1]), float(STOCK[2] + TOOL_H0))


def _hinge(sign):
    """sign +1: the far half, opening about +y; -1: the near half."""
    lo, hi = (0.0, 25.0) if sign > 0 else (-25.0, 0.0)
    return Hinge("y", (0.0, 0.0, -CLEFT_D), lo, hi, damping=0.02, spring=SPLIT_K, rest=-sign * SPLIT_REST,
                 initial=sign * PHI0, armature=1e-5)


STOCK_PART = MeshMechanism(
    "stock", tuple(map(float, STOCK)), STOCK_BASE,
    [Link("near_half", (), _hinge(-1), mass=SPLIT_MASS, color=BARK, friction=(0.5, 0.45)),
     Link("far_half", (), _hinge(+1), mass=SPLIT_MASS, color=BARK, friction=(0.5, 0.45))],
    mass=None, color=BARK, friction=(0.5, 0.45), contact_offset=0.001,
    meshes={"near_half": (_half("near_half_wood", 90.0, 270.0, BARK),),
            "far_half": (_half("far_half_wood", -90.0, 90.0, BARK),)})

TOOL = MeshPart("tool", TOOL_START, TOOL_SHAPES, mass=TOOL_MASS, color=STEEL, friction=(0.35, 0.3),
                contact_offset=0.001, meshes=(_tool_prong(),))

# Scions (scion frame: origin on the scion's axis at the top of its wedge cut; z up along the scion).
SCION_R = 0.0045                # 9 mm
SCION_BARK = 0.0010             # 1 mm bark: cambium 3.5 mm from the axis
CUT_L = 0.032                   # wedge cut 32 mm (1.25 in) long
ABOVE = 0.053                   # 53 mm above the cut: 85 mm overall
TIP_T = 2 * SCION_R * (1 - CUT_L / CLEFT_D)   # 1.8 mm at the tip: the wedge tapers as the closed cleft does
SCION_MASS = 0.006


def _scion_wedge():
    pts = []
    for z, t in ((0.0, 2 * SCION_R), (-CUT_L, TIP_T)):
        for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
            x, y = SCION_R * math.cos(a), SCION_R * math.sin(a)
            pts.append((float(np.clip(x, -t / 2, t / 2)), y, z))
    pts = list(dict.fromkeys(pts))
    return _convex("wedge", pts, WOOD, proxy=(box((0.0, 0.0, -CUT_L / 2), (2 * SCION_R, 2 * SCION_R, CUT_L)),))


STAND_FLOOR = 0.007
SOCKET = 0.011                  # 11 mm square sockets, 18 mm deep
STAND_TOP = 0.025
SOCKETS = {"scion_left": (0.175, 0.115), "scion_right": (0.175, -0.115)}


def _socket_walls(cx, cy):
    w, s, h = 0.006, SOCKET / 2, STAND_TOP - STAND_FLOOR
    zc = TOP + STAND_FLOOR + h / 2
    return [box((cx, cy, TOP + STAND_FLOOR / 2), (SOCKET + 2 * w, SOCKET + 2 * w, STAND_FLOOR), WOOD),
            box((cx + s + w / 2, cy, zc), (w, SOCKET + 2 * w, h), WOOD),
            box((cx - s - w / 2, cy, zc), (w, SOCKET + 2 * w, h), WOOD),
            box((cx, cy + s + w / 2, zc), (SOCKET, w, h), WOOD),
            box((cx, cy - s - w / 2, zc), (SOCKET, w, h), WOOD)]


STAND = MeshPart("stand", (0.0, 0.0, 0.0), [b for c in SOCKETS.values() for b in _socket_walls(*c)], color=WOOD,
                 friction=(0.5, 0.45), contact_offset=0.001)


def _scion(name):
    cx, cy = SOCKETS[name]
    pos = (cx, cy, TOP + STAND_FLOOR + CUT_L + 0.0005)
    return MeshPart(name, pos, [cyl((0.0, 0.0, ABOVE / 2), SCION_R, ABOVE, color=GREEN)], mass=SCION_MASS,
                    color=GREEN, friction=(0.5, 0.45), contact_offset=0.0008, meshes=(_scion_wedge(),))


SCIONS = ("scion_left", "scion_right")
OBJECTS = [STOCK_PART, TOOL, STAND] + [_scion(n) for n in SCIONS]

# ------------------------------------------------------------------ scoring
INSET = (0.0005, 0.0025)        # scion bark 0.5-2.5 mm inside the stub's bark (cambium lines within 1 mm)
TOP_OF_CUT = (-0.005, 0.005)    # top of the wedge cut within 5 mm of the stub's cut face
TILT_DEG = 15.0                 # scion within 15 degrees of upright
IN_CLEFT_X = (-0.004, 0.004)    # its axis in the cleft (between the halves) at the cut face
TIP_DEPTH = 0.020               # and its wedge at least 20 mm down in it
TOOL_OUT = 0.002                # the prong's tip at least 2 mm above the cut face (or the tool well away)
TEST_STICKY = ()


def scion_fit(scion, side):
    """The scion against the stub: where its axis crosses the stub's cut face (stub frame), its bark inset at that
    end of the cleft, the height of its top of cut, its tilt and how deep its wedge tip is."""
    o = np.asarray(scion.origin, float)
    axis = scion.rotation()[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))
    u = (STOCK[2] - o[2]) / axis[2] if abs(axis[2]) > 0.2 else 0.0
    p = o + u * axis - STOCK
    tip = np.asarray(scion.to_world((0.0, 0.0, -CUT_L)), float) - STOCK
    inset = STOCK_R - (side * p[1] + SCION_R)
    return {"x": float(p[0]), "y": float(p[1]), "inset": float(inset), "top_of_cut": float(o[2] - STOCK[2]),
            "tilt": tilt, "tip_depth": float(-tip[2])}


def scion_ok(f):
    return (INSET[0] <= f["inset"] <= INSET[1] and TOP_OF_CUT[0] <= f["top_of_cut"] <= TOP_OF_CUT[1]
            and f["tilt"] <= TILT_DEG and IN_CLEFT_X[0] <= f["x"] <= IN_CLEFT_X[1] and f["tip_depth"] >= TIP_DEPTH)


def tool_out(tool):
    tip = np.asarray(tool.to_world((0.0, 0.0, -PRONG_L)), float)
    away = math.hypot(tip[0] - STOCK[0], tip[1] - STOCK[1]) > 0.040
    return bool(tip[2] - STOCK[2] >= TOOL_OUT or away)


def score(objects):
    """Per scion set right (bark 0.5-2.5 mm inside the stub's bark at its own end of the cleft, top of cut within
    5 mm of the cut face, upright within 15 degrees, wedge in the cleft) and let go of: 0.25 while the tool's wedge
    is still in the cleft; 0.5 once it is out and the cleft has closed on the scions. Only while every gripper is
    3 cm clear of the scions and both scions have been still for 1 s."""
    sides = {"scion_left": 1.0, "scion_right": -1.0}
    n_ok, clear, pts = 0, True, []
    for name in SCIONS:
        s = objects[name]
        f = scion_fit(s, sides[name])
        n_ok += scion_ok(f)
        clear = clear and released(s)
        pts += [s.origin, s.to_world((0.0, 0.0, ABOVE))]
    still = pose_quiet("scions", pts)
    out = tool_out(objects["tool"])
    seconds = held_for("released and still", clear and still)
    if seconds + 1e-9 < STILL_S:
        return 0.0
    return (0.5 if out else 0.25) * n_ok


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ closed-loop scripted demo (cabledemo.py)
TIP = 0.0128                    # fingertips 12.8 mm past the TCP
GRIP_BELOW_TOP = 0.012          # pads' tips 12 mm below a scion's top
SCION_OPEN, SCION_PINCH = -0.013, -0.0025
GAP_MID_X = 0.0                 # the cleft's middle plane
TARGET_Y = STOCK_R - 0.0015 - SCION_R      # 1.5 mm bark inset: 13.0 mm off the stub's axis
SET_TOP = -0.0015               # lower each scion's top of cut to 1.5 mm below the cut face, then let go
SERVO_GAIN = 0.5
TEST_TARGET_Y = None            # cheat variants override these
TEST_SKIP = ()
TEST_HOLD = ()                  # scions whose arm never lets go (a check variant)


def demo_script(ctx):
    target_y = TARGET_Y if TEST_TARGET_Y is None else TEST_TARGET_Y
    hands = {"scion_left": (ctx.left, "left", +1.0), "scion_right": (ctx.right, "right", -1.0)}

    def actual_tcp(arm_name, arm):
        h = grippers()
        return np.asarray(h[arm_name][0], float) if h else np.asarray(arm.tcp())

    def set_scion(name):
        arm, arm_name, side = hands[name]
        bias = np.zeros(3)

        def go(tcp_target, duration, finger=None, servo=True):
            nonlocal bias
            yield from arm.move(np.asarray(tcp_target) + bias, yaw=0.0, finger=finger, duration=duration)
            if servo:
                bias = np.clip(bias + SERVO_GAIN * (np.asarray(tcp_target) - actual_tcp(arm_name, arm)), -0.03, 0.03)

        s0 = ctx.states()[name]
        top = np.asarray(s0.to_world((0.0, 0.0, ABOVE)), float)
        grip_at = top + [0.0, 0.0, -GRIP_BELOW_TOP + TIP]
        yield from go(grip_at + [0, 0, 0.050], 2.0, finger=SCION_OPEN)
        yield from go(grip_at + [0, 0, 0.010], 1.0)
        yield from go(grip_at, 0.8)
        yield from go(grip_at, 0.3)
        yield from arm.grip(SCION_PINCH, duration=0.8)
        yield from arm.wait(0.3)
        held = ctx.states()[name]
        off = actual_tcp(arm_name, arm) - np.asarray(held.origin)

        def carry(origin_target, duration):
            nonlocal bias
            yield from arm.move(np.asarray(origin_target) + off + bias, yaw=0.0, duration=duration)
            err = np.asarray(origin_target) - np.asarray(ctx.states()[name].origin)
            bias = np.clip(bias + SERVO_GAIN * err, -0.03, 0.03)

        start = np.asarray(held.origin)
        yield from carry(start + [0, 0, 0.040], 1.2)
        goal = STOCK + [GAP_MID_X, side * target_y, SET_TOP]
        above = goal + [0.0, 0.0, CUT_L + 0.012]
        yield from carry(above, 2.0)
        for _ in range(8):
            yield from carry(above, 0.2)
        n = 16
        for k in range(1, n + 1):
            yield from carry(above + (goal - above) * k / n, 0.2)
        for _ in range(30):                     # settle on the measured position: radially within 0.2 mm
            o = np.asarray(ctx.states()[name].origin)
            if abs(o[1] - goal[1]) < 0.0002 and abs(o[2] - goal[2]) < 0.0015:
                break
            yield from carry(goal, 0.15)
        yield from arm.wait(0.3)
        if name in TEST_HOLD:
            return
        yield from arm.grip(SCION_OPEN, duration=0.6)
        yield from arm.move(actual_tcp(arm_name, arm) + [0, 0, 0.050], yaw=0.0, duration=1.0)
        park = np.array([0.12, side * 0.20, TOP + 0.16])
        yield from arm.move(park, yaw=0.0, duration=1.2)

    def pull_tool():
        R = ctx.right
        tool = ctx.states()["tool"]
        bar = np.asarray(tool.to_world((GRIP_X, 0.0, (BAR_Z[0] + BAR_Z[1]) / 2)), float)
        grip_at = bar + [0.0, 0.0, -0.002 + TIP]
        yield from R.move(grip_at + [0, 0, 0.050], yaw=0.0, finger=-0.012, duration=1.5)
        yield from R.move(grip_at + [0, 0, 0.008], yaw=0.0, duration=0.8)
        yield from R.move(grip_at, yaw=0.0, duration=0.6)
        yield from R.grip(-0.003, duration=0.8)
        yield from R.wait(0.3)
        n = 20
        for k in range(1, n + 1):
            yield from R.move(grip_at + [0, 0, 0.045 * k / n], yaw=0.0, duration=0.15)
        yield from R.move(grip_at + [-0.06, -0.10, 0.06], yaw=0.0, duration=1.5)
        yield from R.move(np.array([0.13, -0.17, TOP + 0.035]), yaw=0.0, duration=1.2)
        yield from R.grip(-0.012, duration=0.5)
        yield from R.move(np.array([0.10, -0.20, TOP + 0.15]), yaw=0.0, duration=1.0)

    if "scion_left" not in TEST_SKIP:
        yield from set_scion("scion_left")
    if "scion_right" not in TEST_SKIP:
        yield from set_scion("scion_right")
    if "tool" not in TEST_SKIP:
        yield from pull_tool()
    yield from ctx.right.wait(4.0)
