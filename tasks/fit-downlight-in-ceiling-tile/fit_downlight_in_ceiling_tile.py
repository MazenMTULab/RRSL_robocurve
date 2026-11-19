"""A suspended-ceiling tile (600 x 600 mm, 15 mm mineral fibre) has been lifted out of its grid and laid face up
across two battens on the bench; a 36 mm hole has been cut in it for a mini LED downlight. The downlight (40 mm
white rim, 32 mm aluminium body, 55 mm tall) stands on its back end on the tile, rim up. On opposite sides of its
body are two torsion-spring clips; at rest they spring out and up toward the rim, wider than the hole, so the light
cannot go into the hole until both clips are pressed back flat along the body. Pressed back, the clips fit the hole;
once the clips' pivots pass the back of the tile they spring out again and clamp the tile between the rim and the
clips. The left arm is not needed. Zero object jitter; 20 Hz.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box, cyl  # noqa: E402
from yam_joints import Hinge, Link, Mechanism  # noqa: E402
from yam_mesh import Mesh, MeshPart  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"

# Release-and-stillness block, as in the accepted clip-pulse-oximeter-on-patient-finger task.
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
RELEASE_CLEAR = 0.030
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


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

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


_POSE_PREV = {}


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


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None:
        return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]


# ------------------------------------------------------------------ layout (metres, world frame)
TOP = TABLE_TOP[SCENE]

# Ceiling tile, face up on two battens. Tile frame: origin at the hole's centre on the tile's face, axes as world.
BATTEN_H = 0.050                    # battens 50 x 25 mm, under the tile's two edges
TILE_T = 0.015                      # 15 mm mineral fibre
TILE_TOP = TOP + BATTEN_H + TILE_T
HOLE = np.array([0.300, -0.080, TILE_TOP])
HOLE_R = 0.018                      # 36 mm cut-out (the light's specified cut-out)
TILE_X = (-0.210, 0.390)            # 600 mm, relative to the hole
TILE_Y = (-0.300, 0.300)
_SQ = 0.040                         # the hole is cut out of an 80 x 80 mm patch built from convex sectors
_SECTORS = 32
TILE_COLOR = (0.93, 0.93, 0.90)
BATTEN_COLOR = (0.72, 0.56, 0.38)


def _square_hit(t):
    """Where the ray at angle t from the hole's centre leaves the patch square."""
    c, s = math.cos(t), math.sin(t)
    k = _SQ / max(abs(c), abs(s))
    return (k * c, k * s)


def _sector(i):
    t0, t1 = 2 * math.pi * i / _SECTORS, 2 * math.pi * (i + 1) / _SECTORS
    poly = [(HOLE_R * math.cos(t0), HOLE_R * math.sin(t0)), _square_hit(t0)]
    for k in range(4):                                   # patch corners inside the sector
        tc = math.pi / 4 + k * math.pi / 2
        if t0 < tc < t1:
            poly.append((math.copysign(_SQ, math.cos(tc)), math.copysign(_SQ, math.sin(tc))))
    poly += [_square_hit(t1), (HOLE_R * math.cos(t1), HOLE_R * math.sin(t1))]
    n = len(poly)
    pts = np.array([(x, y, z) for z in (-TILE_T, 0.0) for (x, y) in poly], float)
    faces = []
    for a in range(n):
        b = (a + 1) % n
        faces += [(a, b, n + b), (a, n + b, n + a)]
    for k in range(1, n - 1):
        faces += [(0, k + 1, k), (n, n + k, n + k + 1)]
    faces = np.array(faces, int)
    tri = pts[faces]
    if np.einsum('ij,ij->i', tri[:, 0] - pts.mean(0), np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() < 0:
        faces = faces[:, ::-1]
    return Mesh(f"rim{i}", pts, faces, collision="convex", color=TILE_COLOR, proxy=())


def _bx(x0, x1, y0, y1, z0, z1, color=None):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


_Z0, _Z1 = -TILE_T, 0.0
TILE_SHAPES = [
    _bx(TILE_X[0], -_SQ, TILE_Y[0], TILE_Y[1], _Z0, _Z1),            # the tile around the 80 mm patch
    _bx(_SQ, TILE_X[1], TILE_Y[0], TILE_Y[1], _Z0, _Z1),
    _bx(-_SQ, _SQ, TILE_Y[0], -_SQ, _Z0, _Z1),
    _bx(-_SQ, _SQ, _SQ, TILE_Y[1], _Z0, _Z1),
    _bx(TILE_X[0] + 0.010, TILE_X[0] + 0.035, TILE_Y[0], TILE_Y[1], -TILE_T - BATTEN_H, -TILE_T, BATTEN_COLOR),
    _bx(TILE_X[1] - 0.035, TILE_X[1] - 0.010, TILE_Y[0], TILE_Y[1], -TILE_T - BATTEN_H, -TILE_T, BATTEN_COLOR),
]
TILE = MeshPart("tile", tuple(map(float, HOLE)), TILE_SHAPES, color=TILE_COLOR, friction=(0.6, 0.5),
                contact_offset=0.001, meshes=tuple(_sector(i) for i in range(_SECTORS)))

# Mini LED downlight (KSR Mimas mini baffle downlight: 40 mm rim, 36 mm cut-out, 55 mm tall). Light frame: origin at
# the centre of the rim's back face (the face that bears on the ceiling), +z out through the lens, the body along -z.
RIM_R, RIM_T = 0.020, 0.003
BODY_R, BODY_L = 0.016, 0.052       # 32 mm body; 3 + 52 = 55 mm overall
PIVOT_Z = -0.020                    # clip pivots 20 mm behind the rim face, on the body's sides (+y and -y)
CLIP_L, CLIP_W, CLIP_T = 0.024, 0.006, 0.001   # spring-steel clip arms
CLIP_OPEN = 140.0                   # at rest each arm springs out to 140 degrees from flat along the body
CLIP_REST = 160.0                   # spring preload against that stop
CLIP_SPRING = 0.015                 # N m/rad: 0.04 N m (1.7 N at the arm's end) pressed flat
LIGHT_MASS, CLIP_MASS = 0.045, 0.002
WHITE, ALU, STEEL = (0.96, 0.96, 0.95), (0.70, 0.71, 0.73), (0.55, 0.56, 0.58)
_PY = BODY_R + CLIP_T / 2           # arm mid-plane, lying flat on the body


def _clip(name, side):
    s = 1.0 if side > 0 else -1.0
    shapes = (box((0.0, s * _PY, PIVOT_Z - CLIP_L / 2), (CLIP_W, CLIP_T, CLIP_L), STEEL),)
    lo, hi = (0.0, CLIP_OPEN) if side > 0 else (-CLIP_OPEN, 0.0)
    joint = Hinge("x", (0.0, s * _PY, PIVOT_Z), lo, hi, damping=2e-4, spring=CLIP_SPRING, rest=s * CLIP_REST,
                  initial=s * CLIP_OPEN, armature=1e-6)
    return Link(name, shapes, joint, mass=CLIP_MASS, color=STEEL, friction=(0.35, 0.3))


CLIPS = ("clip_left", "clip_right")         # on the +y and -y sides of the body (the robot's left and right)
LIGHT_START = (0.230, -0.170, TILE_TOP + BODY_L + 0.0005)   # standing on its back end on the tile, rim up
LIGHT = Mechanism(
    "light", LIGHT_START,
    [cyl((0.0, 0.0, RIM_T / 2), RIM_R, RIM_T, color=WHITE), cyl((0.0, 0.0, -BODY_L / 2), BODY_R, BODY_L, color=ALU)],
    [_clip(CLIPS[0], +1), _clip(CLIPS[1], -1)],
    mass=LIGHT_MASS, color=WHITE, friction=(0.4, 0.35), contact_offset=0.001)

OBJECTS = [TILE, LIGHT]

# ------------------------------------------------------------------ scoring
CENTRE_TOL = 0.004        # the light's axis within 4 mm of the hole's centre
IN_DEPTH = 0.003          # in the hole: both clip pivots at least 3 mm below the tile's face (resting on open clips: 1.8 mm)
FLUSH = 0.0015            # seated: the rim's back face at most 1.5 mm off the tile's face
TILT_DEG = 3.0            # and the light within 3 degrees of square to the tile
CLIP_OUT_DEG = 60.0       # a clip has sprung out: its arm at least 60 degrees out from the body...
DROP_Z = TOP - 0.020      # ...behind the tile (its pivot below the tile's back face)
TEST_STICKY = ()


def sticky(key, cond):
    return ever(key, bool(cond) or key in TEST_STICKY)


def fit(light):
    """Measurements of the light against the hole (tile frame)."""
    p = np.asarray(light.origin, float) - HOLE
    axis = light.rotation()[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))
    pivots = [light.to_world(LIGHT._link(c).joint.anchor) - HOLE for c in CLIPS]
    angles = [abs(light.angle_deg(c)) for c in CLIPS]
    out = [pv[2] < -TILE_T and a >= CLIP_OUT_DEG for pv, a in zip(pivots, angles)]
    return {
        "centre": float(math.hypot(p[0], p[1])),
        "gap": float(p[2]),
        "tilt": tilt,
        "pivot_depth": [float(-pv[2]) for pv in pivots],
        "angles": angles,
        "out": out,
    }


def score(objects):
    """0.25 with the light in the hole (both clips' pivots inside it); 0.5 with at least one clip sprung out behind
    the tile; 1.0 with both clips out and the rim flush and square on the tile. Only once the gripper is 3 cm clear
    and the light has been still for 1 s."""
    light = objects["light"]
    dropped = sticky("light off the table", light.origin[2] < DROP_Z)
    f = fit(light)
    in_hole = f["centre"] <= CENTRE_TOL and min(f["pivot_depth"]) >= IN_DEPTH and f["tilt"] <= 20.0
    seated = in_hole and abs(f["gap"]) <= FLUSH and f["tilt"] <= TILT_DEG
    n_out = sum(f["out"]) if in_hole else 0
    clear = released(light)
    still = pose_quiet("light", [light.origin, light.to_world((0.0, 0.0, -BODY_L)),
                                 light.to_world((RIM_R, 0.0, 0.0))])
    seconds = held_for("in, released, still", in_hole and clear and still)
    if dropped or seconds + 1e-9 < STILL_S:
        return 0.0
    if seated and n_out == 2:
        return 1.0
    return 0.5 if n_out >= 1 else 0.25


def success(objects):
    return score(objects) >= 1.0


# ------------------------------------------------------------------ closed-loop scripted demo (cabledemo.py)
TIP = 0.0128                        # fingertips 12.8 mm past the TCP
SWEEP_HALF = 0.025                  # pads 25 mm either side of the axis: outside the rim, over the sprung clips
SWEEP_TIP_Z = -0.037                # sweep the fingertips down to 37 mm behind the rim face (clips pressed past flat)
PINCH = -(2 * (BODY_R + CLIP_T) - 0.006) / 2   # 3 mm past contact on each side
INSERT_CLEAR = 0.003                # lower until the fingertips are 3 mm off the tile's face
PRESS_TO = RIM_T - 0.016            # press the rim target 16 mm below the tile face (the arm stops on contact), for a few newtons
SERVO_GAIN = 0.5


def demo_script(ctx):
    R = ctx.right
    yaw = 0.0                                       # pads close along world y, across the two clips
    bias = np.zeros(3)                              # correction for the compliant arm's sag, from its measured pose

    def actual_tcp():
        hands = grippers()
        return np.asarray(hands["right"][0], float) if hands else np.asarray(R.tcp())

    def tcp_for(tip_point):
        return np.asarray(tip_point, float) + [0.0, 0.0, TIP]

    def go(tcp_target, duration, finger=None, servo=True):
        """Move the TCP, aiming off by the measured sag so the real TCP ends on target."""
        nonlocal bias
        yield from R.move(np.asarray(tcp_target) + bias, yaw=yaw, finger=finger, duration=duration)
        if servo:
            bias = np.clip(bias + SERVO_GAIN * (np.asarray(tcp_target) - actual_tcp()), -0.03, 0.03)

    def go_settled(tcp_target, duration, settle=8):
        yield from go(tcp_target, duration)
        for _ in range(settle):
            if np.linalg.norm(np.asarray(tcp_target) - actual_tcp()) < 0.001:
                break
            yield from go(tcp_target, 0.15)

    rim = np.asarray(ctx.states()["light"].origin, float)
    # 1. Pads open 25 mm either side, over the clips; sweep them down past the rim so the fingertips press both
    #    clips back past flat, then close: the pads hold the clips flat against the body.
    yield from go(tcp_for(rim + [0, 0, 0.040]), 2.5, finger=-SWEEP_HALF)
    yield from go_settled(tcp_for(rim + [0, 0, 0.008]), 1.0)
    for k in range(1, 11):
        yield from go(tcp_for(rim + [0, 0, 0.008 + (SWEEP_TIP_Z - 0.008) * k / 10]), 0.2)
    yield from go_settled(tcp_for(rim + [0, 0, SWEEP_TIP_Z]), 0.4)
    yield from R.grip(PINCH, duration=1.0)
    yield from R.wait(0.3)
    # 2. Lift and carry it over the hole; from here aim with the light's own measured position.
    held = ctx.states()["light"]
    off = actual_tcp() - np.asarray(held.origin)    # TCP relative to the rim, as held

    def carry(rim_target, duration):
        nonlocal bias
        yield from R.move(np.asarray(rim_target) + off + bias, yaw=yaw, duration=duration)
        err = np.asarray(rim_target) - np.asarray(ctx.states()["light"].origin)
        bias = np.clip(bias + SERVO_GAIN * err, -0.03, 0.03)

    start = np.asarray(held.origin)
    yield from carry(start + [0, 0, 0.050], 1.5)
    above = HOLE + [0.0, 0.0, BODY_L + 0.030]
    yield from carry(above, 2.0)
    for _ in range(6):
        yield from carry(above, 0.2)
    # 3. Lower it into the hole until the fingertips are 3 mm off the tile: the clips' ends are then inside the hole.
    stop = HOLE + [0.0, 0.0, -SWEEP_TIP_Z + INSERT_CLEAR]
    n = 16
    for k in range(1, n + 1):
        yield from carry(above + (stop - above) * k / n, 0.2)
    for _ in range(25):                                        # settle on the measured depth
        if np.linalg.norm(np.asarray(ctx.states()["light"].origin) - stop) < 0.0015:
            break
        yield from carry(stop, 0.15)
    yield from R.wait(0.3)
    # 4. Let go: the clips spring out against the hole's wall. Then press the rim down with the closed fingertips
    #    until the clips' pivots pass the back of the tile and they spring out behind it.
    yield from R.grip(-SWEEP_HALF, duration=0.6)
    yield from go(actual_tcp() + [0, 0, 0.070], 1.2, servo=False)    # well clear of the rim before closing
    yield from R.grip(FINGER_CLOSED, duration=0.6)
    top = HOLE + [0.0, 0.0, -SWEEP_TIP_Z + INSERT_CLEAR + RIM_T]
    yield from go_settled(tcp_for(top + [0, 0, 0.008]), 1.0)
    press = HOLE + [0.0, 0.0, PRESS_TO]
    n = 20
    for k in range(1, n + 1):
        yield from R.move(tcp_for(top + (press - top) * k / n) + bias * [1, 1, 0], yaw=yaw, duration=0.15)
    yield from R.wait(0.8)
    yield from go(tcp_for(HOLE + [0.0, 0.0, 0.050]), 1.0, servo=False)
    yield from go(tcp_for(HOLE + [-0.08, -0.06, 0.12]), 1.2, servo=False)
    yield from R.wait(4.0)
