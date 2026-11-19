"""A 1:1 clear PET hinged berry clamshell punnet (16.0 x 12.0 x 6.5 cm closed) packed with ripe
strawberries sits on the station table. Its rear-hinged lid starts sprung open (~24 deg under hinge
spring stiffness pulling open) above the berries, and a front sliding snap catch sits open on the
front rim rail (randomized left/right staging via HAND_SIDE=L|R).

Bimanual skill:
1. One YAM arm presses and holds the springy hinged lid's structural side shoulder flush down onto
   the base rim (<= 3.0 deg) without slamming it onto the rim/berries or reaching into the berries.
2. While the holding arm keeps the lid flush closed, the opposite arm pinches and slides the front
   snap catch cleanly home to center (0 +/- 4 mm) over the lid's front lock tongue so the catch
   locks the clamshell closed.
3. Both grippers release and withdraw clear for 1.0 s (20 steps at 20 Hz) while the punnet stays
   upright and latched closed (<= 3.0 deg).
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import guardkit as gk  # noqa: E402
import mechkit as mk  # noqa: E402
import yamkit as kit  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, ObjectState  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
DT = 1.0 / 20
RUN_LIMIT_STEPS = 2400
INSTRUCTION = (
    "Press and hold the spring-open berry clamshell lid flush down onto the base rim with one hand "
    "without slamming or crushing the strawberries, slide the front snap catch cleanly home with the "
    "other hand to latch it shut, then release both hands clear."
)

# -- randomized arm assignment (L = left arm holds lid, right latches; R = mirrored) -----------------------------------
SIDES = ("L", "R")


def _draw_side() -> str:
    env = os.environ.get("HAND_SIDE", "").strip().upper()
    return env if env in SIDES else random.SystemRandom().choice(SIDES)


HAND_SIDE = _draw_side()
SIDE_SIGN = 1.0 if HAND_SIDE == "L" else -1.0
print(f"punnet: HAND_SIDE={HAND_SIDE}", file=sys.stderr)

# -- 1:1 clear PET clamshell punnet dimensions (16.0 x 12.0 x 6.5 cm closed, metres) ------------------------------------
PUNNET_X, PUNNET_Y = 0.275, 0.0
PUNNET_POS = (PUNNET_X, PUNNET_Y, TOP)

FLANGE_X = 0.120  # 12.0 cm deep along X (base x in [-0.060, +0.060])
FLANGE_Y = 0.160  # 16.0 cm wide along Y (base y in [-0.080, +0.080])
TUB_X = 0.112
TUB_Y = 0.152
WALL_T = 0.004
FLOOR_T = 0.004
BASE_H = 0.043  # lower tub rim top at z_local = 0.043
CLOSED_H = 0.065  # 6.5 cm closed overall height (dome top at z_local = 0.065)

# Rear living hinge at +X rim turning about +Y: positive angle lifts front -X lid edge upward
HINGE_LOCAL = (0.058, 0.0, BASE_H)
HINGE_WORLD = (PUNNET_X + HINGE_LOCAL[0], PUNNET_Y, TOP + HINGE_LOCAL[2])

LID_LEN = 0.120  # spans x_lid in [-0.120, 0.0] -> closed front rim at x_base = -0.062
LID_W = 0.160  # spans y_lid in [-0.080, +0.080]
LID_DOME_Y = 0.048  # central raised berry dome spans |y_lid| <= 0.048; |y| in [0.048, 0.080] are flat shoulders
LID_DOME_H = 0.022  # dome top at z_lid = 0.022 -> closed world height 0.065 m
LID_RIM_T = 0.004
LID_MASS = 0.040

LID_SPRING_REST_DEG = 26.5  # pulls lid open to ~24 deg equilibrium against 40 g lid gravity
LID_START_DEG = 24.0
LID_UPPER_DEG = 28.0
LID_SPRING_K = 0.55  # N*m/rad (~2.4 N downward press needed at shoulder lever arm)
LID_DAMP = 0.018

# Front sliding snap catch on -X rim rail (PrismaticJoint along Y)
CATCH_ANCHOR = (-0.065, 0.0, BASE_H)
CATCH_START_Y = -SIDE_SIGN * 0.032
CATCH_TAB_X_LOCAL = -0.019  # tab center in catch frame -> world x = 0.275 - 0.065 - 0.019 = 0.191
CATCH_MASS = 0.025
BASE_MASS = 0.460  # 1-lb (460 g) filled strawberry punnet + PET tub

# Press pad center on lid shoulder (in lid frame)
PRESS_X_LID = -0.088
PRESS_Y_LID = SIDE_SIGN * 0.062
PRESS_Z_LID = 0.007

# -- scoring & safety thresholds ---------------------------------------------------------------------------------------
LID_CLOSED_MAX_DEG = 3.0
LID_CLOSED_MAX_RAD = math.radians(LID_CLOSED_MAX_DEG)
LID_STILL_VEL = 0.10  # rad/s
CATCH_LOCK_TOL = 0.004  # +/-4 mm around center 0.0
CATCH_ENGAGE_TOL = 0.012  # +/-12 mm partial engage
CATCH_STILL_VEL = 0.010  # m/s

SLAM_SPEED_RAD_S = 2.20  # downward closing speed faster than 2.2 rad/s near closed slams lid/berries
SLAM_ANGLE_MAX = math.radians(15.0)
PUNNET_TILT_MAX_DEG = 12.0
PUNNET_END_TILT_DEG = 4.0
PUNNET_YAW_MAX_DEG = 15.0
PUNNET_DRIFT_MAX = 0.045
STILL_SPEED = 0.005
STILL_SPIN = math.radians(5.0)
HOLD_STEPS = round(SUCCESS_HOLD_S / DT)
GRIPPERS = ("left_gripper", "right_gripper")

# -- visual materials --------------------------------------------------------------------------------------------------
PET_CLEAR = kit.Look((0.86, 0.94, 0.98), roughness=0.10, metallic=0.0, opacity=0.28)
PET_FROST = kit.Look((0.78, 0.89, 0.94), roughness=0.20, metallic=0.0, opacity=0.46)
PET_RIM = kit.Look((0.82, 0.91, 0.96), roughness=0.16, metallic=0.0, opacity=0.58)
HINGE_DARK = kit.Look((0.26, 0.28, 0.32), roughness=0.45, metallic=0.50)
PAD_LINER = kit.Look((0.93, 0.95, 0.91), roughness=0.85)
BERRY_RED = kit.Look((0.80, 0.06, 0.09), roughness=0.38)
BERRY_DARK = kit.Look((0.62, 0.04, 0.07), roughness=0.42)
CALYX_GREEN = kit.Look((0.15, 0.48, 0.14), roughness=0.55)
CATCH_ORANGE = kit.Look((0.95, 0.45, 0.08), roughness=0.35)
LOCK_GREEN = kit.Look((0.12, 0.72, 0.32), roughness=0.40)
PRESS_BLUE = kit.Look((0.18, 0.52, 0.80), roughness=0.45, opacity=0.82)


# -- punnet mechanism builder ------------------------------------------------------------------------------------------
def _strawberry_meshes(cx: float, cy: float, z0: float, scale: float, dark: bool) -> list:
    """Plump lathe strawberry + green calyx cap seated at (cx, cy, z0) in base frame."""
    skin = BERRY_DARK if dark else BERRY_RED
    h = 0.031 * scale
    r = 0.0155 * scale
    profile = [
        (0.0, 0.0),
        (0.32 * r, 0.14 * h),
        (0.74 * r, 0.38 * h),
        (1.00 * r, 0.66 * h),
        (0.86 * r, 0.86 * h),
        (0.46 * r, 0.97 * h),
        (0.0, h),
    ]
    berry = kit.lathe(profile, skin, center=(cx, cy, z0), axis="z", n=14)
    leaf_a = kit.box((cx, cy, z0 + h + 0.001), (0.015 * scale, 0.004 * scale, 0.002), CALYX_GREEN, yaw=0.35)
    leaf_b = kit.box((cx, cy, z0 + h + 0.001), (0.004 * scale, 0.015 * scale, 0.002), CALYX_GREEN, yaw=0.35)
    stem = kit.cylinder((cx, cy, z0 + h + 0.003), 0.0016, 0.004, CALYX_GREEN, axis="z", n=8)
    return [berry, leaf_a, leaf_b, stem]


def _build_punnet() -> kit.Mechanism:
    meshes, cols = [], []

    # 1. Clear PET lower tub floor + 4 sidewalls + top flange rim
    meshes.append(kit.box((0.0, 0.0, FLOOR_T / 2.0), (TUB_X, TUB_Y, FLOOR_T), PET_FROST))
    cols.append(
        kit.Collider(
            (0.0, 0.0, FLOOR_T / 2.0),
            size=(TUB_X, TUB_Y, FLOOR_T),
            friction=0.55,
            combine="min",
            contact_offset=0.001,
        )
    )
    wall_h = BASE_H - FLOOR_T
    zc_w = FLOOR_T + wall_h / 2.0
    # Front (-X) & Rear (+X) walls
    for sx in (-1.0, 1.0):
        wx = sx * (TUB_X / 2.0 - WALL_T / 2.0)
        meshes.append(kit.box((wx, 0.0, zc_w), (WALL_T, TUB_Y, wall_h), PET_CLEAR))
        cols.append(kit.Collider((wx, 0.0, zc_w), size=(WALL_T, TUB_Y, wall_h), friction=0.50))
    # Left (+Y) & Right (-Y) walls
    for sy in (-1.0, 1.0):
        wy = sy * (TUB_Y / 2.0 - WALL_T / 2.0)
        meshes.append(kit.box((0.0, wy, zc_w), (TUB_X - 2 * WALL_T, WALL_T, wall_h), PET_CLEAR))
        cols.append(kit.Collider((0.0, wy, zc_w), size=(TUB_X - 2 * WALL_T, WALL_T, wall_h), friction=0.50))

    # Top perimeter flange (12.0 x 16.0 cm outer footprint)
    zf = BASE_H - 0.002
    for sx in (-1.0, 1.0):
        fx = sx * (FLANGE_X / 2.0 - 0.004)
        meshes.append(kit.box((fx, 0.0, zf), (0.008, FLANGE_Y, 0.004), PET_RIM))
        cols.append(kit.Collider((fx, 0.0, zf), size=(0.008, FLANGE_Y, 0.004)))
    for sy in (-1.0, 1.0):
        fy = sy * (FLANGE_Y / 2.0 - 0.004)
        meshes.append(kit.box((0.0, fy, zf), (FLANGE_X - 0.016, 0.008, 0.004), PET_RIM))
        cols.append(kit.Collider((0.0, fy, zf), size=(FLANGE_X - 0.016, 0.008, 0.004)))

    # Rear hinge barrels
    for sy in (-0.048, 0.048):
        meshes.append(kit.cylinder((HINGE_LOCAL[0], sy, BASE_H), 0.004, 0.016, HINGE_DARK, axis="y", n=12))

    # Front catch rail + center green LOCK alignment mark
    meshes.append(kit.box((-0.065, 0.0, 0.0365), (0.010, 0.098, 0.005), PET_RIM))
    cols.append(kit.Collider((-0.065, 0.0, 0.0365), size=(0.010, 0.098, 0.005), friction=0.25, combine="min"))
    meshes.append(kit.box((-0.0705, 0.0, 0.0365), (0.003, 0.020, 0.007), LOCK_GREEN))
    for sy in (-0.047, 0.047):
        meshes.append(kit.box((-0.065, sy, 0.041), (0.010, 0.006, 0.010), HINGE_DARK))

    # Absorbent liner pad + 8 ripe strawberries inside the tub
    meshes.append(kit.box((0.0, 0.0, FLOOR_T + 0.0008), (0.098, 0.138, 0.0016), PAD_LINER))
    berry_specs = [
        (-0.025, -0.049, 1.00, False),
        (-0.023, -0.016, 0.96, True),
        (-0.026, +0.017, 1.02, False),
        (-0.024, +0.050, 0.98, True),
        (+0.024, -0.050, 0.97, True),
        (+0.026, -0.017, 1.03, False),
        (+0.023, +0.016, 0.99, True),
        (+0.025, +0.049, 1.01, False),
    ]
    for bx, by, sc, dk in berry_specs:
        meshes.extend(_strawberry_meshes(bx, by, FLOOR_T + 0.0016, sc, dk))
    # Berry pack collider inside tub (kept low <= 0.034 so closed lid never intersects)
    cols.append(kit.Collider((0.0, 0.0, 0.019), size=(0.096, 0.136, 0.030), friction=0.45))

    # 2. Front sliding snap catch (PrismaticJoint along Y, starts open on latch arm's side)
    if HAND_SIDE == "L":
        c_lo, c_hi = -0.036, 0.003
    else:
        c_lo, c_hi = -0.003, 0.036

    catch_meshes = (
        kit.box((0.0, 0.0, -0.002), (0.010, 0.024, 0.006), CATCH_ORANGE),
        kit.box((-0.006, 0.0, 0.004), (0.006, 0.024, 0.016), CATCH_ORANGE),
        kit.box((0.001, 0.0, 0.0095), (0.016, 0.024, 0.005), CATCH_ORANGE),  # upper retaining hook
        kit.box((CATCH_TAB_X_LOCAL, 0.0, 0.005), (0.014, 0.014, 0.024), CATCH_ORANGE),  # gripper pull tab
    )
    catch_cols = (
        kit.Collider((0.0, 0.0, -0.002), size=(0.010, 0.024, 0.006), friction=0.35),
        kit.Collider((0.001, 0.0, 0.0095), size=(0.016, 0.024, 0.005), friction=0.50),
        kit.Collider((CATCH_TAB_X_LOCAL, 0.0, 0.005), size=(0.014, 0.014, 0.024), friction=0.95),
    )
    catch_link = kit.Link(
        "catch",
        kind="prismatic",
        axis="Y",
        anchor=CATCH_ANCHOR,
        lower=c_lo,
        upper=c_hi,
        start=CATCH_START_Y,
        mass=CATCH_MASS,
        floating=True,
        meshes=catch_meshes,
        colliders=catch_cols,
        law=mk.SnapCatch(lid_link="lid", closed_rad=math.radians(4.0)),
    )

    # 3. Rear-hinged clear PET domed lid with flat structural side shoulders & front lock tongue
    lid_meshes, lid_cols = [], []
    # Left & Right flat structural pressing shoulders (|y| in [0.048, 0.080])
    sh_w = LID_W / 2.0 - LID_DOME_Y  # 0.032 m
    sh_y = LID_DOME_Y + sh_w / 2.0  # 0.064 m
    for sy in (-1.0, 1.0):
        lid_meshes.append(kit.box((-LID_LEN / 2.0, sy * sh_y, 0.003), (LID_LEN, sh_w, 0.006), PET_FROST))
        lid_cols.append(
            kit.Collider(
                (-LID_LEN / 2.0, sy * sh_y, 0.003),
                size=(LID_LEN, sh_w, 0.006),
                friction=0.70,
                contact_offset=0.001,
            )
        )
        # Subtle textured press-target rib on each shoulder
        lid_meshes.append(kit.box((PRESS_X_LID, sy * 0.062, 0.0065), (0.028, 0.020, 0.002), PRESS_BLUE))

    # Front & rear flat rim strips bridging the shoulders
    lid_meshes.append(kit.box((-LID_LEN + 0.007, 0.0, 0.003), (0.014, 2 * LID_DOME_Y, 0.006), PET_FROST))
    lid_cols.append(kit.Collider((-LID_LEN + 0.007, 0.0, 0.003), size=(0.014, 2 * LID_DOME_Y, 0.006), friction=0.65))
    lid_meshes.append(kit.box((-0.008, 0.0, 0.003), (0.016, 2 * LID_DOME_Y, 0.006), PET_FROST))
    lid_cols.append(kit.Collider((-0.008, 0.0, 0.003), size=(0.016, 2 * LID_DOME_Y, 0.006), friction=0.65))

    # Central raised clear PET berry dome (x_lid in [-0.106, -0.016], |y| <= 0.048, z in [0.004, 0.022])
    dome_len = 0.090
    dome_xc = -0.061
    dome_wh = LID_DOME_H - 0.004
    dome_zc = 0.004 + dome_wh / 2.0
    for sy in (-1.0, 1.0):
        lid_meshes.append(kit.box((dome_xc, sy * (LID_DOME_Y - 0.002), dome_zc), (dome_len, 0.004, dome_wh), PET_CLEAR))
        lid_cols.append(kit.Collider((dome_xc, sy * (LID_DOME_Y - 0.002), dome_zc), size=(dome_len, 0.004, dome_wh)))
    for sx in (-0.104, -0.018):
        lid_meshes.append(kit.box((sx, 0.0, dome_zc), (0.004, 2 * LID_DOME_Y - 0.008, dome_wh), PET_CLEAR))
        lid_cols.append(kit.Collider((sx, 0.0, dome_zc), size=(0.004, 2 * LID_DOME_Y - 0.008, dome_wh)))
    # Clear dome top window
    lid_meshes.append(kit.box((dome_xc, 0.0, LID_DOME_H - 0.0015), (dome_len, 2 * LID_DOME_Y, 0.003), PET_CLEAR))
    lid_cols.append(kit.Collider((dome_xc, 0.0, LID_DOME_H - 0.0015), size=(dome_len, 2 * LID_DOME_Y, 0.003)))

    # Front lock tongue at y=0 (engages beneath the sliding snap catch's upper hook when closed)
    lid_meshes.append(kit.box((-0.123, 0.0, 0.0025), (0.010, 0.020, 0.005), PET_RIM))
    lid_cols.append(kit.Collider((-0.123, 0.0, 0.0025), size=(0.010, 0.020, 0.005), friction=0.50))

    lid_link = kit.Link(
        "lid",
        kind="revolute",
        axis="Y",
        anchor=HINGE_LOCAL,
        lower=0.0,
        upper=math.radians(LID_UPPER_DEG),
        start=math.radians(LID_START_DEG),
        mass=LID_MASS,
        meshes=tuple(lid_meshes),
        colliders=tuple(lid_cols),
        law=mk.SprungLid(
            rest=math.radians(LID_SPRING_REST_DEG),
            k=LID_SPRING_K,
            damp=LID_DAMP,
            catch_link="catch",
            lower_rad=0.0,
            upper_rad=math.radians(LID_UPPER_DEG),
            latched_upper_rad=LID_CLOSED_MAX_RAD,
            jam_rad=math.radians(6.5),
            hold_k=10.0,
            hold_damp=0.20,
        ),
    )

    return kit.Mechanism(
        "punnet",
        pos=PUNNET_POS,
        size=(FLANGE_X + 0.03, FLANGE_Y, CLOSED_H),
        mass=BASE_MASS,
        meshes=tuple(meshes),
        colliders=tuple(cols),
        links=(catch_link, lid_link),
    )


PUNNET = _build_punnet()
OBJECTS = [PUNNET]

# -- guardkit shapes (base, berry cavity, lid, catch) ------------------------------------------------------------------
PUNNET_BASE_PARTS = (
    gk.Block((0.0, 0.0, BASE_H / 2.0), (FLANGE_X, FLANGE_Y, BASE_H)),
)
BERRY_CAVITY_PARTS = (
    gk.Block((0.0, 0.0, 0.020), (0.088, 0.128, 0.030)),
)
LID_PARTS = (
    gk.Block((-LID_LEN / 2.0, 0.0, 0.004), (LID_LEN + 0.012, LID_W, 0.008)),
    gk.Block((-0.061, 0.0, 0.013), (0.092, 2 * LID_DOME_Y, 0.018)),
)
CATCH_PARTS = (
    gk.Block((CATCH_TAB_X_LOCAL / 2.0, 0.0, 0.004), (0.030, 0.024, 0.024)),
)


# -- measurements, hazards & scoring -----------------------------------------------------------------------------------
def _tilt_deg(obj) -> float:
    x, y, _, _ = (float(v) for v in obj.quat_xyzw)
    return float(math.degrees(math.acos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _yaw_deg(obj) -> float:
    R = gk._rot(obj.quat_xyzw)
    return float(math.degrees(math.atan2(R[1, 0], R[0, 0])))


def _drift_m(obj) -> float:
    return float(np.linalg.norm(np.asarray(obj.pos[:2], float) - np.array([PUNNET_X, PUNNET_Y])))


def _lid_slam(lid_q: float, lid_qd: float) -> bool:
    return bool(lid_q <= SLAM_ANGLE_MAX and lid_qd < -SLAM_SPEED_RAD_S)


def _berry_crush(objects, lid_q: float) -> bool:
    if lid_q <= math.radians(5.0):
        return False
    return bool(gk.touching(objects, "punnet", BERRY_CAVITY_PARTS, tol=0.0, grippers=GRIPPERS))


def _punnet_disturbed(p) -> bool:
    return bool(
        _tilt_deg(p) > PUNNET_TILT_MAX_DEG
        or abs(_yaw_deg(p)) > PUNNET_YAW_MAX_DEG
        or _drift_m(p) > PUNNET_DRIFT_MAX
        or float(np.asarray(p.low)[2]) < TOP - 0.015
    )


def _touched_any(objects) -> bool:
    return bool(
        gk.touching(objects, "punnet", PUNNET_BASE_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "punnet/lid", LID_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "punnet/catch", CATCH_PARTS, grippers=GRIPPERS)
    )


HAZARD = gk.Latch(steps=1)


class _Tracker:
    def __init__(self):
        self.ep = kit.episode()
        self.key = None
        self.objs = None
        self.hold_steps = 0
        self.reason = None


_T = _Tracker()


def _step(objects) -> None:
    global _T
    if _T.ep != kit.episode():
        _T = _Tracker()
    n = kit._EPISODE.get("last")
    key = (kit.episode(), n) if n is not None else None
    if (key is not None and key == _T.key) or (key is None and objects is _T.objs):
        return
    _T.key, _T.objs = key, objects

    p = objects["punnet"]
    lid_st = kit.joint(objects, "punnet/lid")
    catch_st = kit.joint(objects, "punnet/catch")
    lid_q, lid_qd = float(lid_st.pos), float(lid_st.vel)
    catch_q, catch_qd = float(catch_st.pos), float(catch_st.vel)

    slam = _lid_slam(lid_q, lid_qd)
    crush = _berry_crush(objects, lid_q)
    disturbed = _punnet_disturbed(p)
    HAZARD.feed(objects, slam or crush or disturbed)
    if HAZARD.on and _T.reason is None:
        _T.reason = "slam" if slam else ("berry_crush" if crush else "punnet_disturbed")

    catch_law = mk.state("punnet/catch")
    jammed = bool(catch_law.get("jammed_under_lid", False))
    catch_locked = bool(catch_law.get("locked", False)) and not jammed and abs(catch_q) <= CATCH_LOCK_TOL

    lid_closed = lid_q <= LID_CLOSED_MAX_RAD
    lid_still = abs(lid_qd) <= LID_STILL_VEL
    catch_still = abs(catch_qd) <= CATCH_STILL_VEL
    punnet_ok = (
        _tilt_deg(p) <= PUNNET_END_TILT_DEG
        and abs(float(np.asarray(p.low)[2]) - TOP) <= 0.005
        and p.resting(max_speed=STILL_SPEED, max_spin=STILL_SPIN)
    )
    clear = not _touched_any(objects)

    settled = (
        not HAZARD.on
        and lid_closed
        and lid_still
        and catch_locked
        and catch_still
        and punnet_ok
        and clear
    )
    _T.hold_steps = _T.hold_steps + 1 if settled else 0

    if os.environ.get("PUNNET_TRACE") and n is not None and (n % 10 == 0 or HAZARD.on or _T.hold_steps):
        print(
            f"TRACE step={n} haz={HAZARD.on}({ _T.reason }) lid={math.degrees(lid_q):.1f}deg "
            f"lid_v={lid_qd:.2f} catch={catch_q*1000:+.1f}mm lock={catch_locked} jam={jammed} "
            f"tilt={_tilt_deg(p):.1f} clear={clear} hold={_T.hold_steps}",
            file=sys.stderr,
            flush=True,
        )


def success(objects) -> bool:
    _step(objects)
    return not HAZARD.on and _T.hold_steps > 0


def score(objects) -> float:
    _step(objects)
    if HAZARD.on:
        return 0.0
    if _T.hold_steps >= HOLD_STEPS:
        return 1.0

    lid_q = float(kit.joint(objects, "punnet/lid").pos)
    catch_q = float(kit.joint(objects, "punnet/catch").pos)
    catch_law = mk.state("punnet/catch")
    jammed = bool(catch_law.get("jammed_under_lid", False))
    catch_locked = bool(catch_law.get("locked", False)) and not jammed and abs(catch_q) <= CATCH_LOCK_TOL

    if lid_q <= LID_CLOSED_MAX_RAD and catch_locked:
        return 0.75
    if lid_q <= LID_CLOSED_MAX_RAD and not jammed and abs(catch_q) <= CATCH_ENGAGE_TOL:
        return 0.50
    if lid_q <= LID_CLOSED_MAX_RAD:
        return 0.25
    return 0.0


# -- symmetric bimanual scripted demo & negative controls --------------------------------------------------------------
def _press_pad_world(q_rad: float, y_side: float) -> tuple[float, float, float]:
    """World point on the lid's structural side shoulder top surface at hinge angle ``q_rad``."""
    x = HINGE_WORLD[0] + PRESS_X_LID * math.cos(q_rad) + PRESS_Z_LID * math.sin(q_rad)
    z = HINGE_WORLD[2] - PRESS_X_LID * math.sin(q_rad) + PRESS_Z_LID * math.cos(q_rad)
    return (x, y_side, z)


def demo() -> dict[str, list]:
    """Bimanual clamshell closing & latching sequence:
    - Hold arm (L when HAND_SIDE=L, R when HAND_SIDE=R) gently presses the spring-open lid's
      structural shoulder flush down onto the base rim and holds it closed.
    - Latch arm (opposite arm) pinches the front snap-catch tab and slides it cleanly to center
      (y=0) over the lid's front lock tongue until the catch locks.
    - Both grippers release and return to READY clear for 1.0 s settle.

    Negative controls:
    - DEMO_NO_LATCH=1: hold arm presses lid flush then releases without sliding catch -> springs open -> 0.0.
    - DEMO_LATCH_BEFORE_CLOSE=1: latch arm slides catch to center while lid is still open -> jams -> 0.0.
    - DEMO_CRUSH_BERRIES=1: hold arm plunges into the open berry cavity -> berry_crush hazard -> 0.0.
    """
    s = SIDE_SIGN
    hold_side = "left" if HAND_SIDE == "L" else "right"
    latch_side = "right" if HAND_SIDE == "L" else "left"

    yh = s * 0.062  # holding shoulder y
    p_open = _press_pad_world(math.radians(LID_START_DEG), yh)
    p_mid = _press_pad_world(math.radians(12.0), yh)
    p_closed = _press_pad_world(0.0, yh)

    # TCP heights for flat closed-tip press (PAD_TIP_PAST_TCP = 0.013)
    z_hover_h = TOP + 0.145
    z_touch_open = p_open[2] + 0.018
    z_mid = p_mid[2] + 0.012
    z_press = p_closed[2] + 0.0085  # ~4.5 mm firm compression into rim against hinge spring

    # Catch tab grasp & slide waypoints
    xc = PUNNET_X + CATCH_ANCHOR[0] + CATCH_TAB_X_LOCAL  # 0.191
    yc_start = CATCH_START_Y  # -s * 0.032
    yc_home = 0.0
    zc_tab = TOP + CATCH_ANCHOR[2] + 0.006  # TOP + 0.049 (tip covers z in [0.036, 0.052])
    zc_app = TOP + 0.125
    pitch_c = -0.22  # lean wrist slightly toward -X away from punnet front wall
    open_tab = float(0.018 / 0.047) * FINGER_OPEN  # 36 mm jaw gap straddling 14 mm tab
    pinch_tab = float(0.005 / 0.047) * FINGER_OPEN  # 10 mm firm pinch on 14 mm tab

    if os.environ.get("DEMO_CRUSH_BERRIES", "0") == "1":
        # Negative control 3: plunge gripper straight down into open strawberry bed -> berry_crush -> 0.0
        hold = [
            Waypoint(1.0, (PUNNET_X - 0.010, s * 0.025, z_hover_h), FINGER_OPEN, "over_berries"),
            Waypoint(1.0, (PUNNET_X - 0.010, s * 0.025, TOP + 0.028), FINGER_OPEN, "crush_berries"),
            Waypoint(1.5, (PUNNET_X - 0.010, s * 0.025, TOP + 0.028), FINGER_CLOSED, "pinch_berries"),
            Waypoint(1.0, (PUNNET_X - 0.010, s * 0.025, z_hover_h), FINGER_OPEN, "lift"),
            kit.Home(1.5),
        ]
        latch = [kit.Home(sum(w.duration for w in hold), label="stay_clear")]
        return {hold_side: hold, latch_side: latch}

    if os.environ.get("DEMO_LATCH_BEFORE_CLOSE", "0") == "1":
        # Negative control 2: slide catch to center BEFORE lid is closed -> jams open -> 0.0
        latch = [
            kit.Tilted(1.0, (xc - 0.025, yc_start, zc_app), open_tab, "approach_catch", pitch=pitch_c),
            kit.Tilted(0.9, (xc, yc_start, zc_tab), open_tab, "down_catch", pitch=pitch_c),
            kit.Tilted(0.5, (xc, yc_start, zc_tab), pinch_tab, "grip_catch", pitch=pitch_c),
            kit.Tilted(1.2, (xc, yc_home, zc_tab), pinch_tab, "slide_catch_early", pitch=pitch_c),
            kit.Tilted(0.5, (xc, yc_home, zc_tab), open_tab, "release_catch", pitch=pitch_c),
            kit.Tilted(0.9, (xc - 0.025, yc_home, zc_app), open_tab, "retreat_catch", pitch=pitch_c),
            kit.Home(4.5),
        ]
        hold = [
            Waypoint(4.8, (p_open[0], yh, z_hover_h), FINGER_CLOSED, "wait_early_latch"),
            Waypoint(1.0, (p_open[0], yh, z_touch_open), FINGER_CLOSED, "touch_open"),
            Waypoint(1.6, (p_closed[0], yh, z_press), FINGER_CLOSED, "press_against_jam"),
            Waypoint(0.8, (p_closed[0], yh, z_hover_h), FINGER_CLOSED, "lift_clear"),
            kit.Home(1.5),
        ]
        return {hold_side: hold, latch_side: latch}

    if os.environ.get("DEMO_NO_LATCH", "0") == "1":
        # Negative control 1: press lid flush, then release WITHOUT sliding catch -> springs back open -> 0.0
        hold = [
            Waypoint(1.0, (p_open[0], yh, z_hover_h), FINGER_CLOSED, "approach_shoulder"),
            Waypoint(0.9, (p_open[0], yh, z_touch_open), FINGER_CLOSED, "contact_lid"),
            Waypoint(0.9, (p_mid[0], yh, z_mid), FINGER_CLOSED, "lower_mid"),
            Waypoint(1.0, (p_closed[0], yh, z_press), FINGER_CLOSED, "press_flush"),
            Waypoint(1.5, (p_closed[0], yh, z_press), FINGER_CLOSED, "hold_no_latch"),
            Waypoint(0.9, (p_closed[0], yh, z_hover_h), FINGER_CLOSED, "release_unlatched"),
            kit.Home(2.0),
        ]
        latch = [kit.Home(sum(w.duration for w in hold), label="stay_clear")]
        return {hold_side: hold, latch_side: latch}

    # Positive bimanual demo:
    # t = 0.0..3.8s: hold arm presses lid smoothly down to flush and holds through t = 8.4s
    hold = [
        Waypoint(1.0, (p_open[0], yh, z_hover_h), FINGER_CLOSED, "approach_shoulder"),
        Waypoint(0.9, (p_open[0], yh, z_touch_open), FINGER_CLOSED, "contact_lid"),
        Waypoint(0.9, (p_mid[0], yh, z_mid), FINGER_CLOSED, "lower_mid"),
        Waypoint(1.0, (p_closed[0], yh, z_press), FINGER_CLOSED, "press_flush"),
        Waypoint(4.6, (p_closed[0], yh, z_press), FINGER_CLOSED, "hold_flush_for_latch"),
        Waypoint(0.9, (p_closed[0], yh, z_hover_h), FINGER_CLOSED, "lift_clear"),
        kit.Home(1.5),
        kit.Home(2.0, label="hold_clear"),
    ]

    # Latch arm stages clear until t = 2.8s, approaches tab as lid reaches flush (t=3.8s),
    # pinches & slides catch home (t=4.4..6.0s), releases & retracts clear by t=8.0s
    latch = [
        kit.Tilted(2.8, (xc - 0.035, yc_start - s * 0.040, zc_app), open_tab, "stage_clear", pitch=pitch_c),
        kit.Tilted(1.0, (xc - 0.015, yc_start, zc_app), open_tab, "over_catch", pitch=pitch_c),
        kit.Tilted(1.0, (xc, yc_start, zc_tab), open_tab, "down_catch", pitch=pitch_c),
        kit.Tilted(0.5, (xc, yc_start, zc_tab), pinch_tab, "grip_catch", pitch=pitch_c),
        kit.Tilted(1.4, (xc, yc_home, zc_tab), pinch_tab, "slide_catch_home", pitch=pitch_c),
        kit.Tilted(0.5, (xc, yc_home, zc_tab), pinch_tab, "seat_detent", pitch=pitch_c),
        kit.Tilted(0.5, (xc, yc_home, zc_tab), open_tab, "release_catch", pitch=pitch_c),
        kit.Tilted(0.9, (xc - 0.025, yc_home, zc_app), open_tab, "lift_catch", pitch=pitch_c),
        kit.Home(1.8),
        kit.Home(2.4, label="hold_clear"),
    ]

    return {hold_side: hold, latch_side: latch}
