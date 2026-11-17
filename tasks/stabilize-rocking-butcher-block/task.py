"""A solid hard-maple butcher-block cutting board rests on three 20 mm black rubber feet and one bare 12 mm brass
corner stud (8 mm shorter, front-left by default; BOARD_CORNER=FR is an optional mirror check).
An open ceramic mug sits on top of the board above that corner, pressing the short brass stud onto the tabletop so
there is zero gap underneath at rest (about 3.8 deg tilt). A spare 44 x 36 x 8 mm black EPDM rubber pad lies on the
table in front of the board.

The task is to slide/fit the spare rubber pad squarely under the bare corner stud while keeping the open mug upright
on the board, leaving the board resting flat and steady with grippers clear.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import flowkit as fk  # noqa: E402
import guardkit as gk  # noqa: E402
import yamkit as kit  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
DT = 1.0 / 20
RUN_LIMIT_STEPS = 2400
INSTRUCTION = (
    "Fit the spare black pad under the short foot so the cutting board sits level. "
    "Keep the mug on the board within 4.5 cm of its starting spot on the board and within 25 degrees of upright. "
    "Let go when finished."
)

# -- missing corner (FL = front-left +y, FR = front-right -y) ----------------------------------------------
CORNERS = ("FL", "FR")


def _draw_corner() -> str:
    env = os.environ.get("BOARD_CORNER", "").strip().upper()
    return env if env in CORNERS else "FL"


MISSING_CORNER = _draw_corner()
SIDE_SIGN = 1.0 if MISSING_CORNER == "FL" else -1.0  # +1 for FL (+y), -1 for FR (-y)
print(f"rocking-board: missing pad at {MISSING_CORNER}", file=sys.stderr)

# -- board geometry (world frame origin = board center projected onto z = TOP when flat on 20 mm feet) ----------------
X_B, Y_B = 0.280, 0.0
BOARD_X, BOARD_Y, BOARD_Z = 0.180, 0.240, 0.028  # 18 x 24 x 2.8 cm maple butcher block
BOARD_MASS = 0.95  # kg (~0.70 g/cm^3 maple slab + foot hardware)

FOOT_X, FOOT_Y = 0.074, 0.104  # corner foot centers: 1.6 cm inset from +/-0.090 and +/-0.120 edges
FOOT_R = 0.016  # 32 mm diameter feet (outer edge flush with board corners, clearly visible in top_cam)
FOOT_H = 0.020  # 20 mm intact rubber bumper foot height
PAD_H = 0.008  # 8 mm replacement rubber foot pad height
SHORT_H = FOOT_H - PAD_H  # 12 mm bare brass corner stud (8 mm shorter)

# Foot corner signs (sx, sy) and missing corner coordinates in board frame
ALL_FEET = {
    "FL": (-1.0, 1.0),
    "FR": (-1.0, -1.0),
    "RL": (1.0, 1.0),
    "RR": (1.0, -1.0),
}
MISS_FOOT_LOCAL = (ALL_FEET[MISSING_CORNER][0] * FOOT_X, ALL_FEET[MISSING_CORNER][1] * FOOT_Y, FOOT_H - SHORT_H)

# Initial rest tilt around the diagonal connecting the two adjacent good feet so FR/RL/FL all start touching z = TOP
# Diagonal vector from FR (-FOOT_X, -S*FOOT_Y) to RL (+FOOT_X, +S*FOOT_Y) is along (+FOOT_X, +S*FOOT_Y, 0).
_diag_len = math.hypot(FOOT_X, FOOT_Y)
_D_PERP = 2.0 * FOOT_X * FOOT_Y / _diag_len  # ~0.1206 m perpendicular lever arm from FL/FR to opposite diagonal
THETA0 = math.asin(PAD_H / _D_PERP)  # ~0.0664 rad (3.80 deg)
# Axis u in XY pointing from adjacent (-FOOT_X, -S*FOOT_Y) to (+FOOT_X, +S*FOOT_Y); note u x (-FOOT_X, +S*FOOT_Y)
# has z-component (+2*S*FOOT_X*FOOT_Y / diag_len), so rotation angle (-SIDE_SIGN * THETA0) tips MISSING_CORNER down!
_U_AXIS = np.array([FOOT_X / _diag_len, SIDE_SIGN * FOOT_Y / _diag_len, 0.0])
_HALF_A = -SIDE_SIGN * THETA0 / 2.0
BOARD_TILT_XYZW = (
    float(_U_AXIS[0] * math.sin(_HALF_A)),
    float(_U_AXIS[1] * math.sin(_HALF_A)),
    0.0,
    float(math.cos(_HALF_A)),
)


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


BOARD_R0 = _quat_to_mat(BOARD_TILT_XYZW)
BOARD_POS0 = (X_B, Y_B, TOP)

# Spare replacement EPDM rubber pad: 44 x 36 x 8 mm, sitting in front-center on tabletop
PAD_SIZE = (0.044, 0.036, PAD_H)
PAD_MASS = 0.025
PAD_START = (0.125, 0.0, TOP + PAD_H / 2.0)

# Open ceramic mug standing on top of the cutting board over the missing-foot quadrant so gravity holds that corner down
CUP_R = 0.035  # 70 mm diameter
CUP_H = 0.095  # 95 mm tall
CUP_MASS = 0.42  # kg (ceramic mug + water fill)
CUP_FRICTION = 0.45  # realistic dry ceramic on oiled maple (~24 deg sliding threshold)
CUP_BOARD_LOCAL_XY = (-0.036, SIDE_SIGN * 0.056)  # in missing-corner quadrant -> presses bare corner down onto table
_cup_local0 = np.array([CUP_BOARD_LOCAL_XY[0], CUP_BOARD_LOCAL_XY[1], FOOT_H + BOARD_Z + CUP_H / 2.0])
CUP_POS0 = tuple(float(v) for v in (np.array(BOARD_POS0) + BOARD_R0 @ _cup_local0))

# -- threshold constants (shared with scorer, tests, README, task.json) -----------------------------------------------
CUP_MAX_TILT_DEG = 25.0  # spill/tip threshold (matching water fill 2 cm below rim)
CUP_END_TILT_DEG = 5.0
CUP_SLIDE_MAX = 0.045  # m: cup stays within 4.5 cm of its spot on the board (prevents taking cup off)
BOARD_MAX_TILT_DEG = 18.0
BOARD_DRIFT_MAX = 0.060  # m: board stays within 6 cm of start XY
BOARD_LEVEL_DEG = 0.8  # deg: leveled flat on 4 feet (vs 3.8 deg unshimmed)
BOARD_ALMOST_LEVEL_DEG = 1.2
PAD_SEAT_TOL = 0.012  # m: pad center within 1.2 cm of bare brass foot center
PAD_NEAR_TOL = 0.025  # m: pad partway under/at the corner
PAD_PROGRESS_TOL = 0.090  # m: pad moved toward corner (start is ~0.130 m away)
PAD_FLAT_TILT_DEG = 10.0
PAD_ON_TABLE_TOL = 0.004  # m: pad lowest point within 4 mm of table
LIFT_CLEAR_Z = 0.006  # m: bare brass stud bottom lifted at least 6 mm above table
STILL_SPEED = 0.005  # m/s
STILL_SPIN = math.radians(5.0)
HOLD_STEPS = round(SUCCESS_HOLD_S / DT)
GRIPPERS = ("left_gripper", "right_gripper")

# -- scene props ------------------------------------------------------------------------------------------------------
MAPLE = kit.Look((0.48, 0.28, 0.14), 0.55)  # rich oiled walnut/dark-maple butcher block (high contrast on tan table)
WALNUT_INLAY = kit.Look((0.86, 0.74, 0.52), 0.55)  # pale maple inlay strip
RUBBER = kit.Look((0.12, 0.12, 0.14), 0.85)
WASHER = kit.Look((0.64, 0.66, 0.68), 0.35, metallic=0.8)
BRASS = kit.Look((0.86, 0.66, 0.22), 0.30, metallic=0.85)
RED_COLLAR = kit.Look((0.80, 0.16, 0.12), 0.5)
CERAMIC = kit.Look((0.95, 0.95, 0.93), 0.35)
WATER = kit.Look((0.18, 0.52, 0.78), 0.20)

_board_meshes = [
    kit.box((0.0, 0.0, FOOT_H + BOARD_Z / 2.0), (BOARD_X, BOARD_Y, BOARD_Z), MAPLE),
    kit.box((0.0, 0.0, FOOT_H + BOARD_Z + 0.0003), (BOARD_X - 0.024, 0.010, 0.0006), WALNUT_INLAY),
]
_board_colliders = [kit.Collider((0.0, 0.0, FOOT_H + BOARD_Z / 2.0), size=(BOARD_X, BOARD_Y, BOARD_Z))]

for key, (sx, sy) in ALL_FEET.items():
    fx, fy = sx * FOOT_X, sy * FOOT_Y
    if key == MISSING_CORNER:
        # Bare 12 mm brass mounting stud (8 mm shorter than good feet) ending at z_local = 0.008 m
        _board_meshes += [
            kit.cylinder((fx, fy, FOOT_H - SHORT_H / 2.0), FOOT_R * 0.68, SHORT_H, BRASS),
            kit.cylinder((fx, fy, FOOT_H - 0.002), FOOT_R * 0.88, 0.004, RED_COLLAR),
        ]
        _board_colliders.append(kit.Collider((fx, fy, FOOT_H - SHORT_H / 2.0), radius=FOOT_R * 0.68, height=SHORT_H))
    else:
        _board_meshes += [
            kit.cylinder((fx, fy, FOOT_H / 2.0 - 0.002), FOOT_R, FOOT_H - 0.004, RUBBER),
            kit.cylinder((fx, fy, FOOT_H - 0.002), FOOT_R * 0.95, 0.004, WASHER),
        ]
        _board_colliders.append(kit.Collider((fx, fy, FOOT_H / 2.0), radius=FOOT_R, height=FOOT_H))

BOARD = kit.Prop(
    "cutting_board",
    pos=BOARD_POS0,
    mass=BOARD_MASS,
    tilt_xyzw=BOARD_TILT_XYZW,
    size=(BOARD_X, BOARD_Y, FOOT_H + BOARD_Z),
    meshes=tuple(_board_meshes),
    colliders=tuple(_board_colliders),
)

PAD = kit.Prop(
    "rubber_pad",
    pos=PAD_START,
    mass=PAD_MASS,
    size=PAD_SIZE,
    meshes=(
        kit.box((0.0, 0.0, 0.0), PAD_SIZE, RUBBER),
        kit.box((0.0, 0.0, PAD_H / 2.0 + 0.0002), (0.014, 0.014, 0.0004), WASHER),
    ),
    colliders=(kit.Collider((0.0, 0.0, 0.0), size=PAD_SIZE),),
)

CUP = fk.FrictionProp(
    "cup",
    pos=CUP_POS0,
    mass=CUP_MASS,
    tilt_xyzw=BOARD_TILT_XYZW,
    size=(2 * CUP_R, 2 * CUP_R, CUP_H),
    meshes=(
        kit.lathe([
            (0.0, -CUP_H / 2),
            (CUP_R, -CUP_H / 2),
            (CUP_R, CUP_H / 2),
            (CUP_R - 0.003, CUP_H / 2),
            (CUP_R - 0.003, -CUP_H / 2 + 0.004),
            (0.0, -CUP_H / 2 + 0.004),
        ], CERAMIC),
        kit.cylinder((0.0, 0.0, CUP_H / 2 - 0.022), CUP_R - 0.0032, 0.004, WATER),
    ),
    colliders=(kit.Collider((0.0, 0.0, 0.0), radius=CUP_R, height=CUP_H),),
    friction=CUP_FRICTION,
    combine="min",
)

OBJECTS = [BOARD, PAD, CUP]

# Guardkit shape definitions in each object's own frame
BOARD_PARTS = (gk.Block((0.0, 0.0, FOOT_H + BOARD_Z / 2.0), (BOARD_X, BOARD_Y, BOARD_Z)),)
PAD_PARTS = (gk.Block((0.0, 0.0, 0.0), PAD_SIZE),)
CUP_PARTS = (gk.Ring(0.0, CUP_R, -CUP_H / 2.0, CUP_H / 2.0),)


# -- geometric measurements & per-trial memory -------------------------------------------------------------------------
def _tilt_deg(obj) -> float:
    x, y, _, _ = (float(v) for v in obj.quat_xyzw)
    return float(math.degrees(math.acos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _board_world_point(board_obj, local_pt) -> np.ndarray:
    R = _quat_to_mat(board_obj.quat_xyzw)
    return np.asarray(board_obj.pos, float) + R @ np.asarray(local_pt, float)


def _cup_in_board_ok(objects) -> bool:
    b, c = objects["cutting_board"], objects["cup"]
    c_local = _quat_to_mat(b.quat_xyzw).T @ (np.asarray(c.pos, float) - np.asarray(b.pos, float))
    dx = float(c_local[0] - CUP_BOARD_LOCAL_XY[0])
    dy = float(c_local[1] - CUP_BOARD_LOCAL_XY[1])
    z_ok = FOOT_H + BOARD_Z - 0.006 <= float(c_local[2]) - CUP_H / 2.0 <= FOOT_H + BOARD_Z + 0.015
    return math.hypot(dx, dy) <= CUP_SLIDE_MAX and z_ok


def _cup_hazard(objects) -> bool:
    c = objects["cup"]
    if _tilt_deg(c) > CUP_MAX_TILT_DEG:
        return True
    if float(np.asarray(c.low)[2]) < TOP + FOOT_H + BOARD_Z - 0.025:
        return True
    return not _cup_in_board_ok(objects)


def _board_hazard(objects) -> bool:
    b = objects["cutting_board"]
    if _tilt_deg(b) > BOARD_MAX_TILT_DEG:
        return True
    if float(np.asarray(b.low)[2]) < TOP - 0.010:
        return True
    return float(np.linalg.norm(np.asarray(b.pos[:2], float) - np.array([X_B, Y_B]))) > BOARD_DRIFT_MAX


def _pad_to_foot_xy(objects) -> float:
    foot_w = _board_world_point(objects["cutting_board"], (MISS_FOOT_LOCAL[0], MISS_FOOT_LOCAL[1], 0.0))
    pad_w = np.asarray(objects["rubber_pad"].pos, float)
    return float(np.linalg.norm(pad_w[:2] - foot_w[:2]))


def _pad_start_dist(objects) -> float:
    foot = _board_world_point(objects["cutting_board"], MISS_FOOT_LOCAL)
    return float(np.linalg.norm(np.asarray(PAD_START[:2]) - foot[:2]))


def _foot_bottom_z(objects) -> float:
    return float(_board_world_point(objects["cutting_board"], MISS_FOOT_LOCAL)[2])


def _pad_flat_on_table(objects) -> bool:
    p = objects["rubber_pad"]
    return _tilt_deg(p) <= PAD_FLAT_TILT_DEG and abs(float(np.asarray(p.low)[2]) - TOP) <= PAD_ON_TABLE_TOL


def _pad_seated(objects) -> bool:
    pad = objects["rubber_pad"]
    foot = _board_world_point(objects["cutting_board"], MISS_FOOT_LOCAL)
    local_foot = _quat_to_mat(pad.quat_xyzw).T @ (foot - np.asarray(pad.pos))
    return (
        _pad_flat_on_table(objects)
        and _pad_to_foot_xy(objects) <= PAD_SEAT_TOL
        and abs(float(local_foot[2]) - PAD_H / 2) <= 0.003
        and TOP + PAD_H - 0.003 <= foot[2] <= TOP + PAD_H + 0.003
    )


def _touched_any(objects) -> bool:
    return bool(
        gk.touching(objects, "cutting_board", BOARD_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "rubber_pad", PAD_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "cup", CUP_PARTS, grippers=GRIPPERS)
    )


def _still_all(objects) -> bool:
    return all(objects[n].resting(max_speed=STILL_SPEED, max_spin=STILL_SPIN) for n in ("cutting_board", "rubber_pad", "cup"))


HAZARD = gk.Latch(steps=2)


class _Tracker:
    def __init__(self):
        self.ep = kit.episode()
        self.key = None
        self.objs = None
        self.lifted = False
        self.hold_steps = 0


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
    HAZARD.feed(objects, _cup_hazard(objects) or _board_hazard(objects))
    if not HAZARD.on and _foot_bottom_z(objects) >= TOP + LIFT_CLEAR_Z:
        _T.lifted = True
    pad_ok = _pad_seated(objects)
    board_ok = _tilt_deg(objects["cutting_board"]) <= BOARD_LEVEL_DEG
    cup_ok = _tilt_deg(objects["cup"]) <= CUP_END_TILT_DEG and _cup_in_board_ok(objects)
    settled = pad_ok and board_ok and cup_ok and _still_all(objects) and all(g in objects for g in GRIPPERS) and not _touched_any(objects)
    _T.hold_steps = _T.hold_steps + 1 if settled else 0
    if os.environ.get("BOARD_TRACE") and n is not None and (_T.hold_steps or n % 10 == 0 or HAZARD.on):
        b, p, c = objects["cutting_board"], objects["rubber_pad"], objects["cup"]
        lg, rg = objects.get("left_gripper"), objects.get("right_gripper")
        print(
            f"TRACE step={n} fbz={_foot_bottom_z(objects)-TOP:.4f} btilt={_tilt_deg(b):.2f} "
            f"ctilt={_tilt_deg(c):.2f} pad=({p.pos[0]:.3f},{p.pos[1]:.3f},{p.pos[2]-TOP:.4f}) "
            f"pd={_pad_to_foot_xy(objects):.4f} haz={HAZARD.on} "
            f"lg_z={(lg.pos[2]-TOP if lg else 0):.3f} rg=({rg.pos[0]:.3f},{rg.pos[1]:.3f},{rg.pos[2]-TOP:.3f})",
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
    pad_dist = _pad_to_foot_xy(objects)
    pad_flat = _pad_flat_on_table(objects)
    b_tilt = _tilt_deg(objects["cutting_board"])
    c_tilt = _tilt_deg(objects["cup"])
    if _T.hold_steps >= HOLD_STEPS:
        return 1.0
    if _pad_seated(objects) and b_tilt <= BOARD_ALMOST_LEVEL_DEG and c_tilt <= CUP_END_TILT_DEG:
        return 0.75
    if (pad_flat and pad_dist <= PAD_NEAR_TOL) or (_foot_bottom_z(objects) >= TOP + LIFT_CLEAR_Z and pad_dist <= 0.06):
        return 0.50
    if pad_dist <= PAD_PROGRESS_TOL or _pad_start_dist(objects) - pad_dist >= 0.025:
        return 0.25
    return 0.0


# -- scripted demo: holds side edge horizontally, slides pad under bare front corner, lowers smoothly ----------------
def demo() -> dict[str, list]:
    """Lifting arm grips the missing-corner side edge horizontally (one finger underneath the wood) and lifts +13 mm;
    placing arm picks the spare rubber pad from front-center with a 35-deg forward pitch and places/slides it squarely
    under the bare corner foot; lifting arm sets the board flat and both arms back clear."""
    s = SIDE_SIGN
    lift_side = "left" if MISSING_CORNER == "FL" else "right"
    place_side = "right" if MISSING_CORNER == "FL" else "left"

    # 1. Lifting arm trajectory: enter horizontally from outer side (y = s * 0.24 -> s * 0.12) at x = 0.238
    # When FL (s = +1), left arm approaches along -Y -> roll = -pi/2; when FR (s = -1), right arm approaches along +Y -> roll = +pi/2.
    roll = -s * 1.35
    # At x_lift = 0.238 (x_rel = -0.042), wood underside sits at z_bot ~ TOP + 0.0155; wood top at TOP + 0.0435 (mid TOP + 0.0295).
    # Slight 13 deg wrist angle keeps joint5 inside its +-1.57 rad limit and lifts the palm clear of the tabletop.
    x_lift = 0.242
    y_approach = s * 0.200
    y_grip = s * 0.106  # ~14 mm past side edge so full finger pads clamp firmly without hitting front/rear feet
    z_grip = TOP + 0.033
    z_raised = z_grip + 0.030  # overcomes ~9 mm PD sag cleanly so bare brass stud clears TOP + 15 mm
    open_entry = float(0.027 / 0.047) * FINGER_OPEN  # 54 mm gap: lower finger stays above table & below wood edge
    pinch_board = float(0.011 / 0.047) * FINGER_OPEN  # firm clamp on 28 mm maple slab
    z_hover = TOP + 0.110

    lift_plan = [
        Waypoint(1.5, (x_lift, y_approach, z_hover), open_entry, "move above side entry"),
        kit.Tilted(2.0, (x_lift, y_approach, z_grip), open_entry, "orient horizontally outside board", 0.0, roll, 0.0),
        kit.Tilted(1.5, (x_lift, y_grip, z_grip), open_entry, "slip lower finger under board edge", 0.0, roll, 0.0),
        kit.Tilted(0.8, (x_lift, y_grip, z_grip), pinch_board, "pinch board edge", 0.0, roll, 0.0),
        kit.Tilted(2.0, (x_lift, y_grip, z_raised), pinch_board, "lift missing corner clear", 0.0, roll, 0.0),
        kit.Tilted(9.0, (x_lift, y_grip, z_raised), pinch_board, "hold corner while pad goes under", 0.0, roll, 0.0),
        kit.Tilted(2.0, (x_lift, y_grip, z_grip + 0.009), pinch_board, "lower gently onto pad", 0.0, roll, 0.0),
        kit.Tilted(0.8, (x_lift, y_grip, z_grip + 0.009), open_entry, "open fingers", 0.0, roll, 0.0),
        kit.Tilted(1.5, (x_lift, y_approach, z_grip + 0.009), open_entry, "withdraw sideways clear", 0.0, roll, 0.0),
        Waypoint(1.5, (x_lift, s * 0.24, z_hover), FINGER_OPEN, "lift clear"),
    ]

    # 2. Placing arm trajectory: pick rear edge of 44 x 36 x 8 mm pad with pitch = -0.58 rad (~33 deg forward tilt),
    # carry under lifted foot at (x_foot, s * FOOT_Y), release and retract smoothly along -X.
    pitch_pad = -0.58
    x_foot = X_B - FOOT_X  # 0.206
    y_foot = s * FOOT_Y  # +-0.104
    # Grip the pad at its rear/near half (-0.011 m behind pad center along X) with yaw = 0 (fingers close along Y across 36 mm)
    # Wait: let's check gripper tip offset along tilted Z_gripper (+0.548, 0, -0.836) * PAD_TIP_PAST_TCP (0.013):
    # Tip center relative to TCP = (+0.0071, 0, -0.0109). We want tip bottom around z_table + 0.002 -> z_tcp = TOP + 0.0135!
    # Holding 10 mm back from pad center -> TCP sits at x_pad - 0.010 - 0.007 = x_pad - 0.017.
    px0, py0, _ = PAD_START
    pad_hold = float(0.014 / 0.047) * FINGER_OPEN  # ~28 mm target -> squeezes 36 mm pad firmly
    pad_release = float(0.028 / 0.047) * FINGER_OPEN  # open slightly above 36 mm without bumping neighboring edge
    z_pick = TOP + 0.0145
    z_carry = TOP + 0.0175
    z_high = TOP + 0.095
    x_pick_tcp = px0 - 0.018
    x_place_tcp = x_foot - 0.019  # keeps finger_body (< 0.180) and pad_box (< 0.189) completely outside board x=0.190!

    place_plan = [
        Waypoint(1.5, (px0, py0 - s * 0.08, z_high), FINGER_OPEN, "wait for lift / move near pad"),
        Waypoint(3.5, (px0, py0, z_high), FINGER_OPEN, "wait above pad while corner lifts"),
        kit.Tilted(1.5, (x_pick_tcp, py0, z_high), FINGER_OPEN, "tilt for forward pad pick", 0.0, 0.0, pitch_pad),
        kit.Tilted(1.5, (x_pick_tcp, py0, z_pick), FINGER_OPEN, "lower onto rear edge of pad", 0.0, 0.0, pitch_pad),
        kit.Tilted(0.8, (x_pick_tcp, py0, z_pick), pad_hold, "grip rear edge of pad", 0.0, 0.0, pitch_pad),
        kit.Tilted(1.2, (x_pick_tcp, py0, z_carry), pad_hold, "lift pad 3 mm", 0.0, 0.0, pitch_pad),
        kit.Tilted(2.0, (x_foot - 0.065, y_foot, z_carry), pad_hold, "align outside bare corner", 0.0, 0.0, pitch_pad),
        kit.Tilted(2.0, (x_place_tcp, y_foot, z_pick), pad_hold, "slide pad under lifted foot", 0.0, 0.0, pitch_pad),
        kit.Tilted(0.8, (x_place_tcp, y_foot, z_pick), pad_release, "release pad flat", 0.0, 0.0, pitch_pad),
        kit.Tilted(1.5, (x_foot - 0.080, y_foot, z_pick + 0.004), pad_release, "retract along table", 0.0, 0.0, pitch_pad),
        Waypoint(1.5, (x_foot - 0.090, -s * 0.16, z_high), FINGER_OPEN, "park clear"),
        Waypoint(4.0, (x_foot - 0.090, -s * 0.16, z_high), FINGER_OPEN, "hold clear to settle"),
    ]

    if os.environ.get("DEMO_NO_LIFT"):
        # Negative one-arm control: lifting arm stays idle while placing arm tries to push the pad under unlifted board.
        lift_plan = [Waypoint(22.0, (x_lift, y_approach, z_hover), FINGER_OPEN, "idle")]
    return {lift_side: lift_plan, place_side: place_plan}
