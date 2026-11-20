"""A 1:1 rotary spreader hopper gate assembly (18.0 x 14.0 x 10.0 cm) sits center-table on a workbench alongside a
staging stand holding a locking stop pin, randomized symmetrically left or right (SIDE=L|R).

The horizontal metering slide gate (PrismaticJoint with return spring pulling toward 0.0 cm closed) closes the hopper
chute at rest and covers the Setting 4 (4.0 cm) clevis bore. Over-opening past Setting 5 (> 5.2 cm) spills granules
from the hopper throat (latched 0.0 hazard).

Bimanual skill: one YAM arm grasps the gate T-handle, slides the spring-loaded gate open to ~4.5 cm (aligning the
Setting 4 gate slot with the clevis bore without over-opening), and holds it open against the return spring while the
other arm picks the locking stop pin from its staging stand, inserts it through the clevis bridge, gate slot, and
lower socket, and releases; the gate arm then eases the gate back against the seated pin at 4.0 +/- 0.4 cm and both
arms release and hold clear for 1.0 s (20 steps at 20 Hz).
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
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
DT = 1.0 / 20
RUN_LIMIT_STEPS = 2400
INSTRUCTION = (
    "Slide and hold the spring-loaded spreader hopper gate open to Setting 4 (4.0 cm) without over-opening, "
    "insert and seat the locking stop pin through the Setting 4 clevis slot, and release both hands clear so "
    "the gate stays pinned at 4.0 +/- 0.4 cm."
)

# -- fixed arm assignment, default L (set SIDE=R for the mirrored layout) -------------------------------------
SIDES = ("L", "R")


def _draw_side() -> str:
    env = os.environ.get("SIDE", "").strip().upper()
    return env if env in SIDES else "L"


SIDE = _draw_side()
SIDE_SIGN = 1.0 if SIDE == "L" else -1.0

# -- 1:1 rotary spreader hopper gate assembly (18.0 x 14.0 x 10.0 cm, m) ------------------------------------------------
GATE_X, GATE_Y = 0.295, 0.0
HOUSING_X, HOUSING_Y, HOUSING_Z = 0.140, 0.180, 0.100  # 14.0 x 18.0 x 10.0 cm envelope

BORE_X_LOCAL = -0.048
BORE_Y_LOCAL = -SIDE_SIGN * 0.015
BORE_X = GATE_X + BORE_X_LOCAL  # 0.247 m
BORE_Y = GATE_Y + BORE_Y_LOCAL
BORE_W = 0.020  # 20 x 20 mm square bore through clevis bridge and lower socket

# Prismatic slide gate travel & return spring
GATE_STROKE = 0.065  # 6.5 cm full travel
SETTING4_M = 0.040  # 4.0 cm calibrated Setting 4 target
SETTING4_TOL_M = 0.004  # +/- 0.4 cm -> [3.6, 4.4] cm
OVER_OPEN_M = 0.052  # > 5.2 cm (past Setting 5) spills granules -> latched 0.0 hazard
SPRING_K = 70.0  # N/m pulling toward 0.0 cm closed
SPRING_DAMP = 4.0  # N*s/m
GATE_MASS = 0.18

# Gate pull handle & scale pointer coordinates
HANDLE_Y0_LOCAL = SIDE_SIGN * 0.118
HANDLE_X = GATE_X - 0.010  # 0.285 m
HANDLE_Y0 = GATE_Y + HANDLE_Y0_LOCAL
HANDLE_Z_LOCAL = 0.060  # z in [0.046, 0.074] above front rail (0.046)
SCALE_Y0_LOCAL = SIDE_SIGN * 0.020  # 0 cm mark clear of clevis bridge

# Pin staging stand & locking stop pin
STAND_X = BORE_X  # 0.247 m
STAND_Y = -SIDE_SIGN * 0.185
STAND_H = 0.052
PIN_SHAFT_W = 0.010  # 10 x 10 mm square steel tang
PIN_SHAFT_LEN = 0.044
PIN_COLLAR_W = 0.028
PIN_COLLAR_H = 0.008
PIN_KNOB_SIZE = (0.028, 0.014, 0.026)
PIN_MASS = 0.10
PIN_STAND_POS = (STAND_X, STAND_Y, TOP)
PIN_START_POS = (STAND_X, STAND_Y, TOP + STAND_H)  # origin at underside of stop collar

# Scoring & safety thresholds
PIN_SEAT_Z = TOP + 0.060  # clevis bridge top surface
PIN_SEAT_XY_TOL = 0.010
PIN_SEAT_Z_TOL = 0.006
PIN_TILT_MAX_DEG = 10.0
PIN_LIFT_DELTA_Z = 0.012
GATE_STILL_VEL = 0.005  # m/s
PIN_STILL_SPEED = 0.005
PIN_STILL_SPIN = math.radians(5.0)
HOLD_STEPS = round(SUCCESS_HOLD_S / DT)
GRIPPERS = ("left_gripper", "right_gripper")

# -- visual materials --------------------------------------------------------------------------------------------------
DARK_STEEL = kit.Look((0.18, 0.20, 0.22), roughness=0.45, metallic=0.70)
BRUSHED_STEEL = kit.Look((0.76, 0.78, 0.81), roughness=0.28, metallic=0.82)
HOPPER_GREEN = kit.Look((0.12, 0.44, 0.20), roughness=0.45, metallic=0.05)
CLEVIS_YELLOW = kit.Look((0.92, 0.74, 0.10), roughness=0.35, metallic=0.15)
SCALE_WHITE = kit.Look((0.94, 0.94, 0.92), roughness=0.45)
TICK_DARK = kit.Look((0.14, 0.15, 0.17), roughness=0.50)
SET4_GREEN = kit.Look((0.14, 0.72, 0.28), roughness=0.40)
HAZARD_RED = kit.Look((0.86, 0.16, 0.14), roughness=0.40)
POINTER_ORANGE = kit.Look((0.96, 0.36, 0.08), roughness=0.35)
PIN_RED = kit.Look((0.88, 0.20, 0.12), roughness=0.38)
GRANULE_TAN = kit.Look((0.78, 0.68, 0.38), roughness=0.75)
GRANULE_TEAL = kit.Look((0.28, 0.64, 0.52), roughness=0.70)


def _box_span(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, look: kit.Look, collider: bool = True,
              friction: float | None = None):
    c = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    sz = (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
    m = kit.box(c, sz, look)
    col = kit.Collider(c, size=sz, friction=friction) if collider else None
    return m, col


def _build_hopper_gate() -> kit.Mechanism:
    s = SIDE_SIGN
    meshes, cols = [], []

    def add(x0, x1, y0, y1, z0, z1, look, collider=True, friction=None):
        m, c = _box_span(x0, x1, y0, y1, z0, z1, look, collider=collider, friction=friction)
        meshes.append(m)
        if c is not None:
            cols.append(c)

    # 1. Base plate (14 x 18 x 1.0 cm) & rear guide rail
    add(-0.070, 0.070, -0.090, 0.090, 0.000, 0.010, DARK_STEEL)
    add(0.039, 0.057, -0.090, 0.090, 0.010, 0.046, DARK_STEEL)

    # 2. Front rail lower socket (z=0.010..0.026) with 20x20 mm bore at (BORE_X_LOCAL, BORE_Y_LOCAL)
    bx0, bx1 = BORE_X_LOCAL - BORE_W / 2, BORE_X_LOCAL + BORE_W / 2  # [-0.058, -0.038]
    by0, by1 = BORE_Y_LOCAL - BORE_W / 2, BORE_Y_LOCAL + BORE_W / 2
    add(-0.066, -0.030, -0.090, by0, 0.010, 0.026, DARK_STEEL)
    add(-0.066, -0.030, by1, 0.090, 0.010, 0.026, DARK_STEEL)
    add(-0.066, bx0, by0, by1, 0.010, 0.026, DARK_STEEL)
    add(bx1, -0.030, by0, by1, 0.010, 0.026, DARK_STEEL)

    # 3. Front rail rate-bar U-channel walls (z=0.026..0.046)
    add(-0.066, bx0, -0.090, 0.090, 0.026, 0.046, DARK_STEEL)
    add(bx1, -0.030, -0.090, 0.090, 0.026, 0.046, DARK_STEEL)

    # 4. Upper clevis bridge around Setting 4 bore (z=0.046..0.060, safety yellow)
    cy0, cy1 = BORE_Y_LOCAL - 0.028, BORE_Y_LOCAL + 0.028
    add(-0.066, -0.030, cy0, by0, 0.046, 0.060, CLEVIS_YELLOW)
    add(-0.066, -0.030, by1, cy1, 0.046, 0.060, CLEVIS_YELLOW)
    add(-0.066, bx0, by0, by1, 0.046, 0.060, CLEVIS_YELLOW)
    add(bx1, -0.030, by0, by1, 0.046, 0.060, CLEVIS_YELLOW)

    # 5. Calibrated rate scale plate (0..6 cm) on front rail top & front face (visual ticks only)
    sy_lo = min(SCALE_Y0_LOCAL - s * 0.005, SCALE_Y0_LOCAL + s * 0.065)
    sy_hi = max(SCALE_Y0_LOCAL - s * 0.005, SCALE_Y0_LOCAL + s * 0.065)
    add(-0.0655, -0.0575, sy_lo, sy_hi, 0.046, 0.0472, SCALE_WHITE, collider=False)
    add(-0.0668, -0.0660, sy_lo, sy_hi, 0.034, 0.0465, SCALE_WHITE, collider=False)
    for cm in range(7):
        yc = SCALE_Y0_LOCAL + s * (cm * 0.010)
        if cm == 4:
            lk_t, w_t = SET4_GREEN, 0.0040
        elif cm >= 5:
            lk_t, w_t = HAZARD_RED, 0.0022
        else:
            lk_t, w_t = TICK_DARK, 0.0018
        add(-0.0652, -0.0578, yc - w_t / 2, yc + w_t / 2, 0.0472, 0.0480, lk_t, collider=False)
        add(-0.0672, -0.0665, yc - w_t / 2, yc + w_t / 2, 0.035, 0.0462, lk_t, collider=False)
    # Continuous red over-opening hazard zone bar (> 5.2 cm)
    oz0 = SCALE_Y0_LOCAL + s * OVER_OPEN_M
    oz1 = SCALE_Y0_LOCAL + s * 0.064
    add(-0.0645, -0.0585, min(oz0, oz1), max(oz0, oz1), 0.0470, 0.0476, HAZARD_RED, collider=False)

    # 6. Hopper funnel throat (z=0.046..0.100, x in [-0.020, +0.068], y in [-0.056, +0.056])
    add(-0.020, -0.012, -0.056, 0.056, 0.046, 0.100, HOPPER_GREEN)
    add(0.060, 0.068, -0.056, 0.056, 0.046, 0.100, HOPPER_GREEN)
    add(-0.012, 0.060, -0.056, -0.048, 0.046, 0.100, HOPPER_GREEN)
    add(-0.012, 0.060, 0.048, 0.056, 0.046, 0.100, HOPPER_GREEN)
    # Upper hopper rim lip (4 narrow border strips leaving throat open) & interior granular fertilizer bed
    add(-0.022, -0.010, -0.058, 0.058, 0.094, 0.100, DARK_STEEL, collider=False)
    add(0.058, 0.070, -0.058, 0.058, 0.094, 0.100, DARK_STEEL, collider=False)
    add(-0.010, 0.058, -0.058, -0.046, 0.094, 0.100, DARK_STEEL, collider=False)
    add(-0.010, 0.058, 0.046, 0.058, 0.094, 0.100, DARK_STEEL, collider=False)
    add(0.012, 0.059, -0.047, 0.047, 0.046, 0.066, GRANULE_TAN, collider=False)
    for gx in (0.020, 0.034, 0.048):
        for gy in (-0.032, -0.011, 0.011, 0.032):
            meshes.append(kit.cylinder((gx, gy, 0.067), 0.0055, 0.005, GRANULE_TEAL))

    # 7. Moving spring-returned horizontal slide gate (PrismaticJoint along Y)
    g_meshes, g_cols = [], []

    def g_add(x0, x1, y0, y1, z0, z1, look, collider=True):
        m, c = _box_span(x0, x1, y0, y1, z0, z1, look, collider=collider)
        g_meshes.append(m)
        if c is not None:
            g_cols.append(c)

    # Stainless metering shutter plate covering chute [-0.045, +0.045] at q=0 (visual plate + throat sweep)
    sh_y0 = -0.045 if s > 0 else -0.090
    sh_y1 = 0.090 if s > 0 else 0.045
    g_add(-0.032, 0.036, sh_y0, sh_y1, 0.014, 0.022, BRUSHED_STEEL, collider=False)

    # Rate-bar stop blocks in front clevis channel (x in [-0.054, -0.042], z in [0.028, 0.044])
    # Leading stop block (+s side of Setting 4 window): -s face at -s * 0.055 stops against pin (-s*0.015) at 4.0 cm
    lead_a, lead_b = -s * 0.055, s * 0.096
    g_add(-0.054, -0.042, min(lead_a, lead_b), max(lead_a, lead_b), 0.028, 0.044, BRUSHED_STEEL)
    # Trailing stop block (-s side of 26 mm Setting 4 window): [-s*0.090, -s*0.081]
    tr_a, tr_b = -s * 0.090, -s * 0.081
    g_add(-0.054, -0.042, min(tr_a, tr_b), max(tr_a, tr_b), 0.028, 0.044, BRUSHED_STEEL)
    # Connecting bridge bar behind bore on gate (x in [-0.034, -0.026], z in [0.016, 0.024]) linking trailing & leading blocks
    g_add(-0.034, -0.026, -0.090, 0.090, 0.016, 0.024, DARK_STEEL, collider=False)

    # Bright orange Setting pointer riding along the calibrated front scale
    g_add(-0.065, -0.055, SCALE_Y0_LOCAL - 0.0025, SCALE_Y0_LOCAL + 0.0025, 0.047, 0.056, POINTER_ORANGE, collider=False)

    # Rigid outer neck & T-grip pull handle (at y = s * 0.118, z in [0.046, 0.074])
    nk_a, nk_b = s * 0.088, s * 0.118
    g_add(-0.054, 0.014, min(nk_a, nk_b), max(nk_a, nk_b), 0.028, 0.046, BRUSHED_STEEL, collider=False)
    g_add(-0.034, 0.014, HANDLE_Y0_LOCAL - 0.007, HANDLE_Y0_LOCAL + 0.007, 0.046, 0.074, POINTER_ORANGE)

    q_lo, q_hi = (0.0, GATE_STROKE) if s > 0 else (-GATE_STROKE, 0.0)
    gate_link = kit.Link(
        "gate",
        kind="prismatic",
        axis="Y",
        anchor=(0.0, 0.0, 0.0),
        lower=q_lo,
        upper=q_hi,
        start=0.0,
        mass=GATE_MASS,
        floating=True,
        meshes=tuple(g_meshes),
        colliders=tuple(g_cols),
        law=mk.Spring(rest=0.0, k=SPRING_K, damp=SPRING_DAMP),
    )

    return kit.Mechanism(
        "hopper_gate",
        pos=(GATE_X, GATE_Y, TOP),
        size=(HOUSING_X, HOUSING_Y, HOUSING_Z),
        mass=None,
        meshes=tuple(meshes),
        colliders=tuple(cols),
        links=(gate_link,),
    )


def _build_pin_stand() -> kit.Prop:
    meshes, cols = [], []

    def add(x0, x1, y0, y1, z0, z1, look):
        m, c = _box_span(x0, x1, y0, y1, z0, z1, look)
        meshes.append(m)
        cols.append(c)

    # 6.4 x 6.4 x 5.2 cm block with 16 x 16 mm central socket (z = 0.007..0.052)
    add(-0.032, 0.032, -0.032, 0.032, 0.000, 0.007, DARK_STEEL)
    add(-0.032, -0.008, -0.032, 0.032, 0.007, STAND_H, DARK_STEEL)
    add(0.008, 0.032, -0.032, 0.032, 0.007, STAND_H, DARK_STEEL)
    add(-0.008, 0.008, -0.032, -0.008, 0.007, STAND_H, DARK_STEEL)
    add(-0.008, 0.008, 0.008, 0.032, 0.007, STAND_H, DARK_STEEL)
    meshes.append(kit.box((0.0, 0.0, STAND_H + 0.0004), (0.044, 0.044, 0.0008), CLEVIS_YELLOW))

    return kit.Prop(
        "pin_stand",
        pos=PIN_STAND_POS,
        mass=None,
        size=(0.064, 0.064, STAND_H),
        meshes=tuple(meshes),
        colliders=tuple(cols),
    )


def _build_stop_pin() -> kit.Prop:
    hw = PIN_SHAFT_W / 2
    cw = PIN_COLLAR_W / 2
    kx, ky, kz = PIN_KNOB_SIZE
    meshes = (
        kit.box((0.0, 0.0, -PIN_SHAFT_LEN / 2), (PIN_SHAFT_W, PIN_SHAFT_W, PIN_SHAFT_LEN), BRUSHED_STEEL),
        kit.box((0.0, 0.0, PIN_COLLAR_H / 2), (PIN_COLLAR_W, PIN_COLLAR_W, PIN_COLLAR_H), DARK_STEEL),
        kit.box((0.0, 0.0, PIN_COLLAR_H + 0.004), (0.012, 0.012, 0.008), BRUSHED_STEEL),
        kit.box((0.0, 0.0, PIN_COLLAR_H + 0.008 + kz / 2), (kx, ky, kz), PIN_RED),
    )
    cols = (
        kit.Collider((0.0, 0.0, -PIN_SHAFT_LEN / 2), size=(PIN_SHAFT_W, PIN_SHAFT_W, PIN_SHAFT_LEN), friction=0.35),
        kit.Collider((0.0, 0.0, PIN_COLLAR_H / 2), size=(PIN_COLLAR_W, PIN_COLLAR_W, PIN_COLLAR_H), friction=0.50),
        kit.Collider((0.0, 0.0, PIN_COLLAR_H + 0.008 + kz / 2), size=(kx, ky, kz), friction=0.90),
    )
    return kit.Prop(
        "stop_pin",
        pos=PIN_START_POS,
        mass=PIN_MASS,
        size=(kx, cw * 2, PIN_SHAFT_LEN + PIN_COLLAR_H + 0.008 + kz),
        meshes=meshes,
        colliders=cols,
    )


HOPPER_GATE = _build_hopper_gate()
PIN_STAND = _build_pin_stand()
STOP_PIN = _build_stop_pin()
OBJECTS = [HOPPER_GATE, PIN_STAND, STOP_PIN]

# Guardkit shape definitions
BASE_PARTS = (
    gk.Block((0.0, 0.0, 0.050), (HOUSING_X, HOUSING_Y, HOUSING_Z)),
)
GATE_PARTS = (
    gk.Block((-0.010, HANDLE_Y0_LOCAL, HANDLE_Z_LOCAL), (0.052, 0.020, 0.032)),
    gk.Block((-0.048, 0.0, 0.036), (0.016, 0.190, 0.020)),
)
PIN_PARTS = (
    gk.Block((0.0, 0.0, PIN_COLLAR_H + 0.008 + PIN_KNOB_SIZE[2] / 2), PIN_KNOB_SIZE),
    gk.Block((0.0, 0.0, PIN_COLLAR_H / 2), (PIN_COLLAR_W, PIN_COLLAR_W, PIN_COLLAR_H)),
)


# -- measurements & scoring --------------------------------------------------------------------------------------------
def _gate_open_m(objects) -> float:
    return float(SIDE_SIGN * kit.joint(objects, "hopper_gate/gate").pos)


def _gate_vel_ms(objects) -> float:
    return float(abs(kit.joint(objects, "hopper_gate/gate").vel))


def _tilt_deg(obj) -> float:
    x, y, _, _ = (float(v) for v in obj.quat_xyzw)
    return float(math.degrees(math.acos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _pin_dropped(pin) -> bool:
    return float(np.asarray(pin.low)[2]) < TOP - 0.015


def _pin_lifted(pin) -> bool:
    return float(pin.pos[2]) > TOP + STAND_H + PIN_LIFT_DELTA_Z or float(np.hypot(pin.pos[0] - STAND_X, pin.pos[1] - STAND_Y)) > 0.040


def _pin_seated(pin) -> bool:
    dx = abs(float(pin.pos[0]) - BORE_X)
    dy = abs(float(pin.pos[1]) - BORE_Y)
    dz = abs(float(pin.pos[2]) - PIN_SEAT_Z)
    return bool(dx <= PIN_SEAT_XY_TOL and dy <= PIN_SEAT_XY_TOL and dz <= PIN_SEAT_Z_TOL and _tilt_deg(pin) <= PIN_TILT_MAX_DEG)


def _gate_at_setting4(gate_m: float) -> bool:
    return abs(gate_m - SETTING4_M) <= SETTING4_TOL_M


def _touched_any(objects) -> bool:
    return bool(
        gk.touching(objects, "hopper_gate", BASE_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "hopper_gate/gate", GATE_PARTS, grippers=GRIPPERS)
        or gk.touching(objects, "stop_pin", PIN_PARTS, grippers=GRIPPERS)
    )


def _still_all(objects) -> bool:
    pin_still = objects["stop_pin"].resting(max_speed=PIN_STILL_SPEED, max_spin=PIN_STILL_SPIN)
    return bool(_gate_vel_ms(objects) <= GATE_STILL_VEL and pin_still)


HAZARD = gk.Latch(steps=2)


class _Tracker:
    def __init__(self):
        self.ep = kit.episode()
        self.key = None
        self.objs = None
        self.hold_steps = 0
        self.steady = []


_T = _Tracker()


def _step_still(gate_m: float, pin) -> bool:
    sig = (gate_m, np.asarray(pin.pos, float).copy(), np.asarray(pin.quat_xyzw, float).copy())
    _T.steady.append(sig)
    del _T.steady[:-HOLD_STEPS]
    if len(_T.steady) < 2:
        return True
    g0, p0, q0 = _T.steady[-2]
    g1, p1, q1 = _T.steady[-1]
    step_ok = (
        abs(g1 - g0) <= 0.001
        and float(np.linalg.norm(p1 - p0)) <= 0.001
        and abs(float(np.dot(q1, q0))) >= math.cos(math.radians(0.5))
    )
    if not step_ok:
        return False
    if len(_T.steady) == HOLD_STEPS:
        gw, pw, qw = _T.steady[0]
        return (
            abs(g1 - gw) <= 0.002
            and float(np.linalg.norm(p1 - pw)) <= 0.002
            and abs(float(np.dot(q1, qw))) >= math.cos(math.radians(1.0))
        )
    return True


def _step(objects) -> None:
    global _T
    if _T.ep != kit.episode():
        _T = _Tracker()
    n = kit._EPISODE.get("last")
    key = (kit.episode(), n) if n is not None else None
    if (key is not None and key == _T.key) or (key is None and objects is _T.objs):
        return
    _T.key, _T.objs = key, objects

    gate_m = _gate_open_m(objects)
    pin = objects["stop_pin"]
    HAZARD.feed(objects, bool(gate_m > OVER_OPEN_M or _pin_dropped(pin)))

    goal_clear = (
        not HAZARD.on
        and _gate_at_setting4(gate_m)
        and _pin_seated(pin)
        and not _touched_any(objects)
    )
    if goal_clear and (_still_all(objects) or _step_still(gate_m, pin)):
        _T.hold_steps += 1
    else:
        _T.hold_steps = 0
        if not goal_clear:
            _T.steady.clear()

    if os.environ.get("SPREADER_TRACE") and n is not None and (n % 10 == 0 or HAZARD.on or _T.hold_steps):
        gp = objects["right_gripper" if SIDE == "L" else "left_gripper"].pos
        print(
            f"TRACE step={n} gate={gate_m*100:.2f}cm vel={_gate_vel_ms(objects):.4f} "
            f"pin=({pin.pos[0]:.3f},{pin.pos[1]:.3f},{pin.pos[2]-TOP:.4f}) "
            f"grip=({gp[0]:.3f},{gp[1]:.3f},{gp[2]-TOP:.4f}) tilt={_tilt_deg(pin):.1f}deg "
            f"haz={HAZARD.on} touch={_touched_any(objects)} hold={_T.hold_steps}",
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
    gate_m = _gate_open_m(objects)
    pin = objects["stop_pin"]
    if _gate_at_setting4(gate_m) and _pin_seated(pin):
        return 0.75
    if 0.035 <= gate_m <= 0.049 and _pin_lifted(pin):
        return 0.50
    if 0.032 <= gate_m <= OVER_OPEN_M:
        return 0.25
    return 0.0


# -- symmetric bimanual scripted demo & negative controls --------------------------------------------------------------
def demo() -> dict[str, list]:
    """Symmetric bimanual calibration:
    - Gate arm pulls and holds the spring-returned slide gate open past 4.0 cm (~4.6 cm) without exceeding 5.2 cm.
    - Pin arm picks the locking stop pin from the opposite-side stand, inserts it through the Setting 4 clevis bore,
      releases, and clears back to READY.
    - Gate arm eases the gate back against the seated pin at 4.0 cm, releases, and clears to READY for 1.0 s settle.

    Negative controls:
    - DEMO_NO_PIN=1: gate arm opens to Setting 4 and releases early without pinning -> spring snaps gate to 0 cm (0.0).
    - DEMO_OVER_OPEN=1: gate arm yanks gate open to 6.0 cm (> 5.2 cm spill threshold) -> latched 0.0 hazard.
    """
    s = SIDE_SIGN
    gate_side = "left" if SIDE == "L" else "right"
    pin_side = "right" if SIDE == "L" else "left"
    no_pin = os.environ.get("DEMO_NO_PIN", "0") == "1"
    over_open = os.environ.get("DEMO_OVER_OPEN", "0") == "1"

    open_handle = 0.55 * FINGER_OPEN
    open_pin = 0.50 * FINGER_OPEN
    z_handle = TOP + HANDLE_Z_LOCAL + 0.001  # tips at TOP + 0.048, cleanly above 0.046 rail
    z_pin_stand = TOP + STAND_H + 0.029
    z_pin_bore = PIN_SEAT_Z + 0.029

    # Target pull distances (includes ~4 mm arm PD compliance against ~3.3 N return spring)
    pull_m = 0.064 if over_open else 0.051
    ease_m = 0.042

    if no_pin:
        gate_plan = [
            Waypoint(1.0, (HANDLE_X, HANDLE_Y0, TOP + 0.140), open_handle, "above_handle"),
            Waypoint(0.8, (HANDLE_X, HANDLE_Y0, z_handle), open_handle, "at_handle"),
            Waypoint(0.4, (HANDLE_X, HANDLE_Y0, z_handle), FINGER_CLOSED, "grip_handle"),
            Waypoint(1.2, (HANDLE_X, HANDLE_Y0 + s * 0.049, z_handle), FINGER_CLOSED, "pull_open"),
            Waypoint(1.5, (HANDLE_X, HANDLE_Y0 + s * 0.049, z_handle), FINGER_CLOSED, "hold_no_pin"),
            Waypoint(0.4, (HANDLE_X, HANDLE_Y0 + s * 0.049, z_handle), open_handle, "release_early"),
            Waypoint(0.8, (HANDLE_X, HANDLE_Y0 + s * 0.049, TOP + 0.145), FINGER_OPEN, "lift_clear"),
            kit.Home(1.4, label="return_home"),
            kit.Home(1.5, label="wait_snap_shut"),
        ]
        pin_plan = [kit.Home(sum(w.duration for w in gate_plan), label="stay_home")]
        return {gate_side: gate_plan, pin_side: pin_plan}

    gate_plan = [
        Waypoint(1.0, (HANDLE_X, HANDLE_Y0, TOP + 0.140), open_handle, "above_handle"),
        Waypoint(0.8, (HANDLE_X, HANDLE_Y0, z_handle), open_handle, "at_handle"),
        Waypoint(0.4, (HANDLE_X, HANDLE_Y0, z_handle), FINGER_CLOSED, "grip_handle"),
        Waypoint(1.2, (HANDLE_X, HANDLE_Y0 + s * pull_m, z_handle), FINGER_CLOSED,
                 "over_open_gate" if over_open else "pull_to_setting4"),
        Waypoint(6.8, (HANDLE_X, HANDLE_Y0 + s * pull_m, z_handle), FINGER_CLOSED, "hold_open_for_pin"),
        Waypoint(0.8, (HANDLE_X, HANDLE_Y0 + s * ease_m, z_handle), FINGER_CLOSED, "seat_against_pin"),
        Waypoint(0.4, (HANDLE_X, HANDLE_Y0 + s * ease_m, z_handle), open_handle, "release_handle"),
        Waypoint(0.8, (HANDLE_X, HANDLE_Y0 + s * ease_m, TOP + 0.145), FINGER_OPEN, "lift_clear"),
        kit.Home(1.4, label="return_home"),
        kit.Home(2.4, label="hold_clear"),
    ]

    pin_plan = [
        Waypoint(1.0, (STAND_X, STAND_Y, TOP + 0.155), open_pin, "above_pin_stand"),
        Waypoint(0.8, (STAND_X, STAND_Y, z_pin_stand), open_pin, "at_pin"),
        Waypoint(0.4, (STAND_X, STAND_Y, z_pin_stand), FINGER_CLOSED, "grip_pin"),
        Waypoint(1.0, (STAND_X, STAND_Y, TOP + 0.165), FINGER_CLOSED, "lift_pin"),
        Waypoint(1.4, (BORE_X, BORE_Y, TOP + 0.165), FINGER_CLOSED, "over_bore"),
        Waypoint(0.8, (BORE_X, BORE_Y, TOP + 0.148), FINGER_CLOSED, "settle_over_bore"),
        Waypoint(1.4, (BORE_X, BORE_Y, z_pin_bore - 0.002), FINGER_CLOSED, "seat_pin"),
        Waypoint(0.4, (BORE_X, BORE_Y, z_pin_bore - 0.002), FINGER_CLOSED, "rest_pin"),
        Waypoint(0.4, (BORE_X, BORE_Y, z_pin_bore - 0.002), open_pin, "release_pin"),
        Waypoint(0.8, (BORE_X, BORE_Y, TOP + 0.165), FINGER_OPEN, "clear_pin"),
        kit.Home(1.4, label="pin_arm_home"),
        kit.Home(5.2, label="wait_clear"),
    ]

    return {gate_side: gate_plan, pin_side: pin_plan}
