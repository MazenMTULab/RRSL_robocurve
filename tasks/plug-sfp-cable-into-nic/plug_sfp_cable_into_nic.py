"""On the station table: a PCIe network card stands upright on a foot in front of the left arm, its face toward that arm, its
two SFP ports at its upper edge facing straight up. An SFP-to-SC fibre patch cable lies across the table: its SC plug (blue)
rests in a small tray, fixed in place, and its SFP plug (gold) lies in a slanted holder to the left of the card, tip down at
45 degrees, the cable leaving its back end and draping to the tray. The robot must lift the SFP plug out, stand it upright,
and plug it, tip first, all the way into the card's port nearer the arms (x about 0.29), with the cable's other end staying in
its tray.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sfp_cable_scoring as sc
from fibrekit import Plug, cable
from scenekit import Shape
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

R_CARD = sc.card_rotation()  # the card's upright frame in the world: leaned back 45 degrees, turned 8
HOLD_TILT = math.radians(45.0)  # the plug lies leaning 45 degrees, its back end toward the cable tray (-y)
GRIP_YAW = -math.pi / 2  # the pads close along x, across the plug's width
R_HOLD = sc.rx(HOLD_TILT) @ sc.rz(math.pi / 2)  # the holder's frame: the plug's axis is its z, its plate underneath
CLEARANCE = 0.0  # m a side added to the AIC port; 0 = the AIC model's own size
CARD_Z_CUT = 0.022  # WHY: only the card's upper part is modelled (the rest sits inside the cradle)
BASE_T = 0.006
SKIN, METAL, GOLD, STEEL, BLUE = (0.10, 0.45, 0.20), (0.72, 0.74, 0.77), (0.85, 0.70, 0.20), (0.35, 0.36, 0.38), (0.10, 0.30, 0.75)
CAGE = "10099100-011lfc001"
MOUTH_XY = (0.30, 0.08)
HOLDER_XY = (0.24, 0.30)  # where the SFP plug's origin is, from above
SC_XY = (0.14, -0.10)  # the SC plug's centre, from above
GEOM = sc.GEOM
_M_CARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # card model frame -> its upright frame

# the AIC plugs: SFP module body 47.3 x 13.75 x 8.6 mm, 20 g; SC plug 26 x 25 x 10 mm, 40 g
SFP_BODY = (sc.PLUG_LEN, 0.01375, 0.0086)
# the module's latch end (AIC head and latch colliders): 14 x 12.2 mm, 11.5 mm long, on the cable side of the body, 8 mm beyond it
SFP_HEAD = ((-0.026, 0.0, 0.0), (0.0115, 0.014, 0.0122), (0.80, 0.65, 0.18))
SC_BODY = (0.026, 0.025, 0.010)


def _upright(centre, size, m):
    return tuple((m @ np.array(centre)).tolist()), (size[2], size[0], size[1])


def _card_boxes():
    """(centre, size, colour) in the card's upright frame, plus the cradle."""
    centres = [(b["x_min"] + b["x_max"]) / 2 for b in sc.NIC["bays"]]
    out = []
    for b in sc.NIC["boxes"]:
        cx, cy, cz = b["pos"]
        sx, sy, sz = b["size"]
        color = SKIN
        if b["name"].startswith(CAGE):
            color = METAL
            near = min(centres, key=lambda c: abs(c - cx))
            if sx < 0.002 and sz > 0.01:  # a port's side wall
                cx += CLEARANCE if cx > near else -CLEARANCE
            elif sy > 0.04 and sx > 0.01 and sz < 0.0015:  # a port's floor or roof
                cz += -CLEARANCE if cz < 0.005 else CLEARANCE
        if CLEARANCE and b["name"] in ("plane001_collider_box", "plane001_collider_box.001"):  # the bracket's slot
            cz += -CLEARANCE if cz < sc.BAY_X else CLEARANCE
        c, s = _upright((cx, cy, cz), (sx, sy, sz), _M_CARD)
        lo, hi = c[2] - s[2] / 2, c[2] + s[2] / 2
        if hi <= CARD_Z_CUT:
            continue
        if lo < CARD_Z_CUT:  # clip the box to the modelled part of the card
            c, s = (c[0], c[1], (CARD_Z_CUT + hi) / 2), (s[0], s[1], hi - CARD_Z_CUT)
        out.append((c, s, color))
    out.append(((-0.0046, 0.0, CARD_Z_CUT + 0.004), (0.006, 0.12, 0.024), STEEL))  # back plate behind the PCB
    out.append(((0.0, 0.0, CARD_Z_CUT - 0.004), (0.060, 0.110, 0.008), STEEL))  # foot under the cut edge
    return out


def _holder_boxes():
    """(centre, size, colour) in the holder's frame (the plug's frame when it rests): a plate under the plug's body, two
    side rails with lips over its top face, and an end stop under its tip."""
    tip = sc.MOD["body_y"][0]
    return [
        ((-0.0066, 0.0, -0.0075), (0.004, 0.025, 0.045), STEEL),  # plate: inner face at x = -4.6 mm, 0.3 mm under the body
        ((0.0015, 0.0100, -0.0075), (0.012, 0.003, 0.040), STEEL),  # rails: 17 mm apart, the plug is 13.75 mm wide
        ((0.0015, -0.0100, -0.0075), (0.012, 0.003, 0.040), STEEL),
        ((0.0, 0.0, tip - 0.0005 - 0.0015), (0.016, 0.017, 0.003), STEEL),  # stop, 0.5 mm under the tip
        ((0.0070, 0.00725, -0.0075), (0.002, 0.0025, 0.040), STEEL),  # lips over the plug's top face: it can slide out along
        ((0.0070, -0.00725, -0.0075), (0.002, 0.0025, 0.040), STEEL),  # its axis but the cable cannot lift it out
    ]


FUNNEL = 0.002  # m: how far the lead-in flares outward at the port mouth (0 = none)


def _funnel():
    """Lead-in chamfers around each port's mouth: four thin 45-degree plates per port, as a real cage's flared entry.
    Returns (centre, size, local rotation, colour) in the card's upright frame."""
    out = []
    if FUNNEL <= 0:
        return out
    f, t = FUNNEL, 0.0004
    k = 0.7 * t  # shifts each plate outward so its inner face meets the port's edge instead of cutting into the opening
    length = f * math.sqrt(2) + 0.0004
    x_lo, x_hi, z_m = GEOM["nic"]["cage_floor_z"], GEOM["nic"]["cage_roof_z"], sc.MOUTH_Z
    for b in GEOM["nic"]["bays"]:
        y_lo, y_hi = -b["x_max"], -b["x_min"]  # the bay's inner faces along the card's y
        yc, wy = (y_lo + y_hi) / 2, (y_hi - y_lo) + 2 * f + 0.001
        out.append(((x_hi + f / 2 + k, yc, z_m + f / 2), (length, wy, t), sc.ry(-math.pi / 4), METAL))  # roof side
        out.append(((x_lo - f / 2 - k, yc, z_m + f / 2), (length, wy, t), sc.ry(math.pi / 4), METAL))  # floor side
        wx = (x_hi - x_lo) + 2 * f + 0.001
        xc = (x_hi + x_lo) / 2
        rx = lambda a: np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])  # noqa: E731
        out.append(((xc, y_hi + f / 2 + k, z_m + f / 2), (wx, length, t), rx(math.pi / 4), METAL))
        out.append(((xc, y_lo - f / 2 - k, z_m + f / 2), (wx, length, t), rx(-math.pi / 4), METAL))
    return out


def _lowest(boxes, r):
    return min(float((r @ (np.array(c) + np.array([dx, dy, dz]) * np.array(s) / 2))[2])
               for c, s, _ in boxes for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1))


def _quat(r):
    return sc.quat_xyzw(r)


_CARD, _HOLD = _card_boxes(), _holder_boxes()
_mouth = R_CARD @ np.array([sc.BAY_X, sum(sc.BAY_Y) / 2, sc.MOUTH_Z])
CARD_POS = np.array([MOUTH_XY[0] - _mouth[0], MOUTH_XY[1] - _mouth[1], TOP + BASE_T - _lowest(_CARD, R_CARD)])
HOLD_POS = np.array([*HOLDER_XY, TOP + BASE_T - _lowest(_HOLD, R_HOLD)])  # the SFP plug's frame origin when it rests


def _shapes(prefix, boxes, origin, r):
    q = _quat(r)
    return [Shape(f"{prefix}_{i:02d}", pos=tuple((origin + r @ np.array(c)).tolist()), size=tuple(s), color=col, rot_xyzw=q)
            for i, (c, s, col) in enumerate(boxes)]


# -- the cable: SFP plug in the slanted holder, SC plug in its tray, a smooth drape between -----------------------------------
SC_HEADING = np.array([0.0, -1.0, 0.0])  # the SC plug's tip direction, which is the cable's direction as it reaches it
SC_CENTRE = np.array([SC_XY[0], SC_XY[1], TOP + SC_BODY[2] / 2 + 0.0005])
_P0 = HOLD_POS + R_HOLD @ np.array([0.0, 0.0, sc.MOD["body_y"][1]])  # the plug's back end: where the cable starts
_D0 = R_HOLD @ np.array([0.0, 0.0, 1.0])  # from the plug's tip toward its back end: the cable's first direction
_E = SC_CENTRE - SC_HEADING * SC_BODY[0] / 2  # the SC plug's back end: where the cable ends
CABLE_R = 0.0025
CABLE_SEG = 0.02


def _drape():
    """From the plug's back end: straight on along the plug's axis, a smooth bend down to the table, then straight on into
    the SC plug along its axis, so each plug sits in line with the cable's first and last segments."""
    a, z = _P0 + _D0 * 0.05, _E - SC_HEADING * 0.05
    b = [a, a + _D0 * 0.07, z - SC_HEADING * 0.10, z]
    pts = [_P0]
    for t in np.linspace(0, 1, 80):
        pts.append(sum(w * p for w, p in zip(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), b)))
    pts.append(_E)
    pts = np.array(pts)
    pts[1:-1, 2] = np.maximum(pts[1:-1, 2], TOP + CABLE_R + 0.001)  # the drape never dips into the table
    # the cable starts at the SC plug (the anchored, root end) and ends at the SFP plug
    return [tuple(p) for p in pts[::-1]]


CABLE = cable(
    "cable", start=tuple(_E), path=_drape(), radius=CABLE_R, friction=1.2, segment=0.02, cable_mass=0.012, color=(0.95, 0.80, 0.10),
    bend_stiffness=0.1, bend_damping=0.01,
    start_plug=Plug(size=SC_BODY, mass=0.04, color=BLUE), end_plug=Plug(size=SFP_BODY, mass=0.02, color=GOLD, extras=(SFP_HEAD,)),
    anchor_start=True,
)


def _tray():
    """A tray for the SC plug, where the cable actually puts it: two side rails and a stop at its tip, fixed."""
    ch = CABLE.chain()
    tip_dir = (ch[0] - ch[1]) / np.linalg.norm(ch[0] - ch[1])  # the SC plug's tip points away from the cable
    along = np.array([tip_dir[0], tip_dir[1], 0.0]) / np.hypot(tip_dir[0], tip_dir[1])
    centre = ch[0] + tip_dir * SC_BODY[0] / 2
    side = np.array([-along[1], along[0], 0.0])
    z = TOP + 0.007
    shapes = []
    for k, sgn in enumerate((-1, 1)):
        c = centre + side * sgn * (SC_BODY[1] / 2 + 0.0035)
        shapes.append(Box(f"tray_rail_{k}", pos=(c[0], c[1], z), size=(0.004, SC_BODY[0] + 0.006, 0.014), color=STEEL, yaw=math.atan2(along[1], along[0]) - math.pi / 2))
    stop = centre + along * (SC_BODY[0] / 2 + 0.0035)
    shapes.append(Box("tray_stop", pos=(stop[0], stop[1], z), size=(SC_BODY[1] + 0.012, 0.004, 0.014), color=STEEL, yaw=math.atan2(along[1], along[0]) - math.pi / 2))
    shapes.append(Box("tray_base", pos=(centre[0], centre[1], TOP + 0.0005), size=(SC_BODY[1] + 0.012, SC_BODY[0] + 0.012, 0.001), color=STEEL, yaw=math.atan2(along[1], along[0]) - math.pi / 2))
    return shapes


DOCK_XY = (0.30, 0.20)  # where the plug's tip rests while it is turned upright
DOCK_FLOOR = TOP + BASE_T
DOCK = [
    Box("dock_base", pos=(DOCK_XY[0] - 0.01, DOCK_XY[1], TOP + BASE_T / 2), size=(0.08, 0.06, BASE_T), color=STEEL),
    Box("dock_wall", pos=(DOCK_XY[0] + 0.0045, DOCK_XY[1], DOCK_FLOOR + 0.004), size=(0.004, 0.030, 0.008), color=STEEL),
    Box("dock_rail_0", pos=(DOCK_XY[0] - 0.015, DOCK_XY[1] + 0.0135, DOCK_FLOOR + 0.0125), size=(0.040, 0.003, 0.025), color=STEEL),
    Box("dock_rail_1", pos=(DOCK_XY[0] - 0.015, DOCK_XY[1] - 0.0135, DOCK_FLOOR + 0.0125), size=(0.040, 0.003, 0.025), color=STEEL),
]


OBJECTS = (
    _shapes("card", _CARD, CARD_POS, R_CARD)
    + _shapes("holder", _HOLD, HOLD_POS, R_HOLD)
    + [Shape(f"funnel_{i:02d}", pos=tuple((CARD_POS + R_CARD @ np.array(c)).tolist()), size=tuple(sz), color=col, rot_xyzw=_quat(R_CARD @ rl))
       for i, (c, sz, rl, col) in enumerate(_funnel())]
    + _tray()
    + [Box("card_base", pos=(MOUTH_XY[0] - 0.02, MOUTH_XY[1], TOP + BASE_T / 2), size=(0.14, 0.15, BASE_T), color=STEEL),
       Box("holder_base", pos=(HOLDER_XY[0], HOLDER_XY[1], TOP + BASE_T / 2), size=(0.07, 0.06, BASE_T), color=STEEL),
       CABLE]
)


def _plug(state):
    """Tip face centre and unit axis toward the tip of the SFP plug (the cable's end plug), from its own pose."""
    axis = sc.rotation(state.end_plug_quat)[:, 0]  # the plug's x axis points out of the cable: toward the tip
    return state.end_plug + axis * sc.PLUG_LEN / 2, axis


def score(objects):
    st = objects["cable"]
    tip, axis = _plug(st)
    return sc.score(tip, axis, CARD_POS, R_CARD, plug_z=float(st.end_plug[2]), table_z=TOP)


def success(objects):
    return score(objects) >= 1.0 and float(np.linalg.norm(objects["cable"].lin_vel)) < 0.03


def distance(objects):
    tip, axis = _plug(objects["cable"])
    return max(0.0, sc.FULL_DEPTH - sc.measure(tip, axis, CARD_POS, R_CARD)["depth"])


# -- the scripted demo: closed loop on the plug's measured pose (run it with cabledemo.py) ------------------------------
GRIP_BACK = -SFP_HEAD[0][0]  # the pads close on the latch end, this far behind the plug body's middle
GRIP_UP = 0.010  # the TCP rides this far above the grip point so the pads' tips (13 mm below it) close on the plug
DEMO_DEPTH = 0.047
A_IN = R_CARD @ np.array([0.0, 0.0, -1.0])  # the port axis, pointing into the port
MOUTH = CARD_POS + R_CARD @ np.array([sc.BAY_X, sc.BAY_Y[sc.TARGET_BAY], sc.MOUTH_Z])
UP = np.array([0.0, 0.0, 1.0])

# extra cameras for the recording only (cabledemo.py): name -> (position, look-at point, focal length mm), world frame
RECORD_CAMS = {
    "side_cam": (MOUTH + np.array([0.0, 0.62, 0.12]), MOUTH + np.array([-0.03, 0.10, -0.03]), 20.0),  # side-on to the insertion
}


def _rz(a):
    return sc.rz(a)




def demo_script(ctx):
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN

    arm = ctx.left
    yield from arm.wait(1.5)  # let the cable settle

    # -- the right arm takes the cable's slack: it picks the cable up 0.28 m behind the plug and holds it clear of the card
    rarm = ctx.right
    pts = ctx.states()["cable"].points  # segment centres, SC plug end to SFP plug end
    k = int(round(0.28 / CABLE_SEG))
    c, c_dir = pts[-1 - k], pts[-k] - pts[-2 - k]
    ryaw = math.atan2(c_dir[1], c_dir[0])  # the pads close across the cable
    hold = c + np.array([0.0, 0.0, 0.013])
    yield from rarm.move(hold + np.array([0.0, 0.0, 0.08]), yaw=ryaw, finger=FINGER_OPEN, duration=2.5, label="right arm over the cable")
    yield from rarm.move(hold, yaw=ryaw, duration=1.5, label="down on the cable")
    yield from rarm.grip(FINGER_CLOSED, 0.8)
    yield from rarm.move(hold + np.array([0.0, 0.0, 0.04]), yaw=ryaw, duration=1.2, label="lift the cable's slack")
    st = ctx.states()["cable"]
    tip, axis = _plug(st)
    grip = st.end_plug - axis * GRIP_BACK + UP * GRIP_UP
    back = -axis  # from the tip toward the plug's back end (out of the holder)
    yield from arm.move(grip + back * 0.08, yaw=GRIP_YAW, finger=FINGER_OPEN, duration=1.5, label="to the plug")
    yield from arm.move(grip, yaw=GRIP_YAW, duration=1.5, label="down on it")
    yield from arm.grip(FINGER_CLOSED, 0.8)
    out = grip + back * 0.065
    for f in np.linspace(0.1, 1.0, 10):  # draw it out in short steps along its axis
        yield from arm.move(grip + (out - grip) * f, yaw=GRIP_YAW, duration=0.15, label="draw out along its axis")
    for _ in range(10):
        yield

    # -- turn it upright: tilt the hand 45 degrees about the pad line, so the plug (rigid in the pads) stands vertical
    yield from arm.move(out + np.array([0.0, 0.0, 0.03]), yaw=GRIP_YAW, duration=1.0, label="lift")
    tcp = out + np.array([0.0, 0.0, 0.03])
    for k in np.linspace(0.0, -HOLD_TILT, 7)[1:]:
        ctx.tilt = np.array([float(k), 0.0, 0.0])
        yield from arm.move(tcp, yaw=GRIP_YAW, duration=0.8, label="tilt the plug upright")
    for _ in range(12):  # the plug lags the hand a little: keep tilting until it reads upright
        for _ in range(10):
            yield
        axis = _plug(ctx.states()["cable"])[1]
        lean = math.asin(max(-1.0, min(1.0, float(axis[1]))))
        if abs(lean) < math.radians(0.7):
            break
        ctx.tilt = ctx.tilt + np.array([-0.9 * lean, 0.0, 0.0])
        yield from arm.move(tcp, yaw=GRIP_YAW, duration=0.8, label="straighten the plug")
    for _ in range(20):
        yield
    tip, axis = _plug(ctx.states()["cable"])
    d0 = tip - tcp

    # -- carry it over the port and in, as before
    yaw_c = GRIP_YAW
    pre_tip = MOUTH - A_IN * 0.04
    pre = pre_tip - d0
    top_z = max(tcp[2], pre[2]) + 0.01
    yield from arm.move((tcp[0], tcp[1], top_z), yaw=GRIP_YAW, duration=1.0, label="lift")
    yield from arm.move((pre[0], pre[1], top_z), yaw=yaw_c, duration=2.0, label="carry to the card")
    yield from arm.move(pre, yaw=yaw_c, duration=1.0, label="down to the port axis")
    cmd = np.array(pre, dtype=float)
    nominal = lambda depth: MOUTH + A_IN * depth - d0  # noqa: E731

    yaw_now = yaw_c

    def measure():
        nonlocal yaw_now
        tip_, axis_ = _plug(ctx.states()["cable"])
        a_ = R_CARD.T @ axis_
        w_ = R_CARD.T @ sc.rotation(ctx.states()["cable"].end_plug_quat)[:, 1]  # the plug's width axis in the card's frame: along y when it is square to the port
        roll = math.atan2(w_[0], w_[1])
        roll = (roll + math.pi / 2) % math.pi - math.pi / 2  # a plug is the same turned half way round
        yaw_now = float(np.clip(yaw_now + ROLL_GAIN * roll, yaw_c - 0.12, yaw_c + 0.12))
        ctx.roll = roll
        ctx.tilt = ctx.tilt + np.array([-TILT_GAIN * axis_[1], TILT_GAIN * axis_[0], 0.0])  # hold the plug on the port's axis by tilting the hand
        return R_CARD.T @ (tip_ - MOUTH)

    GAIN = 0.12
    TILT_GAIN = 0.05
    ROLL_GAIN = 0.3

    def step_to(depth, gain=None):
        nonlocal cmd
        e = measure()
        want = np.array([0.0, 0.0, -depth])
        cmd = cmd + R_CARD @ np.clip((GAIN if gain is None else gain) * (want - e), -0.0015, 0.0015)
        cmd = np.clip(cmd, nominal(depth) - 0.008, nominal(depth) + 0.008)
        return e

    depth = -0.04
    while depth < -0.012:
        depth += 0.002
        step_to(depth)
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="approach along the port axis")
    e0 = measure()
    calm = 0
    for _ in range(300):
        e = step_to(-0.012)
        calm = calm + 1 if max(abs(e[0]), abs(e[1])) < 0.00015 else 0
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="line up on the port")
        if calm >= 8:
            break
    depth = -0.012
    held = 0
    while depth < DEMO_DEPTH:
        e = measure()
        off = max(abs(e[0]), abs(e[1]))
        if off < (0.00035 if depth < 0.0 else 0.00025) and -e[2] > depth - 0.0025:
            depth += 0.0005
            held = 0
        else:
            held += 1
        if held > 150:
            break
        step_to(depth)
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="push in along the port axis")
    for _ in range(20):
        yield
    yield from arm.grip(FINGER_OPEN, 0.6)
    yield from arm.move(cmd - A_IN * 0.06, yaw=yaw_c, duration=1.0, label="retreat along the axis")
    yield from rarm.grip(FINGER_OPEN, 0.6)  # the right arm lets the cable go
