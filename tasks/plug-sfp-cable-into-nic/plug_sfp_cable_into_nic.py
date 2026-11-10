"""On the station table: a PCIe network card leans back in a slanted cradle in front of the left arm, turned 8 degrees from
square to the arms, its two SFP ports at its upper edge facing up and toward the arms at 45 degrees. An SFP-to-SC fibre
patch cable lies across the table: its SC plug (blue) rests in a small tray, fixed in place, and its SFP plug (gold) lies
in a slanted holder to the left of the card, tip down at 45 degrees like the card's ports, the cable leaving its back end
and draping to the tray. The robot must lift the SFP plug out and plug it, tip first, all the way into the card's left port
(the one at the larger y), with the cable's other end staying in its tray.
"""

import math
import os
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
R_HOLD = sc.ry(-sc.CARD_TILT)  # the holder's frame: leaned the same way, not turned; the plug's axis is its z
CLEARANCE = float(os.environ.get("SFP_CLEARANCE", "0.0"))  # m a side added to the AIC port; 0 = the AIC model's own size
CARD_Z_CUT = -0.032  # WHY: only the card's upper part is modelled (the rest sits inside the cradle)
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
    out.append(((-0.0046, 0.0, -0.040 + 0.0225), (0.006, 0.12, 0.045), STEEL))  # back plate behind the PCB
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
    "cable", start=tuple(_E), path=_drape(), radius=CABLE_R, friction=1.2, segment=0.02, cable_mass=float(os.environ.get("CABLE_M", "0.012")), color=(0.95, 0.80, 0.10),
    bend_stiffness=float(os.environ.get("CABLE_K", "0.1")), bend_damping=float(os.environ.get("CABLE_D", "0.01")),
    start_plug=Plug(size=SC_BODY, mass=0.04, color=BLUE), end_plug=Plug(size=SFP_BODY, mass=0.02, color=GOLD, extras=(SFP_HEAD,)),
    anchor_start=os.environ.get("SFP_ANCHOR", "1") == "1",
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


OBJECTS = (
    _shapes("card", _CARD, CARD_POS, R_CARD)
    + _shapes("holder", _HOLD, HOLD_POS, R_HOLD)
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
    if os.environ.get("SFP_DEBUG"):  # development only: one line per step with the plug's tip in the card's frame
        with open(os.environ["SFP_DEBUG"], "a") as f:
            loc = R_CARD.T @ (tip - CARD_POS)
            a = R_CARD.T @ axis
            f.write(f"tip(x,y)mm=({(loc[0]-sc.BAY_X)*1000:.2f},{(loc[1]-sc.BAY_Y[0])*1000:.2f}) depth={(sc.MOUTH_Z-loc[2])*1000:.1f} "
                    f"axis=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) plug_z={st.end_plug[2]:.3f} sfp={st.end_plug.round(4).tolist()} "
                    f"sc={st.start_plug.round(4).tolist()} seg0={st.points[0].round(4).tolist()} segN={st.points[-1].round(4).tolist()}\n")
    if os.environ.get("SFP_CHAIN"):  # development only: how far each segment is from its designed place
        dev = np.linalg.norm(st.points - np.stack([c for c, _ in CABLE.frames()]), axis=1) * 1000
        with open(os.environ["SFP_CHAIN"], "a") as f:
            f.write(" ".join(f"{d:.1f}" for d in dev) + "\n")
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


def _rz(a):
    return sc.rz(a)


def demo_script(ctx):
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN

    arm = ctx.left
    yield from arm.wait(1.5)  # let the cable settle
    st = ctx.states()["cable"]
    tip, axis = _plug(st)
    grip = st.end_plug - axis * GRIP_BACK + UP * GRIP_UP
    back = -axis  # from the tip toward the plug's back end (out of the holder)
    yield from arm.move(grip + back * 0.08, yaw=0.0, finger=FINGER_OPEN, duration=1.5, label="to the plug")
    yield from arm.move(grip, yaw=0.0, duration=1.5, label="down on it")
    yield from arm.grip(FINGER_CLOSED, 0.8)
    out = grip + back * 0.065
    yield from arm.move(out, yaw=0.0, duration=1.5, label="draw out along its axis")
    for _ in range(10):
        yield  # let the grip settle
    if os.environ.get("DEMO_HOLD"):  # development only: hold the plug still to see whether it swings in the pads
        yield from arm.move(out + np.array([0.0, 0.0, 0.05]), yaw=0.0, duration=1.0, label="lift test")
        yield from arm.wait(3.0)
        return
    st = ctx.states()["cable"]
    tip, axis = _plug(st)
    d0 = tip - out  # from the TCP to the tip while the grip is at yaw 0; it turns with the wrist
    yaw_c = sc.CARD_YAW
    pre_tip = MOUTH - A_IN * 0.08
    pre = pre_tip - _rz(yaw_c) @ d0
    top_z = max(out[2], pre[2]) + 0.01
    yield from arm.move((out[0], out[1], top_z), yaw=0.0, duration=1.0, label="lift")
    yield from arm.move((pre[0], pre[1], top_z), yaw=yaw_c, duration=2.0, label="carry, turning to the card")
    yield from arm.move(pre, yaw=yaw_c, duration=1.0, label="down to the port axis")
    cmd = np.array(pre, dtype=float)
    nominal = lambda depth: MOUTH + A_IN * depth - d0  # noqa: E731  the TCP that puts the tip `depth` m into the port, if the grip never slips

    def measure():
        tip_, axis_ = _plug(ctx.states()["cable"])
        return R_CARD.T @ (tip_ - MOUTH)  # (across, along the bays, up the port axis): depth = -z

    def step_to(depth, gain=0.6):
        """One control step: feed the tip's error from the port's centre line back into the TCP, and its depth error too."""
        nonlocal cmd
        e = measure()
        want = np.array([0.0, 0.0, -depth])
        cmd = cmd + R_CARD @ np.clip(gain * (want - e), -0.002, 0.002)
        cmd = np.clip(cmd, nominal(depth) - 0.008, nominal(depth) + 0.008)  # never wind up far from the plan
        return e

    # 1. approach to 12 mm outside the port, then settle on its centre line before going in
    depth = -0.08
    while depth < -0.012:
        depth += 0.002
        step_to(depth)
        yield from arm.move(cmd, yaw=yaw_c, duration=0.05, label="approach along the port axis")
    calm = 0
    for _ in range(300):
        e = step_to(-0.012, gain=0.5)
        calm = calm + 1 if max(abs(e[0]), abs(e[1])) < 0.00015 else 0
        yield from arm.move(cmd, yaw=yaw_c, duration=0.05, label="line up on the port")
        if calm >= 8:
            break
    # 2. in, holding back whenever the tip drifts off the centre line
    depth = -0.012
    held = 0
    while depth < DEMO_DEPTH:
        e = measure()
        off = max(abs(e[0]), abs(e[1]))
        if off < 0.00025 and -e[2] > depth - 0.0025:  # on the centre line and keeping up: go deeper
            depth += 0.0003
            held = 0
        else:
            held += 1
        if held > 150:
            break
        step_to(depth, gain=0.5)
        yield from arm.move(cmd, yaw=yaw_c, duration=0.05, label="push in along the port axis")
        if os.environ.get("SFP_IK"):
            print(f"[ik] depth_target={depth*1000:.1f} measured_depth={-e[2]*1000:.2f} across={e[0]*1000:.2f} along={e[1]*1000:.2f} held={held}", flush=True)
    for _ in range(20):
        yield
    yield from arm.grip(FINGER_OPEN, 0.6)
    yield from arm.move(cmd - A_IN * 0.06, yaw=yaw_c, duration=1.0, label="retreat along the axis")
