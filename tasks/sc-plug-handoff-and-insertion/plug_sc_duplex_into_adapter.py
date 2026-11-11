"""On the station table: a PCIe network card stands upright on a foot to the left of the middle, its face toward the left arm, with the
gold SFP plug of a fibre patch cable already seated in its nearer port (fixed in place). The cable hangs from the card's edge and
across the table to its blue duplex SC plug, which lies flat on the table, tip toward the robot's right. A duplex SC adapter port
stands on a small pedestal in front of that plug, its keyed opening facing the left arm. The robot must pick up the SC plug and push
it, tip first and right side up, straight into the adapter port until its rear shoulder sits against the port's mouth, with the SFP
end staying seated in the card.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sc_scoring as sc
from fibrekit import Plug, cable
from scenekit import Shape, env as scene_env, on_reset
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
SUCCESS_HOLD_S = 20.0  # seated and at rest after release: the plug must stay in after the gripper lets go
TOP = TABLE_TOP[SCENE]
GEOM = sc.GEOM
NIC = GEOM["nic"]

SKIN, METAL, GOLD, STEEL, BLUE = (0.10, 0.45, 0.20), (0.72, 0.74, 0.77), (0.85, 0.70, 0.20), (0.35, 0.36, 0.38), (0.10, 0.30, 0.75)
CAGE = "10099100-011lfc001"
BASE_T = 0.006

# -- the network card with the SFP plug seated in its nearer port (the same AIC card as the SFP task) ----------------------
R_CARD = sc.rz(math.pi / 2)  # the card's upright frame in the world: its face toward the left arm, its ports side by side along x
CARD_Z_CUT = 0.022  # only the card's upper part is modelled (the rest sits inside the cradle)
NIC_MOUTH_XY = (0.30, 0.30)  # the target port's mouth centre, from above
MOUTH_Z = -NIC["cage_open_y"]  # cage mouth, above the card origin (card y runs downward)
BAY_Y = [-(b["x_min"] + b["x_max"]) / 2 for b in NIC["bays"]]
BAY_X = (NIC["cage_floor_z"] + NIC["cage_roof_z"]) / 2
_M_CARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # card model frame -> its upright frame
SFP_BODY = (0.0473, 0.01375, 0.0086)  # the AIC SFP module's body, 20 g
SFP_HEAD = ((-0.026, 0.0, 0.0), (0.0115, 0.014, 0.0122), (0.80, 0.65, 0.18))  # its latch end: 14 x 12.2 mm, on the cable side
SFP_DEPTH = 0.0415  # where the plug stops against its shoulder


def _upright(centre, size, m):
    return tuple((m @ np.array(centre)).tolist()), (size[2], size[0], size[1])


def _card_boxes():
    """(centre, size, colour) in the card's upright frame, plus its back plate and foot."""
    out = []
    for b in NIC["boxes"]:
        c, s = _upright(b["pos"], b["size"], _M_CARD)
        color = METAL if b["name"].startswith(CAGE) else SKIN
        lo, hi = c[2] - s[2] / 2, c[2] + s[2] / 2
        if hi <= CARD_Z_CUT:
            continue
        if lo < CARD_Z_CUT:  # clip the box to the modelled part of the card
            c, s = (c[0], c[1], (CARD_Z_CUT + hi) / 2), (s[0], s[1], hi - CARD_Z_CUT)
        out.append((c, s, color))
    out.append(((-0.0046, 0.0, CARD_Z_CUT + 0.004), (0.006, 0.12, 0.024), STEEL))  # back plate behind the PCB
    out.append(((0.0, 0.0, CARD_Z_CUT - 0.004), (0.060, 0.110, 0.008), STEEL))  # foot under the cut edge
    return out


def _lowest(boxes, r):
    return min(float((r @ (np.array(c) + np.array([dx, dy, dz]) * np.array(s) / 2))[2])
               for c, s, _ in boxes for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1))


_CARD = _card_boxes()
_mouth = R_CARD @ np.array([BAY_X, BAY_Y[0], MOUTH_Z])
CARD_POS = np.array([NIC_MOUTH_XY[0] - _mouth[0], NIC_MOUTH_XY[1] - _mouth[1], TOP + BASE_T - _lowest(_CARD, R_CARD)])
NIC_MOUTH = CARD_POS + _mouth

# -- the SC adapter port on a pedestal, mouth facing the left arm ---------------------------------------------------------------
PORT_MOUTH = np.array([0.22, -0.05, TOP + 0.045])  # the port's mouth centre
PORT_POS = PORT_MOUTH - sc.R_PORT @ np.array([0.0, sc.MOUTH_Y, 0.0])

# -- the cable: the SC plug (start) lies on the table; the SFP plug (end) is seated and fixed -------------------------------------
_REAR = next(b for b in sc.PLUG if b["name"].startswith("cube002"))
SC_REAR = tuple(_REAR["size"])  # the plug's rear body: 12.55 x 25 x 10.1 mm, the box the cable's start plug is built on
SC_START_XY = (0.22, -0.006)  # the rear body's centre, from above
CABLE_R = 0.0025
CABLE_SEG = 0.02


def _sc_extras():
    """The plug's two front housings and its two keys, in the start-plug frame (x into the cable, y mirrored, z up)."""
    out = []
    for b in sc.PLUG:
        if b["name"].startswith("cube002") or abs(b["rot"][0][1]) > 0.1:  # skip the rear body and the small chamfers
            continue
        dx, dy, dz = (b["pos"][0] - sc.REAR_CENTRE_X, b["pos"][1], b["pos"][2])
        out.append(((-dx, -dy, dz), tuple(b["size"]), (0.12, 0.35, 0.80)))
    return tuple(out)


SC_CENTRE = np.array([SC_START_XY[0], SC_START_XY[1], TOP + SC_REAR[2] / 2 + 0.0005])
_S = SC_CENTRE + np.array([0.0, SC_REAR[0] / 2, 0.0])  # the plug's back face: where the cable starts (the plug's tip points -y)
_E = NIC_MOUTH + np.array([0.0, 0.0, SFP_BODY[0] - SFP_DEPTH])  # the SFP plug's back end: where the cable ends (its tip points -z)


def _bezier(b, n=60):
    return [sum(w * np.asarray(p, float) for w, p in zip(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), b))
            for t in np.linspace(0, 1, n)]


def _drape():
    """From the SFP plug up and over the card's edge, down to the table and along it to the SC plug's back face, in the order the cable
    runs (SC plug first)."""
    up = np.array([0.0, 0.0, 1.0])
    to_s = np.array([_S[0] - _E[0], _S[1] - _E[1], 0.0])
    run = to_s / np.linalg.norm(to_s)
    q0 = _E + up * 0.045  # straight up out of the plug
    landing = np.array([_E[0], _E[1], TOP + CABLE_R + 0.001]) + run * 0.10
    leg1 = _bezier([q0, q0 + up * 0.03, landing + up * 0.05 - run * 0.02, landing])
    side = np.array([1.0, 0.0, 0.0])
    side = np.array([1.0, 0.0, 0.0])  # the slack bows away from the robot, well away from the plug
    p1 = _S + np.array([0.0, 0.06, 0.0])  # the last 6 cm run straight in along the plug's axis, so the plug lies in line with the cable
    leg2 = _bezier([landing, landing + run * 0.12 + side * 0.30, p1 + np.array([0.0, 0.10, 0.0]), p1], 100)
    pts = np.array([_E, q0] + leg1[1:] + leg2[1:] + [_S])
    pts[2:, 2] = np.maximum(pts[2:, 2], TOP + CABLE_R + 0.001)  # the drape never dips into the table
    return [tuple(p) for p in pts[::-1]]


CABLE = cable(
    "cable", start=tuple(_S), path=_drape(), radius=CABLE_R, friction=1.2, segment=CABLE_SEG, cable_mass=0.012, color=(0.95, 0.80, 0.10),
    bend_stiffness=0.03, bend_damping=0.01,
    start_plug=Plug(size=SC_REAR, mass=0.04, color=BLUE, extras=_sc_extras()),
    end_plug=Plug(size=SFP_BODY, mass=0.02, color=GOLD, extras=(SFP_HEAD,)),
    anchor_end=True,
)


def _quat(r):
    return sc.quat_xyzw(r)


def _port_shapes():
    out = []
    for i, b in enumerate(sc.PORT):
        r = sc.R_PORT @ np.array(b["rot"])
        out.append(Shape(f"port_{i:02d}", pos=tuple((PORT_POS + sc.R_PORT @ np.array(b["pos"])).tolist()), size=tuple(b["size"]),
                         color=METAL, rot_xyzw=_quat(r)))
    return out


def _shapes(prefix, boxes, origin, r):
    q = _quat(r)
    return [Shape(f"{prefix}_{i:02d}", pos=tuple((origin + r @ np.array(c)).tolist()), size=tuple(s), color=col, rot_xyzw=q)
            for i, (c, s, col) in enumerate(boxes)]


_PED_TOP = PORT_MOUTH[2] - 0.0047  # just under the port's bottom plate
OBJECTS = (
    _shapes("card", _CARD, CARD_POS, R_CARD)
    + _port_shapes()
    + [Box("card_base", pos=(NIC_MOUTH_XY[0] - 0.02, NIC_MOUTH_XY[1], TOP + BASE_T / 2), size=(0.14, 0.15, BASE_T), color=STEEL),
       Box("port_pedestal", pos=(PORT_MOUTH[0], PORT_MOUTH[1] - 0.0137, (TOP + _PED_TOP) / 2), size=(0.034, 0.030, _PED_TOP - TOP), color=STEEL),
       Box("port_base", pos=(PORT_MOUTH[0], PORT_MOUTH[1] - 0.0137, TOP + BASE_T / 2), size=(0.08, 0.06, BASE_T), color=STEEL),
       CABLE]
)


def _plug(state):
    """Tip face centre, unit axis toward the tip and unit up vector of the SC plug (the cable's start plug), from its own pose."""
    return sc.plug_tip(state.start_plug, state.start_plug_quat)


_HANDOFF = {
    "donor_held": False,
    "receiver_held": False,
    "transfer_pending": False,
    "transfer_tcp": None,
    "transfer_plug": None,
    "complete": False,
}


@on_reset
def _reset_handoff(_environment):
    _HANDOFF.update(
        donor_held=False,
        receiver_held=False,
        transfer_pending=False,
        transfer_tcp=None,
        transfer_plug=None,
        complete=False,
    )


def _handoff_complete(st) -> bool:
    """Require measured two-arm overlap, then donor release with receiver still closed and near the plug."""
    if _HANDOFF["complete"]:
        return True
    environment = scene_env()
    if environment is None:
        return False
    from yam_arena.runtime import tcp_world

    centre = np.asarray(st.start_plug, float)
    robots = {side: environment.unwrapped.scene[f"{side}_robot"] for side in ("left", "right")}
    near, closed = {}, {}
    for side, robot in robots.items():
        near[side] = float(np.linalg.norm(tcp_world(robot) - centre)) <= 0.04
        ids = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
        fingers = robot.data.joint_pos[0, ids].detach().cpu().numpy()
        closed[side] = bool(np.all(fingers >= -0.014))
    _HANDOFF["donor_held"] |= near["right"] and closed["right"]
    _HANDOFF["receiver_held"] |= near["left"] and closed["left"]
    # Opening and retreating alone could be faked by closing a gripper beside
    # the plug. After donor release, require the connector to follow a receiver
    # movement, which is evidence that the receiver actually retained it.
    donor_released = not near["right"] and not closed["right"]
    if (_HANDOFF["donor_held"] and _HANDOFF["receiver_held"] and donor_released
            and near["left"] and closed["left"] and not _HANDOFF["transfer_pending"]):
        _HANDOFF["transfer_pending"] = True
        _HANDOFF["transfer_tcp"] = tcp_world(robots["left"]).copy()
        _HANDOFF["transfer_plug"] = centre.copy()
    if _HANDOFF["transfer_pending"] and near["left"] and closed["left"]:
        tcp_delta = tcp_world(robots["left"]) - _HANDOFF["transfer_tcp"]
        plug_delta = centre - _HANDOFF["transfer_plug"]
        if (float(np.linalg.norm(tcp_delta)) >= 0.008
                and float(np.linalg.norm(plug_delta - tcp_delta)) <= 0.004):
            _HANDOFF["complete"] = True
    return _HANDOFF["complete"]


def score(objects):
    st = objects["cable"]
    if not _handoff_complete(st):
        return 0.0
    tip, axis, up = _plug(st)
    return sc.score(tip, axis, up, PORT_POS, plug_z=float(st.start_plug[2]), table_z=TOP)


def success(objects):
    return score(objects) >= 1.0 and float(np.linalg.norm(objects["cable"].lin_vel)) < 0.03


def distance(objects):
    tip, axis, up = _plug(objects["cable"])
    return max(0.0, sc.FULL_DEPTH - sc.measure(tip, axis, up, PORT_POS)["depth"])


# extra cameras for the recording only (cabledemo.py): name -> (position, look-at point, focal length mm), world frame
RECORD_CAMS = {
    "side_cam": (PORT_MOUTH + np.array([0.55, 0.0, 0.14]), PORT_MOUTH + np.array([0.0, 0.05, -0.03]), 20.0),
}


# -- the scripted demo: closed loop on the plug's measured pose (run it with cabledemo.py) ------------------------------
GRIP_YAW = -math.pi / 2  # the pads close along x, across the plug's 25 mm width
GRIP_BACK = 0.007  # grip this far toward the cable from the rear body's centre: the 17 mm pads must stay behind the plug's shoulder
GRIP_UP = 0.010  # the TCP rides this far above the plug's rear body centre so the pads' tips close on its sides
DEMO_DEPTH = 0.0145  # push a little past the seat: the plug stops where its shoulder meets the mouth
INTO = sc.R_PORT @ np.array([0.0, -1.0, 0.0])  # the way into the port: world -y
UP = np.array([0.0, 0.0, 1.0])


def demo_script(ctx):
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN

    arm = ctx.left
    donor = ctx.right
    yield from donor.wait(1.5)
    st = ctx.states()["cable"]
    # A holds the front housings; B takes the rear body for insertion.
    front = st.start_plug - sc.rotation(st.start_plug_quat)[:, 0] * 0.012 + UP * GRIP_UP
    yield from donor.move(front + UP * 0.08, yaw=GRIP_YAW + math.pi, finger=FINGER_OPEN, duration=2.5, label="A above connector")
    yield from donor.move(front, yaw=GRIP_YAW + math.pi, duration=1.5, label="A grasps front housings")
    yield from donor.grip(FINGER_CLOSED, 0.8)
    pickup_before = np.array(ctx.states()["cable"].start_plug)
    yield from donor.move(front + UP * 0.09, yaw=GRIP_YAW + math.pi, duration=1.5, label="A presents connector")
    pickup_motion = np.array(ctx.states()["cable"].start_plug) - pickup_before
    print("DONOR pickup motion", pickup_motion.tolist(), flush=True)
    if np.linalg.norm(pickup_motion - UP * 0.09) > 0.015:
        raise RuntimeError("A failed pickup; handoff not attempted")
    st = ctx.states()["cable"]
    arm.tilt = np.array([-math.pi / 3, 0.0, 0.0])
    grip = st.start_plug + np.array([0.0, 0.00866, 0.005])
    yield from arm.move(grip + np.array([0.0, 0.12, 0.0]), yaw=GRIP_YAW, finger=FINGER_OPEN, duration=2.5, label="B approaches horizontally")
    yield from arm.move(grip, yaw=GRIP_YAW, duration=1.5, label="B takes rear body from side")
    yield from arm.grip(FINGER_CLOSED, 0.8)
    from yam_arena import ik
    receiver_error = float("inf")
    for side in ("left", "right"):
        robot = ctx.emb._base.scene[side + "_robot"]
        q = robot.data.joint_pos[0].cpu().numpy()
        names = list(robot.joint_names)
        angles = np.array([q[names.index("joint" + str(k))] for k in range(1, 7)])
        a = getattr(ctx, side)
        actual, _ = ik.tcp_pose(angles, a.base)
        if side == "left":
            receiver_error = float(np.linalg.norm(actual - grip))
        print("GRASP", side, "actual", actual.tolist(), "command", a.tcp().tolist(), "joints", dict(zip(names, q.tolist())), flush=True)
    print("GRASP connector", np.array(ctx.states()["cable"].start_plug).tolist(), "target", grip.tolist(), flush=True)
    # Pose stability is a diagnostic, not yet a contact-based secure-grip test.
    before = np.array(ctx.states()["cable"].start_plug)
    yield from arm.wait(0.5)
    if np.linalg.norm(ctx.states()["cable"].start_plug - before) > 0.003:
        raise RuntimeError("Connector unstable during dual grasp; donor kept closed")
    if receiver_error > 0.003:
        raise RuntimeError("Receiver missed grasp target; donor kept closed")
    yield from donor.grip(FINGER_OPEN, 0.8)
    yield from donor.move(donor.tcp() + UP * 0.09, yaw=GRIP_YAW + math.pi, duration=1.5, label="A clears handoff")
    lift = arm.tcp()
    before = np.array(ctx.states()["cable"].start_plug)
    yield from arm.move(lift + UP * 0.01, yaw=GRIP_YAW, duration=0.8, label="B retention check")
    print("TRANSFER before", before.tolist(), "after", np.array(ctx.states()["cable"].start_plug).tolist(), flush=True)
    movement = np.array(ctx.states()["cable"].start_plug) - before
    if np.linalg.norm(movement - UP * 0.01) > 0.004:
        raise RuntimeError("B did not retain connector after handoff")
    lift = arm.tcp()
    print("HANDOFF retention check passed", flush=True)
    # Level the connector while preserving its centre through the rotation.
    st = ctx.states()["cable"]
    current = sc.rotation(st.start_plug_quat)
    change = sc.rz(math.pi / 2) @ current.T
    from yam_arena.urdf import quat_xyzw_to_matrix
    from yam_arena.demo import DOWN_GRIP
    down = sc.rz(GRIP_YAW) @ quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
    tool = ik.tcp_pose(arm.q, arm.base)[1]
    arm.tilt = ik._rot_error(change @ tool, down)
    centre = np.array(st.start_plug)
    target = centre + change @ (arm.tcp() - centre)
    yield from arm.move(target, yaw=GRIP_YAW, duration=2.0, label="B levels connector")
    lift = arm.tcp()

    tip, axis, up = _plug(ctx.states()["cable"])
    d0 = tip - lift  # tip relative to the TCP while it is held

    # -- carry it in front of the port at the port's height, on its axis, 4 cm short of the mouth
    pre_tip = PORT_MOUTH - INTO * 0.04
    pre = pre_tip - d0
    top_z = max(lift[2], pre[2]) + 0.005
    yield from arm.move((pre[0], pre[1], top_z), yaw=GRIP_YAW, duration=2.0, label="carry to the port")
    yield from arm.move(pre, yaw=GRIP_YAW, duration=1.0, label="to the port's height")
    cmd = np.array(pre, dtype=float)
    yaw_now = GRIP_YAW
    GAIN, DEPTH_GAIN, YAW_GAIN, TILT_GAIN = 0.12, 0.5, 0.3, 0.05
    Z_BIAS = -0.00012  # the top clearance inside the port is only 0.14 mm, the bottom 0.4 mm: aim low
    nominal = lambda depth: pre + INTO * (depth + 0.04)  # noqa: E731

    insertion_tilt = np.array(ctx.tilt, copy=True)

    def measure():
        nonlocal yaw_now
        tip_, axis_, up_ = _plug(ctx.states()["cable"])
        yaw_now = float(np.clip(yaw_now - YAW_GAIN * axis_[0], GRIP_YAW - 0.12, GRIP_YAW + 0.12))  # square the plug to the port's walls
        t = ctx.tilt
        ctx.tilt = np.clip(np.array([t[0] + TILT_GAIN * axis_[2], t[1] - TILT_GAIN * up_[0], t[2]]), insertion_tilt - 0.04, insertion_tilt + 0.04)  # keep it level and the right way up
        return tip_ - PORT_MOUTH  # the tip relative to the mouth's centre, world frame

    def step_to(depth):
        nonlocal cmd
        e = measure()
        want = INTO * depth + np.array([0.0, 0.0, Z_BIAS if depth > 0.004 else 0.0])
        cmd = cmd + np.clip(np.array([GAIN, DEPTH_GAIN, GAIN]) * (want - e), -0.0015, 0.0015)
        cmd = np.clip(cmd, nominal(depth) - 0.006, nominal(depth) + 0.006)
        return e

    depth = -0.04
    while depth < -0.006:
        depth += 0.002
        step_to(depth)
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="approach along the port axis")
    calm = 0
    for _ in range(300):
        e = step_to(-0.006)
        if _ % 50 == 0:
            print("ALIGN", sc.measure(*_plug(ctx.states()["cable"]), PORT_POS), "tilt", ctx.tilt.tolist(), flush=True)
        calm = calm + 1 if max(abs(e[0]), abs(e[2])) < 0.00015 and abs(float(e @ INTO) + 0.006) < 0.0005 else 0
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="line up on the port")
        if calm >= 8:
            break
    held = 0
    while depth < DEMO_DEPTH:
        e = measure()
        off = max(abs(e[0]), abs(e[2]))
        if off < 0.0002 and float(e @ INTO) > depth - 0.002:
            depth += 0.0005
            held = 0
        else:
            held += 1
        if held > 150:
            break
        step_to(depth)
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="push in along the port axis")
    prev = -1.0
    for _ in range(160):  # keep pushing until the plug stops moving in: its shoulder is then against the port's mouth
        e = measure()
        here = float(e @ INTO)
        step_to(DEMO_DEPTH)
        yield from arm.move(cmd, yaw=yaw_now, duration=0.05, label="seat it")
        if here > 0.0132 or (_ % 10 == 9 and abs(here - prev) < 0.00003):
            break
        if _ % 10 == 9:
            prev = here
    for _ in range(20):
        yield
    yield from arm.grip(FINGER_OPEN, 0.6)
    for _ in range(40):
        yield
    yield from arm.move(cmd - INTO * 0.06 + UP * 0.03, yaw=yaw_now, duration=1.0, label="retreat")
    for _ in range(40):
        yield

    yield from arm.wait(22.0)  # verify retained seating after release and retreat
