"""An adult trauma-care training manikin lies face up on the table with its head toward the robot and its feet away:
head, neck and the top of the chest. The patient is unconscious and must be treated as having a neck injury. Its
head rests on the back of the skull and, if nothing holds it, it rolls to one side (toward the robot's left): about
10 degrees after 3 seconds, and on until it lies on its side. A light hand on the forehead, pressing on the side that
rises, is enough to stop it. The face and forehead are flat, 12 cm across. The neck arches up off the table, leaving
a gap about 3 cm high under it.

A two-piece rigid neck collar lies on the table on the robot's left: the back half (a flat blue foam plate with one
raised end, lying beside the neck) and the front half (a white shell with short side flanges and a blue chin rest
along one edge, standing on its flanges beside the head). Fitting the collar means sliding the back half flat under
the neck until it is centred, then setting the front half on the front of the neck, centred, flanges down either
side and chin rest toward the chin.
The head must be kept straight the whole time: not rolled more than 10 degrees either way.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import parts  # noqa: E402  (articulated parts; hooks yam_arena so the head's hinge is simulated)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SKIN = (0.85, 0.70, 0.60)
COLLAR = (0.93, 0.93, 0.90)    # off-white plastic shell
FOAM = (0.20, 0.35, 0.60)      # blue foam lining / ends

# --- Manikin ---------------------------------------------------------------------------------------------------
# World frame: x away from the robot, y to the robot's left, z up. The head is toward the robot (small x), the
# feet away (+x). Face up, the patient's right is toward -y and its left toward +y.
HEAD_R, HEAD_LEN = 0.08, 0.20  # skull as a cylinder along the body: 16 cm across, crown to chin 20 cm
HEAD_X = 0.12                  # centre of the head; crown at x 0.02, chin end at 0.22
HEAD_LIFT = 0.002              # the skull clears the table by 2 mm: the table's friction would otherwise hold it
NECK_X0, NECK_X1 = 0.22, 0.31  # neck between chin and collarbones: 9 cm
NECK_X = (NECK_X0 + NECK_X1) / 2
NECK_W = 0.11                  # side to side
NECK_GAP = 0.03                # height of the gap under the arched neck
NECK_TOP = 0.115               # front of the neck above the table
CHEST_X1, CHEST_W, CHEST_H = 0.58, 0.38, 0.14
FACE_LEN, FACE_W = 0.14, 0.12 # the face and forehead: a flat 14 x 12 cm area over the skull
FACE_TOP = 2 * HEAD_R + 0.005  # forehead height above the skull's lowest point
FOREHEAD_X = 0.075             # middle of the forehead, between crown and brow
DARK = (0.15, 0.12, 0.10)

# The head rolls about its long axis. Its weight pulls it gently toward its left: a weak spring whose rest point is
# the head lying on its side (ROLL_STOP), against damping. Unheld it passes 10 degrees about 3 s after the start and
# keeps going, ending up about 45 degrees over after 30 s. A light hand on the forehead holds it: the pull is about
# 0.04 N m, half a newton sideways at the top of the head.
ROLL_STOP = 0.9                # rad: lying on its side (also the hinge's limits)
HEAD_SPRING = 0.045            # N m/rad: pulls toward ROLL_STOP with about 0.04 N m near straight
HEAD_DAMPING = 0.7             # N m s/rad: about 3.3 degrees a second near straight

MANIKIN = parts.Part(
    "manikin",
    pos=(0.0, 0.0, TOP + HEAD_LIFT),
    shapes=[
        parts.box((NECK_X1 - NECK_X0 + 0.01, NECK_W, NECK_TOP - NECK_GAP),
                  at=((NECK_X0 + NECK_X1) / 2 + 0.005, 0.0, (NECK_TOP + NECK_GAP) / 2 - HEAD_LIFT), color=SKIN),
        parts.box((CHEST_X1 - NECK_X1, CHEST_W, CHEST_H),
                  at=((NECK_X1 + CHEST_X1) / 2, 0.0, CHEST_H / 2 - HEAD_LIFT), color=SKIN),
    ],
    joints=[
        parts.Hinge(
            "head",
            shapes=[
                parts.cylinder(HEAD_R, HEAD_LEN, at=(HEAD_X, 0.0, HEAD_R), color=SKIN, axis="x"),   # skull
                parts.box((FACE_LEN, FACE_W, 0.02), at=(HEAD_X + 0.01, 0.0, FACE_TOP - 0.01), color=SKIN),  # face
                parts.box((0.045, 0.028, 0.02), at=(HEAD_X + 0.04, 0.0, FACE_TOP + 0.01), color=SKIN),  # nose
                parts.box((0.012, 0.022, 0.002), at=(HEAD_X + 0.005, 0.03, FACE_TOP + 0.001), color=DARK),  # eyes
                parts.box((0.012, 0.022, 0.002), at=(HEAD_X + 0.005, -0.03, FACE_TOP + 0.001), color=DARK),
            ],
            pivot=(HEAD_X, 0.0, HEAD_R),   # the skull's axis
            axis=(-1.0, 0.0, 0.0),     # positive roll turns the face toward +y (the patient's left)
            lower=-ROLL_STOP, upper=ROLL_STOP, mass=1.0,
            spring=HEAD_SPRING, damping=HEAD_DAMPING, max_effort=20.0,
        ),
    ],
)

# --- Collar ----------------------------------------------------------------------------------------------------
PLATE_T = 0.012
BACK_W, BACK_LEN = 0.06, 0.24  # back half: 6 cm (along the neck) x 24 cm plate
END_H = 0.045                  # its raised end (+y), 12 mm thick
BACK_START = (NECK_X, 0.27, TOP + PLATE_T / 2 + 0.001)
BACK = parts.Part(
    "collar_back",
    pos=BACK_START,
    shapes=[
        parts.box((BACK_W, BACK_LEN, PLATE_T), color=FOAM),
        parts.box((BACK_W, PLATE_T, END_H), at=(0.0, BACK_LEN / 2 - PLATE_T / 2, PLATE_T / 2 + END_H / 2), color=FOAM),
    ],
    mass=0.05,
)

FRONT_W = 0.07                 # front half: 7 cm along the neck
FRONT_SPAN = 0.146             # its shell, side to side
FLANGE_Y = 0.068               # side flanges' centres (10 mm thick): 12.6 cm between them, over an 11 cm neck
FLANGE_H = 0.035               # they reach 3.5 cm down the sides of the neck
SHELL_T = 0.010
LIP = (0.010, 0.06, 0.020)     # chin rest: a raised lip along the head-side edge, the front half's handle
FRONT_SEATED_Z = TOP + NECK_TOP + SHELL_T / 2      # origin (shell's centre) when resting on the front of the neck
FRONT_START = (0.10, 0.25, TOP + FLANGE_H + SHELL_T / 2 + 0.001)   # standing on its flanges beside the head
FRONT = parts.Part(
    "collar_front",
    pos=FRONT_START,
    shapes=[
        parts.box((FRONT_W, FRONT_SPAN, SHELL_T), color=COLLAR),
        parts.box((FRONT_W, 0.010, FLANGE_H), at=(0.0, FLANGE_Y, -SHELL_T / 2 - FLANGE_H / 2), color=COLLAR),
        parts.box((FRONT_W, 0.010, FLANGE_H), at=(0.0, -FLANGE_Y, -SHELL_T / 2 - FLANGE_H / 2), color=COLLAR),
        parts.box(LIP, at=(-FRONT_W / 2 + LIP[0] / 2, 0.0, SHELL_T / 2 + LIP[2] / 2), color=FOAM),
    ],
    mass=0.06,
)

OBJECTS = [MANIKIN, BACK, FRONT]


# --- Simulator hook: the head's weight pulls it to its left -------------------------------------------------------
@parts.on_reset
def _settle(sim):
    sim.set_target(MANIKIN.name, "head", ROLL_STOP)


@parts.on_step
def _pull(sim):
    sim.set_target(MANIKIN.name, "head", ROLL_STOP)


# --- Scoring ---------------------------------------------------------------------------------------------------
ROLL_LIMIT_DEG = 10            # the head may not roll more than 10 degrees either way, at any time
PLACE_TOL = 0.015              # a half's centre within 1.5 cm of the neck's centre, along and across the neck
LEVEL_DEG = 10                 # each half within 10 degrees of level
SEAT_TOL = 0.008               # front half's shell within 8 mm of resting on the front of the neck
BACK_ON_TABLE = 0.010          # back half's lowest point within 10 mm of the table: lying under the neck, not lifted

_state = {}


def _fresh():
    _state.update(frozen=None, last=None)


_fresh()


def _tilt_deg(obj):
    q = np.asarray(obj.quat_xyzw, float)
    z = parts._quat_to_matrix(q)[:, 2]
    return math.degrees(math.acos(float(np.clip(z[2], -1.0, 1.0))))


def _roll_deg(objects):
    return math.degrees(objects[MANIKIN.name].joints["head"])


def _centred(obj):
    return abs(obj.pos[0] - NECK_X) <= PLACE_TOL and abs(obj.pos[1]) <= PLACE_TOL and _tilt_deg(obj) <= LEVEL_DEG


def _back_fitted(objects):
    b = objects[BACK.name]
    return _centred(b) and b.low[2] - TOP <= BACK_ON_TABLE


def _front_fitted(objects):
    f = objects[FRONT.name]
    return _centred(f) and abs(f.pos[2] - FRONT_SEATED_Z) <= SEAT_TOL


def _placement(objects):
    if not _back_fitted(objects):
        return 0.0
    return 1.0 if _front_fitted(objects) else 0.5


def _new_trial(back):
    """The task module lives across trials, so a reset is recognised by the back half jumping back to its start
    within one control step. An arm carrying it back moves it gradually and never triggers this."""
    pos = np.asarray(back.pos, float).copy()
    last, _state["last"] = _state["last"], pos
    if last is None:
        return True
    return bool(np.linalg.norm(pos - last) > 0.03 and np.linalg.norm(pos - np.asarray(BACK_START)) < 0.01)


def _track(objects):
    if _new_trial(objects[BACK.name]):
        _fresh()
        _state["last"] = np.asarray(objects[BACK.name].pos, float).copy()
    if _state["frozen"] is None and abs(_roll_deg(objects)) > ROLL_LIMIT_DEG:
        _state["frozen"] = _placement(objects)


def _current(objects):
    return _state["frozen"] if _state["frozen"] is not None else _placement(objects)


def score(objects):
    """0.5 with the back half centred flat under the neck; 1.0 with the front half also resting centred on the front
    of the neck. If the head ever rolls more than 10 degrees, the score is frozen at its value at that moment."""
    _track(objects)
    return _current(objects)


def success(objects):
    """A score of 1.0 (read only; score() tracks the head)."""
    return _current(objects) == 1.0


# --- Scripted demonstration --------------------------------------------------------------------------------------
FOREHEAD_Y = -0.02             # right arm presses the forehead 2 cm to the patient's right, the side that rises
HEAD_PRESS = 0.008             # right arm aims its closed fingertips 8 mm into the forehead: the compliant arm
                               # stops at the skin and presses lightly, holding the head about 3 degrees over
LIP_GRIP = 0.012                # left arm pinches the chin rest 12 mm above the shell's centre (fingers along x)
FRONT_YAW = -np.pi / 2


def demo():
    """Right arm: straight to the forehead and presses its right side down with closed fingertips for the whole
    trial, so the head cannot roll. Left arm: pinches the back half's raised end, slides it flat under the neck
    until centred and lets go; then pinches the front half's chin rest, lifts it, carries it beside the head to the
    neck, raises it over the neck's height, moves it over the neck, lowers it onto the front of the neck and lets go."""
    head_top = TOP + HEAD_LIFT + FACE_TOP + PAD_TIP_PAST_TCP
    right = [
        (1.0, (FOREHEAD_X, FOREHEAD_Y, head_top + 0.03), FINGER_CLOSED, "above the forehead"),
        (0.6, (FOREHEAD_X, FOREHEAD_Y, head_top - HEAD_PRESS), FINGER_CLOSED, "press the forehead"),
        (60.0, (FOREHEAD_X, FOREHEAD_Y, head_top - HEAD_PRESS), FINGER_CLOSED, "keep the head straight"),
    ]

    bx, by, _ = BACK_START
    end_y = by + BACK_LEN / 2 - PLATE_T / 2
    end_grip = TOP + 0.045
    seat_end_y = BACK_LEN / 2 - PLATE_T / 2      # raised end's y with the back half centred
    fx, fy, fz = FRONT_START
    lip_x = NECK_X - FRONT_W / 2 + LIP[0] / 2
    start_lip_x = fx - FRONT_W / 2 + LIP[0] / 2
    lift = TOP + 0.075 + LIP_GRIP                       # flanges clear the back half's raised end
    carry = TOP + NECK_TOP + 0.012 + FLANGE_H + SHELL_T / 2 + LIP_GRIP   # flanges 12 mm over the neck
    seat = FRONT_SEATED_Z + LIP_GRIP + 0.001
    left = [
        (2.0, (bx, end_y, end_grip + 0.05), FINGER_OPEN, 0.0, "above the back half's raised end"),
        (1.0, (bx, end_y, end_grip), FINGER_OPEN, 0.0, "down around it"),
        (0.6, (bx, end_y, end_grip), FINGER_CLOSED, 0.0, "pinch the raised end"),
        (3.5, (NECK_X, seat_end_y, end_grip + 0.001), FINGER_CLOSED, 0.0, "slide it flat under the neck"),
        (0.6, (NECK_X, seat_end_y, end_grip + 0.001), FINGER_OPEN, 0.0, "let go"),
        (0.8, (NECK_X, seat_end_y, end_grip + 0.06), FINGER_OPEN, 0.0, "up"),
        (2.0, (start_lip_x, fy, fz + LIP_GRIP + 0.04), FINGER_OPEN, FRONT_YAW, "above the front half's chin rest"),
        (1.0, (start_lip_x, fy, fz + LIP_GRIP), FINGER_OPEN, FRONT_YAW, "down around it"),
        (0.6, (start_lip_x, fy, fz + LIP_GRIP), FINGER_CLOSED, FRONT_YAW, "pinch the chin rest"),
        (1.0, (start_lip_x, fy, lift), FINGER_CLOSED, FRONT_YAW, "lift it"),
        (2.0, (lip_x, fy, lift), FINGER_CLOSED, FRONT_YAW, "carry it beside the head to the neck"),
        (1.2, (lip_x, fy, carry), FINGER_CLOSED, FRONT_YAW, "raise it over the neck's height"),
        (2.0, (lip_x, 0.0, carry), FINGER_CLOSED, FRONT_YAW, "over the neck"),
        (1.5, (lip_x, 0.0, seat), FINGER_CLOSED, FRONT_YAW, "lower it onto the front of the neck"),
        (0.6, (lip_x, 0.0, seat), FINGER_OPEN, FRONT_YAW, "let go"),
        (1.0, (lip_x, 0.0, seat + 0.05), FINGER_OPEN, FRONT_YAW, "up and away"),
        (4.0, (lip_x, 0.05, seat + 0.05), FINGER_OPEN, FRONT_YAW, "wait"),
    ]
    return {
        "left": [Waypoint(float(d), tuple(map(float, p)), f, label, yaw) for d, p, f, yaw, label in left],
        "right": [Waypoint(float(d), tuple(map(float, p)), f, label, 0.0) for d, p, f, label in right],
    }
