"""fibrekit: cablekit with a bent starting shape (a path), a tilted plug at either end, and an anchorable end.

This is the `cablekit` of the other tasks in this repository plus three additions: `cable(..., path=[points])` lays the cable
along a world-frame polyline instead of a straight line (segments of equal length, each joint at rest in that shape);
`anchor_end` welds the end plug to the world; and the plugs follow the cable's first and last segment, so they can stand
tilted. What follows is the original documentation.

cablekit: a flexible cable for yam_arena task files.

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import cablekit
    from cablekit import cable

    CABLE = cable("cable", start=(0.20, 0.10, TOP + 0.005), yaw=0.0, length=0.60, radius=0.003)
    OBJECTS = [CABLE, ...]

The cable is a chain of rigid capsule segments joined by joints that bend (two axes) but neither stretch nor twist.
Each joint has a spring-damper drive back to straight, which sets the cable's bending stiffness. Optional rigid
plugs (boxes) ride on either end, and the start can be anchored to the world (e.g. a cable leaving a wall box).

It plugs into yam_arena without modifying it: importing this module hooks ``yam_arena.scene`` so a ``Cable`` in
``OBJECTS`` is spawned as a PhysX articulation, and ``success()/score()/distance()`` receive a :class:`CableState`
(an ``ObjectState`` with the segment centre points, start/end plug poses) for it.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Cable", "CableState", "Plug", "cable", "install"]

DEFAULT_SEGMENT = 0.02  # m


def _dense(path, step=0.0005) -> np.ndarray:
    pts = np.asarray(path, dtype=float)
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(2, int(np.linalg.norm(b - a) / step))
        dense += [a + (b - a) * j / k for j in range(1, k + 1)]
    return np.array(dense)


def _walk(dense: np.ndarray, n: int, seg: float) -> np.ndarray:
    """n + 1 points along the dense polyline, each exactly `seg` from the last (the last is the polyline's end if it is
    nearer than that)."""
    out, i = [dense[0]], 0
    for _ in range(n):
        p = out[-1]
        j = i + 1
        while j < len(dense) and np.linalg.norm(dense[j] - p) < seg:
            j += 1
        if j >= len(dense):
            out.append(dense[-1])
            continue
        a, b = dense[j - 1], dense[j]
        d = b - a
        # |a + t d - p| = seg, the root in [0, 1]
        qa, qb, qc = d @ d, 2 * d @ (a - p), (a - p) @ (a - p) - seg * seg
        t = (-qb + np.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
        out.append(a + min(max(t, 0.0), 1.0) * d)
        i = j - 1
    return np.array(out)


def _fit_chord(dense: np.ndarray, n: int, seg: float) -> float:
    """The smallest chord length, at most `seg`, for which n chords just reach the polyline's end."""
    lo, hi = 0.5 * seg, seg
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.linalg.norm(_walk(dense, n, mid)[-1] - dense[-1]) < 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class Plug:
    """A rigid box on a cable end. ``size`` is (length along the cable, width, height) in m."""

    size: tuple[float, float, float] = (0.03, 0.015, 0.010)
    mass: float = 0.01
    color: tuple[float, float, float] = (0.12, 0.12, 0.12)
    extras: tuple = ()  # more boxes on the plug, ((centre in the plug's frame), (size), (colour)): a latch, a shoulder


@dataclass(frozen=True)
class Cable(Box):
    """A cable lying straight from ``start`` along ``yaw`` (rad about z; 0 = +x). Build one with :func:`cable`.

    It subclasses Box only so yam_arena's task-file checks accept it; ``pos``/``size`` are its bounding box.
    """

    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # centre of the cable's start end (m)
    length: float = 0.5  # m, excluding plugs
    radius: float = 0.003  # m
    segments: int = 25
    cable_mass: float = 0.03  # kg, whole cable without plugs
    bend_stiffness: float = 0.02  # N·m/rad per joint drive
    bend_damping: float = 0.002  # N·m·s/rad
    collision_sides: int = 8  # collider cross-section: n-gon prism (flats at ``radius``) so a straight cable can't roll
    # forever on a table (PhysX has no rolling resistance); 0 = round capsule collider (old behaviour). The visual is
    # always the round capsule.
    armature: float = 5e-6  # kg·m² per bend DOF: keeps tiny-link drives stable at 240 Hz (stability guide: armature >> k·dt²)
    max_bend: float = 1.2  # rad per joint (limit)
    friction: float = 0.8
    anchor_start: bool = False  # start end welded to the world
    anchor_end: bool = False  # end plug (or end segment) welded to the world
    plug_torsional_patch: float = 0.01  # m: torsional contact patch radius of the plugs (0 = none)
    path: tuple = ()  # world-frame polyline the cable lies along, or () for a straight cable from `start` along `yaw`
    self_collision: bool = True
    joint_kind: str = "d6"  # "d6" (bend 2 axes, no twist) or "spherical" (cone limit, twist free)
    start_plug: Plug | None = None
    end_plug: Plug | None = None
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def free(self) -> bool:  # never jittered by yam_arena (its reset writes rigid poses only)
        return False

    @property
    def seg_len(self) -> float:
        return self.length / self.segments

    def _check(self) -> None:
        super()._check()
        if self.segments < 2 or not self.length > 0 or not self.radius > 0:
            raise ValueError(f"{self.name}: need segments >= 2, length > 0, radius > 0")
        if self.seg_len <= 2 * self.radius:
            raise ValueError(f"{self.name}: segment length {self.seg_len:.4f} m must exceed the diameter")

    # -- geometry of the starting layout (pure numpy; use it in demo()) --------------------------------------

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    def chain(self) -> np.ndarray:
        """(segments + 1, 3) joint points of the starting layout: successive points a segment length apart (chord)."""
        if not self.path:
            return np.stack([self.point_at(i * self.seg_len) for i in range(self.segments + 1)])
        return _walk(_dense(self.path), self.segments, self.seg_len)

    def frames(self) -> list:
        """Per segment (centre, rotation matrix) in the starting layout: x along the segment, z as near up as it can be."""
        ch = self.chain()
        out = []
        for a, b in zip(ch[:-1], ch[1:]):
            d = (b - a) / np.linalg.norm(b - a)
            z = np.array([0.0, 0.0, 1.0]) - d[2] * d
            if np.linalg.norm(z) < 1e-3:
                z = np.array([0.0, 1.0, 0.0]) - d[1] * d
            z /= np.linalg.norm(z)
            y = np.cross(z, d)
            out.append(((a + b) / 2, np.stack([d, y, z], axis=1)))
        return out

    def point_at(self, s: float) -> np.ndarray:
        """Starting position of the point ``s`` metres along the cable from its start (plugs excluded)."""
        return np.asarray(self.start, dtype=float) + s * self.direction

    def start_points(self) -> np.ndarray:
        """Starting segment centres, (segments, 3)."""
        if self.path:
            return np.stack([c for c, _ in self.frames()])
        return np.stack([self.point_at((i + 0.5) * self.seg_len) for i in range(self.segments)])


def cable(
    name: str,
    start: tuple[float, float, float],
    *,
    yaw: float = 0.0,
    length: float = 0.5,
    radius: float = 0.003,
    segment: float = DEFAULT_SEGMENT,
    color: tuple[float, float, float] = (0.10, 0.10, 0.10),
    **kwargs,
) -> Cable:
    """A cable from ``start`` along ``yaw`` with segments about ``segment`` metres long."""
    path = tuple(tuple(float(v) for v in p) for p in kwargs.pop("path", ()))
    if path:
        pts = np.asarray(path)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        start = tuple(pts[0])
    n = max(2, round(length / segment))
    if path:
        length = _fit_chord(_dense(path), n, length / n) * n
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    a, b = np.asarray(start, dtype=float), np.asarray(start, dtype=float) + length * d
    if path:
        a, b = np.min(np.asarray(path), axis=0), np.max(np.asarray(path), axis=0)
    lo, hi = np.minimum(a, b) - radius, np.maximum(a, b) + radius
    return Cable(
        name=name,
        pos=tuple(((lo + hi) / 2).tolist()),
        size=tuple((hi - lo).tolist()),
        color=color,
        yaw=yaw,
        start=tuple(float(v) for v in start),
        length=length,
        radius=radius,
        segments=n,
        path=path,
        **kwargs,
    )


@dataclass(frozen=True)
class CableState(ObjectState):
    """An ObjectState for a cable, plus its shape."""

    points: np.ndarray = None  # (segments, 3) segment centres, start to end
    start_plug: np.ndarray | None = None  # (3,) plug centre, if any
    end_plug: np.ndarray | None = None
    start_dir: np.ndarray | None = None  # unit vector, pointing out of the cable at the start end
    end_dir: np.ndarray | None = None  # unit vector, pointing out of the cable at the end end
    seg_len: float = 0.0
    start_plug_quat: np.ndarray | None = None  # xyzw, the plug body's orientation (x along the cable, pointing out of it)
    end_plug_quat: np.ndarray | None = None

    @property
    def start_tip(self) -> np.ndarray:
        return self.points[0] + self.start_dir * self.seg_len / 2

    @property
    def end_tip(self) -> np.ndarray:
        return self.points[-1] + self.end_dir * self.seg_len / 2

    def arclength(self) -> np.ndarray:
        """Distance along the cable (m) of each segment centre from the start end."""
        return (np.arange(len(self.points)) + 0.5) * self.seg_len

    def polyline(self) -> np.ndarray:
        """Start tip, segment centres, end tip."""
        return np.vstack([self.start_tip, self.points, self.end_tip])


def _wxyz(r: np.ndarray) -> tuple:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = float(np.trace(r))
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        return (0.25 * k, (r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k, (r[1, 0] - r[0, 1]) / k)
    i = int(np.argmax(np.diag(r)))
    j, m = (i + 1) % 3, (i + 2) % 3
    k = math.sqrt(1.0 + r[i, i] - r[j, j] - r[m, m]) * 2
    v = [0.0, 0.0, 0.0]
    v[i], v[j], v[m] = 0.25 * k, (r[j, i] + r[i, j]) / k, (r[m, i] + r[i, m]) / k
    return ((r[m, j] - r[j, m]) / k, v[0], v[1], v[2])


# -- Isaac Sim side ------------------------------------------------------------------------------------------


def _spawn_cable(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the cable articulation under ``prim_path`` (pxr only). ``cfg.cable_name`` keys the Cable spec."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Cable = _SPECS[cfg.cable_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    vis = f"{prim_path}/Looks/cable"
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.5)
    look.func(vis, look)
    phys = f"{prim_path}/Looks/cable_physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)

    L, r, n = spec.seg_len, spec.radius, spec.segments
    seg_mass = spec.cable_mass / n

    def body(path: str, x, mass: float, quat=(1.0, 0.0, 0.0, 0.0)):
        xform = UsdGeom.Xform.Define(stage, path)
        pos = Gf.Vec3d(*map(float, x)) if hasattr(x, "__len__") else Gf.Vec3d(float(x), 0.0, 0.0)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(pos)
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, quat)))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateLinearDampingAttr(0.5)
        rb.CreateAngularDampingAttr(2.0)
        return prim

    def collide(geom_prim, color_path: str, torsional: float = 0.0):
        UsdPhysics.CollisionAPI.Apply(geom_prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(geom_prim)
        cp.CreateContactOffsetAttr(0.002)
        cp.CreateRestOffsetAttr(0.0)
        if torsional:  # lets the pads of a pinch resist twisting the plug about the pinch axis
            cp.CreateTorsionalPatchRadiusAttr(torsional)
            cp.CreateMinTorsionalPatchRadiusAttr(0.8 * torsional)
        bind_physics_material(str(geom_prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(geom_prim.GetPath()), color_path, stage=stage)

    def prism_points(sides: int) -> list:
        """Convex n-gon prism along X with chamfered ends; flats at distance r from the axis, one flat facing -Z
        (the cable spawns lying on a flat). Ends reach about as far as the round capsule's caps."""
        rv = r / math.cos(math.pi / sides)  # vertex radius for flats at r
        pts = []
        for x, rr in ((L / 2 - 0.3 * r, rv), (L / 2 + 0.6 * r, 0.55 * rv)):
            for sx in (-x, x):
                for k in range(sides):
                    a = 2 * math.pi * (k + 0.5) / sides - math.pi / 2
                    pts.append(Gf.Vec3f(sx, rr * math.cos(a), rr * math.sin(a)))
        return pts

    sides = int(spec.collision_sides)
    seg_paths = []
    fr = spec.frames() if spec.path else None
    if fr:
        c0, q0 = fr[0]
        local = [(q0.T @ (c - c0), q0.T @ q) for c, q in fr]  # each segment's pose in seg_0's (the root's) frame
    for i in range(n):
        path = f"{prim_path}/seg_{i}"
        if fr:
            body(path, local[i][0], seg_mass, _wxyz(local[i][1]))
        else:
            body(path, i * L, seg_mass)  # seg_0 centred on the root frame: it is the articulation's root link
        cap = UsdGeom.Capsule.Define(stage, f"{path}/geom")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(r)
        cap.CreateHeightAttr(max(L - 2 * r, 1e-4) + 2 * r * 0.9)  # slight overlap: no visible gaps when bent
        if sides >= 3:
            bind_visual_material(str(cap.GetPath()), vis, stage=stage)
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/collider")
            pts = prism_points(sides)
            mesh.CreatePointsAttr(pts)
            # faces only matter for the hull cook; give it the two end caps + side quads of the outer rings
            m = sides
            counts, idx = [], []
            for ring in range(2):  # 0: inner rings (+-x0), 1: outer rings
                base = ring * 2 * m
                counts += [m, m]
                idx += [base + k for k in range(m)] + [base + m + k for k in range(m)]
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(idx)
            mesh.CreatePurposeAttr("guide")  # not rendered
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            collide(mesh.GetPrim(), vis)
        else:
            collide(cap.GetPrim(), vis)
        seg_paths.append(path)

    def joint(parent: str, child: str, name: str, x0: float, x1: float, bend: bool, rot1=(1.0, 0.0, 0.0, 0.0)):
        jp = f"{child}/{name}"
        if not bend:
            j = UsdPhysics.FixedJoint.Define(stage, jp)
        elif spec.joint_kind == "spherical":
            j = UsdPhysics.SphericalJoint.Define(stage, jp)
        else:
            j = UsdPhysics.Joint.Define(stage, jp)
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(x0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(x1, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(*map(float, rot1)))
        if not bend:
            return
        prim = j.GetPrim()
        if spec.joint_kind == "spherical":
            j.CreateAxisAttr("X")
            j.CreateConeAngle0LimitAttr(math.degrees(spec.max_bend))
            j.CreateConeAngle1LimitAttr(math.degrees(spec.max_bend))
            return
        for dof in ("transX", "transY", "transZ", "rotX"):  # no stretch, no twist
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        for dof in ("rotY", "rotZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, dof)
            lim.CreateLowAttr(-math.degrees(spec.max_bend))
            lim.CreateHighAttr(math.degrees(spec.max_bend))
            drv = UsdPhysics.DriveAPI.Apply(prim, dof)
            drv.CreateTypeAttr("force")
            # USD angular drive gains are per degree
            drv.CreateStiffnessAttr(spec.bend_stiffness * math.pi / 180)
            drv.CreateDampingAttr(spec.bend_damping * math.pi / 180)
            drv.CreateTargetPositionAttr(0.0)

    for i in range(1, n):
        # rest shape = the starting layout: the child's frame is the parent's turned by the layout's relative rotation
        rot1 = (1.0, 0.0, 0.0, 0.0)
        if fr:
            rel = local[i][1].T @ local[i - 1][1]
            rot1 = _wxyz(rel)
        joint(seg_paths[i - 1], seg_paths[i], f"bend_{i}", L / 2, -L / 2, bend=True, rot1=rot1)

    plug_pose = {}  # each plug's pose in the root's frame
    for end, plug in (("start", spec.start_plug), ("end", spec.end_plug)):
        if plug is None:
            continue
        sign = -1.0 if end == "start" else 1.0
        x = (-L / 2 if end == "start" else spec.length - L / 2) + sign * plug.size[0] / 2
        path = f"{prim_path}/{end}_plug"
        if fr and end == "end":
            cl, ql = local[-1]
            plug_pose[end] = (cl + ql[:, 0] * (L / 2 + plug.size[0] / 2), ql)
            body(path, plug_pose[end][0], plug.mass, _wxyz(ql))
        else:
            plug_pose[end] = (np.array([x, 0.0, 0.0]), np.eye(3))
            body(path, x, plug.mass)
        geom = UsdGeom.Cube.Define(stage, f"{path}/geom")
        geom.CreateSizeAttr(1.0)
        geom.AddScaleOp().Set(Gf.Vec3f(*plug.size))
        plook = f"{prim_path}/Looks/{end}_plug"
        pl = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(plug.color), roughness=0.4)
        pl.func(plook, pl)
        collide(geom.GetPrim(), plook, torsional=spec.plug_torsional_patch)
        for k, (ec, es, ecol) in enumerate(plug.extras):
            extra = UsdGeom.Cube.Define(stage, f"{path}/extra_{k}")
            extra.CreateSizeAttr(1.0)
            extra.AddTranslateOp().Set(Gf.Vec3f(*map(float, ec)))
            extra.AddScaleOp().Set(Gf.Vec3f(*map(float, es)))
            elook = f"{prim_path}/Looks/{end}_plug_extra_{k}"
            el = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(ecol), roughness=0.4)
            el.func(elook, el)
            collide(extra.GetPrim(), elook, torsional=spec.plug_torsional_patch)
        seg = seg_paths[0] if end == "start" else seg_paths[-1]
        # plug is the child: the chain stays a tree rooted at seg_0
        joint(seg, path, "weld", sign * L / 2, -sign * plug.size[0] / 2, bend=False)

    root_prim = root.GetPrim()
    # Root API on seg_0 itself, so seg_0 is the root link and Isaac Lab's root pose is seg_0's pose (on a plain
    # Xform PhysX picks its own root link, e.g. a middle segment, and resets then shift the whole cable).
    root_link = stage.GetPrimAtPath(seg_paths[0])
    UsdPhysics.ArticulationRootAPI.Apply(root_link)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_link)
    art.CreateEnabledSelfCollisionsAttr(spec.self_collision)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    for end_name, on in (("start", spec.anchor_start), ("end", spec.anchor_end)):
        if not on:
            continue
        plug_ = spec.start_plug if end_name == "start" else spec.end_plug
        seg_ = seg_paths[0] if end_name == "start" else seg_paths[-1]
        anchor = seg_ if plug_ is None else f"{prim_path}/{end_name}_plug"
        fj = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/anchor_{end_name}")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(anchor)])
        if fr:  # the root's world pose is set after spawning, so build the anchor's world pose from the layout
            t_loc, r_loc = plug_pose[end_name] if plug_ is not None else (local[0 if end_name == "start" else -1][0], local[0 if end_name == "start" else -1][1])
            t_w = c0 + q0 @ t_loc
            qw = _wxyz(q0 @ r_loc)
            fj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, t_w)))
            fj.CreateLocalRot0Attr(Gf.Quatf(*map(float, qw)))
        else:
            world = UsdGeom.Xformable(stage.GetPrimAtPath(anchor)).ComputeLocalToWorldTransform(0)
            t = world.ExtractTranslation()
            q = world.ExtractRotationQuat()
            fj.CreateLocalPos0Attr(Gf.Vec3f(*t))
            fj.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return root_prim


def _arena_object(spec: Cable):
    """The cable as an Arena articulation object."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class CableSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_cable)
        cable_name: str = ""

    class CableObject(Object):
        def _generate_articulation_cfg(self):
            cfg = ArticulationCfg(
                prim_path=self.prim_path,
                spawn=self.spawner_cfg,
                actuators={
                    "bend": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=spec.bend_stiffness,
                        damping=spec.bend_damping,
                        armature=spec.armature,
                    )
                },
            )
            return self._add_initial_pose_to_cfg(cfg)

    yaw = spec.yaw
    if spec.path:
        c0, q0 = spec.frames()[0]
        w, qx, qy, qz = _wxyz(q0)
        pose = Pose(position_xyz=tuple(float(v) for v in c0), rotation_xyzw=(float(qx), float(qy), float(qz), float(w)))
    else:
        pose = Pose(position_xyz=tuple(spec.point_at(spec.seg_len / 2).tolist()), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    return CableObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION,
        spawner_cfg=CableSpawnerCfg(cable_name=spec.name),
        initial_pose=pose,
    )


def _cable_state(env, spec: Cable) -> CableState:
    base = env.unwrapped
    art = base.scene[spec.name]
    origin = base.scene.env_origins[0]

    def t(x):
        return getattr(x, "torch", x)

    names = list(art.body_names)
    pos = t(art.data.body_pos_w)[0] - origin
    quat = t(art.data.body_quat_w)[0]
    vel = t(art.data.body_lin_vel_w)[0]
    angv = t(art.data.body_ang_vel_w)[0]
    idx = [names.index(f"seg_{i}") for i in range(spec.segments)]
    to_np = lambda a: a.detach().cpu().numpy().astype(np.float64)  # noqa: E731
    pts = to_np(pos[idx])
    plugs, quats = {}, {}
    for end in ("start", "end"):
        key = f"{end}_plug"
        plugs[end] = to_np(pos[names.index(key)]) if key in names else None
        quats[end] = to_np(quat[names.index(key)]) if key in names else None  # Isaac Lab 3.0 body quats are xyzw
    sd = pts[0] - pts[1]
    ed = pts[-1] - pts[-2]
    lo = pts.min(0) - spec.radius
    hi = pts.max(0) + spec.radius
    return CableState(
        pos=pts.mean(0),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        lin_vel=to_np(vel[idx]).mean(0),
        ang_vel=to_np(angv[idx]).mean(0),
        low=lo,
        high=hi,
        points=pts,
        start_plug=plugs["start"],
        end_plug=plugs["end"],
        start_dir=sd / (np.linalg.norm(sd) + 1e-12),
        end_dir=ed / (np.linalg.norm(ed) + 1e-12),
        seg_len=spec.seg_len,
        start_plug_quat=quats["start"],
        end_plug_quat=quats["end"],
    )


# -- hooking yam_arena.scene ---------------------------------------------------------------------------------

_SPECS: dict[str, Cable] = {}


def _patch_scene(scene) -> None:
    if getattr(scene, "_cablekit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Cable):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _cable_state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._cablekit_patched = True  # (same flag as cablekit: do not load both kits)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_scene(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena.scene (now if already imported, else when it is). Idempotent."""
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


import importlib.util  # noqa: E402  (used by _Finder)

install()
assert task_file  # re-exported for task files that want TABLE_TOP etc.
