"""yamkit: modeled props, mechanisms, per-collider physics materials, and gripper states for yam_arena tasks.

Adapted for vent-and-unlock-pressure-pot:
- Supports free-standing Mechanism articulations (`mass is not None`) with PhysicsArticulationRootAPI placed on `base`
  so PhysX roots the floating articulation at the pot/lid body rather than a child link.
- Supports per-collider `friction`, `combine`, and explicit `contact_offset=0.001` so tight bayonet fits and sliding
  bases behave accurately without spawner material overrides.
- Populates actual `lin_vel`, `ang_vel`, and rotated bounding boxes (`low`, `high`) in `_mech_states`.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET
from yam_arena.task_file import Box, ObjectState

GRIPPERS = {"left_gripper": "left_robot", "right_gripper": "right_robot"}
TIP_LINKS = ("tip_left", "tip_right")
_OPENING: dict[str, float] = {}
_EPISODE = {"count": 0, "last": None}  # trials seen, and the env's step count within the current one
_PROPS: set[str] = set()
_ORIGIN = np.zeros(3, dtype=np.float64)


# -- shapes ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Look:
    """A plain surface: sRGB color (0..1), roughness, metallic and opacity."""

    color: tuple[float, float, float]
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0


@dataclass(frozen=True)
class Mesh:
    """Triangles and quads in the prop's own frame (metres), with one look."""

    points: tuple
    counts: tuple
    indices: tuple
    look: Look


def _rot(axis: str | np.ndarray) -> np.ndarray:
    """A rotation taking +z to `axis` ("x", "y", "z" or a vector)."""
    a = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}.get(axis, axis) if isinstance(axis, str) else axis
    a = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, a), float(np.dot(z, a))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to USD Gf.Quatf (w, x, y, z)."""
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (0.25 * s, float((R[2, 1] - R[1, 2]) / s), float((R[0, 2] - R[2, 0]) / s),
                float((R[1, 0] - R[0, 1]) / s))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (float((R[2, 1] - R[1, 2]) / s), 0.25 * s, float((R[0, 1] + R[1, 0]) / s),
                float((R[0, 2] + R[2, 0]) / s))
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return (float((R[0, 2] - R[2, 0]) / s), float((R[0, 1] + R[1, 0]) / s), 0.25 * s,
                float((R[1, 2] + R[2, 1]) / s))
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return (float((R[1, 0] - R[0, 1]) / s), float((R[0, 2] + R[2, 0]) / s),
            float((R[1, 2] + R[2, 1]) / s), 0.25 * s)


def _mesh(points, counts, indices, look: Look) -> Mesh:
    return Mesh(tuple(map(tuple, np.round(np.asarray(points, float), 6))), tuple(counts), tuple(indices), look)


def box(center, size, look: Look, yaw: float = 0.0) -> Mesh:
    """An axis-aligned (or yawed about Z) box."""
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    p = [c + Rz @ (h * np.array([sx, sy_, sz])) for sz in (-1, 1) for sy_ in (-1, 1) for sx in (-1, 1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return _mesh(p, [4] * 6, [i for f in faces for i in f], look)


def lathe(profile, look: Look, center=(0, 0, 0), axis="z", n: int = 40) -> Mesh:
    """A surface of revolution: `profile` is (radius, height) pairs from bottom to top, turned about `axis` through
    `center`. A radius of 0 closes the end."""
    r = _rot(axis)
    pts, counts, idx = [], [], []
    for rad, z in profile:
        for k in range(n):
            t = 2 * math.pi * k / n
            pts.append(np.asarray(center) + r @ np.array([rad * math.cos(t), rad * math.sin(t), z]))
    for i in range(len(profile) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            counts.append(4)
            idx += [a, b, b + n, a + n]
    for end, flip in ((0, True), (len(profile) - 1, False)):
        if profile[end][0] > 0:
            ring = [end * n + k for k in range(n)]
            counts.append(n)
            idx += ring[::-1] if flip else ring
    return _mesh(pts, counts, idx, look)


def cylinder(center, radius: float, length: float, look: Look, axis="z", n: int = 32) -> Mesh:
    """A closed cylinder of `length` along `axis`, centered on `center`."""
    return lathe([(radius, -length / 2), (radius, length / 2)], look, center, axis, n)


def band(path, width: float, thickness: float, look: Look, normal=(1, 0, 0)) -> Mesh:
    """A flat strap following `path` (points), `width` wide along `normal` x tangent and `thickness` thick."""
    path = np.asarray(path, float)
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    pts, counts, idx = [], [], []
    for i, p in enumerate(path):
        tan = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        tan /= np.linalg.norm(tan)
        out = np.cross(tan, nrm)
        out /= np.linalg.norm(out)
        for s in (-1, 1):
            for t in (-1, 1):
                pts.append(p + nrm * s * width / 2 + out * t * thickness / 2)
    for i in range(len(path) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for e0, e1 in ((0, 1), (1, 3), (3, 2), (2, 0)):
            counts.append(4)
            idx += [a + e0, a + e1, b + e1, b + e0]
    for end, flip in ((0, True), (len(path) - 1, False)):
        ring = [4 * end + k for k in (0, 1, 3, 2)]
        counts.append(4)
        idx += ring[::-1] if flip else ring
    return _mesh(pts, counts, idx, look)


@dataclass(frozen=True)
class Collider:
    """An invisible box (`size`) or cylinder (`radius`, `height`, along `axis`) collider in the prop's frame.
    Optional `yaw` turns a box collider around Z. Optional `friction` & `combine` override default grippy material.
    Optional `contact_offset` sets PhysxCollisionAPI contact offset (default 0.001 m for tight mechanism fits)."""

    center: tuple
    size: tuple | None = None
    radius: float = 0.0
    height: float = 0.0
    axis: str = "Z"
    yaw: float = 0.0
    friction: float | None = None
    combine: str = "average"
    contact_offset: float | None = 0.001

    def extent(self, k: int) -> float:
        """Half-size along axis k (0, 1, 2) of the prop's frame, from `center`."""
        if self.size is not None:
            if k < 2 and self.yaw != 0.0:
                hx, hy = self.size[0] / 2, self.size[1] / 2
                c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
                return hx * c + hy * s if k == 0 else hx * s + hy * c
            return self.size[k] / 2
        return self.height / 2 if "XYZ"[k] == self.axis.upper() else self.radius


# -- props -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prop(Box):
    """A modeled object."""

    meshes: tuple = field(default=())
    colliders: tuple = field(default=())
    tilt_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _fmt(v) -> str:
    return "(" + ", ".join(f"{float(x):.6g}" for x in v) + ")"


def _phys_mats_usda(colliders) -> tuple[str, dict[tuple, str]]:
    """Emit physics materials for colliders when any custom friction is requested."""
    specs = []
    for c in colliders:
        if c.friction is not None:
            specs.append((float(c.friction), str(c.combine)))
    if not specs:
        return "", {}
    unique = [(1.2, "average")] + [s for s in dict.fromkeys(specs) if s != (1.2, "average")]
    mapping = {s: f"PM{i}" for i, s in enumerate(unique)}
    mapping[(None, "average")] = "PM0"
    lines = []
    for i, (mu, comb) in enumerate(unique):
        dyn = 1.0 if (mu == 1.2 and comb == "average") else mu
        lines.append(
            f'    def Material "PM{i}" (\n'
            f'        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]\n'
            f'    )\n    {{\n'
            f'        float physics:staticFriction = {mu}\n'
            f'        float physics:dynamicFriction = {dyn}\n'
            f'        float physics:restitution = 0\n'
            f'        uniform token physxMaterial:frictionCombineMode = "{comb}"\n'
            f'        uniform token physxMaterial:restitutionCombineMode = "min"\n'
            f'    }}\n'
        )
    return "".join(lines), mapping


def _collider_usda(i: int, c: Collider, indent: str, mat_map: dict) -> str:
    kind = "Cube" if c.size is not None else "Cylinder"
    mat_name = mat_map.get((None if c.friction is None else float(c.friction), str(c.combine)))
    apis = ['"PhysicsCollisionAPI"']
    if c.contact_offset is not None:
        apis.append('"PhysxCollisionAPI"')
    if mat_name is not None:
        apis.append('"MaterialBindingAPI"')
    lines = [
        f'{indent}def {kind} "C{i}" (\n{indent}    prepend apiSchemas = [{", ".join(apis)}]\n{indent})\n{indent}{{\n',
        f'{indent}    uniform token purpose = "guide"\n',
    ]
    if c.contact_offset is not None:
        lines.append(f'{indent}    float physxCollision:contactOffset = {c.contact_offset}\n')
        lines.append(f'{indent}    float physxCollision:restOffset = 0\n')
    if mat_name is not None:
        lines.append(f'{indent}    rel material:binding:physics = </Root/{mat_name}>\n')
    if c.size is not None:
        lines.append(f'{indent}    double size = 1\n')
        lines.append(f'{indent}    double3 xformOp:translate = {_fmt(c.center)}\n')
        if c.yaw != 0.0:
            q = (math.cos(c.yaw / 2), 0.0, 0.0, math.sin(c.yaw / 2))
            lines.append(f'{indent}    quatf xformOp:orient = {_fmt(q)}\n')
            lines.append(f'{indent}    float3 xformOp:scale = {_fmt(c.size)}\n')
            lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n')
        else:
            lines.append(f'{indent}    float3 xformOp:scale = {_fmt(c.size)}\n')
            lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n')
    else:
        lines.append(f'{indent}    uniform token axis = "{c.axis.upper()}"\n')
        lines.append(f'{indent}    double radius = {c.radius}\n')
        lines.append(f'{indent}    double height = {c.height}\n')
        lines.append(f'{indent}    double3 xformOp:translate = {_fmt(c.center)}\n')
        lines.append(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]\n')
    lines.append(f'{indent}}}\n')
    return "".join(lines)


def _usda(prop: Prop) -> str:
    looks = list(dict.fromkeys(m.look for m in prop.meshes))
    mats_str, mat_map = _phys_mats_usda(prop.colliders)
    out = ['#usda 1.0\n(\n    defaultPrim = "Root"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n']
    out.append('def Xform "Root" (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n')
    if prop.mass is not None:
        out.append(f"    float physics:mass = {prop.mass}\n")
    if mats_str:
        out.append(mats_str)
    out.append('    def Scope "Looks"\n    {\n')
    for i, lk in enumerate(looks):
        col = _fmt(_linear(c) for c in lk.color)
        out.append(
            f'        def Material "M{i}"\n        {{\n'
            f'            token outputs:surface.connect = </Root/Looks/M{i}/S.outputs:surface>\n'
            f'            def Shader "S"\n            {{\n'
            f'                uniform token info:id = "UsdPreviewSurface"\n'
            f'                color3f inputs:diffuseColor = {col}\n'
            f'                float inputs:roughness = {lk.roughness}\n'
            f'                float inputs:metallic = {lk.metallic}\n'
            f'                float inputs:opacity = {lk.opacity}\n'
            f'                token outputs:surface\n            }}\n        }}\n'
        )
    out.append("    }\n")
    for i, m in enumerate(prop.meshes):
        out.append(
            f'    def Mesh "V{i}" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {{\n'
            f'        point3f[] points = [{", ".join(_fmt(p) for p in m.points)}]\n'
            f'        int[] faceVertexCounts = [{", ".join(map(str, m.counts))}]\n'
            f'        int[] faceVertexIndices = [{", ".join(map(str, m.indices))}]\n'
            f'        uniform token subdivisionScheme = "none"\n'
            f'        uniform bool doubleSided = 1\n'
            f'        rel material:binding = </Root/Looks/M{looks.index(m.look)}>\n    }}\n'
        )
    for i, c in enumerate(prop.colliders):
        out.append(_collider_usda(i, c, "    ", mat_map))
    out.append("}\n")
    return "".join(out)


def _write(prop: Prop) -> str:
    text = _usda(prop)
    folder = os.path.join(tempfile.gettempdir(), "yamkit_usd")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{prop.name}_{hashlib.sha1(text.encode()).hexdigest()[:12]}.usda")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
    return path


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# -- mechanisms: parts on hinges and slides (Kit A) ------------------------------------------


@dataclass(frozen=True)
class Latch:
    """A catch on a joint."""

    at: float
    zone: float
    hold: float = 200.0


@dataclass(frozen=True)
class Link:
    """A moving part of a `Mechanism`, joined to its base or parent link."""

    name: str
    kind: str
    axis: str
    anchor: tuple
    lower: float
    upper: float
    start: float = 0.0
    mass: float = 0.05
    meshes: tuple = ()
    colliders: tuple = ()
    spring: float = 0.0
    rest: float | None = None
    damping: float = 0.5
    friction: float = 0.0
    floating: bool = False
    latch: Latch | None = None
    stay: float = 0.0
    drag: float = 0.0
    law: object = None
    parent: str | None = None


@dataclass(frozen=True)
class Mechanism(Prop):
    """A `Prop` (its meshes and colliders are the base) with moving `links`."""

    links: tuple = field(default=())


@dataclass(frozen=True)
class JointState:
    """A link's joint: `pos` (rad or m), `vel`, and `latched` once its latch has caught."""

    pos: float
    vel: float
    latched: bool


_MECHS: dict[str, Mechanism] = {}
_JOINTS: dict[str, JointState] = {}
_LATCHED: dict[str, int] = {}
_LAW_MEM: dict[str, tuple] = {}


def _law_mem(key: str, lk) -> dict:
    """The law's memory for this trial; a fresh one (seeded by the law's `fresh`) after every scene reset."""
    ep, mem = _LAW_MEM.get(key, (None, None))
    if ep != _EPISODE["count"]:
        mem = {"start": lk.start}
        if hasattr(lk.law, "fresh"):
            mem.update(lk.law.fresh(lk))
        _LAW_MEM[key] = (_EPISODE["count"], mem)
    return mem


def law_state(key: str) -> dict:
    """What link `key`'s law has recorded this trial."""
    return _LAW_MEM.get(key, (None, {}))[1]


def joint(objects, key: str) -> JointState:
    """The joint state of link `key` ("mechanism/link") at this control step."""
    return _JOINTS[key]


def _body_usda(name: str, translate, mass, meshes, colliders, looks, mat_map: dict,
               floating=False, art_root=False, indent="    ") -> list[str]:
    i2 = indent + "    "
    apis = ['"PhysicsRigidBodyAPI"', '"PhysicsMassAPI"', '"PhysxRigidBodyAPI"']
    if art_root:
        apis += ['"PhysicsArticulationRootAPI"', '"PhysxArticulationAPI"']
    out = [
        f'{indent}def Xform "{name}" (\n{i2}prepend apiSchemas = [{", ".join(apis)}]\n{indent})\n{indent}{{\n',
        f"{i2}float physics:mass = {mass}\n",
        *([f"{i2}float3 physics:diagonalInertia = {_fmt([mass * 1e-4] * 3)}\n"] if not colliders else []),
        f"{i2}bool physxRigidBody:disableGravity = {1 if floating else 0}\n",
        f"{i2}float physxRigidBody:maxDepenetrationVelocity = 1.0\n",
    ]
    if art_root:
        out += [
            f"{i2}bool physxArticulation:enabledSelfCollisions = 0\n",
            f"{i2}int physxArticulation:solverPositionIterationCount = 32\n",
            f"{i2}int physxArticulation:solverVelocityIterationCount = 4\n",
            f"{i2}float physxArticulation:sleepThreshold = 0\n",
            f"{i2}float physxArticulation:stabilizationThreshold = 0\n",
        ]
    out += [
        f"{i2}double3 xformOp:translate = {_fmt(translate)}\n",
        f'{i2}uniform token[] xformOpOrder = ["xformOp:translate"]\n',
    ]
    for i, m in enumerate(meshes):
        out.append(
            f'{i2}def Mesh "V{i}" (\n{i2}    prepend apiSchemas = ["MaterialBindingAPI"]\n{i2})\n{i2}{{\n'
            f'{i2}    point3f[] points = [{", ".join(_fmt(p) for p in m.points)}]\n'
            f'{i2}    int[] faceVertexCounts = [{", ".join(map(str, m.counts))}]\n'
            f'{i2}    int[] faceVertexIndices = [{", ".join(map(str, m.indices))}]\n'
            f'{i2}    uniform token subdivisionScheme = "none"\n'
            f'{i2}    uniform bool doubleSided = 1\n'
            f'{i2}    rel material:binding = </Root/Looks/M{looks.index(m.look)}>\n{i2}}}\n'
        )
    for i, c in enumerate(colliders):
        out.append(_collider_usda(i, c, i2, mat_map))
    out.append(f"{indent}}}\n")
    return out


def _mech_has_custom_mat(mech: Mechanism) -> bool:
    return any(c.friction is not None for c in mech.colliders) or any(
        c.friction is not None for lk in mech.links for c in lk.colliders
    )


def _mech_usda(mech: Mechanism) -> str:
    """The base and links as one articulation."""
    meshes = list(mech.meshes) + [m for lk in mech.links for m in lk.meshes]
    looks = list(dict.fromkeys(m.look for m in meshes))
    all_colliders = list(mech.colliders) + [c for lk in mech.links for c in lk.colliders]
    mats_str, mat_map = _phys_mats_usda(all_colliders)
    free_base = mech.mass is not None
    out = ['#usda 1.0\n(\n    defaultPrim = "Root"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n']
    if free_base:
        # Put PhysicsArticulationRootAPI on `base` (not Root) so PhysX roots the floating body at the base!
        out.append('def Xform "Root"\n{\n')
    else:
        out += [
            'def Xform "Root" (\n    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]\n)\n{\n',
            "    bool physxArticulation:enabledSelfCollisions = 0\n",
            "    int physxArticulation:solverPositionIterationCount = 32\n",
            "    int physxArticulation:solverVelocityIterationCount = 4\n",
            "    float physxArticulation:sleepThreshold = 0\n",
            "    float physxArticulation:stabilizationThreshold = 0\n",
        ]
    if mats_str:
        out.append(mats_str)
    out.append('    def Scope "Looks"\n    {\n')
    for i, lk in enumerate(looks):
        col = _fmt(_linear(c) for c in lk.color)
        out.append(
            f'        def Material "M{i}"\n        {{\n'
            f'            token outputs:surface.connect = </Root/Looks/M{i}/S.outputs:surface>\n'
            f'            def Shader "S"\n            {{\n'
            f'                uniform token info:id = "UsdPreviewSurface"\n'
            f'                color3f inputs:diffuseColor = {col}\n'
            f'                float inputs:roughness = {lk.roughness}\n'
            f'                float inputs:metallic = {lk.metallic}\n'
            f'                float inputs:opacity = {lk.opacity}\n'
            f'                token outputs:surface\n            }}\n        }}\n'
        )
    out.append("    }\n")
    out += _body_usda(
        "base", (0, 0, 0), mech.mass if free_base else 1.0, mech.meshes, mech.colliders, looks,
        mat_map, art_root=free_base,
    )
    if not free_base:
        out.append('    def PhysicsFixedJoint "root_joint"\n    {\n        rel physics:body1 = </Root/base>\n    }\n')
    for lk in mech.links:
        out += _body_usda(lk.name, lk.anchor, lk.mass, lk.meshes, lk.colliders, looks, mat_map, floating=lk.floating)
        kind = {"revolute": "PhysicsRevoluteJoint", "prismatic": "PhysicsPrismaticJoint"}[lk.kind]
        scale = math.degrees(1.0) if lk.kind == "revolute" else 1.0
        lin = "angular" if lk.kind == "revolute" else "linear"
        drive_api = f', "PhysicsDriveAPI:{lin}"' if lk.law is not None else ""
        k_init = getattr(lk.law, "k", 0.0) if (lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)) else 0.0
        d_init = getattr(lk.law, "damp", lk.damping) if (lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)) else lk.damping
        r_init = getattr(lk.law, "rest", 0.0) * scale if (lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)) else 0.0
        drive = (
            f'        uniform token drive:{lin}:physics:type = "force"\n'
            f"        float drive:{lin}:physics:maxForce = 1000\n"
            f"        float drive:{lin}:physics:stiffness = {k_init}\n"
            f"        float drive:{lin}:physics:damping = {d_init}\n"
            f"        float drive:{lin}:physics:targetPosition = {r_init}\n"
        ) if lk.law is not None else ""
        out.append(
            f'    def {kind} "{lk.name}_joint" (\n        prepend apiSchemas = ["PhysxJointAPI", "PhysicsJointStateAPI:{lin}"{drive_api}]\n    )\n    {{\n'
            + drive
            + f'        float state:{lin}:physics:position = {lk.start * scale:.7g}\n'
            f'        float state:{lin}:physics:velocity = 0\n'
            f'        uniform token physics:axis = "{lk.axis}"\n'
            f"        rel physics:body0 = </Root/{lk.parent or 'base'}>\n        rel physics:body1 = </Root/{lk.name}>\n"
            f"        point3f physics:localPos0 = {_fmt(_local_anchor(mech, lk))}\n        quatf physics:localRot0 = (1, 0, 0, 0)\n"
            f"        point3f physics:localPos1 = (0, 0, 0)\n        quatf physics:localRot1 = (1, 0, 0, 0)\n"
            f"        float physics:lowerLimit = {(lk.lower - 1e-4) * scale:.7g}\n"
            f"        float physics:upperLimit = {(lk.upper + 1e-4) * scale:.7g}\n"
            f"        float physxJoint:jointFriction = {lk.friction}\n    }}\n"
        )
    out.append("}\n")
    return "".join(out)


def _local_anchor(mech: Mechanism, lk: Link) -> tuple:
    if lk.parent is None:
        return lk.anchor
    par = next(o for o in mech.links if o.name == lk.parent)
    return tuple(a - b for a, b in zip(lk.anchor, par.anchor))


def _write_text(name: str, text: str) -> str:
    folder = os.path.join(tempfile.gettempdir(), "yamkit_usd")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_{hashlib.sha1(text.encode()).hexdigest()[:12]}.usda")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
    return path


def _mech_object(scene, spec: Mechanism):
    from isaaclab.actuators import ImplicitActuatorCfg

    class _MechObject(scene.Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            cfg.actuators = {
                lk.name: ImplicitActuatorCfg(
                    joint_names_expr=[f"{lk.name}_joint"],
                    stiffness=(getattr(lk.law, "k", 0.0) if (lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)) else 0.0),
                    damping=(getattr(lk.law, "damp", lk.damping) if (lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)) else lk.damping),
                    friction=lk.friction,
                )
                for lk in spec.links
            }
            try:
                from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg

                if hasattr(cfg.spawn, "articulation_props"):
                    cfg.spawn.articulation_props = ArticulationRootPropertiesCfg(
                        sleep_threshold=0.0, stabilization_threshold=0.0
                    )
            except ImportError:
                pass
            cfg.init_state.joint_pos = {f"{lk.name}_joint": lk.start for lk in spec.links}
            cfg.init_state.joint_vel = {f"{lk.name}_joint": 0.0 for lk in spec.links}
            return cfg

    yaw = (0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))
    mat = None if _mech_has_custom_mat(spec) else scene._grippy()
    return _MechObject(
        name=spec.name,
        usd_path=_write_text(spec.name, _mech_usda(spec)),
        object_type=scene.ObjectType.ARTICULATION,
        initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=_quat_mul(yaw, spec.tilt_xyzw)),
        spawn_cfg_addon={"physics_material": mat},
    )


def _set_drive(art, j: int, target: float, stiffness: float, effort: float, damping: float | None = None) -> None:
    import torch

    dev = getattr(art.data.joint_pos, "torch", art.data.joint_pos).device
    one = lambda v: torch.tensor([[v]], dtype=torch.float32, device=dev)  # noqa: E731
    art.set_joint_position_target_index(target=one(target), joint_ids=[j])
    art.set_joint_effort_target_index(target=one(effort), joint_ids=[j])
    art.write_joint_stiffness_to_sim_index(stiffness=one(stiffness), joint_ids=[j])
    if damping is not None:
        art.write_joint_damping_to_sim_index(damping=one(damping), joint_ids=[j])


def _set_limits(art, j: int, lo: float, hi: float) -> None:
    import torch

    dev = getattr(art.data.joint_pos, "torch", art.data.joint_pos).device
    art.write_joint_position_limit_to_sim_index(
        limits=torch.tensor([[[lo, hi]]], dtype=torch.float32, device=dev),
        joint_ids=[j],
        warn_limit_violation=False,
    )


def _put_joint(art, j: int, value: float) -> None:
    import torch

    dev = getattr(art.data.joint_pos, "torch", art.data.joint_pos).device
    art.write_joint_position_to_sim_index(position=torch.tensor([[value]], dtype=torch.float32, device=dev), joint_ids=[j])
    art.write_joint_velocity_to_sim_index(velocity=torch.tensor([[0.0]], dtype=torch.float32, device=dev), joint_ids=[j])


def _watch_resets(base) -> None:
    if getattr(base, "_yamkit_reset", False):
        return
    base._yamkit_reset = True
    original = base.reset

    def reset(*a, **k):
        out = original(*a, **k)
        for name, spec in _MECHS.items():
            try:
                art = base.scene[name]
            except KeyError:
                continue
            names = list(art.joint_names)
            for lk in spec.links:
                j = names.index(f"{lk.name}_joint")
                is_spring = lk.law is not None and hasattr(lk.law, "rest") and not getattr(lk.law, "kinematic", False)
                _set_drive(
                    art, j,
                    target=(getattr(lk.law, "rest", lk.start) if is_spring else lk.start),
                    stiffness=(getattr(lk.law, "k", 0.0) if is_spring else 0.0),
                    effort=0.0,
                    damping=(getattr(lk.law, "damp", None) if is_spring else None),
                )
                if lk.law is not None and getattr(lk.law, "sets_limits", False):
                    _set_limits(art, j, lk.lower - 1e-4, lk.upper + 1e-4)
                _put_joint(art, j, lk.start)
        return out

    base.reset = reset


def _local_bounds(colliders, fallback_size) -> tuple[np.ndarray, np.ndarray]:
    if not colliders:
        h = np.asarray(fallback_size, float) / 2
        return -h, h
    lo = np.array([min(c.center[k] - c.extent(k) for c in colliders) for k in range(3)], float)
    hi = np.array([max(c.center[k] + c.extent(k) for c in colliders) for k in range(3)], float)
    return lo, hi


def _world_aabb(pos, quat_xyzw, lo_local, hi_local) -> tuple[np.ndarray, np.ndarray]:
    R = _quat_to_mat(quat_xyzw)
    corners = np.array([
        [x, y, z]
        for x in (lo_local[0], hi_local[0])
        for y in (lo_local[1], hi_local[1])
        for z in (lo_local[2], hi_local[2])
    ], float)
    w = pos + corners @ R.T
    return w.min(0), w.max(0)


def _mech_states(base, name: str, origin) -> dict:
    spec = _MECHS[name]
    art = base.scene[name]
    np_ = lambda t: getattr(t, "torch", t).detach().cpu().numpy().astype(np.float64)  # noqa: E731
    bpos, bquat = np_(art.data.body_pos_w)[0], np_(art.data.body_quat_w)[0]
    blin = np_(art.data.body_lin_vel_w)[0] if hasattr(art.data, "body_lin_vel_w") else np.zeros((len(bpos), 3))
    bang = np_(art.data.body_ang_vel_w)[0] if hasattr(art.data, "body_ang_vel_w") else np.zeros((len(bpos), 3))
    bodies = list(art.body_names)
    q, qd = np_(art.data.joint_pos)[0], np_(art.data.joint_vel)[0]
    jnames = list(art.joint_names)
    out = {}

    def state(i, lo_local, hi_local):
        p = bpos[i] - origin
        lo_w, hi_w = _world_aabb(p, bquat[i], lo_local, hi_local)
        return ObjectState(
            pos=p, quat_xyzw=bquat[i], lin_vel=blin[i], ang_vel=bang[i], low=lo_w, high=hi_w
        )

    base_lo, base_hi = _local_bounds(spec.colliders, spec.size)
    out[name] = state(bodies.index("base"), base_lo, base_hi)
    for lk in spec.links:
        key = f"{name}/{lk.name}"
        j = jnames.index(f"{lk.name}_joint")
        latched = _LATCHED.get(key) == _EPISODE["count"]
        if lk.latch is not None and not latched and abs(q[j] - lk.latch.at) <= lk.latch.zone:
            _LATCHED[key] = _EPISODE["count"]
            latched = True
        _JOINTS[key] = JointState(pos=float(q[j]), vel=float(qd[j]), latched=latched)
        if os.environ.get("YAMKIT_DEBUG_JOINTS"):
            print(f"yamkit joint {key} q={q[j]:.4f} qd={qd[j]:.4f} latched={latched}", flush=True)
        lk_lo, lk_hi = _local_bounds(lk.colliders, (0.02, 0.02, 0.02))
        out[key] = state(bodies.index(lk.name), lk_lo, lk_hi)
        rest = lk.start if lk.rest is None else lk.rest
        if lk.law is not None and not latched:
            jq = {o.name: float(q[jnames.index(f"{o.name}_joint")]) for o in spec.links}
            out_ = lk.law.drive(key, float(q[j]), float(qd[j]), jq, _law_mem(key, lk))
            target, stiff, effort = out_[:3]
            damp_ = out_[3] if len(out_) > 3 else None
            if getattr(lk.law, "kinematic", False):
                goal = _law_mem(key, lk).get("target", target)
                nxt = float(q[j]) + min(max(goal - float(q[j]), -lk.law.rate), lk.law.rate)
                nxt = min(max(nxt, lk.lower), lk.upper)
                _put_joint(art, j, nxt)
                _set_drive(art, j, target=nxt, stiffness=lk.law.hold, effort=0.0, damping=0.0)
            else:
                _set_drive(art, j, target=target, stiffness=stiff, effort=effort, damping=damp_)
            mem = _law_mem(key, lk)
            want = mem.get("limits") or (lk.lower, lk.upper)
            if mem.get("_limits_set") != want:
                _set_limits(art, j, want[0] - 1e-4, want[1] + 1e-4)
                mem["_limits_set"] = want
        elif latched:
            _set_drive(art, j, target=lk.latch.at, stiffness=lk.latch.hold, effort=0.0)
        else:
            drag = min(lk.drag, 0.8 * lk.mass / 0.05) if lk.kind == "prismatic" else lk.drag
            _set_drive(
                art, j, target=float(q[j]), stiffness=lk.stay,
                effort=-lk.spring * (float(q[j]) - rest) - drag * float(qd[j]),
            )
    return out


# -- tilted grips for demos --------------------------------------------------------------------


@dataclass
class Tilted:
    """A demo waypoint with roll about world x, then pitch about world y (radians), after yaw about vertical."""

    duration: float
    pos: tuple
    finger: float
    label: str = ""
    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0


@dataclass
class Home:
    """A demo waypoint returning the arm to yam_arena's READY pose."""

    duration: float
    finger: float = FINGER_OPEN
    label: str = "home"
    pos: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0


def _rot_xyz(roll: float, pitch: float) -> np.ndarray:
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


def _patch_demo(demo) -> None:
    if getattr(demo, "_yamkit", False):
        return
    demo._yamkit = True
    plain = demo.grip_rotation

    def target(self, t: float):
        prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
        for w in self.waypoints:
            if isinstance(w, Home):
                goal, rot = demo.ik.tcp_pose(np.asarray(demo.READY, dtype=float), self.base_pos)
                goal = np.asarray(goal, dtype=float)
            else:
                goal, rot = np.asarray(w.pos, dtype=float), plain(w.yaw)
            if isinstance(w, Tilted) and (w.roll or w.pitch):
                rot = _rot_xyz(w.roll, w.pitch) @ rot
            if t < w.duration:
                s = demo.ik.smoothstep(t / w.duration)
                return (
                    prev_pos + s * (goal - prev_pos),
                    demo.ik.slerp_matrix(prev_rot, rot, s),
                    prev_finger + s * (w.finger - prev_finger),
                    w.label,
                )
            t -= w.duration
            prev_pos, prev_rot, prev_finger = goal, rot, w.finger
        return prev_pos, prev_rot, prev_finger, "done"

    demo.ArmPlan.target = target


# -- hooks into yam_arena.scene --------------------------------------------------------------


def _patch(scene) -> None:
    if getattr(scene, "_yamkit", False):
        return
    scene._yamkit = True
    original_task_object, original_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Mechanism):
            _MECHS[spec.name] = spec
            return _mech_object(scene, spec)
        if not isinstance(spec, Prop):
            return original_task_object(spec)
        _PROPS.add(spec.name)
        yaw = (0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))
        has_custom = any(c.friction is not None for c in spec.colliders)
        addon = {"physics_material": None if has_custom else scene._grippy()}
        if spec.mass is None:
            addon.update(
                func=scene._spawn_fixture,
                rigid_props=scene.PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
            )
        else:
            addon.update(collision_props=scene._contact_props())
        return scene.Object(
            name=spec.name,
            usd_path=_write(spec),
            object_type=scene.ObjectType.RIGID,
            initial_pose=scene.Pose(position_xyz=tuple(spec.pos), rotation_xyzw=_quat_mul(yaw, spec.tilt_xyzw)),
            spawn_cfg_addon=addon,
        )

    def object_states(env, names):
        out = original_states(env, [n for n in names if n not in GRIPPERS and n not in _MECHS])
        base = env.unwrapped
        buf = getattr(base, "episode_length_buf", None)
        if buf is not None:
            n = int(buf[0].item())
            if _EPISODE["last"] is not None and n < _EPISODE["last"]:
                _EPISODE["count"] += 1
            _EPISODE["last"] = n
        origin = base.scene.env_origins[0].detach().cpu().numpy()
        _ORIGIN[:] = origin
        if _MECHS:
            _watch_resets(base)
        for name in [n for n in names if n in _MECHS]:
            out.update(_mech_states(base, name, origin))
        for name in _PROPS & set(out):
            p = base.arena_world.get_pose_w(name)[0, :3].detach().cpu().numpy().astype(np.float64) - origin
            s = out[name]
            out[name] = ObjectState(
                pos=p, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel, low=s.low, high=s.high
            )
        for key, robot in GRIPPERS.items():
            try:
                art = base.scene[robot]
            except KeyError:
                continue
            i = art.body_names.index("gripper")
            pos = art.data.body_pos_w.torch[0, i].detach().cpu().numpy().astype(np.float64) - origin
            x, y, z, w = art.data.body_quat_w.torch[0, i].detach().cpu().numpy().astype(np.float64)
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            tcp = pos + rot @ np.asarray(TCP_OFFSET)
            tips = tcp + rot @ np.array([0.0, 0.0, PAD_TIP_PAST_TCP])
            j = art.joint_names.index("joint7")
            _OPENING[key] = float(abs(art.data.joint_pos.torch[0, j].item()) / abs(FINGER_OPEN))
            out[key] = ObjectState(
                pos=tcp, quat_xyzw=np.array([x, y, z, w]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                low=np.minimum(tcp, tips), high=np.maximum(tcp, tips),
            )
        return out

    scene.task_object, scene.object_states = task_object, object_states


class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Patches yam_arena.scene as soon as it is imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        self._inner = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module)


def episode() -> int:
    """Which trial this is: 0 for the first, +1 each time the simulator resets the scene."""
    return _EPISODE["count"]


def opening(objects, side: str) -> float:
    """How far a gripper's fingers are open, 0 (closed) to 1 (fully open)."""
    return _OPENING.get(side, float("nan"))


if "yam_arena.scene" in sys.modules:
    _patch(sys.modules["yam_arena.scene"])
elif not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())

import yam_arena.demo as _yam_demo  # noqa: E402

_patch_demo(_yam_demo)
