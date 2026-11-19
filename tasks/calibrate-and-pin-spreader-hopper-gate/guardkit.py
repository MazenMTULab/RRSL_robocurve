"""guardkit: gripper contact, steam-cone intrusion, and per-trial Latch checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import yamkit as kit

GRIPPERS = ("left_gripper", "right_gripper")
CONTACT = 0.002  # m: touching within 2 mm


@dataclass(frozen=True)
class Seg:
    """A capsule from `a` to `b` (object frame, m) with radius `r`."""

    a: tuple
    b: tuple
    r: float = 0.0


@dataclass(frozen=True)
class Ring:
    """A solid ring/cylinder about the object's z axis (object frame, m)."""

    r_in: float
    r_out: float
    z0: float
    z1: float
    center: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Block:
    """A box centered at `center` with full `size` (object frame, m)."""

    center: tuple
    size: tuple


@dataclass(frozen=True)
class Cone:
    """A vertical truncated cone (frustum) along object +z: from `z0` (radius `r0`) to `z1` (radius `r1`),
    centered horizontally on `(center[0], center[1])` (object frame, m)."""

    r0: float
    r1: float
    z0: float
    z1: float
    center: tuple = (0.0, 0.0, 0.0)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _proj_cone_rz(rho: float, dz: float, r0: float, r1: float, h: float) -> tuple[float, float]:
    """Exact closest point in (rho >= 0, dz) to convex frustum {0 <= z <= h, 0 <= r <= r0 + (r1-r0)*z/h}."""
    if dz <= 0.0 and rho <= r0:
        return rho, 0.0
    if dz >= h and rho <= r1:
        return rho, h
    r_at = r0 + (r1 - r0) * (dz / h if h > 0 else 0.0)
    if 0.0 <= dz <= h and rho <= r_at:
        return rho, dz
    # Outside slant or beyond caps: check slant segment (r0, 0) -> (r1, h) and caps.
    dr = r1 - r0
    L2 = dr * dr + h * h
    t = 0.0 if L2 < 1e-18 else max(0.0, min(1.0, ((rho - r0) * dr + dz * h) / L2))
    rs, zs = r0 + t * dr, t * h
    d_slant = (rho - rs) ** 2 + (dz - zs) ** 2
    best = (rs, zs, d_slant)
    if dz < 0.0:
        rb = min(rho, r0)
        d_bot = (rho - rb) ** 2 + dz * dz
        if d_bot < best[2]:
            best = (rb, 0.0, d_bot)
    if dz > h:
        rt = min(rho, r1)
        d_top = (rho - rt) ** 2 + (dz - h) ** 2
        if d_top < best[2]:
            best = (rt, h, d_top)
    return best[0], best[1]


def _proj_local(part, q: np.ndarray) -> np.ndarray:
    """Nearest points on `part` to object-frame points q (n x 3)."""
    if isinstance(part, Block):
        c, h = np.asarray(part.center, float), np.asarray(part.size, float) / 2
        return c + np.clip(q - c, -h, h)
    if isinstance(part, Ring):
        c = np.asarray(part.center, float)
        d = q - c
        rho = np.hypot(d[:, 0], d[:, 1])
        safe = np.where(rho > 1e-12, rho, 1.0)
        ux, uy = np.where(rho > 1e-12, d[:, 0] / safe, 1.0), np.where(rho > 1e-12, d[:, 1] / safe, 0.0)
        r = np.clip(rho, part.r_in, part.r_out)
        return c + np.stack([r * ux, r * uy, np.clip(d[:, 2], part.z0, part.z1)], axis=1)
    if isinstance(part, Cone):
        c = np.asarray(part.center, float)
        d = q - c
        h = part.z1 - part.z0
        out = np.empty_like(q)
        for idx in range(len(q)):
            dx, dy, dz = float(d[idx, 0]), float(d[idx, 1]), float(d[idx, 2] - part.z0)
            rho = math.hypot(dx, dy)
            r_p, z_p = _proj_cone_rz(rho, dz, part.r0, part.r1, h)
            if rho > 1e-12:
                ux, uy = dx / rho, dy / rho
            else:
                ux, uy = 1.0, 0.0
            out[idx] = (c[0] + r_p * ux, c[1] + r_p * uy, c[2] + part.z0 + z_p)
        return out
    a, b = np.asarray(part.a, float), np.asarray(part.b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(q)) if L2 < 1e-18 else np.clip((q - a) @ ab / L2, 0.0, 1.0)
    x = a + t[:, None] * ab
    v = q - x
    n = np.linalg.norm(v, axis=1)
    return x + v * np.minimum(1.0, part.r / np.maximum(n, 1e-12))[:, None]


def _dist_local_any(part, p: np.ndarray) -> float:
    if isinstance(part, (Ring, Cone)):
        return float(np.linalg.norm(p - _proj_local(part, p[None, :])[0]))
    if isinstance(part, Block):
        d = np.abs(p - np.asarray(part.center, float)) - np.asarray(part.size, float) / 2
        return float(np.linalg.norm(np.maximum(d, 0.0)))
    a, b = np.asarray(part.a, float), np.asarray(part.b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-18 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return max(0.0, float(np.linalg.norm(p - (a + t * ab))) - part.r)


PALM = Block((0.0, 0.0, -0.1075), (0.102, 0.065, 0.074))


def hand_boxes(o: float) -> list:
    out = [PALM]
    for s in (-1.0, 1.0):
        out += [
            Block((s * (o + 0.012), 0.0, -0.051), (0.024, 0.068, 0.060)),
            Block((s * (o + 0.005), 0.0, -0.012), (0.010, 0.040, 0.018)),
            Block((s * (o + 0.0032), 0.0, 0.0049), (0.0064, 0.0264, 0.0158)),
        ]
    return out


def _box_gap(cg, Rg, box, co, Ro, part, iters: int = 40) -> float:
    c, h = np.asarray(box.center, float), np.asarray(box.size, float) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] + [[0, 0, 0]], float)
    p = cg + (c + corners * h) @ Rg.T
    q = p
    for _ in range(iters):
        q = co + _proj_local(part, (p - co) @ Ro) @ Ro.T
        p_new = cg + (c + np.clip((q - cg) @ Rg - c, -h, h)) @ Rg.T
        if np.max(np.abs(p_new - p)) < 1e-6:
            p = p_new
            break
        p = p_new
    return float(np.min(np.linalg.norm(p - q, axis=1)))


def _hand_frames(objects, g: str, obj: str):
    from yam_arena.constants import FINGER_OPEN

    op = kit.opening(objects, g)
    o = abs(FINGER_OPEN) * (1.0 if not np.isfinite(op) else float(np.clip(op, 0.0, 1.0)))
    s, so = objects[g], objects[obj]
    return o, np.asarray(s.pos, float), _rot(s.quat_xyzw), np.asarray(so.pos, float), _rot(so.quat_xyzw)


def touching(objects, obj: str, parts, tol: float = CONTACT, grippers=GRIPPERS) -> tuple:
    """Grippers whose palm or fingers touch or overlap any part of `obj` within `tol`."""
    out = []
    for g in grippers:
        if g not in objects:
            continue
        o, cg, Rg, co, Ro = _hand_frames(objects, g, obj)
        hit = False
        for part in (parts if isinstance(parts, (list, tuple)) else (parts,)):
            for box in hand_boxes(o):
                ctr = cg + Rg @ np.asarray(box.center, float)
                reach = float(np.linalg.norm(np.asarray(box.size, float))) / 2
                if _dist_local_any(part, Ro.T @ (ctr - co)) - reach > tol:
                    continue
                if _box_gap(cg, Rg, box, co, Ro, part) <= tol:
                    hit = True
                    break
            if hit:
                break
        if hit:
            out.append(g)
    return tuple(out)


class Latch:
    """Irreversible per-trial latch triggering once `cond` holds for `steps` consecutive control steps."""

    def __init__(self, steps: int = 2):
        self.steps = steps
        self._ep, self._key, self._objs = None, None, None
        self.run, self.on = 0, False

    def feed(self, objects, cond: bool) -> bool:
        ep, n = kit.episode(), kit._EPISODE.get("last")
        if ep != self._ep:
            self._ep, self._key, self._objs = ep, None, None
            self.run, self.on = 0, False
        key = (ep, n) if n is not None else None
        if (key is not None and key == self._key) or (key is None and objects is self._objs):
            return self.on
        self._key, self._objs = key, objects
        self.run = self.run + 1 if cond else 0
        if not self.on and self.run >= self.steps:
            self.on = True
        return self.on
