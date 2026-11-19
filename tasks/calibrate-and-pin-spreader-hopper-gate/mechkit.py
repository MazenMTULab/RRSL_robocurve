"""mechkit: joint behaviors ("laws") for yamkit Mechanism links."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import yamkit as kit


@dataclass(frozen=True)
class Free:
    """No behavior beyond friction-like `stay` drag."""

    drag: float = 0.02

    def drive(self, key, q, qd, J, mem):
        return q, 0.0, 0.0, self.drag


@dataclass(frozen=True)
class OverCenter:
    """A bistable toggle: past `center` (toward upper) a torque/force `push` drives it to its upper stop,
    below it to the lower stop. `locked` is set while q >= `lock_at` (with sign=1)."""

    center: float
    lock_at: float
    push: float = 0.15
    dead: float = 0.03
    damp: float = 0.02
    sign: int = 1

    def drive(self, key, q, qd, J, mem):
        d = self.sign * (q - self.center)
        f = 0.0 if abs(d) < self.dead else math.copysign(self.push, d) * self.sign
        mem["locked"] = self.sign * (q - self.lock_at) >= 0
        mem["ever_locked"] = mem.get("ever_locked", False) or mem["locked"]
        return q, 0.0, f, self.damp


@dataclass(frozen=True)
class Detents:
    """A knob, selector or rocker with click positions."""

    positions: tuple
    labels: tuple = ()
    zone: float = 0.12
    pull: float = 0.6
    damp: float = 0.03
    drag: float = 0.05

    def drive(self, key, q, qd, J, mem):
        i = min(range(len(self.positions)), key=lambda k: abs(q - self.positions[k]))
        p = self.positions[i]
        if abs(q - p) <= self.zone:
            mem["setting"] = i
            mem["label"] = self.labels[i] if self.labels else i
            return p, self.pull, 0.0, self.damp
        mem["setting"] = None
        mem["label"] = None
        return q, 0.0, 0.0, self.drag


@dataclass(frozen=True)
class Follow:
    """A part driven by another joint or callable `fn(J, mem)`, clamped to [lo, hi]."""

    source: str | None = None
    ratio: float = 1.0
    offset: float = 0.0
    fn: object = None
    lo: float = -1e9
    hi: float = 1e9
    k: float = 0.5
    damp: float = 0.05
    kinematic: bool = True
    rate: float = 0.25
    hold: float = 0.01

    def drive(self, key, q, qd, J, mem):
        t = self.fn(J, mem) if self.fn is not None else self.ratio * J[self.source] + self.offset
        t = min(max(t, self.lo), self.hi)
        mem["target"] = t
        return t, self.k, 0.0, self.damp


@dataclass(frozen=True)
class Interlock:
    """A joint held at where it was when the lock engaged while `blocked(J)` is true."""

    blocked: object
    inner: object = field(default_factory=Free)
    hold: float = 400.0
    force_tol: float = 0.02

    def drive(self, key, q, qd, J, mem):
        if self.blocked(J):
            at = mem.setdefault("held_at", q)
            if abs(q - at) > self.force_tol:
                mem["forced"] = mem.get("forced", 0) + 1
            mem["blocked"] = True
            return at, self.hold, 0.0, 1.0
        mem.pop("held_at", None)
        mem["blocked"] = False
        return self.inner.drive(key, q, qd, J, mem)


@dataclass(frozen=True)
class Spring:
    """A plain return spring with damping, pulling to `rest`."""

    rest: float
    k: float = 0.3
    damp: float = 0.02

    def drive(self, key, q, qd, J, mem):
        return self.rest, self.k, 0.0, self.damp


# -- scorer readouts -------------------------------------------------------------------------------


def state(key: str) -> dict:
    """The law's memory for link `key` ("mechanism/link") in this trial."""
    return kit.law_state(key)


def locked(objects, key: str) -> bool:
    """An OverCenter link seated over center right now."""
    return bool(state(key).get("locked", False))


def setting(objects, key: str):
    """A Detents link's current detent label (or index), None between detents."""
    return state(key).get("label")


def forced(objects, key: str) -> int:
    """Steps on which an Interlock was pushed against while blocked."""
    return int(state(key).get("forced", 0))
