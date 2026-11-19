"""mechkit: joint behaviors ("laws") for yamkit Mechanism links."""

from __future__ import annotations

from dataclasses import dataclass

import yamkit as kit


@dataclass(frozen=True)
class Free:
    """No behavior beyond viscous drag."""

    drag: float = 0.02

    def drive(self, key, q, qd, J, mem):
        return q, 0.0, 0.0, self.drag


@dataclass(frozen=True)
class Spring:
    """A plain return spring with damping, pulling to `rest`."""

    rest: float
    k: float = 0.5
    damp: float = 0.02

    def drive(self, key, q, qd, J, mem):
        return self.rest, self.k, 0.0, self.damp


@dataclass(frozen=True)
class SnapCatch:
    """Sliding front lock catch on a clamshell punnet.

    Slides with light drag until reaching `lock_zone` around `0.0`. If the lid is closed
    (`abs(J[lid_link]) <= closed_rad`) and the catch did not enter center prematurely while the
    lid was still open, it snaps smoothly to `0.0` and reports `locked = True` within `latch_zone`.
    If slid into `jam_zone` while the lid is open, `jammed_under_lid` stays True until withdrawn
    outside `jam_zone`, preventing fake latching.
    """

    lid_link: str = "lid"
    closed_rad: float = 0.070  # ~4.0 deg
    lock_zone: float = 0.008
    latch_zone: float = 0.004
    jam_zone: float = 0.013
    snap_k: float = 16.0
    damp: float = 0.45
    slide_drag: float = 0.30

    def drive(self, key, q, qd, J, mem):
        lid_q = float(J[self.lid_link])
        was_locked = bool(mem.get("locked", False))
        if abs(q) > self.jam_zone:
            mem["jammed_under_lid"] = False
            mem["locked"] = False
            return q, 0.0, 0.0, self.slide_drag

        if not was_locked and lid_q > self.closed_rad:
            mem["jammed_under_lid"] = True

        if mem.get("jammed_under_lid", False):
            mem["locked"] = False
            return q, 0.0, 0.0, self.slide_drag

        if abs(q) <= self.lock_zone and (lid_q <= self.closed_rad or was_locked):
            mem["locked"] = abs(q) <= self.latch_zone
            mem["ever_locked"] = mem.get("ever_locked", False) or mem["locked"]
            return 0.0, self.snap_k, 0.0, self.damp

        mem["locked"] = False
        return q, 0.0, 0.0, self.slide_drag


@dataclass(frozen=True)
class SprungLid:
    """Rear-hinged spring-open punnet lid coupled to a front `SnapCatch` link.

    Normally a return spring (`rest`, `k`, `damp`) pulls the lid open across `[lower, upper]`.
    - When `catch` is validly locked (`locked == True`), the catch's retaining hook holds the lid
      closed (`limits = (0.0, latched_upper_rad)` and strong `hold_k` toward `0.0`).
    - When `catch` sits in `jam_zone` prematurely while the lid is still open (`jammed_under_lid`),
      the catch hook sits beneath the lid flange and physically blocks closing below `jam_rad`.
    """

    rest: float
    k: float = 0.55
    damp: float = 0.018
    catch_link: str = "catch"
    lower_rad: float = 0.0
    upper_rad: float = 0.49  # ~28 deg
    latched_upper_rad: float = 0.052  # ~3.0 deg
    jam_rad: float = 0.113  # ~6.5 deg
    hold_k: float = 10.0
    hold_damp: float = 0.20
    sets_limits: bool = True

    def drive(self, key, q, qd, J, mem):
        mech_name = key.split("/", 1)[0]
        catch_st = kit.law_state(f"{mech_name}/{self.catch_link}")
        catch_q = float(J[self.catch_link])

        if catch_st.get("locked", False):
            mem["limits"] = (self.lower_rad, self.latched_upper_rad)
            mem["latched"] = True
            mem["blocked"] = False
            return 0.0, self.hold_k, 0.0, self.hold_damp

        if catch_st.get("jammed_under_lid", False) and abs(catch_q) <= 0.013:
            mem["limits"] = (self.jam_rad, self.upper_rad)
            mem["latched"] = False
            mem["blocked"] = True
            return self.rest, self.k, 0.0, self.damp

        mem["limits"] = (self.lower_rad, self.upper_rad)
        mem["latched"] = False
        mem["blocked"] = False
        return self.rest, self.k, 0.0, self.damp


def state(key: str) -> dict:
    """The law's memory for link `key` ("mechanism/link") in this trial."""
    return kit.law_state(key)


def locked(objects, key: str) -> bool:
    """True when the snap catch is seated locked over the lid tongue."""
    return bool(state(key).get("locked", False))
