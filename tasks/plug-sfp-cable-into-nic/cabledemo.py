"""cabledemo: closed-loop scripted demos for yam_arena task files (privileged state, IK, both arms at once).

A demo script is a function ``script(ctx)`` that drives the arms with motion primitives and can read the live
object states (``ctx.states()``), so grasps follow wherever the cable actually is. Each arm runs its own
generator; ``ctx.both(a, b)`` runs two in parallel.

    python cabledemo.py TASK_FILE OUT_DIR [--max-steps 2400]

TASK_FILE must define ``demo_script(ctx)``. Writes OUT_DIR/{top_cam,left_cam,right_cam}.mp4 (H.264, 20 fps),
OUT_DIR/setup.png (first top-camera frame) and OUT_DIR/result.json (task_score, steps, success).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


_TILT = np.zeros(3)  # the tilt of the arm that is moving (set by Arm.move before each plan step)


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF) from wherever its last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena.constants import FINGER_OPEN, READY

        self.name = name
        self.base = np.asarray(base_pos, dtype=float)
        self.q = np.array(READY, dtype=float)
        self.finger = FINGER_OPEN
        self.cmd = np.r_[self.q, 1.0]
        self._yaw = 0.0
        self.tilt = np.zeros(3)  # rotation vector (world frame, rad) applied on top of this arm's straight-down grip

    def tcp(self) -> np.ndarray:
        from yam_arena import ik

        return ik.tcp_pose(self.q, self.base)[0]

    def move(self, pos, *, yaw: float = 0.0, finger: float | None = None, duration: float = 1.0, label: str = ""):
        """Generator: go to ``pos`` (world m, TCP) with a straight-down grip turned ``yaw``; ``finger`` in m
        (FINGER_OPEN..FINGER_CLOSED; None keeps it). One yield per control step."""
        global _TILT
        from yam_arena.demo import ArmPlan, Waypoint

        _TILT = self.tilt
        f = self.finger if finger is None else finger
        plan = ArmPlan([Waypoint(duration, tuple(map(float, pos)), f, label, yaw)], self.base, q_start=self.q,
                       finger_start=self.finger)
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            _TILT = self.tilt
            self.cmd, _ = plan.action7(k / CONTROL_HZ)
            self.q = plan.q.copy()
            yield
        self.finger = f
        self._yaw = yaw

    def grip(self, finger: float, duration: float = 0.6):
        """Generator: open/close in place."""
        yield from self.move(self.tcp(), yaw=self._yaw, finger=finger, duration=duration, label="grip")

    def wait(self, duration: float):
        for _ in range(max(1, round(duration * CONTROL_HZ))):
            yield


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        from yam_arena import demo as _demo

        self.emb = emb
        orig = _demo.grip_rotation

        def tilted(yaw: float = 0.0):
            th = float(np.linalg.norm(_TILT))
            if th < 1e-9:
                return orig(yaw)
            k = _TILT / th
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            return (np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx) @ orig(yaw)

        _demo.grip_rotation = tilted
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.t = 0.0

    @property
    def tilt(self):
        """The left arm's tilt (the demo scripts the left arm's hand tilt through this)."""
        return self.left.tilt

    @tilt.setter
    def tilt(self, v):
        self.left.tilt = np.asarray(v, float)

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    @staticmethod
    def both(a, b):
        """Run two generators in lockstep until both end."""
        a, b = iter(a), iter(b)
        live = [a, b]
        while live:
            for g in list(live):
                try:
                    next(g)
                except StopIteration:
                    live.remove(g)
            if live:
                yield

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


class _Mp4:
    """Streams frames to ffmpeg as they arrive (no frames held in memory)."""

    def __init__(self, path: Path, w: int, h: int):
        ff = os.environ.get("FFMPEG", "ffmpeg")
        self.proc = subprocess.Popen(
            [ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(CONTROL_HZ),
             "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def add(self, f: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(f[..., :3], dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """World-convention camera orientation (x forward, y left, z up) as xyzw, looking from pos at target."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross(up, f)
    left /= np.linalg.norm(left)
    u = np.cross(f, left)
    m = np.stack([f, left, u], axis=1)
    t = np.trace(m)
    if t > 0:
        k = np.sqrt(t + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k, 0.25 * k)
    i = int(np.argmax(np.diag(m)))
    j, l = (i + 1) % 3, (i + 2) % 3
    k = np.sqrt(1.0 + m[i, i] - m[j, j] - m[l, l]) * 2
    q = [0.0, 0.0, 0.0, (m[l, j] - m[j, l]) / k]
    q[i], q[j], q[l] = 0.25 * k, (m[j, i] + m[i, j]) / k, (m[l, i] + m[i, l]) / k
    return tuple(float(v) for v in q)


def _hook_embodiment(cams: dict, width: int, height: int, cam_w: int, cam_h: int) -> None:
    """When yam_arena.embodiment is imported (after Isaac Sim has started), render its cameras at width x height instead of
    the default 640x360 and add the task's RECORD_CAMS ({name: (pos, target, focal_mm)}) as extra cameras."""
    import importlib.abc
    import importlib.util

    def patch(emod):
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        cls = emod.BimanualYamEmbodiment
        orig_init, orig_make = cls.__init__, emod.make_configclass

        def make(name, fields, bases=()):
            for cam, (pos, target, focal) in (cams.items() if name == "YamCameraCfg" else ()):
                fields.append((cam, CameraCfg, CameraCfg(
                    prim_path="{ENV_REGEX_NS}/" + "".join(p.title() for p in cam.split("_")), height=cam_h, width=cam_w,
                    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.02, 20.0)),
                    offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in pos), rot=_look_at(pos, target), convention="world"),
                    renderer_cfg=emod.RENDERER)))
            return orig_make(name, fields, bases=bases)

        def init(self, *a, **kw):
            kw.setdefault("cam_height", height)
            kw.setdefault("cam_width", width)
            emod.make_configclass = make
            try:
                orig_init(self, *a, **kw)
            finally:
                emod.make_configclass = orig_make

        cls.__init__ = init

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "yam_arena.embodiment":
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            loader = spec.loader
            exec_module = loader.exec_module

            def run_and_patch(module):
                exec_module(module)
                patch(module)

            loader.exec_module = run_and_patch
            return spec

    sys.meta_path.insert(0, Finder())


def run(task_path: str, out_dir: str, max_steps: int = 2400, cam_width: int = 320, cam_height: int = 176,
        extra_size: tuple = (1280, 720),
        hold_s: float = 2.0, record: bool = True, device: str = "cuda:0") -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if record:
        from yam_arena import task_file

        extra = dict(getattr(task_file.load(task_path).module, "RECORD_CAMS", {}))
        _hook_embodiment(extra, cam_width, cam_height, *extra_size)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height, device=device)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    ctx = Ctx(emb)
    script = emb.task.module.demo_script(ctx)
    writers: dict[str, _Mp4] = {}
    info: dict = {}
    steps, ended, success = 0, None, False

    def snap(o):
        if record:
            for cam, img in o.images.items():
                img = np.asarray(img)
                if cam not in writers:
                    writers[cam] = _Mp4(out / f"{cam}.mp4", img.shape[1], img.shape[0])
                    if cam == "top_cam":
                        import imageio

                        imageio.imwrite(out / "setup.png", img[..., :3])
                writers[cam].add(img)
            for cam in extra:
                out_ = emb._base.scene[cam].data.output["rgb"]
                out_ = out_.torch if hasattr(out_, "torch") else out_
                img = np.ascontiguousarray(out_[0, ..., :3].cpu().numpy(), dtype=np.uint8)
                if cam not in writers:
                    writers[cam] = _Mp4(out / f"{cam}.mp4", img.shape[1], img.shape[0])
                writers[cam].add(img)

    snap(obs)
    while steps < max_steps:
        try:
            next(script)
        except StopIteration:
            ended = ended or steps
        res = emb.step(ctx.action())
        steps += 1
        ctx.t += 1.0 / CONTROL_HZ
        info = res.info
        snap(res.observation)
        if steps % 100 == 0:
            import time
            now = time.time()
            rate = 100 / max(now - getattr(run, "_t", now - 1e9), 1e-6)
            run._t = now
            print(f"step {steps} score {info.get('score')} {rate:.1f} steps/s", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "final_info": {k: v for k, v in info.items()
                                                                                       if isinstance(v, int | float | bool)}}
    for w in writers.values():
        w.close()
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("RESULT", json.dumps(result))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extra-size", default="1280x720", help="frame size WxH of the task's RECORD_CAMS (the 8 GB GPU renders about 3.8 megapixels per frame in all)")
    ap.add_argument("--size", default="1280x720", help="camera frame size WxH for recorded runs")
    a = ap.parse_args()
    small = a.no_record  # debug runs: tiny camera frames, much faster rendering
    run(a.task, a.out, a.max_steps, extra_size=tuple(int(v) for v in a.extra_size.split("x")), record=not a.no_record, device=a.device,
        cam_width=64 if small else int(a.size.split("x")[0]), cam_height=36 if small else int(a.size.split("x")[1]))
