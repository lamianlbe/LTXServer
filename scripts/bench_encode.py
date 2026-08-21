#!/usr/bin/env python3
"""Standalone encode benchmark — isolates video encoding from generation.

Self-contained ON PURPOSE: copy this single file to any machine (the
FastVideo pod, the LTXServer pod) and run it there; it carries its own
copies of the encoder functions (line-identical to ltxserver/encoder.py
and FastVideo's ltx23_engine.py — verified, that's the point: the code is
the same, so any speed difference between machines is environment).

Synthesizes DETERMINISTIC mid-complexity frames (seeded texture + motion),
so runs on different machines encode the exact same content. Defaults
mirror the production config (libx265 veryfast 4000kbps, LQ ultrafast
1000kbps blur 10). Phases timed separately:

  gpu_yuv     GPU RGB->YUV420 conversion alone (the swscale bypass)
  lq_frames   GPU half-res + gaussian blur alone
  hq_encode   full production HQ encode (conversion + ffmpeg)
  lq_encode   full production LQ encode
  parallel    HQ + LQ in a 2-thread pool — THE production wall time

    python scripts/bench_encode.py                       # production defaults
    python scripts/bench_encode.py --extra-video-args "" # x265 default asm
    python scripts/bench_encode.py --codec libx264       # codec comparison
    python scripts/bench_encode.py --verbose             # dump x265 banner
                                                         # (pools/asm info)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# encoder functions — line-identical to ltxserver/encoder.py (and FastVideo
# ltx23_engine.py); inlined so this file runs standalone on any machine.
# --------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as err:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: install a system ffmpeg or "
                           "`pip install imageio-ffmpeg`") from err


def _rgb_frames_to_yuv420p_bytes(frames: list[Any], device: str) -> bytes:
    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.stack(frames)).to(device).float()  # N,H,W,3
    n = x.shape[0]
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = (16 + (0.2126 * r + 0.7152 * g + 0.0722 * b) * (219 / 255)).clamp(0, 255)
    u = 128 + (-0.1146 * r - 0.3854 * g + 0.5 * b) * (224 / 255)
    v = 128 + (0.5 * r - 0.4542 * g - 0.0458 * b) * (224 / 255)
    u = F.avg_pool2d(u.unsqueeze(1), 2).squeeze(1).clamp(0, 255)
    v = F.avg_pool2d(v.unsqueeze(1), 2).squeeze(1).clamp(0, 255)
    buf = torch.cat([
        y.to(torch.uint8).reshape(n, -1),
        u.to(torch.uint8).reshape(n, -1),
        v.to(torch.uint8).reshape(n, -1),
    ], dim=1).reshape(-1)
    return buf.cpu().numpy().tobytes()


def encode_video_h264(
    frames: list[Any],
    fps: int,
    output_path: str | Path,
    *,
    bitrate_kbps: int = 3000,
    preset: str = "medium",
    profile: str = "main",
    audio: Any = None,
    audio_sample_rate: int | None = None,
    audio_bitrate_kbps: int | None = None,
    audio_mono: bool = False,
    threads: int = 0,
    gpu_yuv: bool = True,
    codec: str = "libx264",
    extra_video_args: str = "",
    loglevel: str = "error",
    stderr_sink: list | None = None,
) -> float:
    """Production encode path. ``loglevel``/``stderr_sink`` are bench-only
    additions (dump the x265 banner: thread pools, asm capabilities)."""
    import numpy as np

    if not frames:
        raise ValueError("no frames to encode")
    t0 = time.perf_counter()
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    ffmpeg = _resolve_ffmpeg()

    use_gpu = False
    if gpu_yuv:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            use_gpu = False

    if use_gpu:
        video_bytes = _rgb_frames_to_yuv420p_bytes(frames, "cuda")
        in_pix = "yuv420p"
        color_args = ["-colorspace", "bt709", "-color_primaries", "bt709",
                      "-color_trc", "bt709", "-color_range", "tv"]
    else:
        video_bytes = np.ascontiguousarray(np.stack(frames)).tobytes()
        in_pix = "rgb24"
        color_args = []

    cmd = [ffmpeg, "-y", "-loglevel", loglevel,
           "-f", "rawvideo", "-pix_fmt", in_pix, "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "pipe:0"]
    cmd += ["-c:v", codec, "-preset", preset]
    if "264" in codec:
        cmd += ["-profile:v", profile]
    cmd += ["-pix_fmt", "yuv420p",
            "-b:v", f"{int(bitrate_kbps)}k", "-maxrate", f"{int(bitrate_kbps) * 2}k",
            "-bufsize", f"{int(bitrate_kbps) * 4}k", "-threads", str(max(0, int(threads)))]
    cmd += color_args
    if extra_video_args.strip():
        import shlex
        cmd += shlex.split(extra_video_args)
    cmd += ["-movflags", "+faststart", str(output_path)]

    proc = subprocess.run(cmd, input=video_bytes, capture_output=True)
    if stderr_sink is not None:
        stderr_sink.append(proc.stderr.decode(errors="replace"))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed (rc={proc.returncode}): "
                           f"{proc.stderr.decode(errors='replace')[-800:]}")
    return time.perf_counter() - t0


def make_lq_frames(
    frames: list[Any],
    blur_radius: float,
    device: str | None = None,
    chunk_size: int = 16,
) -> list[Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not frames:
        raise ValueError("no frames")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sigma = float(blur_radius)
    kernel_x = kernel_y = None
    pad = 0
    if sigma > 0:
        ksize = 2 * int(math.ceil(3.0 * sigma)) + 1
        pad = ksize // 2
        coords = torch.arange(ksize, dtype=torch.float32, device=dev) - pad
        gauss = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        gauss = gauss / gauss.sum()
        kernel_x = gauss.view(1, 1, 1, ksize).repeat(3, 1, 1, 1)
        kernel_y = gauss.view(1, 1, ksize, 1).repeat(3, 1, 1, 1)

    out: list[Any] = []
    for start in range(0, len(frames), chunk_size):
        batch = torch.from_numpy(np.stack(frames[start:start + chunk_size])).to(dev)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = F.interpolate(batch, scale_factor=0.5, mode="area")
        if kernel_x is not None:
            batch = F.conv2d(F.pad(batch, (pad, pad, 0, 0), mode="reflect"), kernel_x, groups=3)
            batch = F.conv2d(F.pad(batch, (0, 0, pad, pad), mode="reflect"), kernel_y, groups=3)
        batch = batch.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        batch = batch.permute(0, 2, 3, 1).cpu().numpy()
        out.extend(list(batch))
    return out


# --------------------------------------------------------------------------
# bench harness
# --------------------------------------------------------------------------

def synth_frames(width: int, height: int, num_frames: int, seed: int = 1234):
    """Deterministic mid-complexity content: fixed texture layer scrolling
    over a moving gradient — spatial detail + motion, same bytes on every
    machine. Complexity sits between 'flat gradient' (too easy for x265)
    and 'white noise' (unrealistically hard)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    # one texture, HxWx3 in [-28, 28]; scrolled per frame => real motion
    texture = rng.integers(-28, 29, size=(height, width, 3), dtype=np.int16)
    yy = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        phase = 2.0 * math.pi * i / max(1, num_frames)
        base = np.stack([
            (xx + yy * math.sin(phase)) % 256.0 * np.ones_like(yy),
            (yy + 40.0 * math.cos(phase)) % 256.0 * np.ones_like(xx),
            ((xx + yy) / 2.0 + i * 3.0) % 256.0,
        ], axis=-1).astype(np.int16)
        shifted = np.roll(texture, shift=(i * 2) % height, axis=0)
        frames[i] = np.clip(base + shifted, 0, 255).astype(np.uint8)
    return frames


def read_cpu_budget() -> str:
    """Cores the process may actually use: sched affinity + cgroup quota.
    Rental pods often cap containers well below the host's core count —
    x265's auto thread pool sizes itself off the HOST cores and then
    oversubscribes the quota."""
    parts = [f"os.cpu_count={os.cpu_count()}"]
    try:
        parts.append(f"sched_affinity={len(os.sched_getaffinity(0))}")
    except AttributeError:  # macOS
        pass
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        parts.append("cgroup_quota=unlimited" if quota == "max"
                      else f"cgroup_quota={int(quota) / int(period):.1f} cores")
    except OSError:
        try:  # cgroup v1
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            parts.append("cgroup_quota=unlimited" if q < 0 else f"cgroup_quota={q / p:.1f} cores")
        except OSError:
            pass
    return "  ".join(parts)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=249)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--codec", default="libx265")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument("--extra-video-args", default="-x265-params asm=avx512 -tag:v hvc1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--lq-preset", default="ultrafast")
    ap.add_argument("--lq-bitrate", type=int, default=1000)
    ap.add_argument("--lq-blur", type=float, default=10.0)
    ap.add_argument("--no-gpu", action="store_true", help="force the CPU rgb24 fallback path")
    ap.add_argument("--repeat", type=int, default=2, help="timed repetitions (first run shown separately)")
    ap.add_argument("--verbose", action="store_true", help="loglevel info + dump x265 banner (pools/asm)")
    args = ap.parse_args()

    gpu_yuv = not args.no_gpu
    loglevel = "info" if args.verbose else "error"

    print(f"cpu     : {cpu_model()}")
    print(f"budget  : {read_cpu_budget()}")
    print(f"loadavg : {os.getloadavg() if hasattr(os, 'getloadavg') else 'n/a'}")
    ffmpeg = _resolve_ffmpeg()
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout
    print(f"ffmpeg  : {ffmpeg}")
    print(f"          {version.splitlines()[0] if version else '?'}")
    have_265 = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    print(f"          libx265: {'yes' if 'libx265' in have_265 else 'MISSING'}  "
          f"libx264: {'yes' if 'libx264' in have_265 else 'MISSING'}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"torch   : {torch.__version__}  cuda={cuda}"
              + (f"  ({torch.cuda.get_device_name(0)})" if cuda else ""))
    except Exception as err:  # noqa: BLE001
        print(f"torch   : unavailable ({err}) — CPU fallback only")
        gpu_yuv = False
    print(f"content : {args.width}x{args.height} x{args.frames} @ {args.fps}fps (seeded synthetic)")
    print(f"hq      : {args.codec} {args.preset} {args.bitrate}k threads={args.threads} "
          f"extra={args.extra_video_args!r} gpu_yuv={gpu_yuv}")
    print(f"lq      : {args.codec} {args.lq_preset} {args.lq_bitrate}k blur={args.lq_blur}")
    print()

    t = time.perf_counter()
    array = synth_frames(args.width, args.height, args.frames)
    frames = list(array)  # production passes a LIST of HxWx3 frames (recipe.py _package)
    print(f"[synth] {time.perf_counter() - t:.2f}s  ({array.nbytes / 2**20:.0f} MiB)")

    workdir = Path(tempfile.mkdtemp(prefix="ltxs_bench_"))
    try:
        if gpu_yuv:
            t = time.perf_counter()
            n = len(_rgb_frames_to_yuv420p_bytes(frames, "cuda"))
            print(f"[gpu_yuv] first {time.perf_counter() - t:.2f}s  ({n / 2**20:.0f} MiB)", flush=True)
            t = time.perf_counter()
            _rgb_frames_to_yuv420p_bytes(frames, "cuda")
            print(f"[gpu_yuv] warm  {time.perf_counter() - t:.2f}s", flush=True)

            t = time.perf_counter()
            lq_frames = make_lq_frames(frames, args.lq_blur)
            print(f"[lq_frames] {time.perf_counter() - t:.2f}s", flush=True)
        else:
            lq_frames = list(array[:, ::2, ::2])  # crude half-res for the CPU-only case

        def hq(path: Path, sink=None) -> float:
            return encode_video_h264(
                frames, args.fps, path, bitrate_kbps=args.bitrate, preset=args.preset,
                threads=args.threads, gpu_yuv=gpu_yuv, codec=args.codec,
                extra_video_args=args.extra_video_args, loglevel=loglevel, stderr_sink=sink)

        def lq(path: Path) -> float:
            return encode_video_h264(
                lq_frames, args.fps, path, bitrate_kbps=args.lq_bitrate, preset=args.lq_preset,
                profile="baseline", threads=args.threads, gpu_yuv=gpu_yuv, codec=args.codec,
                extra_video_args=args.extra_video_args, loglevel=loglevel)

        for rep in range(max(1, args.repeat)):
            sink: list = [] if args.verbose and rep == 0 else None  # type: ignore[assignment]
            enc = hq(workdir / "hq.mp4", sink)
            size = (workdir / "hq.mp4").stat().st_size
            print(f"[hq_encode] run{rep + 1} {enc:.2f}s  ({args.frames / enc:.0f} fps, "
                  f"{size / 2**20:.1f} MiB)", flush=True)
            if sink:
                banner = [ln for ln in sink[0].splitlines() if "x265" in ln or "pool" in ln.lower()]
                for ln in banner[:12]:
                    print(f"    | {ln.strip()}")

        enc = lq(workdir / "lq.mp4")
        print(f"[lq_encode] {enc:.2f}s  ({len(lq_frames) / enc:.0f} fps)", flush=True)

        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_hq = pool.submit(hq, workdir / "hq2.mp4")
            f_lq = pool.submit(lq, workdir / "lq2.mp4")
            hq_s, lq_s = f_hq.result(), f_lq.result()
        wall = time.perf_counter() - t
        print(f"[parallel] wall {wall:.2f}s  (hq {hq_s:.2f}s + lq {lq_s:.2f}s overlapped) "
              f"<- production number", flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
