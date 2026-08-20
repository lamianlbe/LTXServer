"""Video/audio encoding + S3 delivery.

Ported unchanged from the FastVideo LTX-2.3 server engine (same author,
same deployment targets): GPU RGB->YUV420 conversion piped to one ffmpeg
subprocess, LQ variant generation, and the S3 helpers. Keeping this half
byte-compatible means the two backends serve identical mp4s for identical
frames.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _audio_to_int16(audio: Any) -> tuple[Any, int]:
    """[samples] / [samples, ch] / [ch, samples] float in ~[-1, 1] ->
    (int16 [samples, ch], num_channels)."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2:
        if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
            audio_np = audio_np.T
    else:
        raise ValueError(f"Unexpected audio shape {audio_np.shape}.")
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16, audio_int16.shape[1]


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
    """RGB uint8 frames -> planar yuv420p bytes (BT.709 limited range) on GPU."""
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
) -> float:
    """RGB frames (+ optional audio) -> mp4 via one ffmpeg subprocess.
    VBR at the average bitrate with a 2x/4x VBV envelope, +faststart.
    Returns wall seconds."""
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

    audio_path = None
    if audio is not None and audio_sample_rate:
        import wave
        audio_int16, num_channels = _audio_to_int16(audio)
        audio_path = str(Path(output_path).with_suffix(".wav"))
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(int(audio_sample_rate))
            wf.writeframes(audio_int16.tobytes())

    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", in_pix, "-s", f"{w}x{h}", "-r", str(int(fps)),
           "-i", "pipe:0"]
    if audio_path:
        cmd += ["-i", audio_path]
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
    if audio_path:
        cmd += ["-c:a", "aac"]
        if audio_bitrate_kbps:
            cmd += ["-b:a", f"{int(audio_bitrate_kbps)}k"]
        if audio_mono:
            cmd += ["-ac", "1"]
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", str(output_path)]

    try:
        proc = subprocess.run(cmd, input=video_bytes, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed (rc={proc.returncode}): "
                               f"{proc.stderr.decode(errors='replace')[-800:]}")
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)
    return time.perf_counter() - t0


def make_lq_frames(
    frames: list[Any],
    blur_radius: float,
    device: str | None = None,
    chunk_size: int = 16,
) -> list[Any]:
    """Half-resolution area downscale + separable gaussian blur, on GPU."""
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


def build_s3_key(s3_cfg, filename: str) -> str:
    prefix = s3_cfg.prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def create_s3_client(s3_cfg):
    import boto3
    return boto3.client(
        "s3",
        region_name=s3_cfg.region,
        aws_access_key_id=s3_cfg.access_key,
        aws_secret_access_key=s3_cfg.secret_key,
        **({"endpoint_url": s3_cfg.endpoint_url} if s3_cfg.endpoint_url else {}),
    )


def upload_file_to_s3(client, s3_cfg, local_path: str | Path, key: str) -> str:
    client.upload_file(str(local_path), s3_cfg.bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"s3://{s3_cfg.bucket}/{key}"
