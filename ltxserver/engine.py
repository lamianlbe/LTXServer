"""Engine: boot comfy, hold the recipe, warm up, serve generations."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from .comfy_boot import boot, setup_environment
from .config import Mode, ServerConfig, WARMUP_PROMPT
from .recipe import GenerationRequest, LtxRecipe

logger = logging.getLogger("ltxserver.engine")


def create_recipe(cfg: ServerConfig) -> LtxRecipe:
    """Environment -> embedded comfy -> models resident. Call once."""
    setup_environment(cfg.cuda_visible_devices, inductor_cache_dir=cfg.inductor_cache_dir)
    handles = boot(
        use_sage_attention=cfg.use_sage_attention,
        highvram=cfg.highvram,
        gpu_only=cfg.gpu_only,
        reserve_vram_gb=cfg.reserve_vram_gb,
        model_files={
            "checkpoints": cfg.models.checkpoint,
            "diffusion_models": cfg.models.stage2_transformer,
            "text_encoders": cfg.models.text_encoder,
            "latent_upscale_models": cfg.models.latent_upsampler,
        },
    )
    recipe = LtxRecipe(handles, cfg)

    if cfg.vae_decode_chunk_mib != 0:
        from .perf import set_vae_chunk_budget
        set_vae_chunk_budget(cfg.vae_decode_chunk_mib)

    stage_models = [("stage1", recipe.model_s1, cfg.fa4_fp8_stage1)]
    if recipe.model_s2 is not None:
        stage_models.append(("stage2", recipe.model_s2, cfg.fa4_fp8_stage2))

    if cfg.attention_backend == "fa4":
        from .attention import install_fa4_override
        for label, patcher, fp8 in stage_models:
            install_fa4_override(patcher, fp8=fp8, label=label,
                                 smooth_k=cfg.fa4_fp8_smooth_k)
    elif cfg.attention_backend == "cudnn_mxfp8":
        from .attention_mxfp8 import install_mxfp8_override
        for label, patcher, _fp8 in stage_models:
            install_mxfp8_override(patcher, label=label)

    if cfg.compile:
        from .perf import apply_inductor_settings, compile_model, prepare_model_for_compile
        apply_inductor_settings()
        for label, patcher, _fp8 in stage_models:
            prepare_model_for_compile(patcher, label)
            compile_model(patcher, scope=cfg.compile_scope, label=label)
        if cfg.compile_vae:
            from .perf import compile_vae_codec
            compile_vae_codec(recipe.vae, label="vae")
        if cfg.compile_te:
            from .perf import compile_text_encoder
            compile_text_encoder(recipe.clip, label="te")

    return recipe


def generate_for_mode(recipe: LtxRecipe, cfg: ServerConfig, mode: Mode,
                      request: GenerationRequest) -> dict:
    result = recipe.generate(request, mode)
    if not result.get("frames"):
        raise RuntimeError("generation returned no frames")
    return result


def make_warmup_image(path: str | Path, width: int, height: int,
                      invert: bool = False) -> None:
    """Synthetic but structured conditioning image (diagonal gradient)."""
    import numpy as np
    from PIL import Image

    x = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    arr = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.broadcast_to((x + y) / 2.0, (height, width)),
        ],
        axis=-1,
    ).astype(np.uint8)
    if invert:
        arr = 255 - arr
    Image.fromarray(arr).save(path)


def run_warmup(recipe: LtxRecipe, cfg: ServerConfig, log=print, s3_client=None) -> None:
    """Mirror the REAL request flow before declaring ready, per distinct mode:
    generation with first AND last conditioning frames, then the production
    encode pattern (GPU LQ frames + HQ/LQ ffmpeg in a 2-thread pool), an S3
    connection warm, and finally a steady-state re-generation of the first
    mode — so the first real request pays no first-hit cost (compile caches,
    CUDA allocator segment growth, boto3 TLS/auth) that warmup could have
    absorbed."""
    from concurrent.futures import ThreadPoolExecutor

    from .encoder import encode_video_h264, make_lq_frames

    seen: set[tuple[int, int, int, int]] = set()
    encode_checked = False
    first_request: GenerationRequest | None = None
    first_mode: Mode | None = None
    workdir = Path(tempfile.mkdtemp(prefix="ltxs_warmup_"))
    try:
        for mode in cfg.modes:
            key = (mode.width, mode.height, mode.num_frames, mode.fps)
            if key in seen:
                log(f"[warmup] {mode} duplicates an earlier mode; skipping")
                continue
            seen.add(key)
            first = workdir / f"first_{mode.width}x{mode.height}.png"
            last = workdir / f"last_{mode.width}x{mode.height}.png"
            make_warmup_image(first, mode.width, mode.height)
            make_warmup_image(last, mode.width, mode.height, invert=True)
            # last_frame included: real requests send one, and the last-frame
            # conditioning path (inplace index=-1 / trailing guide) must be
            # exercised before traffic.
            request = GenerationRequest(prompt=WARMUP_PROMPT, first_frame_path=str(first),
                                        last_frame_path=str(last), seed=42,
                                        last_frame_strength=cfg.last_frame_strength)
            if first_request is None:
                first_request, first_mode = request, mode
            t0 = time.perf_counter()
            log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} fps{mode.fps}…")
            result = generate_for_mode(recipe, cfg, mode, request)
            log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} "
                f"wall={time.perf_counter() - t0:.1f}s")
            if cfg.compile:
                from .perf import log_dynamo_counters
                log_dynamo_counters(f"warmup {mode.width}x{mode.height}", log=lambda m, *a: log(m % a))
            if not encode_checked:
                encode_checked = True
                # Production pattern (server /v1/generate_s3): GPU LQ frames,
                # then HQ + LQ encodes concurrently in a 2-thread pool.
                t_enc = time.perf_counter()
                lq_frames = make_lq_frames(result["frames"], cfg.lq_blur_radius)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    hq_f = pool.submit(
                        encode_video_h264, result["frames"], mode.fps, workdir / "hq.mp4",
                        bitrate_kbps=cfg.video_bitrate_kbps, preset=cfg.x264_preset,
                        audio=result.get("audio"),
                        audio_sample_rate=result.get("audio_sample_rate"),
                        threads=cfg.encode_threads, codec=cfg.video_codec,
                        extra_video_args=cfg.extra_video_args)
                    lq_f = pool.submit(
                        encode_video_h264, lq_frames, mode.fps, workdir / "lq.mp4",
                        bitrate_kbps=cfg.lq_bitrate_kbps, preset=cfg.lq_x264_preset,
                        profile="baseline", audio=result.get("audio"),
                        audio_sample_rate=result.get("audio_sample_rate"),
                        audio_bitrate_kbps=64, audio_mono=True,
                        threads=cfg.encode_threads, codec=cfg.video_codec,
                        extra_video_args=cfg.extra_video_args)
                    hq_s, lq_s = hq_f.result(), lq_f.result()
                log(f"[warmup] encode check passed (hq {hq_s:.1f}s + lq {lq_s:.1f}s, "
                    f"wall {time.perf_counter() - t_enc:.1f}s)")
        del result

        if s3_client is not None and cfg.s3 is not None:
            # Warm the server's boto3 client (DNS + TLS + auth) so the first
            # real upload doesn't pay the handshake. No object is written.
            try:
                t_s3 = time.perf_counter()
                s3_client.head_bucket(Bucket=cfg.s3.bucket)
                log(f"[warmup] s3 connection warm ({time.perf_counter() - t_s3:.2f}s)")
            except Exception as err:  # noqa: BLE001
                log(f"[warmup] s3 warm skipped ({err})")

        if first_request is not None and first_mode is not None:
            # Steady-state check: re-run the first mode AFTER the encode so
            # warmup itself absorbs any first-request-after-warmup residue
            # (allocator reshaping, lingering guard misses). These numbers
            # should match steady serving; if not, the dynamo-graph warning
            # in the phase log says why.
            first_request.seed = 43
            t0 = time.perf_counter()
            log("[warmup] steady-state check (re-run of first mode)…")
            generate_for_mode(recipe, cfg, first_mode, first_request)
            log(f"[warmup] steady-state check wall={time.perf_counter() - t0:.1f}s "
                "(this should match real request latency)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
