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
        disable_smart_memory=cfg.disable_smart_memory,
        reserve_vram_gb=cfg.reserve_vram_gb,
        model_files={
            "checkpoints": cfg.models.checkpoint,
            "diffusion_models": cfg.models.stage2_transformer,
            "text_encoders": cfg.models.text_encoder,
            "latent_upscale_models": cfg.models.latent_upsampler,
        },
    )
    recipe = LtxRecipe(handles, cfg)

    stage_models = [("stage1", recipe.model_s1, cfg.fa4_fp8_stage1)]
    if recipe.model_s2 is not None:
        stage_models.append(("stage2", recipe.model_s2, cfg.fa4_fp8_stage2))

    if cfg.attention_backend == "fa4":
        from .attention import install_fa4_override
        for label, patcher, fp8 in stage_models:
            install_fa4_override(patcher, fp8=fp8, label=label)

    if cfg.compile:
        from .perf import apply_inductor_settings, compile_model, prepare_model_for_compile
        apply_inductor_settings()
        for label, patcher, _fp8 in stage_models:
            prepare_model_for_compile(patcher, label)
            compile_model(patcher, scope=cfg.compile_scope, label=label)

    return recipe


def generate_for_mode(recipe: LtxRecipe, cfg: ServerConfig, mode: Mode,
                      request: GenerationRequest) -> dict:
    result = recipe.generate(request, mode)
    if not result.get("frames"):
        raise RuntimeError("generation returned no frames")
    return result


def make_warmup_image(path: str | Path, width: int, height: int) -> None:
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
    Image.fromarray(arr).save(path)


def run_warmup(recipe: LtxRecipe, cfg: ServerConfig, log=print) -> None:
    """One generation per distinct mode shape: pushes every model through a
    full pass (CUDA context, sage kernels, VAE tiling decisions) before real
    traffic, and validates the encode path once."""
    from .encoder import encode_video_h264

    seen: set[tuple[int, int, int, int]] = set()
    encode_checked = False
    workdir = Path(tempfile.mkdtemp(prefix="ltxs_warmup_"))
    try:
        for mode in cfg.modes:
            key = (mode.width, mode.height, mode.num_frames, mode.fps)
            if key in seen:
                log(f"[warmup] {mode} duplicates an earlier mode; skipping")
                continue
            seen.add(key)
            first = workdir / f"first_{mode.width}x{mode.height}.png"
            make_warmup_image(first, mode.width, mode.height)
            request = GenerationRequest(prompt=WARMUP_PROMPT, first_frame_path=str(first),
                                        seed=42, last_frame_strength=cfg.last_frame_strength)
            t0 = time.perf_counter()
            log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} fps{mode.fps}…")
            result = generate_for_mode(recipe, cfg, mode, request)
            log(f"[warmup] {mode.width}x{mode.height} f{mode.num_frames} "
                f"wall={time.perf_counter() - t0:.1f}s")
            if not encode_checked:
                encode_checked = True
                seconds = encode_video_h264(
                    result["frames"], mode.fps, workdir / "warmup.mp4",
                    bitrate_kbps=cfg.video_bitrate_kbps, preset=cfg.x264_preset,
                    audio=result.get("audio"),
                    audio_sample_rate=result.get("audio_sample_rate"),
                    threads=cfg.encode_threads, codec=cfg.video_codec,
                    extra_video_args=cfg.extra_video_args)
                log(f"[warmup] encode check passed ({seconds:.1f}s)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
