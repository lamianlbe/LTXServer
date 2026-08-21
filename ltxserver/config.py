"""Server configuration: YAML -> validated dataclasses.

The recipe knobs default to the reference workflow's values (optimized
LTX-2.3 all-in-one, corrected guider sigma list), so an empty override
section reproduces the workflow; the serving knobs mirror the FastVideo
LTX-2.3 server's config surface so deployments can switch between the two
backends without relearning the file format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The reference recipe (ComfyUI workflow port). stage1_sigmas is the EASED
# schedule the sampler runs; the guider's own sigma list keeps its near-1.0
# entries — they confine cfg 2 to the first three eased steps.
DEFAULT_STAGE1_SIGMAS = [
    1.0, 0.99987238, 0.99820748, 0.99001548, 0.96332988, 0.89394948,
    0.744596, 0.47298248, 0.20186216, 0.04708576, 0.0,
]
DEFAULT_STAGE2_SIGMAS = [0.85, 0.7250, 0.4219, 0.0]
DEFAULT_CFG_SIGMA_LIST = [
    1.0, 0.99375, 0.9875, 0.98125, 0.9550, 0.8925, 0.8120, 0.7150,
    0.6030, 0.4824, 0.3618, 0.2412, 0.1206, 0.0,
]
DEFAULT_CFG_VALUES = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
DEFAULT_NEGATIVE_PROMPT = (
    "still image, bad quality, subtitles, text, watermark, overlay effects, pc game, "
    "yelling, console game, video game, cartoon, childish, ugly, text, blur, logo, "
    "wordmark, static, low quality, noise, white noise, bleep, censoring, censor, "
    "bleeping, beep, beeping, newscast, interview, podcast, non-english, foreign "
    "language, russian, chinese, japanese, mutant, horror, 70's, film grain, "
    "cinematic, comedy, stand-up ")
DEFAULT_GUIDE_LONGER_SIZE = 1536
DEFAULT_GUIDE_STRENGTH = 0.8
DEFAULT_LAST_FRAME_STRENGTH = 0.8
WARMUP_PROMPT = ("A person slowly turns their head toward the camera and smiles, "
                 "soft warm light, gentle camera drift.")


@dataclass(frozen=True)
class Mode:
    """One supported (resolution, frames, fps) combination.

    width/height are the FINAL (post-upscale) output resolution, same as the
    FastVideo server's config: stage 1 renders at mode/upsampler-scale
    (validated against the loaded upsampler at startup — for the x1.5
    upscaler use dims divisible by 96, e.g. 1344x768 -> stage 1 896x512).
    With stage2_enabled: false the mode is the direct render resolution.
    """
    width: int
    height: int
    num_frames: int
    fps: int

    def validate(self) -> None:
        if self.width % 32 or self.height % 32:
            raise ValueError(f"mode {self.width}x{self.height}: dims must be divisible by 32 "
                             "(LTX VAE spatial compression); the x1.5 upscale additionally "
                             "needs the upscaled dims to stay integral")
        if (self.num_frames - 1) % 8:
            raise ValueError(f"mode num_frames={self.num_frames}: must be 8*k+1 "
                             "(temporal VAE compression)")
        if self.fps <= 0:
            raise ValueError(f"mode fps={self.fps}: must be positive")


@dataclass
class ModelPaths:
    """The four ComfyUI-style safetensors files the recipe loads.

    checkpoint          all-in-one base with the STAGE-1 merged DiT inside
                        (build with scripts/merge_stage1_into_base.py)
    stage2_transformer  DiT-only ModelSave export of the stage-2 merge
    text_encoder        Gemma text-encoder file (e.g. the heretic nvfp4)
    latent_upsampler    the x1.5 spatial upscaler
    """
    checkpoint: str
    stage2_transformer: str
    text_encoder: str
    latent_upsampler: str

    def validate(self, stage2_enabled: bool = True) -> None:
        names = ["checkpoint", "text_encoder"]
        if stage2_enabled:
            names += ["stage2_transformer", "latent_upsampler"]
        for name in names:
            path = Path(getattr(self, name))
            if not path.is_file():
                raise ValueError(f"models.{name}: not a file: {path}")


@dataclass
class S3Config:
    region: str
    bucket: str
    access_key: str
    secret_key: str
    endpoint_url: str = ""
    prefix: str = ""

    def validate(self) -> None:
        for field_name in ("region", "bucket", "access_key", "secret_key"):
            if not getattr(self, field_name):
                raise ValueError(f"s3.{field_name} is required")


@dataclass
class ServerConfig:
    models: ModelPaths
    modes: list[Mode]

    # --- process / serving ---------------------------------------------------
    cuda_visible_devices: str = ""  # "" = inherit; one server process per GPU
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = ""
    log_dir: str = ""
    api_keys: list[str] = field(default_factory=list)
    max_consecutive_failures: int = 3
    warmup_on_start: bool = True

    # --- ComfyUI runtime -------------------------------------------------------
    # Attention backend. Default false = ComfyUI's pytorch SDPA — the
    # mathematically exact kernel and the baseline all comparisons use.
    # SageAttention (int8 QK^T) is kept as an opt-in experiment only; it
    # runs poorly on Blackwell and needs a manual source build
    # (thu-ml/SageAttention).
    use_sage_attention: bool = False
    # Pin every model in GPU memory (comfy --highvram): without it comfy
    # unloads models to CPU after use, so a server pays a full PCIe reload
    # per request AND between the stages inside one request. Required for
    # serving; set false only on VRAM-starved debug boxes (comfy's smart
    # memory then offloads as needed).
    highvram: bool = True
    # comfy --gpu-only: text encoders live ON the GPU and intermediate
    # results stay there. Without it every request reloads the 13GB text
    # encoder from RAM (~2s) and VAE-decoded frames round-trip through CPU
    # fp32 (seconds of copy + CPU postprocessing). Set false together with
    # highvram: false on VRAM-starved debug boxes.
    gpu_only: bool = True
    # Reserved VRAM comfy leaves free (GB); 0 = comfy default.
    reserve_vram_gb: float = 0.0

    # --- performance -----------------------------------------------------------
    # torch.compile both DiTs (inductor, via comfy's official wrapper). The
    # fp8 QuantizedTensor linears are first swapped for compile-friendly
    # twins that are probe-verified BIT-IDENTICAL (see ltxserver/perf.py);
    # with compile off, the model runs stock comfy modules end to end.
    # Changing this (or attention settings) changes the compiled graphs —
    # warmup recompiles, so keep inductor_cache_dir persistent.
    compile: bool = False
    # blocks (default) compiles the 48 transformer blocks (~95% of compute)
    # and leaves comfy's outer glue eager — the guide-keyframe bookkeeping in
    # _process_input does a data-dependent .item() that can never be captured
    # (observed breaking whole-model graphs on stage 1). model = one graph
    # per DiT, only sensible for guide-free recipes.
    compile_scope: str = "blocks"  # blocks | model
    inductor_cache_dir: str = ""  # "" = torch default (NOT persistent across restarts)
    # Attention backend for BOTH DiTs. sdpa = comfy's pytorch attention (the
    # exact baseline). fa4 = flash_attn.cute (Hopper/Blackwell datacenter
    # GPUs; needs the pinned source build — see install.sh notes); masked
    # attention segments (the stage-1 guide bias) always fall back to sdpa,
    # mirroring the FastVideo server. fa4_fp8_stage1/2 additionally run that
    # stage's unmasked attention with fp8 q/k/v (per-head descales) at the
    # fp8 tensor-core rate — a numerics/speed trade to A/B per stage.
    attention_backend: str = "sdpa"  # sdpa | fa4
    fa4_fp8_stage1: bool = False
    fa4_fp8_stage2: bool = False

    # --- recipe (defaults = the reference workflow) ---------------------------
    stage1_sigmas: list[float] = field(default_factory=lambda: list(DEFAULT_STAGE1_SIGMAS))
    stage2_sigmas: list[float] = field(default_factory=lambda: list(DEFAULT_STAGE2_SIGMAS))
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    # STGGuiderAdvanced (stage 1). The sigma->params mapping is a lookup:
    # smallest listed sigma still >= the sampler's current sigma.
    cfg_sigma_list: list[float] = field(default_factory=lambda: list(DEFAULT_CFG_SIGMA_LIST))
    cfg_values_by_sigma: list[float] = field(default_factory=lambda: list(DEFAULT_CFG_VALUES))
    cfg_star_rescale: bool = True
    skip_steps_sigma_threshold: float = 1.0
    # STG is numerically inert in the reference workflow: its skip-layer
    # lists are all [9999] (no real block), so the perturbed pass equals the
    # plain pass and scale*(pos - perturbed) is exactly zero for ANY scale.
    # The workflow node still carries non-zero scales (2, 1.5, 1, ...),
    # which makes the guider run that dead perturbed pass — a FULL extra DiT
    # forward EVERY step. Default the scales to zero instead: bit-identical
    # output (the term is zero either way), one forward per step saved.
    # Anyone enabling real STG sets stg_layers_indices AND these scales.
    stg_scale_values: list[float] | None = None   # None = all 0.0 (skip the dead pass)
    stg_rescale_values: list[float] | None = None  # None = all 1.0
    stg_layers_indices: str = ""                   # "" = "[9999]" per entry
    apg_cfg_scale: float = 1.0
    apg_eta: float = 1.0
    apg_norm_threshold: float = 1.0
    # Run only stage 1 (skip the x1.5 upsample + refine pass) and decode
    # the stage-1 result directly at the mode resolution. For stage-level
    # A/B and fast iteration; the stage-2 transformer and the upsampler are
    # not loaded when this is false.
    stage2_enabled: bool = True
    # Stage-1 conditioning mechanism:
    #   guide   — the workflow's appended keyframes (BatchAddGuide): extra
    #             guide-frame tokens, keyframe bookkeeping in every forward,
    #             optional attention bias. The reference-quality recipe.
    #   inplace — the FastVideo recipe: write the first frame INTO latent
    #             frame 0 (hard pin, strength 1.0) and the optional last
    #             frame at -1 (pinned at the request's last_frame_strength).
    #             No appended tokens, no keyframe machinery — stage-1
    #             forwards as clean (and as fast) as stage 2. guide_strength
    #             and guide_attention_bias are unused in this mode.
    #             Switching modes changes the stage-1 sequence length, so
    #             compiled graphs re-warm on first use.
    stage1_conditioning: str = "guide"  # guide | inplace
    # Guide (first/last frame) conditioning.
    # guide_attention_bias: the workflow's log(strength) content<->guide
    # self-attention attenuation. It forces stage-1 attn1 onto a segmented
    # MASKED sdpa path in every block (no FA4 there) — the reference recipe
    # pays this; false skips the mask entirely (single unmasked call,
    # FA4-eligible) while the guide's noise-level semantics keep
    # guide_strength — the same trade FastVideo's guide_attention_bias knob
    # makes, in the same direction.
    guide_attention_bias: bool = True
    guide_strength: float = DEFAULT_GUIDE_STRENGTH
    guide_longer_size: int = DEFAULT_GUIDE_LONGER_SIZE
    last_frame_strength: float = DEFAULT_LAST_FRAME_STRENGTH
    # LTXVPreprocess "motion strength" H.264 crush of the guide images.
    # The reference workflow has no preprocess node, so 0 (off) is parity.
    image_crf: float = 0.0

    # --- encoding / delivery (identical to the FastVideo server) -------------
    video_bitrate_kbps: int = 3000
    x264_preset: str = "medium"
    video_codec: str = "libx264"
    extra_video_args: str = ""
    encode_threads: int = 0
    max_concurrent_encodes: int = 2
    lq_blur_radius: float = 2.0
    lq_bitrate_kbps: int = 1000
    lq_x264_preset: str = "ultrafast"
    s3: S3Config | None = None

    # ------------------------------------------------------------------
    def stage1_steps(self) -> int:
        return len(self.stage1_sigmas) - 1

    def resolved_stg_scale_values(self) -> list[float]:
        return (list(self.stg_scale_values) if self.stg_scale_values
                else [0.0] * len(self.cfg_values_by_sigma))

    def resolved_stg_rescale_values(self) -> list[float]:
        return (list(self.stg_rescale_values) if self.stg_rescale_values
                else [1.0] * len(self.cfg_values_by_sigma))

    def resolved_stg_layers_indices(self) -> str:
        if self.stg_layers_indices.strip():
            return self.stg_layers_indices
        return ", ".join(["[9999]"] * len(self.cfg_sigma_list))


def _validate_sigmas(name: str, sigmas: list[float]) -> None:
    if len(sigmas) < 2:
        raise ValueError(f"{name} needs at least two entries")
    if any(b >= a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError(f"{name} must be strictly decreasing, got {sigmas}")
    if not math.isclose(sigmas[-1], 0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must end at 0.0, got {sigmas[-1]}")
    if sigmas[0] > 1.0:
        raise ValueError(f"{name} entries must be <= 1.0, got {sigmas[0]}")


def validate_config(cfg: ServerConfig, source: str = "config") -> None:
    cfg.models.validate(stage2_enabled=cfg.stage2_enabled)
    if not cfg.modes:
        raise ValueError(f"{source}: 'modes' must list at least one combination")
    for mode in cfg.modes:
        mode.validate()
    _validate_sigmas("stage1_sigmas", cfg.stage1_sigmas)
    _validate_sigmas("stage2_sigmas", cfg.stage2_sigmas)
    sig = cfg.cfg_sigma_list
    if any(b > a for a, b in zip(sig, sig[1:])):
        raise ValueError(f"{source}: cfg_sigma_list must be non-increasing, got {sig}")
    if len(cfg.cfg_values_by_sigma) < len(sig) - 1:
        raise ValueError(f"{source}: cfg_values_by_sigma needs at least len(cfg_sigma_list) - 1 "
                         f"entries ({len(sig) - 1}), got {len(cfg.cfg_values_by_sigma)}")
    if any(v < 1.0 for v in cfg.cfg_values_by_sigma):
        raise ValueError(f"{source}: cfg_values_by_sigma entries must be >= 1.0")
    if not 0.0 < cfg.guide_strength <= 1.0:
        raise ValueError(f"{source}: guide_strength must be in (0, 1], got {cfg.guide_strength}")
    if cfg.guide_longer_size < 64:
        raise ValueError(f"{source}: guide_longer_size must be >= 64")
    if cfg.image_crf < 0:
        raise ValueError(f"{source}: image_crf must be >= 0")
    if any(not isinstance(k, str) or not k.strip() for k in cfg.api_keys):
        raise ValueError(f"{source}: api_keys entries must be non-empty strings")
    cvd = cfg.cuda_visible_devices
    if cvd and not all(p.strip().isdigit() for p in cvd.split(",")):
        raise ValueError(f"{source}: cuda_visible_devices must be comma-separated GPU indices")
    if cfg.stage1_conditioning not in ("guide", "inplace"):
        raise ValueError(f"{source}: stage1_conditioning must be guide | inplace, "
                         f"got {cfg.stage1_conditioning!r}")
    if cfg.compile_scope not in ("model", "blocks"):
        raise ValueError(f"{source}: compile_scope must be model | blocks, got {cfg.compile_scope!r}")
    if cfg.attention_backend not in ("sdpa", "fa4"):
        raise ValueError(f"{source}: attention_backend must be sdpa | fa4, got {cfg.attention_backend!r}")
    if (cfg.fa4_fp8_stage1 or cfg.fa4_fp8_stage2) and cfg.attention_backend != "fa4":
        raise ValueError(f"{source}: fa4_fp8_stage1/2 require attention_backend: fa4")
    if cfg.attention_backend == "fa4" and cfg.use_sage_attention:
        raise ValueError(f"{source}: attention_backend: fa4 and use_sage_attention are mutually exclusive")
    n_stg = len(cfg.resolved_stg_scale_values())
    if n_stg < len(sig) - 1:
        raise ValueError(f"{source}: stg_scale_values needs at least {len(sig) - 1} entries")


def load_config(path: str | Path) -> ServerConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    models_raw = raw.pop("models", None)
    if not isinstance(models_raw, dict):
        raise ValueError(f"{path}: 'models' must be a mapping with checkpoint / "
                         "stage2_transformer / text_encoder / latent_upsampler")
    modes_raw = raw.pop("modes", None)
    if not modes_raw:
        raise ValueError(f"{path}: 'modes' must list at least one "
                         "{width, height, num_frames, fps} combination")
    s3_raw = raw.pop("s3", None)
    s3_cfg = None
    if s3_raw is not None:
        if not isinstance(s3_raw, dict):
            raise ValueError(f"{path}: 's3' must be a mapping")
        s3_cfg = S3Config(**s3_raw)
        s3_cfg.validate()
    if "disable_smart_memory" in raw:
        raise ValueError(f"{path}: 'disable_smart_memory' is gone — it meant the OPPOSITE of "
                         "resident models (comfy's flag forces offload-after-use). Use "
                         "'highvram: true' (the default) to pin models in GPU memory.")
    known = {f for f in ServerConfig.__dataclass_fields__ if f not in ("models", "modes", "s3")}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    cfg = ServerConfig(models=ModelPaths(**models_raw),
                       modes=[Mode(**m) for m in modes_raw],
                       s3=s3_cfg,
                       **raw)
    validate_config(cfg, source=str(path))
    return cfg


def match_mode(modes: list[Mode], width: int, height: int, num_frames: int,
               fps: int) -> tuple[Mode, bool]:
    """Exact match, else the closest resolution (aspect-aware log distance);
    frames/fps break ties."""
    for mode in modes:
        if (mode.width, mode.height, mode.num_frames, mode.fps) == (width, height, num_frames, fps):
            return mode, True

    def distance(mode: Mode) -> tuple[float, int, int]:
        res = (math.log(width / mode.width) ** 2 + math.log(height / mode.height) ** 2)
        return (res, abs(num_frames - mode.num_frames), abs(fps - mode.fps))

    return min(modes, key=distance), False
