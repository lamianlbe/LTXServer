"""FA4 (flash_attn.cute) attention override for the embedded ComfyUI models.

Installed per-model through ComfyUI's official
``transformer_options["optimized_attention_override"]`` hook — no comfy
source is modified. The override receives every ``optimized_attention``
call the model makes and routes it:

  * unmasked calls -> the FA4 kernel (bf16, or fp8 e4m3 with per-(batch,
    head) descales when that stage's fp8 flag is on);
  * masked calls (the stage-1 guide bias segments) -> the original backend
    (pytorch SDPA), which is the only additive-mask path — mirroring the
    FastVideo server exactly;
  * anything else FA4 cannot take (odd dtypes, unsupported head dims,
    mismatched q/k heads) -> the original backend.

Numerics are a verbatim port of FastVideo's FA4 integration
(fastvideo/attention/utils/flash_attn_cute.py + fp8_utils.py):

  * kernel calls go through ``torch.library`` custom ops with fake kernels,
    so the override stays traceable inside torch.compile'd graphs (the
    CuTeDSL kernel itself is an opaque boundary dynamo cannot trace);
  * fp8 quantization: amax over (seqlen, headdim) per (batch, head),
    clamped min 1e-6, scale applied in the input dtype, saturating clamp to
    +-448, descale = amax/448 as float32 (batch, heads); the fp8 kernel
    always returns bf16;
  * Smooth-K (on by default with fp8, ``smooth_k``): K's per-(batch, head,
    channel) mean over the TOKEN axis is subtracted before quantization.
    softmax(Q(K-mu)^T) == softmax(QK^T) exactly — the shift is constant per
    query row — so this is mathematically free, and it removes the channel
    mean offset that otherwise dominates K's per-head amax (SageAttention's
    core trick). It tightens only K; V keeps the coarse per-head scale, so
    for full per-block scaling see attention_backend: cudnn_mxfp8;
  * bf16 FA4 needs sm90+ (H200 ok); fp8 FA4 is sm100-only upstream (B200).

Install pin (see install.sh): flash-attn @ 82d6441e subdirectory
flash_attn/cute — the cutlass-4.5-compatible revision. A present-but-broken
cute install (nvidia-cutlass-dsl skew) surfaces as ImportError here with
the original error attached.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("ltxserver.attention")

FP8_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
_SUPPORTED_HEAD_DIMS = (32, 64, 96, 128, 160, 192, 224, 256)

_OPS_REGISTERED = False


def _import_fa4():
    """flash_attn.cute forward, with the cutlass-skew remap FastVideo uses:
    an installed-but-broken cute (nvidia-cutlass-dsl version mismatch)
    raises AttributeError-ish garbage — re-raise uniformly as ImportError."""
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
        return _flash_attn_fwd
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ImportError(f"flash_attn.cute (FA4) is installed but failed to import ({e!r}); "
                          "usually an nvidia-cutlass-dsl version mismatch — reinstall the "
                          "pinned revision from install.sh") from e


def _register_ops() -> None:
    """torch.library custom ops wrapping the FA4 kernel (idempotent)."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    _flash_attn_fwd = _import_fa4()

    @torch.library.custom_op("ltxserver::fa4_forward", mutates_args=(), device_types="cuda")
    def fa4_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (batch, seqlen, nheads, headdim); default softmax scale 1/sqrt(d).
        # _flash_attn_fwd returns (out, lse[, ...]) — take the output.
        return _flash_attn_fwd(
            q, k, v,
            softmax_scale=None, causal=False,
            window_size_left=None, window_size_right=None,
            softcap=0.0, num_splits=1, pack_gqa=None,
        )[0]

    @fa4_forward.register_fake
    def _(q, k, v):
        return torch.empty_like(q)

    @torch.library.custom_op("ltxserver::fa4_fp8_forward", mutates_args=(), device_types="cuda")
    def fa4_fp8_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        q_descale: torch.Tensor, k_descale: torch.Tensor,
                        v_descale: torch.Tensor) -> torch.Tensor:
        return _flash_attn_fwd(
            q, k, v,
            softmax_scale=None, causal=False,
            window_size_left=None, window_size_right=None,
            softcap=0.0, num_splits=1, pack_gqa=None,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        )[0]

    @fa4_fp8_forward.register_fake
    def _(q, k, v, q_descale, k_descale, v_descale):
        # fp8 inputs always produce a bf16 output upstream.
        batch, seqlen_q, nheads = q.shape[:3]
        return v.new_empty((batch, seqlen_q, nheads, v.shape[-1]), dtype=torch.bfloat16)

    _OPS_REGISTERED = True


def fp8_quantize_for_fa4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(batch, seqlen, nheads, headdim) -> (fp8 tensor, float32 (batch, nheads)
    descale). Verbatim FastVideo math."""
    amax = t.abs().amax(dim=(1, 3)).to(torch.float32).clamp(min=1e-6)
    scale = (FP8_E4M3_MAX / amax).to(t.dtype)
    scaled = (t * scale[:, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), amax / FP8_E4M3_MAX


def _make_override(fp8: bool, smooth_k: bool = True):
    def fa4_override(func, q, k, v, heads, *args, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False, **kwargs):
        # Masked segments (guide bias) and anything FA4 cannot take run the
        # original backend with the arguments untouched.
        def fallback():
            return func(q, k, v, heads, *args, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                        **kwargs)

        if mask is not None or args:
            return fallback()
        if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
            return fallback()

        if skip_reshape:  # (B, H, T, D)
            dim_head = q.shape[-1]
            q_b = q.transpose(1, 2)
            k_b = k.transpose(1, 2)
            v_b = v.transpose(1, 2)
        else:  # (B, T, H*D)
            if q.shape[-1] % heads or k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
                return fallback()
            dim_head = q.shape[-1] // heads
            q_b = q.view(q.shape[0], q.shape[1], heads, dim_head)
            k_b = k.view(k.shape[0], k.shape[1], heads, dim_head)
            v_b = v.view(v.shape[0], v.shape[1], heads, dim_head)
        if dim_head not in _SUPPORTED_HEAD_DIMS:
            return fallback()

        if fp8:
            if smooth_k:
                # Token-axis mean removal: softmax-invariant (constant shift
                # per query row), tightens K's fp8 range.
                k_b = k_b - k_b.mean(dim=1, keepdim=True)
            q_f, q_d = fp8_quantize_for_fa4(q_b)
            k_f, k_d = fp8_quantize_for_fa4(k_b)
            v_f, v_d = fp8_quantize_for_fa4(v_b)
            out = torch.ops.ltxserver.fa4_fp8_forward(q_f, k_f, v_f, q_d, k_d, v_d)
            if out.dtype != q.dtype:
                out = out.to(q.dtype)
        else:
            out = torch.ops.ltxserver.fa4_forward(q_b.contiguous(), k_b.contiguous(),
                                                  v_b.contiguous())

        if skip_output_reshape:  # (B, H, T, D)
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * dim_head)

    return fa4_override


def install_fa4_override(model_patcher, *, fp8: bool, label: str,
                         smooth_k: bool = True) -> None:
    """Preflight the hardware/install, then hook the override onto one model."""
    if not torch.cuda.is_available():
        raise RuntimeError("attention_backend: fa4 needs a CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        raise RuntimeError(f"FA4 needs sm90+ (H200/B200); this device is sm{cap[0]}{cap[1]}")
    if fp8 and cap not in ((10, 0), (10, 3)):
        raise RuntimeError(f"fp8 FA4 is sm100-only upstream (B200); this device is "
                           f"sm{cap[0]}{cap[1]} — use bf16 FA4 here (fa4_fp8_{label}: false)")
    _register_ops()

    options = model_patcher.model_options.setdefault("transformer_options", {})
    options["optimized_attention_override"] = _make_override(fp8, smooth_k)
    mode = "bf16"
    if fp8:
        mode = "fp8 e4m3, per-head descales" + (", smooth-K" if smooth_k else "")
    logger.info("[%s] FA4 attention override installed (%s; masked segments stay on the "
                "default backend)", label, mode)
