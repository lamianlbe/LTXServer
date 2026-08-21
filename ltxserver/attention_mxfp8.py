"""cuDNN MXFP8 (microscaled fp8) attention override — Blackwell only.

Motivation: coarse per-(batch,head) fp8 attention (FA4 fp8) visibly
desaturates LTX-2 output — one amax per head over the whole sequence
crushes small-amplitude signal, and V quantization directly limits output
precision. MXFP8 quantizes per 32-ELEMENT BLOCK with an E8M0 scale each:
Q/K along the head dim (4 scales per row at d=128), V along the sequence
(one scale per 32 tokens per channel), P at a fixed 256, output straight
to bf16. cuDNN's Blackwell SDPA consumes this natively at full fp8
tensor-core rate (block-scale MMA is a tcgen05 hardware feature).

Same integration shape as the FA4 override (attention.py): installed via
``transformer_options["optimized_attention_override"]``, masked calls and
anything unsupported (head_dim != 128, mismatched layouts) fall back to
the original backend, and the kernel call is wrapped in a torch.library
custom op with a fake kernel so torch.compile treats it as an opaque node.

Quantization math (E8M0 rounding, satfinite clamp, F8_128x4 scale swizzle)
is adapted line-for-line from NVIDIA cudnn-frontend
``test/python/sdpa/mxfp8_quant.py`` (MIT), reworked to quantize
BSHD-contiguous tensors in place of the BHSD reference layout —
``scripts/test_mxfp8_quant.py`` bit-verifies this module against the
vendored reference.

Requires: sm100/sm103 (B200/B300), cuDNN backend >= 9.21, and the
``nvidia-cudnn-frontend`` python package (``import cudnn``).
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger("ltxserver.attention_mxfp8")

BLOCK = 32
E4M3_MAX = 448.0
_E4M3_MAX_RCP = torch.tensor(1.0 / 448.0, dtype=torch.float32)  # fp32-rounded, as TE

_OPS_REGISTERED = False
_GRAPH_CACHE: dict = {}


def _import_cudnn():
    try:
        import cudnn
        return cudnn
    except ImportError as e:
        raise ImportError(
            "attention_backend: cudnn_mxfp8 needs the cudnn-frontend python package — "
            "pip install 'nvidia-cudnn-frontend[cutedsl]'") from e


# --------------------------------------------------------------------------
# E8M0 block quantization (TE-equivalent semantics, BSHD-native)
# --------------------------------------------------------------------------

def _e8m0_ceil(v: torch.Tensor) -> torch.Tensor:
    """fp32 (>=0) -> biased E8M0 exponent byte, scale rounded UP to a power
    of two. 0 -> 0x00, inf -> 0xFE, nan -> 0xFF (TE float_to_e8m0)."""
    v = v.float().contiguous()
    bits = v.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    round_up = (mant > 0) & (exp != 0xFE) & ~((exp == 0) & (mant <= 0x400000))
    e = exp + round_up.to(torch.int32)
    e = torch.where(torch.isnan(v), torch.full_like(e, 0xFF), e)
    e = torch.where(torch.isinf(v), torch.full_like(e, 0xFE), e)
    return e.to(torch.uint8)


def _exp2_rcp(e: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 byte -> exact fp32 2^(127-e), incl. the e==254 -> 2^-127
    subnormal special case (TE exp2f_rcp)."""
    e32 = e.to(torch.int32)
    rcp_bits = torch.where(e32 == 254, torch.full_like(e32, 0x00400000), (254 - e32) << 23)
    return rcp_bits.view(torch.float32)


def _swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """F8_128x4 reorder of a logical scale matrix [..., R, C] (uint8),
    R % 128 == 0, C % 4 == 0. Atoms (128 rows x 4 cols) row-major; inside an
    atom scale (r, c) lives at byte (r % 32) * 16 + (r // 32) * 4 + c."""
    assert sf.dtype == torch.uint8
    *lead, R, C = sf.shape
    assert R % 128 == 0 and C % 4 == 0, f"F8_128x4 needs R%128==0, C%4==0, got {sf.shape}"
    v = sf.reshape(*lead, R // 128, 4, 32, C // 4, 4)  # (rt, rg, rr, ct, cc)
    n = len(lead)
    v = v.permute(*range(n), n + 0, n + 3, n + 2, n + 1, n + 4)  # (rt, ct, rr, rg, cc)
    return v.contiguous().reshape(*lead, R, C)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantize_rowwise(t: torch.Tensor):
    """(B, S, H, D) any float -> (fp8 e4m3 (B, S, H, D) BSHD-contiguous,
    swizzled E8M0 scales uint8 (B, H, S_pad, D/32_pad)). Scales along D —
    the Q/K orientation. D must be a multiple of 32."""
    b, s, h, d = t.shape
    assert d % BLOCK == 0
    dsc = _cdiv(d // BLOCK, 4) * 4
    s_pad = _cdiv(s, 128) * 128

    x = t.float()
    amax = x.reshape(b, s, h, d // BLOCK, BLOCK).abs().amax(dim=-1)  # (B,S,H,D/32)
    e = _e8m0_ceil(amax * _E4M3_MAX_RCP.to(x.device))
    rcp = _exp2_rcp(e).unsqueeze(-1)  # (B,S,H,D/32,1)
    data = (x.reshape(b, s, h, d // BLOCK, BLOCK) * rcp).clamp_(-E4M3_MAX, E4M3_MAX)
    data = data.to(torch.float8_e4m3fn).reshape(b, s, h, d).contiguous()

    sf = e.permute(0, 2, 1, 3)  # (B,H,S,D/32)
    pad_c = dsc - d // BLOCK
    pad_r = s_pad - s
    if pad_c or pad_r:
        sf = torch.nn.functional.pad(sf, (0, pad_c, 0, pad_r))
    return data, _swizzle_128x4(sf.contiguous())


def quantize_columnwise(t: torch.Tensor):
    """(B, S, H, D) any float -> (fp8 e4m3 (B, S, H, D) BSHD-contiguous,
    swizzled E8M0 scales uint8 (B, H, S/32_pad, D_pad)). Scales along S —
    the V orientation. Zero-padding the last partial 32-token block cannot
    raise its amax, so arbitrary S is exact."""
    b, s, h, d = t.shape
    ssc = _cdiv(_cdiv(s, BLOCK), 4) * 4
    s_pad32 = _cdiv(s, BLOCK) * BLOCK
    d_pad = _cdiv(d, 128) * 128

    x = t.float()
    if s_pad32 != s:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, s_pad32 - s))
    xb = x.reshape(b, s_pad32 // BLOCK, BLOCK, h, d)
    amax = xb.abs().amax(dim=2)  # (B, S/32, H, D)
    e = _e8m0_ceil(amax * _E4M3_MAX_RCP.to(x.device))
    rcp = _exp2_rcp(e).unsqueeze(2)  # (B, S/32, 1, H, D)
    data = (xb * rcp).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    data = data.reshape(b, s_pad32, h, d)[:, :s].contiguous()

    sf = e.permute(0, 2, 1, 3)  # (B, H, S/32, D)
    pad_r = ssc - s_pad32 // BLOCK
    pad_c = d_pad - d
    if pad_r or pad_c:
        sf = torch.nn.functional.pad(sf, (0, pad_c, 0, pad_r))
    # Columnwise storage keeps [S/32, D]; the 128x4 atom rule applies to the
    # TRANSPOSED matrix [D, S/32] (TE swizzle_col_scaling_kernel).
    swz = _swizzle_128x4(sf.transpose(-1, -2).contiguous()).reshape(sf.shape)
    return data, swz


# --------------------------------------------------------------------------
# cuDNN graph cache + custom op
# --------------------------------------------------------------------------

def _get_graph(b: int, h: int, s_q: int, s_kv: int, d: int):
    """Build (once per shape) the sdpa_mxfp8 pygraph: fp8 BHSD-logical /
    BSHD-strided q/k/v + swizzled E8M0 descales -> bf16 output, no stats."""
    key = (b, h, s_q, s_kv, d)
    hit = _GRAPH_CACHE.get(key)
    if hit is not None:
        return hit
    cudnn = _import_cudnn()

    dsc = _cdiv(d // BLOCK, 4) * 4
    sq_pad = _cdiv(s_q, 128) * 128
    skv_pad = _cdiv(s_kv, 128) * 128
    skv_sc = _cdiv(_cdiv(s_kv, BLOCK), 4) * 4
    d_pad = _cdiv(d, 128) * 128

    g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3,
                      intermediate_data_type=cudnn.data_type.FLOAT,
                      compute_data_type=cudnn.data_type.FLOAT)

    def bshd(seq):
        return dict(dim=[b, h, seq, d], stride=[seq * h * d, d, h * d, 1])

    q = g.tensor(**bshd(s_q), data_type=cudnn.data_type.FP8_E4M3)
    k = g.tensor(**bshd(s_kv), data_type=cudnn.data_type.FP8_E4M3)
    v = g.tensor(**bshd(s_kv), data_type=cudnn.data_type.FP8_E4M3)

    def sf(r, c):
        return g.tensor(dim=[b, h, r, c],
                        stride=[h * r * c, r * c, c, 1],
                        data_type=cudnn.data_type.FP8_E8M0,
                        reordering_type=cudnn.tensor_reordering.F8_128x4)

    dq = sf(sq_pad, dsc)
    dk = sf(skv_pad, dsc)
    dv = sf(skv_sc, d_pad)

    o, _stats, amax_o = g.sdpa_mxfp8(q=q, k=k, v=v,
                                     descale_q=dq, descale_k=dk, descale_v=dv,
                                     attn_scale=1.0 / math.sqrt(d),
                                     generate_stats=False)
    o.set_output(True).set_dim([b, h, s_q, d]).set_stride([s_q * h * d, d, h * d, 1]) \
        .set_data_type(cudnn.data_type.BFLOAT16)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]) \
        .set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    g.check_support()
    g.build_plans()
    ws = g.get_workspace_size()
    entry = (g, (q, k, v, dq, dk, dv, o, amax_o), ws)
    _GRAPH_CACHE[key] = entry
    logger.info("mxfp8 sdpa graph built: b=%d h=%d s_q=%d s_kv=%d d=%d (workspace %d B)",
                b, h, s_q, s_kv, d, ws)
    return entry


def _register_ops() -> None:
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    _import_cudnn()  # fail here, not inside the op

    @torch.library.custom_op("ltxserver::mxfp8_sdpa", mutates_args=(), device_types="cuda")
    def mxfp8_sdpa(q8: torch.Tensor, k8: torch.Tensor, v8: torch.Tensor,
                   sfq: torch.Tensor, sfk: torch.Tensor, sfv: torch.Tensor) -> torch.Tensor:
        # q8/k8/v8: fp8 e4m3 (B, S, H, D) BSHD-contiguous; sf*: swizzled
        # uint8 E8M0 scales from quantize_rowwise / quantize_columnwise.
        b, s_q, h, d = q8.shape
        s_kv = k8.shape[1]
        g, (tq, tk, tv, tdq, tdk, tdv, to, tamax), ws = _get_graph(b, h, s_q, s_kv, d)
        out = torch.empty(b, s_q, h, d, device=q8.device, dtype=torch.bfloat16)
        amax = torch.zeros(1, 1, 1, 1, device=q8.device, dtype=torch.float32)
        workspace = torch.empty(max(ws, 1), device=q8.device, dtype=torch.uint8)
        # cudnn consumes BHSD-logical views over our BSHD-contiguous memory.
        g.execute({tq: q8.transpose(1, 2), tk: k8.transpose(1, 2), tv: v8.transpose(1, 2),
                   tdq: sfq, tdk: sfk, tdv: sfv,
                   to: out.transpose(1, 2), tamax: amax}, workspace)
        return out

    @mxfp8_sdpa.register_fake
    def _(q8, k8, v8, sfq, sfk, sfv):
        return q8.new_empty(q8.shape, dtype=torch.bfloat16)

    _OPS_REGISTERED = True


# --------------------------------------------------------------------------
# comfy override
# --------------------------------------------------------------------------

def _make_override():
    def mxfp8_override(func, q, k, v, heads, *args, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
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
            q_b, k_b, v_b = (x.transpose(1, 2) for x in (q, k, v))
        else:  # (B, T, H*D)
            if q.shape[-1] % heads or k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
                return fallback()
            dim_head = q.shape[-1] // heads
            q_b = q.view(q.shape[0], q.shape[1], heads, dim_head)
            k_b = k.view(k.shape[0], k.shape[1], heads, dim_head)
            v_b = v.view(v.shape[0], v.shape[1], heads, dim_head)
        # The native cuDNN MXFP8 engine is D128; other head dims (the audio
        # transformer's d=64) stay on the default backend — negligible math.
        if dim_head != 128 or k_b.shape[2] != heads:
            return fallback()

        q8, sfq = quantize_rowwise(q_b)
        k8, sfk = quantize_rowwise(k_b)
        v8, sfv = quantize_columnwise(v_b)
        out = torch.ops.ltxserver.mxfp8_sdpa(q8, k8, v8, sfq, sfk, sfv)
        if out.dtype != q.dtype:
            out = out.to(q.dtype)

        if skip_output_reshape:  # (B, H, T, D)
            return out.transpose(1, 2)
        return out.reshape(out.shape[0], out.shape[1], heads * dim_head)

    return mxfp8_override


def install_mxfp8_override(model_patcher, *, label: str) -> None:
    """Preflight (arch, cudnn frontend/backend, tiny graph build), then hook
    the MXFP8 override onto one model."""
    if not torch.cuda.is_available():
        raise RuntimeError("attention_backend: cudnn_mxfp8 needs a CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap not in ((10, 0), (10, 3)):
        raise RuntimeError(f"cudnn_mxfp8 needs Blackwell sm100/sm103 (B200/B300); "
                           f"this device is sm{cap[0]}{cap[1]}")
    cudnn = _import_cudnn()
    backend = cudnn.backend_version()
    if backend < 92100:
        raise RuntimeError(f"cudnn_mxfp8 needs cuDNN backend >= 9.21, found {backend} — "
                           "upgrade the nvidia-cudnn wheel torch loads")
    _register_ops()
    # Tiny real graph build catches engine/env problems at startup instead of
    # mid-request; serving shapes compile during warmup.
    _get_graph(1, 2, 256, 256, 128)
    options = model_patcher.model_options.setdefault("transformer_options", {})
    options["optimized_attention_override"] = _make_override()
    logger.info("[%s] cuDNN MXFP8 attention override installed (per-32-block E8M0 scales; "
                "d=128 unmasked calls only, everything else stays on the default backend; "
                "cudnn backend %s)", label, backend)
