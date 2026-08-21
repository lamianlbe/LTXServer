"""torch.compile support for the embedded ComfyUI models.

Two obstacles stand between comfy's LTX models and useful compilation, both
solved here without touching comfy source:

1. The fp8 layers hold ``QuantizedTensor`` weights whose forward constructs a
   layout ``Params`` dataclass per call — dynamo cannot build that object and
   graph-breaks TWICE per quantized linear (~2500 splits per DiT forward,
   measured). ``prepare_model_for_compile`` swaps each such layer for
   :class:`Fp8ScaledLinear`, a plain-tensor equivalent of the exact same
   numerics (input quantized at scale 1.0 with a saturating e4m3 cast, fp8
   GEMM through comfy_kitchen's own ``scaled_mm_v2`` with the bias fused,
   bf16 out). Every swap is probe-verified BIT-IDENTICAL against the layer
   it replaces before it is accepted; the eager (compile: false) path keeps
   stock comfy modules end to end.

2. Compilation itself goes through comfy's official
   ``set_torch_compile_wrapper`` (object-swap at APPLY_MODEL time), either
   whole-model or per-transformer-block ("blocks" scope: 48 small shared-code
   graphs, faster warmup, kernel cache shared across blocks).

Inductor/dynamo knobs mirror the FastVideo LTX-2.3 production server —
notably ``shape_padding = False``, which is load-bearing on Blackwell
(pad_mm crashes).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ltxserver.perf")

_FP8_LAYOUTS = ("TensorCoreFP8E4M3Layout", "TensorCoreFP8Layout")


def apply_inductor_settings() -> None:
    """Process-wide compile settings, applied once before the first compile."""
    import torch
    import torch._inductor.config as inductor

    inductor.shape_padding = False  # mandatory on Blackwell (pad_mm crash)
    inductor.conv_1x1_as_mm = True
    inductor.coordinate_descent_tuning = True
    inductor.coordinate_descent_check_all_directions = True
    inductor.epilogue_fusion = False
    # Shapes accumulate per code object: modes x stages x (cond batched with
    # uncond or not) x blocks sharing one forward. Keep well clear of the
    # default limit so dynamo never silently falls back to eager.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)


class Fp8ScaledLinear:  # replaced below — kept as a name for type checks
    pass


def _build_fp8_linear_class():
    """Deferred so importing this module never drags torch in early."""
    import torch
    from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

    global Fp8ScaledLinear

    class _Fp8ScaledLinear(torch.nn.Module):
        """Compile-friendly twin of a comfy MixedPrecisionOps fp8 Linear.

        Same math as comfy_kitchen's dispatch chain for a
        ``float8_e4m3fn``-format layer with no input_scale: activation cast
        to e4m3 at scale 1.0 (saturating), ``scaled_mm_v2(x_q, w_q.t(),
        1.0, w_scale, bias, out_dtype)`` — bias added on the fp32
        accumulator, one rounding to the output dtype. Plain tensors and
        aten ops only, so dynamo captures the whole thing in one graph.
        """

        FP8_MAX = 448.0

        def __init__(self, payload: torch.Tensor, scale: torch.Tensor,
                     bias: torch.Tensor | None, out_dtype: torch.dtype):
            super().__init__()
            self.register_buffer("weight_fp8", payload.contiguous(), persistent=False)
            self.register_buffer("weight_scale", scale.detach().to(torch.float32).reshape(()),
                                 persistent=False)
            self.register_buffer("input_scale_one",
                                 torch.ones((), dtype=torch.float32, device=payload.device),
                                 persistent=False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias.detach().clone(), requires_grad=False)
            else:
                self.bias = None
            self.out_dtype = out_dtype
            self.out_features = payload.shape[0]

        @property
        def weight(self):
            # comfy.ops.linear_input_act reads ``linear.weight`` to detect its
            # fused-INT8 fast path (the FFN down-projection goes through it).
            # Hand back the raw fp8 payload — a plain tensor, NOT a
            # QuantizedTensor — so that caller (and any other introspection)
            # falls through to ``linear(act(x))``, i.e. our forward.
            return self.weight_fp8

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            # Round to the compute dtype BEFORE the fp8 cast: eager comfy
            # materializes the incoming activation (e.g. the gelu in
            # linear_input_act) as bf16 before quantizing, so keep that
            # boundary under fusion too (no-op in eager). NOTE: compiled
            # mode is still not bit-identical to eager — inductor's
            # elementwise math (libdevice tanh etc.) differs in fp32 ULPs,
            # which occasionally flips an fp8 rounding bucket (~one fp8 ULP
            # per affected element). That is inherent to torch.compile;
            # quality is gated by the same-seed bench A/B, and the SWAP
            # itself (compile: false) remains bit-identical.
            x2 = x.reshape(-1, shape[-1]).to(self.out_dtype)
            x_q = x2.clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn).contiguous()
            out = scaled_mm_v2(
                x_q,
                self.weight_fp8.t(),
                scale_a=self.input_scale_one,
                scale_b=self.weight_scale,
                bias=self.bias,
                out_dtype=self.out_dtype,
            )
            return out.reshape(*shape[:-1], self.out_features)

    Fp8ScaledLinear = _Fp8ScaledLinear
    return _Fp8ScaledLinear


def _build_fp8_dequant_linear_classes():
    """Weight-only fp8 twins: dequantize the weight, run a plain GEMM.

    comfy runs TEXT ENCODERS this way (activations stay bf16; only the
    weight is stored fp8), while diffusion models run the fp8 x fp8
    scaled_mm path — the probe ladder keeps whichever twin matches the
    real layer bit-for-bit. Two dequant rounding orders exist in the
    wild, so both variants are offered: multiply in fp32 then cast, or
    cast the payload first and multiply in the compute dtype.
    """
    import torch

    class _DequantBase(torch.nn.Module):
        def __init__(self, payload, scale, bias, out_dtype):
            super().__init__()
            self.register_buffer("weight_fp8", payload.contiguous(), persistent=False)
            self.register_buffer("weight_scale", scale.detach().to(torch.float32).reshape(()),
                                 persistent=False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias.detach().clone(), requires_grad=False)
            else:
                self.bias = None
            self.out_dtype = out_dtype

        @property
        def weight(self):
            # Type-check compatibility for comfy.ops.linear_input_act (a
            # plain tensor routes that caller into ``linear(act(x))``).
            return self.weight_fp8

    class Fp8DequantF32Linear(_DequantBase):
        def forward(self, x):
            w = (self.weight_fp8.to(torch.float32) * self.weight_scale).to(x.dtype)
            return torch.nn.functional.linear(x, w, self.bias)

    class Fp8DequantCastLinear(_DequantBase):
        def forward(self, x):
            w = self.weight_fp8.to(x.dtype) * self.weight_scale.to(x.dtype)
            return torch.nn.functional.linear(x, w, self.bias)

    return (Fp8DequantF32Linear, Fp8DequantCastLinear)


def swap_fp8_linears(root_module, label: str) -> int:
    """Swap QuantizedTensor fp8 linears under ``root_module`` for
    compile-friendly twins, in place.

    Every swapped layer is verified BIT-IDENTICAL to the module it replaces
    on a random probe input before the swap is kept; any deviation aborts
    (a comfy/comfy_kitchen upgrade changing fp8 numerics must fail loudly
    here, never ship silently different pixels).
    """
    import torch
    import comfy.utils
    from comfy.quant_ops import QuantizedTensor

    cls = _build_fp8_linear_class()

    swapped = 0
    kind_counts: dict[str, int] = {}
    with torch.no_grad():
        for name, module in list(root_module.named_modules()):
            weight = getattr(module, "weight", None)
            if not isinstance(weight, QuantizedTensor):
                continue
            layout = getattr(weight, "_layout_cls", None)
            if layout not in _FP8_LAYOUTS:
                raise RuntimeError(f"{label}.{name}: unsupported quantized layout {layout!r} "
                                   "for compile (only per-tensor fp8 is handled)")
            payload, scale = _plain_tensors(weight)
            bias = getattr(module, "bias", None)
            out_dtype = bias.dtype if bias is not None else torch.bfloat16

            probe = torch.randn(2, 16, payload.shape[1], device=payload.device,
                                dtype=out_dtype) * 3.0
            import comfy.ops as comfy_ops
            ref_fwd = module(probe)
            ref_act = comfy_ops.linear_input_act(module, probe, "gelu_tanh")

            # comfy picks the compute path per MODEL: diffusion models run
            # fp8 x fp8 scaled_mm, text encoders dequantize the weight and
            # run a bf16 GEMM. Offer one twin per known semantics and keep
            # whichever matches bit-for-bit on BOTH probes (the plain
            # forward, and comfy.ops.linear_input_act — the FFN
            # down-projection caller that reads .weight first).
            dequant_f32_cls, dequant_cast_cls = _build_fp8_dequant_linear_classes()
            chosen = None
            diffs = []
            for kind, twin_cls in (("scaled_mm", cls),
                                   ("dequant_f32", dequant_f32_cls),
                                   ("dequant_cast", dequant_cast_cls)):
                twin = twin_cls(payload, scale, bias, out_dtype).to(payload.device)
                got_fwd = twin(probe)
                got_act = comfy_ops.linear_input_act(twin, probe, "gelu_tanh")
                if torch.equal(ref_fwd, got_fwd) and torch.equal(ref_act, got_act):
                    chosen = (kind, twin)
                    break
                diffs.append((kind, (ref_fwd.float() - got_fwd.float()).abs().max().item()))
            if chosen is None:
                raise RuntimeError(
                    f"{label}.{name}: no compile-friendly fp8 twin matches the comfy layer "
                    f"bit-for-bit (max forward diffs: {diffs}); refusing to compile. A comfy/"
                    "comfy_kitchen upgrade likely changed fp8 numerics — re-verify.")
            kind_counts[chosen[0]] = kind_counts.get(chosen[0], 0) + 1
            comfy.utils.set_attr(root_module, name, chosen[1])
            swapped += 1

    logger.info("[%s] %d fp8 linear(s) swapped for compile-friendly twins "
                "(probe: bit-identical; kinds: %s)", label, swapped, kind_counts or {})
    return swapped


def prepare_model_for_compile(model_patcher, label: str) -> int:
    """fp8-twin swap over a ModelPatcher's diffusion model."""
    return swap_fp8_linears(model_patcher.get_model_object("diffusion_model"), label)

def _plain_tensors(weight):
    """(payload, scale) from a QuantizedTensor via its layout class."""
    from comfy.quant_ops import get_layout_class
    layout_cls = get_layout_class(weight._layout_cls)
    return layout_cls.get_plain_tensors(weight)


def compile_model(model_patcher, *, scope: str, label: str) -> None:
    """torch.compile a DiT, FastVideo-style.

    ``blocks`` (default) patches each transformer block's BOUND ``forward``
    in place. Every block — across BOTH stage models — shares one forward
    code object, so dynamo traces a single block per served shape and
    reuses the entry everywhere; per-shape warmup covers one block's trace
    instead of a fully-inlined 48-block model (the difference between
    ~minutes and ~tens of minutes, measured on FastVideo). The in-place
    patch survives ModelPatcher clones (guiders clone the patcher but share
    the nn.Modules), so no APPLY_MODEL wrapper is needed.

    ``model`` compiles the whole diffusion_model through comfy's official
    wrapper — only sensible for guide-free recipes (see compile_scope).
    """
    import torch

    if scope == "blocks":
        diffusion_model = model_patcher.get_model_object("diffusion_model")
        blocks = list(diffusion_model.transformer_blocks)
        # Shared code objects accumulate one dynamo entry per served shape —
        # and the ACCUMULATED limit counts every entry across all code
        # objects (DiT blocks x shapes + TE + VAE). Overflow is a silent
        # eager fallback; raise both proportionally to the fan-out, and only
        # ever raise (import order must not clobber a higher value).
        cfg = torch._dynamo.config
        needed = max(64, 16 * len(blocks))
        cfg.recompile_limit = max(cfg.recompile_limit, needed)
        cfg.accumulated_recompile_limit = max(cfg.accumulated_recompile_limit, 4 * needed)
        for block in blocks:
            block.forward = torch.compile(block.forward, backend="inductor", mode="default",
                                          fullgraph=False, dynamic=False)
        logger.info("[%s] torch.compile attached to %d block forwards "
                    "(shared code object — one trace per shape)", label, len(blocks))
    elif scope == "model":
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        set_torch_compile_wrapper(model_patcher, backend="inductor", mode="default",
                                  fullgraph=False, dynamic=False, keys=["diffusion_model"])
        logger.info("[%s] torch.compile attached (whole diffusion_model)", label)
    else:
        raise ValueError(f"compile_scope must be 'model' or 'blocks', got {scope!r}")


def log_dynamo_counters(tag: str, log=logger.info) -> None:
    """One-line dynamo health readout (graphs captured / graph breaks)."""
    from torch._dynamo.utils import counters

    stats = counters["stats"]
    breaks = sum(counters["graph_break"].values())
    log("[%s] dynamo: %s graph(s) captured, %s graph break(s)%s", tag,
        stats.get("unique_graphs", 0), breaks,
        "" if not breaks else " — top: " + "; ".join(
            f"{k.strip()[:90]} x{v}" for k, v in
            sorted(counters["graph_break"].items(), key=lambda kv: -kv[1])[:3]))


def _build_singleshot_conv_class():
    import torch

    class SingleShotCausalConv3d(torch.nn.Module):
        """Stateless twin of comfy's CausalConv3d for single-shot runs.

        The original keys a temporal streaming cache by threading.get_ident()
        on EVERY forward — an untraceable builtin that graph-breaks dynamo at
        each conv. With the chunk budget unlimited every forward sees the
        whole sequence, so the cache is semantically dead: the fresh-state +
        ended path reduces to replicate-padding the first frame (and, for
        non-causal calls, the last frame) and running the conv. This module
        is exactly that math and nothing else — pure aten, fully traceable.

        Deliberately NOT a CausalConv3d subclass and carries no
        ``temporal_cache_state``: comfy's mark_conv3d_ended / cache-pop walks
        skip it by isinstance/hasattr.
        """

        def __init__(self, orig):
            super().__init__()
            self.conv = orig.conv
            self.time_kernel_size = orig.time_kernel_size
            self.out_channels = orig.out_channels

        def forward(self, x, causal: bool = True):
            import torch as _t
            padding_length = self.time_kernel_size - 1
            if not causal:
                padding_length = padding_length // 2
            pieces = [x[:, :, :1, :, :].repeat((1, 1, padding_length, 1, 1)), x]
            if not causal:
                pieces.append(x[:, :, -1:, :, :].repeat(
                    (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)))
            return self.conv(_t.cat(pieces, dim=2))

    return SingleShotCausalConv3d


def swap_causal_convs(first_stage_model, label: str) -> int:
    """Swap every CausalConv3d for its stateless single-shot twin, in place.

    Probe-verified BIT-IDENTICAL against the original conv in its
    single-shot state (fresh cache + ended). Only valid when the VAE chunk
    budget is unlimited — the caller enforces that.
    """
    import threading

    import torch
    import comfy.utils
    from comfy.ldm.lightricks.vae.causal_conv3d import CausalConv3d

    cls = _build_singleshot_conv_class()
    device = next(first_stage_model.parameters()).device
    dtype = next(first_stage_model.parameters()).dtype
    tid = threading.get_ident()

    swapped = 0
    with torch.no_grad():
        for name, module in list(first_stage_model.named_modules()):
            if not isinstance(module, CausalConv3d):
                continue
            twin = cls(module)
            in_ch = module.conv.in_channels
            probe = torch.randn(1, in_ch, 5, 8, 8, device=device, dtype=dtype)
            for causal in (True, False):
                module.temporal_cache_state[tid] = (None, True)  # fresh + ended
                ref = module(probe, causal=causal)
                module.temporal_cache_state.pop(tid, None)
                got = twin(probe, causal=causal)
                if not torch.equal(ref, got):
                    max_diff = (ref.float() - got.float()).abs().max().item()
                    raise RuntimeError(
                        f"{label}.{name} (causal={causal}): single-shot conv twin is NOT "
                        f"bit-identical (max diff {max_diff}); refusing to compile the VAE.")
            comfy.utils.set_attr(first_stage_model, name, twin)
            swapped += 1

    logger.info("[%s] %d causal conv(s) swapped for stateless single-shot twins "
                "(probe: bit-identical)", label, swapped)
    return swapped


def compile_vae_codec(vae, *, label: str) -> None:
    """Compile the video VAE's decode AND encode paths.

    Requires single-shot mode (unlimited chunk budget): shapes are then
    static per mode, and after the causal convs are swapped for stateless
    twins nothing on the path graph-breaks. Comfy's OOM->tiled retry at the
    wrapper level still works (it calls the same compiled methods).
    """
    import torch

    model = getattr(vae, "first_stage_model", None)
    if model is None or not hasattr(model, "decode"):
        logger.warning("[%s] VAE has no first_stage_model.decode — not compiled", label)
        return
    swap_causal_convs(model, label)
    model.decode = torch.compile(model.decode, backend="inductor", mode="default",
                                 fullgraph=False, dynamic=False)
    if hasattr(model, "encode"):
        model.encode = torch.compile(model.encode, backend="inductor", mode="default",
                                     fullgraph=False, dynamic=False)
    logger.info("[%s] torch.compile attached to the video VAE decode + encode", label)


def compile_text_encoder(clip, *, label: str) -> None:
    """fp8-twin swap + torch.compile for the LTXAV gemma text encoder.

    The tokenizer left-pads every prompt to 1024 tokens, so shapes are
    static. Works with bf16 TEs (nothing to swap) and comfy_quant
    fp8_e4m3fn TEs (per-tensor scaled — the same layout as the DiT, handled
    by the same bit-verified twin swap). Block-scaled formats (mxfp8/nvfp4)
    are rejected by the swap's layout check with a clear error.
    """
    import torch

    te_model = getattr(clip, "cond_stage_model", None)
    if te_model is None:
        logger.warning("[%s] clip has no cond_stage_model — not compiled", label)
        return
    swap_fp8_linears(te_model, label)
    gemma = getattr(te_model, "gemma3_12b", None)
    transformer = getattr(gemma, "transformer", None)
    if transformer is None:
        logger.warning("[%s] gemma transformer not found — not compiled", label)
        return
    gemma.transformer = torch.compile(transformer, backend="inductor", mode="default",
                                      fullgraph=False, dynamic=False)
    logger.info("[%s] torch.compile attached to the gemma text encoder", label)

def set_vae_chunk_budget(chunk_mib: int) -> None:
    """Raise the causal video VAE's temporal chunk budget.

    comfy's ``get_max_chunk_size`` caps the per-level activation chunk at
    128 MiB on any GPU with >= 24GB VRAM — a B200-class decode gets split
    into hundreds of tiny temporal chunks whose launch and conv-cache
    overhead dominate the wall time. Chunked and unchunked decodes are
    exactly equivalent (streaming causal convolutions carry state across
    chunk edges), so the budget is a pure speed/VRAM trade. Overwrites the
    module constants comfy reads on every decode; both bounds are set so
    the heuristic returns the requested budget regardless of VRAM size.
    """
    from comfy.ldm.lightricks.vae import causal_video_autoencoder as cva

    if chunk_mib < 0:
        budget = 2 ** 62  # effectively unlimited: single-shot decode/encode
        note = "unlimited (single-shot)"
    else:
        budget = int(chunk_mib) * 1024 ** 2
        note = f"{chunk_mib} MiB"
    logger.info("video VAE chunk budget: %d MiB -> %s",
                cva.MAX_CHUNK_SIZE // 1024 ** 2, note)
    cva.MIN_CHUNK_SIZE = budget
    cva.MAX_CHUNK_SIZE = budget
