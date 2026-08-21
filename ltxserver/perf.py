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


def prepare_model_for_compile(model_patcher, label: str) -> int:
    """Swap QuantizedTensor fp8 linears for compile-friendly twins, in place.

    Every swapped layer is verified BIT-IDENTICAL to the module it replaces
    on a random probe input before the swap is kept; any deviation aborts
    (a comfy/comfy_kitchen upgrade changing fp8 numerics must fail loudly
    here, never ship silently different pixels).
    """
    import torch
    import comfy.utils
    from comfy.quant_ops import QuantizedTensor

    cls = _build_fp8_linear_class()
    diffusion_model = model_patcher.get_model_object("diffusion_model")

    swapped = 0
    with torch.no_grad():
        for name, module in list(diffusion_model.named_modules()):
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
            twin = cls(payload, scale, bias, out_dtype).to(payload.device)

            probe = torch.randn(2, 16, payload.shape[1], device=payload.device,
                                dtype=out_dtype) * 3.0
            # Two probes: the plain forward, and comfy.ops.linear_input_act —
            # the one external caller that reaches around the module's forward
            # (the FFN down-projection), which reads .weight first.
            import comfy.ops as comfy_ops
            for tag, ref, got in (
                ("forward", module(probe), twin(probe)),
                ("linear_input_act", comfy_ops.linear_input_act(module, probe, "gelu_tanh"),
                 comfy_ops.linear_input_act(twin, probe, "gelu_tanh")),
            ):
                if not torch.equal(ref, got):
                    max_diff = (ref.float() - got.float()).abs().max().item()
                    raise RuntimeError(
                        f"{label}.{name} [{tag}]: compile-friendly fp8 linear is NOT bit-identical "
                        f"to the comfy layer it would replace (max diff {max_diff}); refusing to "
                        "compile. A comfy/comfy_kitchen upgrade likely changed fp8 numerics — re-verify.")
            comfy.utils.set_attr(diffusion_model, name, twin)
            swapped += 1

    logger.info("[%s] %d fp8 linear(s) swapped for compile-friendly twins (probe: bit-identical)",
                label, swapped)
    return swapped


def _plain_tensors(weight):
    """(payload, scale) from a QuantizedTensor via its layout class."""
    from comfy.quant_ops import get_layout_class
    layout_cls = get_layout_class(weight._layout_cls)
    return layout_cls.get_plain_tensors(weight)


def compile_model(model_patcher, *, scope: str, label: str) -> None:
    """Attach comfy's official torch.compile wrapper to a ModelPatcher."""
    from comfy_api.torch_helpers import set_torch_compile_wrapper

    if scope == "blocks":
        diffusion_model = model_patcher.get_model_object("diffusion_model")
        num_blocks = len(diffusion_model.transformer_blocks)
        keys = [f"diffusion_model.transformer_blocks.{i}" for i in range(num_blocks)]
    elif scope == "model":
        keys = ["diffusion_model"]
    else:
        raise ValueError(f"compile_scope must be 'model' or 'blocks', got {scope!r}")

    set_torch_compile_wrapper(model_patcher, backend="inductor", mode="default",
                              fullgraph=False, dynamic=False, keys=keys)
    logger.info("[%s] torch.compile attached (scope=%s, %d module(s))", label, scope, len(keys))


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


def compile_vae_decode(vae, *, label: str) -> None:
    """torch.compile the video VAE's decode path.

    The comfy ``VAE`` wrapper calls ``self.first_stage_model.decode(...)``;
    wrapping that bound method compiles the whole decoder while leaving
    comfy's tiling/memory fallbacks (which call the same method) intact.
    Decode shapes are fixed per mode, so warmup covers every graph.
    """
    import torch

    model = getattr(vae, "first_stage_model", None)
    if model is None or not hasattr(model, "decode"):
        logger.warning("[%s] VAE has no first_stage_model.decode — not compiled", label)
        return
    model.decode = torch.compile(model.decode, backend="inductor", mode="default",
                                 fullgraph=False, dynamic=False)
    logger.info("[%s] torch.compile attached to the video VAE decode", label)
