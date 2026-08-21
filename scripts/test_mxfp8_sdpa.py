#!/usr/bin/env python3
"""MXFP8 vs FA4 attention: quality + speed on real serving shapes (GPU).

For each shape, builds bf16 q/k/v, computes an fp32 SDPA ground truth, then
measures every available backend's error against it and its wall time
(including quantization — that is what serving pays):

  sdpa_bf16    torch SDPA on bf16 (the quality baseline the model was
               validated against — its error vs fp32 is the noise floor)
  fa4_bf16     flash_attn.cute bf16
  fa4_fp8      flash_attn.cute fp8, per-(batch,head) descales (the backend
               that desaturates output)
  cudnn_mxfp8  cuDNN Blackwell microscaled fp8 (per-32-block E8M0 scales)

Expected result: cudnn_mxfp8 error lands near the bf16 noise floor, an
order of magnitude below fa4_fp8, at a similar speed.

    python scripts/test_mxfp8_sdpa.py                # production shapes
    python scripts/test_mxfp8_sdpa.py --seq 14336    # 896x512 mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bench(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def report(name: str, out: torch.Tensor, ref: torch.Tensor, ms: float) -> None:
    diff = (out.float() - ref).abs()
    rel = diff.mean().item() / ref.abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0).item()
    print(f"  {name:<12} {ms:8.3f} ms   max_err {diff.max().item():.5f}   "
          f"mean_rel {rel:.2e}   cos {cos:.6f}")


def run_shape(b: int, h: int, s_q: int, s_kv: int, d: int) -> None:
    print(f"shape: B={b} H={h} S_q={s_q} S_kv={s_kv} D={d}")
    torch.manual_seed(0)
    dev = "cuda"
    q = (torch.randn(b, s_q, h, d, device=dev) * 0.7).to(torch.bfloat16)
    k = (torch.randn(b, s_kv, h, d, device=dev) * 0.7).to(torch.bfloat16)
    v = (torch.randn(b, s_kv, h, d, device=dev) * 0.7).to(torch.bfloat16)

    # fp32 ground truth (BHSD for sdpa)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2).contiguous()

    def sdpa_bf16():
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    report("sdpa_bf16", sdpa_bf16(), ref, bench(sdpa_bf16))

    try:
        from ltxserver.attention import _register_ops as fa4_register, fp8_quantize_for_fa4
        fa4_register()

        def fa4_bf16():
            return torch.ops.ltxserver.fa4_forward(q.contiguous(), k.contiguous(), v.contiguous())

        report("fa4_bf16", fa4_bf16(), ref, bench(fa4_bf16))

        def fa4_fp8():
            qf, qd = fp8_quantize_for_fa4(q)
            kf, kd = fp8_quantize_for_fa4(k)
            vf, vd = fp8_quantize_for_fa4(v)
            return torch.ops.ltxserver.fa4_fp8_forward(qf, kf, vf, qd, kd, vd)

        report("fa4_fp8", fa4_fp8(), ref, bench(fa4_fp8))
    except ImportError as err:
        print(f"  fa4          skipped ({err})")

    try:
        from ltxserver.attention_mxfp8 import (
            _register_ops as mx_register, quantize_columnwise, quantize_rowwise)
        mx_register()

        def mxfp8():
            q8, sfq = quantize_rowwise(q)
            k8, sfk = quantize_rowwise(k)
            v8, sfv = quantize_columnwise(v)
            return torch.ops.ltxserver.mxfp8_sdpa(q8, k8, v8, sfq, sfk, sfv)

        report("cudnn_mxfp8", mxfp8(), ref, bench(mxfp8))
    except Exception as err:  # noqa: BLE001
        print(f"  cudnn_mxfp8  FAILED: {err}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--seq", type=int, default=28160, help="video tokens (1280x704x249 mode)")
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    print(f"gpu: {torch.cuda.get_device_name(0)}  sm{'.'.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"torch {torch.__version__}  cudnn backend {torch.backends.cudnn.version()}")
    try:
        import cudnn
        print(f"cudnn-frontend {cudnn.__version__}  (backend_version {cudnn.backend_version()})")
    except ImportError:
        print("cudnn-frontend NOT installed — pip install 'nvidia-cudnn-frontend[cutedsl]'")
    print()

    run_shape(args.batch, args.heads, args.seq, args.seq, args.dim)   # self-attn
    run_shape(args.batch, args.heads, args.seq, 1024, args.dim)       # text cross-attn


if __name__ == "__main__":
    main()
