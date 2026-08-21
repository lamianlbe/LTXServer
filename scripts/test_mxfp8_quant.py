#!/usr/bin/env python3
"""Bit-verify ltxserver's BSHD-native MXFP8 quantizer against the vendored
NVIDIA reference (scripts/mxfp8_quant_reference.py, MIT — the exact
generator cudnn-frontend's own sdpa_mxfp8 tests use).

The reference quantizes BHSD-contiguous input; ours quantizes the
BSHD-contiguous layout comfy hands the attention override, producing the
same fp8 payload and the same F8_128x4-swizzled E8M0 scale bytes. Any
mismatch here means wrong scales reaching the cuDNN kernel — so this must
pass EXACTLY (torch.equal), not approximately.

CPU-only, no cudnn needed:

    python scripts/test_mxfp8_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltxserver.attention_mxfp8 import quantize_columnwise, quantize_rowwise  # noqa: E402
from mxfp8_quant_reference import quantize_to_mxfp8  # noqa: E402


def check(b: int, h: int, s: int, d: int, seed: int) -> None:
    torch.manual_seed(seed)
    # bf16 like production, with outliers to exercise the satfinite clamp
    t_bshd = (torch.randn(b, s, h, d, dtype=torch.float32) * 3.0)
    t_bshd[0, 0, 0, :4] = 3000.0  # amax outlier
    t_bshd[0, min(1, s - 1), 0, :] = 0.0  # all-zero block -> e8m0 0x00
    t_bshd = t_bshd.to(torch.bfloat16)
    t_bhsd = t_bshd.permute(0, 2, 1, 3).contiguous()

    dsc = -(-(d // 32) // 4) * 4
    s_pad = -(-s // 128) * 128
    ssc = -(-(-(-s // 32)) // 4) * 4
    d_pad = -(-d // 128) * 128

    ref_d, _, ref_sfd, ref_s, _, ref_sfs = quantize_to_mxfp8(
        t_bhsd, b, h, s, d, 32, torch.float8_e4m3fn, with_ref=False)

    my_d, my_sfd = quantize_rowwise(t_bshd)
    my_s, my_sfs = quantize_columnwise(t_bshd)

    ref_d_bshd = ref_d.permute(0, 2, 1, 3).contiguous()
    ref_s_bshd = ref_s.permute(0, 2, 1, 3).contiguous()
    ref_sfd_g = ref_sfd.view(torch.uint8).reshape(b, h, s_pad, dsc)
    ref_sfs_g = ref_sfs.view(torch.uint8).reshape(b, h, ssc, d_pad)

    tag = f"b={b} h={h} s={s} d={d}"
    assert torch.equal(my_d.view(torch.uint8), ref_d_bshd.view(torch.uint8)), \
        f"{tag}: rowwise fp8 payload mismatch"
    assert torch.equal(my_sfd, ref_sfd_g), f"{tag}: rowwise swizzled scales mismatch"
    assert torch.equal(my_s.view(torch.uint8), ref_s_bshd.view(torch.uint8)), \
        f"{tag}: columnwise fp8 payload mismatch"
    assert torch.equal(my_sfs, ref_sfs_g), f"{tag}: columnwise swizzled scales mismatch"
    print(f"OK {tag}")


def main() -> None:
    check(1, 2, 256, 128, 0)      # aligned everything
    check(2, 4, 300, 128, 1)      # S not 32- or 128-aligned
    check(1, 3, 1024, 128, 2)     # text cross-attn kv length
    check(1, 2, 257, 64, 3)       # d=64 (audio) + odd S
    check(2, 1, 128, 128, 4)
    check(1, 32, 1408, 128, 5)    # production-ish head count, S%128=0
    print("ALL QUANTIZER CHECKS PASSED (bit-exact vs NVIDIA reference)")


if __name__ == "__main__":
    main()
