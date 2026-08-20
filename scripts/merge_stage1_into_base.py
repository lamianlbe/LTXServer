#!/usr/bin/env python3
"""Fold the stage-1 merged DiT back into the all-in-one base checkpoint.

The reference workflow builds two models at load time by stacking LoRAs on
the base checkpoint. This server loads pre-merged weights instead, from
exactly two files:

  * the BASE checkpoint with its ``model.diffusion_model.*`` payload
    REPLACED by the stage-1 ComfyUI ``ModelSave`` export (this script's
    output) — VAE, audio VAE, vocoder, text-embedding connectors and all
    quantization sidecars stay byte-identical to the base;
  * the stage-2 ``ModelSave`` export, loaded standalone via UNETLoader.

Everything is copied as raw bytes — fp8 payloads, ``.weight_scale`` and
``.comfy_quant`` sidecars ride along untouched, so the merged file keeps
the exact per-layer mixed-precision profile ComfyUI saved.

Key naming: ``ModelSave`` writes comfy's INTERNAL module names, which may
differ from the base checkpoint's on-disk names for the gated-attention
weight (``to_gate_logits`` vs ``to_gate_compress``). The base's naming
wins — each base DiT key is sourced from the export under either spelling.

    python scripts/merge_stage1_into_base.py \\
        --base 10Eros_v1-fp8mixed_learned.safetensors \\
        --stage1 stage1_00001_.safetensors \\
        --output 10Eros_v1-stage1merged.safetensors

Streaming: peak RAM is one tensor.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from safetensors import safe_open

DIT_PREFIX = "model.diffusion_model."
GATE_SWAPS = (("to_gate_compress", "to_gate_logits"), ("to_gate_logits", "to_gate_compress"))
DTYPE_SIZE = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2,
              "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta


def nbytes(dtype: str, shape) -> int:
    size = DTYPE_SIZE[dtype]
    for d in shape:
        size *= d
    return size


def export_key_for(base_key: str, export_header: dict) -> str | None:
    if base_key in export_header:
        return base_key
    for old, new in GATE_SWAPS:
        if old in base_key:
            candidate = base_key.replace(old, new)
            if candidate in export_header:
                return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="original all-in-one checkpoint")
    ap.add_argument("--stage1", required=True, help="ComfyUI ModelSave export of the stage-1 merge")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_path, export_path, out_path = Path(args.base), Path(args.stage1), Path(args.output)
    base_header, base_meta = read_header(base_path)
    export_header, _ = read_header(export_path)

    plan: list[tuple[str, str, dict, str]] = []  # (out_key, src_key, entry, src)
    consumed: set[str] = set()
    missing: list[str] = []
    replaced = kept = fp8 = 0
    for key in base_header:
        if key.startswith(DIT_PREFIX):
            src_key = export_key_for(key, export_header)
            if src_key is None:
                missing.append(key)
                continue
            entry = export_header[src_key]
            plan.append((key, src_key, entry, "export"))
            consumed.add(src_key)
            replaced += 1
            if entry["dtype"].startswith("F8_"):
                fp8 += 1
        else:
            plan.append((key, key, base_header[key], "base"))
            kept += 1

    if missing:
        raise SystemExit(f"{len(missing)} base DiT key(s) have no counterpart in the export "
                         f"(same name or gate-swapped): {missing[:8]}")
    leftover = sorted(set(export_header) - consumed)
    if leftover:
        raise SystemExit(f"{len(leftover)} export key(s) were not consumed — the export carries "
                         f"tensors the base lacks: {leftover[:8]}")

    out_header, offset = {}, 0
    for out_key, _src_key, entry, _src in plan:
        size = nbytes(entry["dtype"], entry["shape"])
        out_header[out_key] = {"dtype": entry["dtype"], "shape": entry["shape"],
                               "data_offsets": [offset, offset + size]}
        offset += size

    meta = dict(base_meta)
    meta["ltxserver_merge"] = json.dumps({"base": base_path.name, "stage1": export_path.name})
    blob = json.dumps({"__metadata__": meta, **out_header}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    print(f"base    : {base_path}  ({len(base_header)} tensors)")
    print(f"stage1  : {export_path}  ({len(export_header)} tensors)")
    print(f"output  : {out_path}  ({replaced} DiT tensors replaced [{fp8} fp8], "
          f"{kept} non-DiT kept verbatim, {offset / 1e9:.1f} GB)")

    import torch
    with safe_open(str(base_path), framework="pt", device="cpu") as bf, \
         safe_open(str(export_path), framework="pt", device="cpu") as ef, \
         out_path.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for i, (_out_key, src_key, _entry, src) in enumerate(plan):
            handle = ef if src == "export" else bf
            tensor = handle.get_tensor(src_key)
            out.write(tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes())
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
