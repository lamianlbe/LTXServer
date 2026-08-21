#!/usr/bin/env python3
"""Build a FULL checkpoint from a full base + a DiT-only ModelSave export.

Takes any complete checkpoint (``--base``) and replaces its
``model.diffusion_model.*`` payload with a ComfyUI ``ModelSave`` export
(``--dit``). Everything else — VAE, audio VAE, vocoder, text-embedding
connectors, the safetensors ``config``/``model_version`` metadata and all
quantization sidecars — stays byte-identical to the base. The output loads
through comfy's normal checkpoint path (CheckpointLoaderSimple), no
metadata grafting or UNETLoader needed.

The reference workflow builds two models at load time by stacking LoRAs on
one base checkpoint; this server loads pre-merged weights instead. Run
this script once per stage:

    # stage 1: fold the stage-1 merge into the original base
    python scripts/merge_dit_into_checkpoint.py \\
        --base 10Eros_v1-fp8mixed_learned.safetensors \\
        --dit stage1_00001_.safetensors \\
        --output 10Eros_v1-stage1merged.safetensors

    # stage 2: same operation — the stage-1 merged file works as the base
    # (its non-DiT payload is byte-identical to the original base's)
    python scripts/merge_dit_into_checkpoint.py \\
        --base 10Eros_v1-stage1merged.safetensors \\
        --dit 10Eros_v1-stage2.safetensors \\
        --output 10Eros_v1-stage2merged.safetensors

Everything is copied as raw bytes — fp8 payloads, ``.weight_scale`` and
``.comfy_quant`` sidecars ride along untouched, so the merged file keeps
the exact per-layer mixed-precision profile ComfyUI saved. A ``--dit``
file that already carries grafted metadata (embed_config_metadata.py) is
fine: only its tensors are read, the base's metadata wins.

Key naming: ``ModelSave`` writes comfy's INTERNAL module names, which may
differ from the base checkpoint's on-disk names for the gated-attention
weight (``to_gate_logits`` vs ``to_gate_compress``). The base's naming
wins — each base DiT key is sourced from the export under either spelling.

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
    ap.add_argument("--base", required=True, help="complete checkpoint donating everything but the DiT")
    ap.add_argument("--dit", "--stage1", dest="dit", required=True,
                    help="ComfyUI ModelSave export whose DiT tensors replace the base's")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_path, export_path, out_path = Path(args.base), Path(args.dit), Path(args.output)
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
    provenance = {"base": base_path.name, "dit": export_path.name}
    if "ltxserver_merge" in meta:  # base was itself a merge — keep the chain
        provenance["parent"] = json.loads(meta["ltxserver_merge"])
    meta["ltxserver_merge"] = json.dumps(provenance)
    blob = json.dumps({"__metadata__": meta, **out_header}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    print(f"base    : {base_path}  ({len(base_header)} tensors)")
    print(f"dit     : {export_path}  ({len(export_header)} tensors)")
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
