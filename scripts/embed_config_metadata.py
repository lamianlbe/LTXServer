#!/usr/bin/env python3
"""Copy safetensors metadata (the ``config`` JSON) from one file to another.

ComfyUI's model detection needs the ``config`` metadata to build the exact
LTX-2.3 architecture: without it the LTXV branch falls back to older-model
defaults (6-row scale-shift tables instead of 9, wrong caption-projection
variants) and the load fails or silently mis-shapes. The base checkpoint
ships that metadata; ``ModelSave`` exports do NOT (they carry only
prompt/workflow) — so a stage-2 DiT export must have the base's config
grafted on before UNETLoader can load it:

    python scripts/embed_config_metadata.py \\
        --from-file 10Eros_v1-fp8mixed_learned.safetensors \\
        --file stage2_00001_.safetensors \\
        --output stage2_00001_config.safetensors

Tensor bytes are copied verbatim (streaming); only the header changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

COPY_KEYS = ("config", "model_version")


def read_header(path: Path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta, 8 + n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file", required=True, help="donor (e.g. the base checkpoint)")
    ap.add_argument("--file", required=True, help="file whose tensors are kept")
    ap.add_argument("--output", required=True)
    ap.add_argument("--keys", nargs="*", default=list(COPY_KEYS),
                    help=f"metadata keys to copy (default: {list(COPY_KEYS)})")
    args = ap.parse_args()

    donor = Path(args.from_file)
    src = Path(args.file)
    out = Path(args.output)
    _, donor_meta, _ = read_header(donor)
    header, meta, payload_start = read_header(src)

    copied = []
    for key in args.keys:
        if key in donor_meta:
            meta[key] = donor_meta[key]
            copied.append(key)
    if "config" not in copied:
        raise SystemExit(f"{donor} has no 'config' metadata — nothing useful to copy "
                         f"(donor metadata keys: {sorted(donor_meta)})")
    meta["ltxserver_config_from"] = donor.name

    blob = json.dumps({"__metadata__": meta, **header}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    payload_size = src.stat().st_size - payload_start
    print(f"copying metadata {copied} from {donor.name} -> {out.name} "
          f"({payload_size / 1e9:.1f} GB payload verbatim)")

    with src.open("rb") as fin, out.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        fin.seek(payload_start)
        shutil.copyfileobj(fin, fout, length=64 * 2**20)
    print("done")


if __name__ == "__main__":
    main()
