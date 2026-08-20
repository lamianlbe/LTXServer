#!/usr/bin/env python3
"""Timed A/B benchmark: one server config, optional field overrides, N runs.

Boots the engine exactly like the server, warms up ONE mode, then times N
generations and (optionally) writes each output mp4 — so two invocations
with different overrides give a clean same-seed speed + quality comparison:

    python scripts/bench.py --config config.yaml --tag sdpa
    python scripts/bench.py --config config.yaml --tag fa4 \\
        --set attention_backend=fa4
    python scripts/bench.py --config config.yaml --tag fa4fp8_compiled \\
        --set attention_backend=fa4 --set fa4_fp8_stage2=true --set compile=true

Overrides accept bool/int/float/str ServerConfig fields. Outputs land in
--out-dir as <tag>_run<i>.mp4 plus a timing summary line per run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _coerce(value: str, current):
    if isinstance(current, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"expected a bool, got {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="FIELD=VALUE", help="override a ServerConfig field")
    ap.add_argument("--mode-index", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=None, help="default: the warmup prompt")
    ap.add_argument("--first-frame", default=None, help="default: synthetic gradient")
    ap.add_argument("--out-dir", default="bench_out")
    ap.add_argument("--no-save", action="store_true", help="skip mp4 encode/save")
    args = ap.parse_args()

    from ltxserver.config import WARMUP_PROMPT, load_config, validate_config
    cfg = load_config(args.config)
    for item in args.overrides:
        field_name, _, value = item.partition("=")
        if not hasattr(cfg, field_name):
            raise SystemExit(f"--set {item}: ServerConfig has no field {field_name!r}")
        setattr(cfg, field_name, _coerce(value, getattr(cfg, field_name)))
    validate_config(cfg, source="bench overrides")
    cfg.warmup_on_start = False  # bench does its own warmup below

    from ltxserver.engine import create_recipe, generate_for_mode, make_warmup_image
    from ltxserver.recipe import GenerationRequest

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg.modes[args.mode_index]
    print(f"[bench:{args.tag}] mode {mode.width}x{mode.height} f{mode.num_frames} fps{mode.fps} "
          f"| overrides: {args.overrides or 'none'}", flush=True)

    recipe = create_recipe(cfg)

    first = args.first_frame
    if first is None:
        first = str(out_dir / "bench_first.png")
        make_warmup_image(first, mode.width, mode.height)
    request = GenerationRequest(prompt=args.prompt or WARMUP_PROMPT,
                                first_frame_path=first, seed=args.seed)

    t0 = time.perf_counter()
    generate_for_mode(recipe, cfg, mode, request)
    print(f"[bench:{args.tag}] warmup (compile) wall: {time.perf_counter() - t0:.1f}s", flush=True)
    if cfg.compile:
        from ltxserver.perf import log_dynamo_counters
        log_dynamo_counters(args.tag, log=lambda msg, *a: print(msg % a, flush=True))

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = generate_for_mode(recipe, cfg, mode, request)
        wall = time.perf_counter() - t0
        times.append(wall)
        print(f"[bench:{args.tag}] run {i + 1}/{args.runs}: {wall:.1f}s", flush=True)
        if not args.no_save:
            from ltxserver.encoder import encode_video_h264
            encode_video_h264(result["frames"], mode.fps, out_dir / f"{args.tag}_run{i}.mp4",
                              bitrate_kbps=cfg.video_bitrate_kbps, preset=cfg.x264_preset,
                              audio=result.get("audio"),
                              audio_sample_rate=result.get("audio_sample_rate"))
    best = min(times)
    print(f"[bench:{args.tag}] best {best:.1f}s | mean {sum(times) / len(times):.1f}s "
          f"over {args.runs} run(s); outputs in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
