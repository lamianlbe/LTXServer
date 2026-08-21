# LTXServer

The LTX-2.3 two-stage I2V workflow as a production HTTP service, running on
an **embedded ComfyUI** — every compute step calls the exact node
implementation the reference workflow runs, so output parity is by
construction, not by porting. FastAPI wrapper, per-mode warmup, JSON request
logs, S3 delivery, one process per GPU.

```
third_party/ComfyUI              ComfyUI v0.33.2 (submodule, unmodified)
custom_nodes_ext/ComfyUI-LTXVideo  Lightricks plugin (submodule, unmodified)
custom_nodes_ext/ltxplus         vendored LTXPlusBatchAddGuide (@26ddd98)
ltxserver/                       config / comfy bootstrap / recipe / server
scripts/merge_stage1_into_base.py  model prep (see below)
```

## Model preparation — 4 files

The workflow's LoRA stacking happens OFFLINE (in ComfyUI, `ModelSave` the
two merged models), so the server loads pre-merged weights:

| config key | file | how to get it |
|---|---|---|
| `models.checkpoint` | base ckpt with the **stage-1** DiT folded in | `python scripts/merge_stage1_into_base.py --base <base ckpt> --stage1 <stage1 ModelSave export> --output ...` |
| `models.stage2_transformer` | **stage-2** DiT-only export **with config metadata** | ComfyUI `ModelSave` of the stage-2 merge, then `python scripts/embed_config_metadata.py --from-file <base ckpt> --file <export> --output ...` (ModelSave drops the `config` metadata comfy's LTX-2.3 detection needs) |
| `models.text_encoder` | Gemma text encoder | same file the workflow's `LTXAVTextEncoderLoader` uses |
| `models.latent_upsampler` | x1.5 spatial upscaler | `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` |

All quantization sidecars (fp8 payloads, `weight_scale`, `comfy_quant`)
pass through byte-identically — the merged files keep ComfyUI's exact
per-layer mixed-precision profile.

## Install & run

```bash
# activate YOUR prepared venv/conda first — install.sh installs into the
# current environment (torch is skipped when already present)
./install.sh                         # submodules + deps + FA4 pin
cp config.example.yaml config.yaml   # edit model paths + modes
python -m ltxserver --config config.yaml            # GPU from config
python -m ltxserver --config config.yaml --gpu 1 --port 8001   # per-GPU instance
```

Startup boots comfy headless, loads the four models resident, binds the
port, and warms up one generation per distinct mode in the background —
`/v1/*` returns 503 (and `/readyz` stays 503) until warmup finishes.

## API (drop-in compatible with the FastVideo LTX-2.3 server)

* `POST /v1/generate` — multipart: `prompt, width, height, num_frames,
  fps, first_frame` (+ optional `last_frame, negative_prompt, seed,
  last_frame_strength, image_crf, video_bitrate_kbps`); returns the mp4
  with `X-LTX23-*` headers (served mode, seed, timings).
* `POST /v1/generate_s3` — same, plus `generate_lq`; uploads HQ (+ LQ)
  to S3 and returns JSON.
* `GET /v1/modes`, `GET /healthz`, `GET /readyz`.

Unmatched (w, h, frames, fps) combos are served with the
closest-resolution configured mode. FLF: a `last_frame` upload adds a
tail keyframe in stage 1 (`frame_indices "0, -1"`, batched when its
strength equals `guide_strength`, appended separately otherwise).

## Stage-1 conditioning modes

`stage1_conditioning: guide` (default) is the workflow's appended-keyframe
mechanism — guide-frame tokens in the sequence, per-forward keyframe
bookkeeping, optional attention bias. `inplace` is the FastVideo recipe:
the first frame is hard-pinned (strength 1.0) into latent frame 0 and an
optional last frame partially pinned at the request's
`last_frame_strength` — no keyframe tokens exist, so stage-1 forwards are
as clean and as fast as stage 2's. `guide_strength` /
`guide_attention_bias` only apply to guide mode. The two modes produce
different stage-1 sequence lengths, so switching re-warms the compiled
graphs (both sets coexist in a persistent inductor cache). Whether inplace
costs quality ON THE CORRECTED STAGE-2 WEIGHTS is an open A/B — the old
comparison predates that fix.

## Performance: torch.compile + FA4

Both are off by default; the default configuration runs stock comfy modules
and pytorch SDPA end to end.

**torch.compile** (`compile: true`) wraps both DiTs via comfy's official
`set_torch_compile_wrapper`. Comfy's fp8 `QuantizedTensor` layers graph-break
twice per linear under dynamo (the layout `Params` dataclass is untraceable
— measured ~2500 splits per forward), so at startup every quantized linear
is swapped for a compile-friendly twin: same payload, same scale, same
saturating scale-1.0 input cast, same `scaled_mm_v2` call with the bias
fused. Each swap is probe-verified **bit-identical** to the layer it
replaces and the server refuses to start otherwise, so a comfy upgrade that
changes fp8 numerics can never ship silently different pixels.
`compile_scope: blocks` (the default) compiles per transformer block —
48 small shared-code graphs covering ~95% of the compute — and leaves
comfy's outer glue eager; that glue includes the stage-1 guide-keyframe
bookkeeping whose data-dependent `.item()` can never be captured, so
`model` scope graph-breaks there and is only sensible for guide-free
recipes. First warmup per mode
compiles for minutes — point `inductor_cache_dir` at a persistent volume;
after that, restarts reuse the kernel cache. Inductor knobs mirror the
FastVideo production server (`shape_padding=False` is load-bearing on
Blackwell).

**FA4** (`attention_backend: fa4`) installs an `optimized_attention`
override per model via `transformer_options` — again comfy's official hook.
Unmasked attention runs `flash_attn.cute` (needs sm90+, i.e. H200/B200, and
the pinned install from `install.sh`); the stage-1 guide-bias segments are
masked and always stay on SDPA, mirroring the FastVideo server.
`fa4_fp8_stage1/2` additionally quantize that stage's q/k/v to fp8 e4m3
with per-(batch, head) descales — both attention GEMMs at the fp8
tensor-core rate, bf16 out. fp8 attention is sm100-only upstream: **B200
yes, H200 bf16-FA4 only.** The kernel calls are torch custom ops with fake
kernels, so FA4 composes with `compile: true`.

`compile_vae: true` (default, needs single-shot mode) compiles the video
VAE decode+encode after swapping comfy's causal convs for stateless
bit-identical twins (the originals consult a thread-keyed streaming cache
on every forward — untraceable, and dead code in single-shot mode).
`compile_te: true` (default) compiles the gemma text encoder — prompts are
left-padded to a fixed 1024 tokens so shapes are static; bf16 and
fp8_e4m3fn (per-tensor scaled) TEs are supported, block-scaled TEs
(mxfp8/nvfp4) are rejected with a clear error (set `compile_te: false`).

Every combination changes the compiled graphs — re-warm after toggling.
A/B with `scripts/bench.py`:

```bash
python scripts/bench.py --config config.yaml --tag sdpa
python scripts/bench.py --config config.yaml --tag fa4 --set attention_backend=fa4
python scripts/bench.py --config config.yaml --tag fa4_compiled \
    --set attention_backend=fa4 --set compile=true
```

Same seed per tag ⇒ the mp4s are directly comparable for quality; the
summary lines give the speed. Recommended ladder: `sdpa` (baseline) → `fa4`
→ `compile` → `fa4+compile` → `+fa4_fp8_stage2` — one variable at a time.

## Fidelity notes

* **Attention**: pytorch SDPA by default — the exact kernel, and the
  baseline every comparison uses. When A/B-ing against a desktop ComfyUI,
  launch it WITHOUT `--use-sage-attention` so both sides run SDPA.
  (`use_sage_attention: true` remains as an opt-in experiment; it needs a
  manual source build and runs poorly on Blackwell.)
* The guider (STGGuiderAdvanced), guide injection (LTXPlusBatchAddGuide),
  sampling (euler_ancestral / euler_ancestral_cfg_pp over ManualSigmas),
  upsample, inplace keyframe, crop and decodes are the plugin/core node
  implementations called directly — see `ltxserver/recipe.py` for the
  1:1 transcription (value-plumbing nodes became plain Python; the
  RES4LYF easing node became the pre-eased `stage1_sigmas` list; VHS
  encode became the ffmpeg pipeline shared with the FastVideo server).
* Not yet here (deliberately, Tier 2/3): torch.compile, FA4 attention.
  Each lands as a single-variable A/B against this baseline.

## Multi-GPU

One process per GPU behind your supervisor/load balancer:

```
python -m ltxserver --config config.yaml --gpu 0 --port 8000 --log-dir logs/gpu0
python -m ltxserver --config config.yaml --gpu 1 --port 8001 --log-dir logs/gpu1
```
