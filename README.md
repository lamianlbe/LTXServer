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
./install.sh                         # submodules + venv + torch cu130 + deps + sage
cp config.example.yaml config.yaml   # edit model paths + modes
source .venv/bin/activate
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
