#!/usr/bin/env bash
# One-shot dependency install for LTXServer — into the CURRENT environment.
# Prepare and activate your venv/conda first; this script does not create one.
#   PYTHON=...           interpreter to install into (default: python3 on PATH)
#   TORCH_INDEX_URL=...  torch wheel index if torch must be installed (default cu130)
#   FORCE_TORCH=1        (re)install torch even if one is already importable
#   SKIP_FA4=1           skip the FA4 (flash_attn.cute) install
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
PIP="$PY -m pip"
echo "== installing into: $($PY -c 'import sys; print(sys.prefix)') =="
if [ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  echo "NOTE: no active venv/conda detected — installing into the interpreter above."
fi

echo "== submodules (ComfyUI + ComfyUI-LTXVideo, pinned) =="
git submodule update --init --recursive

echo "== torch (before comfy requirements so pip resolves against it) =="
if [ "${FORCE_TORCH:-0}" != "1" ] && $PY -c "import torch" 2>/dev/null; then
  $PY -c "import torch; print('torch already present:', torch.__version__, 'cuda', torch.version.cuda, '— skipping (FORCE_TORCH=1 to reinstall)')"
else
  $PIP install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
fi

echo "== ComfyUI requirements =="
$PIP install -r third_party/ComfyUI/requirements.txt

if [ -f custom_nodes_ext/ComfyUI-LTXVideo/requirements.txt ]; then
  echo "== ComfyUI-LTXVideo requirements =="
  $PIP install -r custom_nodes_ext/ComfyUI-LTXVideo/requirements.txt
fi

echo "== server requirements =="
$PIP install -r requirements.txt

# Attention runs on pytorch SDPA by default (exact; no extra install).
# SageAttention is an opt-in experiment only — it runs poorly on Blackwell.
# If you ever want it: pip install "sageattention @ git+https://github.com/thu-ml/SageAttention@v2.2.0"

echo "== FA4 (flash_attn.cute) — for attention_backend: fa4 =="
# Pure-Python CuTe DSL package (kernels JIT at first use), pinned to the
# cutlass-4.5-compatible revision. Needs sm90+ at runtime (H200/B200);
# fp8 attention additionally needs sm100 (B200). Skippable: SKIP_FA4=1.
FA4_REV="${FA4_REV:-82d6441eec5d4dfec120153db2c0145ae855a083}"
if [ "${SKIP_FA4:-0}" != "1" ]; then
  $PIP install "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@${FA4_REV}#subdirectory=flash_attn/cute" || {
    echo "WARNING: FA4 install failed; attention_backend: fa4 will not be available"
    echo "         (attention_backend: sdpa — the default — is unaffected)."
  }
fi

$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
command -v ffmpeg >/dev/null || echo "NOTE: no system ffmpeg — the imageio-ffmpeg bundle will be used"
echo "OK. Run:  $PY -m ltxserver --config config.yaml"
