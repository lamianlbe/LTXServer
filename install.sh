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

echo "== torch >= 2.13 (before comfy requirements so pip resolves against it) =="
# 2.13 is a hard floor: older dynamo cannot trace torch._scaled_mm_v2, so the
# fp8-linear twins graph-break at EVERY quantized linear (compile still works
# but the DiT shatters into dozens of graphs — the FastVideo-image torch 2.12
# failure mode).
if [ "${FORCE_TORCH:-0}" != "1" ] && $PY -c "
import sys
try:
    import torch
except Exception:
    sys.exit(1)
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
sys.exit(0 if v >= (2, 13) else 1)
" 2>/dev/null; then
  $PY -c "import torch; print('torch already present and >= 2.13:', torch.__version__, 'cuda', torch.version.cuda, '— skipping (FORCE_TORCH=1 to reinstall)')"
else
  $PY -c "import torch; print('torch', torch.__version__, 'is < 2.13 — upgrading')" 2>/dev/null || true
  $PIP install -U torch torchvision torchaudio --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
fi

echo "== cuDNN backend >= 9.21 (attention_backend: cudnn_mxfp8 needs it) =="
CUDNN_VER=$($PY -c "import torch; print(torch.backends.cudnn.version() or 0)" 2>/dev/null || echo 0)
if [ "${CUDNN_VER}" -lt 92100 ]; then
  CU_MAJ=$($PY -c "import torch; print((torch.version.cuda or '13').split('.')[0])" 2>/dev/null || echo 13)
  echo "cudnn backend ${CUDNN_VER} < 92100 — upgrading nvidia-cudnn-cu${CU_MAJ}."
  echo "(pip will warn that torch pins an older nvidia-cudnn — expected and harmless:"
  echo " cuDNN minor versions are ABI-compatible and torch loads the newer one.)"
  $PIP install -U "nvidia-cudnn-cu${CU_MAJ}" || {
    echo "WARNING: cudnn upgrade failed; attention_backend: cudnn_mxfp8 will refuse"
    echo "         to start (sdpa/fa4 are unaffected)."
  }
else
  echo "cudnn backend ${CUDNN_VER} — OK"
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

echo "== cudnn-frontend — for attention_backend: cudnn_mxfp8 =="
# Python bindings for cuDNN's graph API (microscaled fp8 attention on
# B200/B300; the backend >= 9.21 requirement is checked/upgraded above).
# Skippable: SKIP_CUDNN_FE=1.
if [ "${SKIP_CUDNN_FE:-0}" != "1" ]; then
  $PIP install "nvidia-cudnn-frontend[cutedsl]" || {
    echo "WARNING: cudnn-frontend install failed; attention_backend: cudnn_mxfp8"
    echo "         will not be available (sdpa/fa4 are unaffected)."
  }
fi

$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'available', torch.cuda.is_available())"
command -v ffmpeg >/dev/null || echo "NOTE: no system ffmpeg — the imageio-ffmpeg bundle will be used"
echo "OK. Run:  $PY -m ltxserver --config config.yaml"
