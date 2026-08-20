#!/usr/bin/env bash
# One-shot dependency install for LTXServer.
#   TORCH_INDEX_URL=...  override the torch wheel index (default cu130)
#   PYTHON=...           interpreter used to create .venv (default python3)
set -euo pipefail
cd "$(dirname "$0")"

echo "== submodules (ComfyUI + ComfyUI-LTXVideo, pinned) =="
git submodule update --init --recursive

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== torch (before comfy requirements so pip resolves against it) =="
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

echo "== ComfyUI requirements =="
pip install -r third_party/ComfyUI/requirements.txt

if [ -f custom_nodes_ext/ComfyUI-LTXVideo/requirements.txt ]; then
  echo "== ComfyUI-LTXVideo requirements =="
  pip install -r custom_nodes_ext/ComfyUI-LTXVideo/requirements.txt
fi

echo "== server requirements =="
pip install -r requirements.txt

echo "== SageAttention 2.2 (the reference attention backend) =="
# PyPI only carries sageattention 1.x, whose kernels/semantics differ from
# the 2.x the reference deployment runs — build 2.2.0 from source (needs an
# nvcc matching torch's CUDA; compiles for the LOCAL GPU arch, ~5-15 min).
SAGE_REF="${SAGE_REF:-v2.2.0}"
pip install "sageattention @ git+https://github.com/thu-ml/SageAttention@${SAGE_REF}" || {
  echo "WARNING: sageattention build failed. Either build it manually on this"
  echo "         machine (git clone thu-ml/SageAttention && pip install .) or"
  echo "         set use_sage_attention: false in the config."
}

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
command -v ffmpeg >/dev/null || echo "NOTE: no system ffmpeg — the imageio-ffmpeg bundle will be used"
echo "OK. Run:  source .venv/bin/activate && python -m ltxserver --config config.yaml"
