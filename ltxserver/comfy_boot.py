"""Embed ComfyUI as a library (no web UI, no queue).

Mirrors the import choreography of ComfyUI's own main.py — the order is
load-bearing:

  1. environment (CUDA_VISIBLE_DEVICES) before anything touches torch;
  2. ``comfy.options`` stays in its default no-argparse mode so importing
     ``comfy.cli_args`` yields the DEFAULT args namespace, which we then
     mutate programmatically (sage attention, smart memory, ...);
  3. ``cuda_malloc`` before torch (it stages PYTORCH_CUDA_ALLOC_CONF);
  4. only then the torch-heavy comfy modules, folder registration, a
     headless PromptServer instance (many custom nodes touch
     ``PromptServer.instance`` at import), and ``nodes.init_extra_nodes``.

Nothing in the ComfyUI tree is modified; plugins load from this repo's
``custom_nodes_ext/`` via an extra custom-node search path.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

logger = logging.getLogger("ltxserver.boot")

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_DIR = REPO_ROOT / "third_party" / "ComfyUI"
CUSTOM_NODES_EXT = REPO_ROOT / "custom_nodes_ext"


@dataclass
class ComfyHandles:
    """Everything the recipe needs after boot."""
    nodes: ModuleType                 # ComfyUI nodes.py (core classes + NODE_CLASS_MAPPINGS)
    nodes_lt: ModuleType              # comfy_extras.nodes_lt
    nodes_lt_audio: ModuleType        # comfy_extras.nodes_lt_audio
    nodes_lt_upsampler: ModuleType    # comfy_extras.nodes_lt_upsampler (import side effects)
    nodes_hunyuan: ModuleType         # LatentUpscaleModelLoader lives here
    nodes_custom_sampler: ModuleType  # RandomNoise / KSamplerSelect / SamplerCustomAdvanced / ...
    model_management: ModuleType
    model_names: SimpleNamespace      # basenames registered per folder category


def call_node(node_cls: Any, **kwargs: Any):
    """Invoke a ComfyUI node class regardless of API generation.

    New-style (io.ComfyNode): classmethod ``execute`` returning NodeOutput —
    unwrap ``.result``. Old-style: instantiate and call the method named by
    ``FUNCTION``. Both return the plain results tuple.
    """
    if hasattr(node_cls, "define_schema"):
        out = node_cls.execute(**kwargs)
    else:
        fn_name = getattr(node_cls, "FUNCTION", "execute")
        out = getattr(node_cls(), fn_name)(**kwargs)
    result = getattr(out, "result", out)
    if result is None:
        result = ()
    return result


def find_node_class(nodes_module: ModuleType, name: str) -> Any:
    """A registered node class by its NODE_CLASS_MAPPINGS key."""
    mapping = nodes_module.NODE_CLASS_MAPPINGS
    if name in mapping:
        return mapping[name]
    raise KeyError(f"node class {name!r} is not registered — did the custom node "
                   f"packages under {CUSTOM_NODES_EXT} load? "
                   f"(have: {[k for k in mapping if 'LTX' in k or 'STG' in k][:20]})")


def boot(*, use_sage_attention: bool, highvram: bool, gpu_only: bool,
         reserve_vram_gb: float, model_files: dict[str, str]) -> ComfyHandles:
    """Initialize embedded ComfyUI and register model/plugin paths.

    ``model_files`` maps folder categories to absolute file paths:
    checkpoints / diffusion_models / text_encoders / latent_upscale_models.
    Returns handles plus each file's basename for the loader nodes.
    """
    if not (COMFY_DIR / "comfy").is_dir():
        raise RuntimeError(f"ComfyUI submodule missing at {COMFY_DIR} — run install.sh "
                           "(git submodule update --init)")
    sys.path.insert(0, str(COMFY_DIR))

    # -- args (defaults, then programmatic overrides) — BEFORE torch ----------
    import comfy.options  # noqa: F401  (args_parsing stays False -> defaults)
    from comfy.cli_args import args
    args.use_sage_attention = bool(use_sage_attention)
    # --highvram pins models in GPU memory (comfy otherwise unloads to CPU
    # after each use — per-request PCIe reloads). NOTE: --disable-smart-memory
    # is the OPPOSITE of resident (it forces aggressive offload); never set it
    # for serving.
    args.highvram = bool(highvram)
    # --gpu-only additionally puts TEXT ENCODERS on the GPU
    # (text_encoder_device) and keeps INTERMEDIATE results there
    # (intermediate_device). Without it comfy reloads the text encoder from
    # RAM every request (~2s for the 13GB gemma) and ships VAE-decoded
    # frames to the CPU as fp32 (a 1.4GB copy + CPU-side postprocessing).
    if gpu_only and hasattr(args, "gpu_only"):
        args.gpu_only = True
    if reserve_vram_gb and hasattr(args, "reserve_vram"):
        args.reserve_vram = float(reserve_vram_gb)

    # -- allocator staging, exactly like main.py ------------------------------
    import cuda_malloc  # noqa: F401

    import folder_paths
    folder_paths.add_model_folder_path("custom_nodes", str(CUSTOM_NODES_EXT))
    model_names: dict[str, str] = {}
    for category, file_path in model_files.items():
        path = Path(file_path).resolve()
        folder_paths.add_model_folder_path(category, str(path.parent))
        model_names[category] = path.name

    import comfy.model_management as model_management
    logger.info("comfy device: %s | sage attention: %s | highvram (models pinned): %s | "
                "gpu_only (TE + intermediates on GPU): %s | vram state: %s",
                model_management.get_torch_device(), args.use_sage_attention,
                args.highvram, getattr(args, "gpu_only", False), model_management.vram_state.name)

    # -- node registry (needs a PromptServer instance like main.py makes) -----
    import nodes
    import server as comfy_server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    comfy_server.PromptServer(loop)  # published as PromptServer.instance
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    logger.info("comfy nodes initialized: %d classes", len(nodes.NODE_CLASS_MAPPINGS))

    handles = ComfyHandles(
        nodes=nodes,
        nodes_lt=importlib.import_module("comfy_extras.nodes_lt"),
        nodes_lt_audio=importlib.import_module("comfy_extras.nodes_lt_audio"),
        nodes_lt_upsampler=importlib.import_module("comfy_extras.nodes_lt_upsampler"),
        nodes_hunyuan=importlib.import_module("comfy_extras.nodes_hunyuan"),
        nodes_custom_sampler=importlib.import_module("comfy_extras.nodes_custom_sampler"),
        model_management=model_management,
        model_names=SimpleNamespace(**model_names),
    )
    return handles


def setup_environment(cuda_visible_devices: str, inductor_cache_dir: str = "") -> None:
    """Process env staged BEFORE boot() (and therefore before torch)."""
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if inductor_cache_dir:
        # Already-exported env wins (operator override), same as FastVideo.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache_dir))
