"""Vendored LTXPlus nodes (lamianlbe/ComfyUI-LTXPlus @ 26ddd98, trimmed to
the one node the LTX-2.3 workflow uses). Shaped like a ComfyUI custom-node
package so comfy's node scan can load it, though the server recipe imports
the class directly."""

from .batch_add_guide import LTXPlusBatchAddGuide

NODE_CLASS_MAPPINGS = {
    "LTXPlusBatchAddGuide": LTXPlusBatchAddGuide,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXPlusBatchAddGuide": "🅛🅣🅧 LTX Plus Batch Add Guide",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
