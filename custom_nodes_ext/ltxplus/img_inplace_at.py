"""In-place image conditioning at an arbitrary latent index.

Comfy core's LTXVImgToVideoInplace only writes frame 0. This node is the
same math — bilinear cover-resize to the latent's pixel dims, vae.encode,
write the latent rows and set noise_mask = 1 - strength — generalized to
any latent index (negative counts from the end, so -1 pins the final
frame). Used by the server's `stage1_conditioning: inplace` mode, which
mirrors the FastVideo recipe: conditioning frames live IN the latent under
a partial denoise-mask pin instead of as appended guide keyframes, so the
DiT sees no keyframe tokens and none of their bookkeeping.
"""

from __future__ import annotations

import comfy.utils
from comfy_extras.nodes_lt import get_noise_mask


class LTXPlusImgInplaceAt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": "Latent frame index to write at; negative "
                                   "counts from the end (-1 = final frame).",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "noise_mask is set to 1 - strength on the "
                                   "written rows (1.0 = hard pin).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "ltx-plus"
    DESCRIPTION = "LTXVImgToVideoInplace at an arbitrary latent index."

    def execute(self, vae, image, latent, index, strength=1.0):
        samples = latent["samples"].clone()
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        _, _, latent_length, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        if image.shape[1] != height or image.shape[2] != width:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            pixels = image
        t = vae.encode(pixels[:, :, :, :3])

        frames = t.shape[2]
        start = index if index >= 0 else latent_length + index - frames + 1
        if start < 0 or start + frames > latent_length:
            raise ValueError(
                f"LTXPlusImgInplaceAt: index {index} resolves to rows "
                f"[{start}, {start + frames}) outside latent_length={latent_length}"
            )

        samples[:, :, start:start + frames] = t
        noise_mask = get_noise_mask(latent)
        noise_mask[:, :, start:start + frames] = 1.0 - strength
        return (
            {"samples": samples, "noise_mask": noise_mask},
        )
