"""
LTX Plus Batch Add Guide — append-mode batch keyframe injection.

Vendored verbatim from lamianlbe/ComfyUI-LTXPlus @ 26ddd98 (the version the
reference workflow runs), with exactly one change: the package-local
``@comfy_node`` registration decorator is replaced by the plain class +
NODE_CLASS_MAPPINGS in this package's __init__ so the file has no
dependency on the LTXPlus package scaffolding.

Functionally equivalent to chaining N copies of comfy_extras'
`LTXVAddGuide`, but in a single node that takes:

  - an IMAGE batch (one image per reference frame)
  - a comma-separated string of frame indices ("0, 30, 60, -1")
  - a single strength value applied to all keyframes

The node loops `min(len(images), len(indices))` times and folds each
keyframe into the latent + conditioning using the exact same
encode / get_latent_index / append_keyframe / guide_attention_entry
sequence as the upstream `LTXVAddGuide.execute()`. Frame indices follow
LTXV's conventions:
  - Negative values count from the end of the original video
  - Multi-frame keyframes (guide_length > 1) get frame_idx aligned to
    the time-scale grid (multiple of 8 + 1, or 0)
  - Each iteration's `get_latent_index` consults the running positive
    conditioning, so negative indices resolve against the unchanged
    base video length even after earlier iterations grew the latent
"""

import logging

import torch  # noqa: F401 — used implicitly via tensor ops in LTXVAddGuide

import comfy.utils  # noqa: F401 — same as above
import comfy_extras.nodes_lt as _lt_nodes
import node_helpers  # noqa: F401 — same as above

logger = logging.getLogger(__name__)


def _parse_frame_indices(s):
    """Parse a comma-separated string of integers, ignoring blanks/bad tokens."""
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            logger.warning(
                f"LTXPlusBatchAddGuide: ignoring non-integer token {tok!r}"
            )
    return out


class LTXPlusBatchAddGuide:
    """Append a batch of keyframes using LTXVAddGuide semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae":      ("VAE",),
                "latent":   ("LATENT",),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Image batch. Each item in the batch is one "
                                   "keyframe; pair with a frame_indices entry.",
                    },
                ),
                "frame_indices": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": "Comma-separated frame indices for each "
                                   "keyframe (e.g. '0, 30, 60, -1'). Negative "
                                   "values count from the end. For multi-frame "
                                   "guides, frame_idx must be 0 or (8k+1); "
                                   "non-aligned values are floored. The loop "
                                   "runs min(len(images), len(indices)) times "
                                   "— extras on either side are dropped.",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Strength applied uniformly to every "
                                   "keyframe in this batch (same semantics as "
                                   "LTXVAddGuide.strength).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "ltx-plus"
    DESCRIPTION = (
        "Append a batch of keyframes to a video latent in one shot, with "
        "exactly the same per-keyframe behavior as chaining LTXVAddGuide N "
        "times."
    )

    def execute(self, positive, negative, vae, latent, images, frame_indices,
                strength=0.8):
        indices = _parse_frame_indices(frame_indices)
        n_images = int(images.shape[0]) if images is not None else 0
        n = min(n_images, len(indices))

        # Pass-through if there's nothing to do.
        if n == 0:
            logger.info(
                f"LTXPlusBatchAddGuide: nothing to inject "
                f"(images={n_images}, indices={len(indices)}) — pass-through."
            )
            return (positive, negative, latent)

        if n_images != len(indices):
            logger.info(
                f"LTXPlusBatchAddGuide: image count ({n_images}) and index "
                f"count ({len(indices)}) differ — using min ({n}); the extras "
                f"are dropped."
            )

        scale_factors = vae.downscale_index_formula

        # Working state — every iteration consumes the previous outputs so
        # keyframe_idxs accumulate correctly in `positive`/`negative`, and
        # the latent grows along the temporal dim.
        latent_image = latent["samples"]
        noise_mask = _lt_nodes.get_noise_mask(latent)

        for fi in range(n):
            # Take one image out of the batch (keep the leading batch dim
            # so encode() still receives a 4D BHWC tensor).
            single_image = images[fi:fi + 1]
            frame_idx_in = indices[fi]

            _, _, latent_length, latent_height, latent_width = latent_image.shape

            # Encode the keyframe at the running latent's spatial dims.
            # encode() also crops the image to (8k+1) frames if needed.
            _, t = _lt_nodes.LTXVAddGuide.encode(
                vae, latent_width, latent_height, single_image, scale_factors,
            )

            # Resolve user-facing frame_idx into a latent index. Critical
            # detail: get_latent_index reads keyframe_idxs out of the
            # current `positive` conditioning so previously-appended
            # keyframes (from earlier loop iterations) are correctly
            # subtracted from latent_count — that's why negative frame
            # indices keep their meaning across the loop.
            frame_idx, latent_idx = _lt_nodes.LTXVAddGuide.get_latent_index(
                positive, latent_length, len(single_image),
                frame_idx_in, scale_factors,
            )

            # Mirror the assertion from LTXVAddGuide.execute so the user
            # gets a clear error if the resolved position would overflow.
            if latent_idx + t.shape[2] > latent_length:
                raise ValueError(
                    f"LTXPlusBatchAddGuide[{fi}]: resolved latent_idx="
                    f"{latent_idx} + guide_length={t.shape[2]} exceeds "
                    f"latent_length={latent_length}. Lower frame_idx or "
                    f"shrink the guide image."
                )

            positive, negative, latent_image, noise_mask = (
                _lt_nodes.LTXVAddGuide.append_keyframe(
                    positive, negative, frame_idx,
                    latent_image, noise_mask, t, strength, scale_factors,
                )
            )

            # Per-reference attention tracking — same call LTXVAddGuide
            # does at the end of execute(). Without this the model can't
            # apply per-keyframe attention control during sampling.
            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            guide_latent_shape = list(t.shape[2:])
            positive, negative = _lt_nodes._append_guide_attention_entry(
                positive, negative, pre_filter_count, guide_latent_shape,
                strength=strength,
            )

            logger.info(
                f"LTXPlusBatchAddGuide[{fi}]: input pixel_idx={frame_idx_in} "
                f"→ resolved frame_idx={frame_idx}, latent_idx={latent_idx}, "
                f"strength={strength}, guide_shape={guide_latent_shape}"
            )

        return (
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
        )
