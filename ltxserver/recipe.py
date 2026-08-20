"""The LTX-2.3 two-stage workflow, transcribed node-for-node into code.

Every compute step calls the SAME node implementation ComfyUI's executor
would call for the reference workflow (optimized all-in-one v2), in the
same order with the same inputs — the graph plumbing (primitive value
nodes, casts, the dead resize branch) is replaced by plain Python, and the
LoRA-loader chain is gone because the checkpoints are pre-merged:

  stage 1  CheckpointLoaderSimple(merged base)      # DiT = stage-1 merge
           CLIPTextEncode -> LTXVConditioning(fps)
           EmptyLTXVLatentVideo + LTXVEmptyLatentAudio
           [LTXVPreprocess crf]  (off in the reference workflow)
           lanczos longer-edge resize (verified identical to
           ResizeImageMaskNode "scale longer dimension")
           LTXPlusBatchAddGuide(strength, frame indices)
           LTXVConcatAVLatent
           STGGuiderAdvanced (sigma-lookup cfg, CFG-Zero* rescale)
           SamplerCustomAdvanced(euler_ancestral, ManualSigmas eased)
  stage 2  LTXVSeparateAVLatent -> LTXVLatentUpsampler (x1.5)
           LTXVImgToVideoInplace(first frame, strength 1)
           LTXVConcatAVLatent(audio from stage 1)
           CFGGuider(stage-2 DiT, cfg 1)
           SamplerCustomAdvanced(euler_ancestral_cfg_pp, ManualSigmas)
           LTXVSeparateAVLatent -> LTXVCropGuides -> VAEDecode
           LTXVAudioVAEDecode

The same RandomNoise object drives both samplers (workflow node 926:950
feeds both SamplerCustomAdvanced nodes).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .comfy_boot import ComfyHandles, call_node, find_node_class
from .config import ServerConfig, Mode

logger = logging.getLogger("ltxserver.recipe")


@dataclass
class GenerationRequest:
    prompt: str
    first_frame_path: str
    last_frame_path: str | None = None
    negative_prompt: str | None = None
    seed: int = 0
    last_frame_strength: float = 0.8
    image_crf: float | None = None  # None = config default


def _csv(values) -> str:
    return ", ".join(f"{float(v):g}" for v in values)


def load_image_tensor(path: str | Path):
    """PIL -> [1, H, W, C] float32 in [0, 1] — the same pixel math as
    ComfyUI's LoadImage (EXIF transpose, RGB, /255)."""
    import torch
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def scale_longer_dimension(size: tuple[int, int], longer_size: int) -> tuple[int, int]:
    """(w, h) with the longer edge scaled to ``longer_size`` — verbatim
    rounding of comfy's ResizeImageMaskNode "scale longer dimension"."""
    width, height = size
    if height > width:
        return max(1, round((width / height) * longer_size)), longer_size
    if width > height:
        return longer_size, max(1, round((height / width) * longer_size))
    return longer_size, longer_size


def load_guide_image(path: str | Path, longer_size: int):
    """Load + lanczos-resize so the longer edge is ``longer_size``.

    Byte-identical to LoadImage -> ResizeImageMaskNode(lanczos): comfy's
    lanczos helper round-trips through uint8 PIL, which is exactly resizing
    the decoded PIL image before tensor conversion.
    """
    import torch
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    target = scale_longer_dimension(image.size, longer_size)
    if target != image.size:
        image = image.resize(target, resample=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


class LtxRecipe:
    """Loads the four model files once; runs the two-stage workflow per call."""

    def __init__(self, handles: ComfyHandles, cfg: ServerConfig) -> None:
        self.h = handles
        self.cfg = cfg
        self._lock = threading.Lock()  # comfy execution is not thread-safe

        names = handles.model_names
        nodes = handles.nodes
        t0 = time.perf_counter()
        logger.info("loading checkpoint (stage-1 merged base): %s", names.checkpoints)
        model_s1, _base_clip, vae = call_node(nodes.CheckpointLoaderSimple,
                                              ckpt_name=names.checkpoints)[:3]
        logger.info("loading stage-2 transformer: %s", names.diffusion_models)
        (model_s2,) = call_node(nodes.UNETLoader, unet_name=names.diffusion_models,
                                weight_dtype="default")
        logger.info("loading text encoder: %s (+ connectors from the checkpoint)",
                    names.text_encoders)
        (clip,) = call_node(handles.nodes_lt_audio.LTXAVTextEncoderLoader,
                            text_encoder=names.text_encoders,
                            ckpt_name=names.checkpoints,
                            device="default")
        (audio_vae,) = call_node(handles.nodes_lt_audio.LTXVAudioVAELoader,
                                 ckpt_name=names.checkpoints)
        (upscale_model,) = call_node(handles.nodes_hunyuan.LatentUpscaleModelLoader,
                                     model_name=names.latent_upscale_models)
        self.model_s1, self.model_s2 = model_s1, model_s2
        self.clip, self.vae, self.audio_vae = clip, vae, audio_vae
        self.upscale_model = upscale_model
        self.stg_guider_cls = find_node_class(nodes, "STGGuiderAdvanced")
        # Vendored under custom_nodes_ext/ltxplus and registered by comfy's
        # custom-node scan during boot.
        self.batch_add_guide_cls = find_node_class(nodes, "LTXPlusBatchAddGuide")
        logger.info("models loaded in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def generate(self, request: GenerationRequest, mode: Mode) -> dict:
        """One full two-stage generation. Returns raw frames + audio."""
        import torch
        cfg, h = self.cfg, self.h
        n = h.nodes
        lt, lta, ncs = h.nodes_lt, h.nodes_lt_audio, h.nodes_custom_sampler
        t0 = time.perf_counter()

        with self._lock, torch.no_grad():
            # --- text conditioning (CLIPTextEncode -> LTXVConditioning) ------
            (pos,) = call_node(n.CLIPTextEncode, clip=self.clip, text=request.prompt)
            negative_text = (request.negative_prompt
                             if request.negative_prompt is not None else cfg.negative_prompt)
            (neg,) = call_node(n.CLIPTextEncode, clip=self.clip, text=negative_text)
            pos, neg = call_node(lt.LTXVConditioning, positive=pos, negative=neg,
                                 frame_rate=float(mode.fps))

            # --- empty AV latents --------------------------------------------
            (lat_video,) = call_node(lt.EmptyLTXVLatentVideo, width=mode.width,
                                     height=mode.height, length=mode.num_frames,
                                     batch_size=1)
            (lat_audio,) = call_node(lta.LTXVEmptyLatentAudio,
                                     frames_number=mode.num_frames,
                                     frame_rate=int(mode.fps), batch_size=1,
                                     audio_vae=self.audio_vae)

            # --- guide images -------------------------------------------------
            guide_first = load_guide_image(request.first_frame_path, cfg.guide_longer_size)
            image_crf = cfg.image_crf if request.image_crf is None else request.image_crf
            if image_crf and image_crf > 0:
                (guide_first,) = call_node(lt.LTXVPreprocess, image=guide_first,
                                           img_compression=int(image_crf))
            # LTXPlusBatchAddGuide applies ONE strength per call, so keyframes
            # are grouped by strength: the first frame always at
            # cfg.guide_strength (the reference workflow's i2v guide), and an
            # optional FLF tail keyframe — batched into the same call when it
            # shares the strength, else appended in its own call. Chained calls
            # are exactly chained LTXVAddGuide nodes (append semantics).
            guide_batches: list[tuple[object, str, float]] = []
            strengths_note = f"strength={cfg.guide_strength}"
            if request.last_frame_path:
                guide_last = load_guide_image(request.last_frame_path, cfg.guide_longer_size)
                if image_crf and image_crf > 0:
                    (guide_last,) = call_node(lt.LTXVPreprocess, image=guide_last,
                                              img_compression=int(image_crf))
                if abs(request.last_frame_strength - cfg.guide_strength) < 1e-6:
                    guide_batches.append((torch.cat([guide_first, guide_last], dim=0),
                                          "0, -1", cfg.guide_strength))
                else:
                    guide_batches.append((guide_first, "0", cfg.guide_strength))
                    guide_batches.append((guide_last, "-1", request.last_frame_strength))
                    strengths_note += f", tail strength={request.last_frame_strength}"
            else:
                guide_batches.append((guide_first, "0", cfg.guide_strength))

            for batch_images, frame_indices, strength in guide_batches:
                pos, neg, lat_video = call_node(self.batch_add_guide_cls, positive=pos,
                                                negative=neg, vae=self.vae, latent=lat_video,
                                                images=batch_images,
                                                frame_indices=frame_indices,
                                                strength=strength)

            (av_latent,) = call_node(lt.LTXVConcatAVLatent, video_latent=lat_video,
                                     audio_latent=lat_audio)

            # --- stage 1: STG guider + euler_ancestral over eased sigmas -----
            (guider1,) = call_node(
                self.stg_guider_cls, model=self.model_s1, positive=pos, negative=neg,
                skip_steps_sigma_threshold=cfg.skip_steps_sigma_threshold,
                cfg_star_rescale=cfg.cfg_star_rescale,
                sigmas=_csv(cfg.cfg_sigma_list),
                cfg_values=_csv(cfg.cfg_values_by_sigma),
                stg_scale_values=_csv(cfg.resolved_stg_scale_values()),
                stg_rescale_values=_csv(cfg.resolved_stg_rescale_values()),
                stg_layers_indices=cfg.resolved_stg_layers_indices(),
                apply_apg=False, apg_cfg_scale=cfg.apg_cfg_scale,
                eta=cfg.apg_eta, norm_threshold=cfg.apg_norm_threshold,
            )
            (noise,) = call_node(ncs.RandomNoise, noise_seed=int(request.seed))
            (sampler1,) = call_node(ncs.KSamplerSelect, sampler_name="euler_ancestral")
            (sigmas1,) = call_node(ncs.ManualSigmas, sigmas=_csv(cfg.stage1_sigmas))
            out1 = call_node(ncs.SamplerCustomAdvanced, noise=noise, guider=guider1,
                             sampler=sampler1, sigmas=sigmas1, latent_image=av_latent)[0]
            t_stage1 = time.perf_counter()
            v1, a1 = call_node(lt.LTXVSeparateAVLatent, av_latent=out1)

            # --- stage 2: x1.5 upsample + inplace keyframe + cfg_pp refine ---
            (upsampled,) = call_node(h.nodes_lt_upsampler.LTXVLatentUpsampler,
                                     samples=v1, upscale_model=self.upscale_model,
                                     vae=self.vae)
            (inplace,) = call_node(lt.LTXVImgToVideoInplace, vae=self.vae,
                                   image=guide_first[0:1], latent=upsampled,
                                   strength=1.0, bypass=False)
            (av2,) = call_node(lt.LTXVConcatAVLatent, video_latent=inplace,
                               audio_latent=a1)
            (guider2,) = call_node(ncs.CFGGuider, model=self.model_s2, positive=pos,
                                   negative=neg, cfg=1.0)
            (sampler2,) = call_node(ncs.KSamplerSelect,
                                    sampler_name="euler_ancestral_cfg_pp")
            (sigmas2,) = call_node(ncs.ManualSigmas, sigmas=_csv(cfg.stage2_sigmas))
            out2 = call_node(ncs.SamplerCustomAdvanced, noise=noise, guider=guider2,
                             sampler=sampler2, sigmas=sigmas2, latent_image=av2)[0]
            t_stage2 = time.perf_counter()
            v2, a2 = call_node(lt.LTXVSeparateAVLatent, av_latent=out2)
            _, _, v2_cropped = call_node(lt.LTXVCropGuides, positive=pos, negative=neg,
                                         latent=v2)

            # --- decode -------------------------------------------------------
            (image_batch,) = call_node(n.VAEDecode, vae=self.vae, samples=v2_cropped)
            (audio_out,) = call_node(lta.LTXVAudioVAEDecode, samples=a2,
                                     audio_vae=self.audio_vae)

            frames = [(frame.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
                      for frame in image_batch]
            waveform = audio_out["waveform"]
            audio_np = waveform[0].float().cpu().numpy() if waveform is not None else None
            sample_rate = int(audio_out.get("sample_rate", 0)) or None

        gen_seconds = time.perf_counter() - t0
        logger.info("generated %dx%d f%d seed=%s (%s): stage1 %.1fs, stage2 %.1fs, total %.1fs",
                    mode.width, mode.height, mode.num_frames, request.seed, strengths_note,
                    t_stage1 - t0, t_stage2 - t_stage1, gen_seconds)
        return {
            "frames": frames,
            "audio": audio_np,
            "audio_sample_rate": sample_rate,
            "gen_seconds": gen_seconds,
        }
