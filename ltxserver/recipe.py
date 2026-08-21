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
        model_s2 = None
        if cfg.stage2_enabled:
            logger.info("loading stage-2 transformer: %s", names.diffusion_models)
            (model_s2,) = call_node(nodes.UNETLoader, unet_name=names.diffusion_models,
                                    weight_dtype="default")
        else:
            logger.info("stage 2 disabled: stage-2 transformer and upsampler are not loaded")
        logger.info("loading text encoder: %s (+ connectors from the checkpoint)",
                    names.text_encoders)
        (clip,) = call_node(handles.nodes_lt_audio.LTXAVTextEncoderLoader,
                            text_encoder=names.text_encoders,
                            ckpt_name=names.checkpoints,
                            device="default")
        (audio_vae,) = call_node(handles.nodes_lt_audio.LTXVAudioVAELoader,
                                 ckpt_name=names.checkpoints)
        upscale_model = None
        self.upsampler_scale = 1.0
        if cfg.stage2_enabled:
            (upscale_model,) = call_node(handles.nodes_hunyuan.LatentUpscaleModelLoader,
                                         model_name=names.latent_upscale_models)
            inner = getattr(upscale_model, "model", upscale_model)
            self.upsampler_scale = float(getattr(inner, "spatial_scale", 0.0) or 0.0)
            if self.upsampler_scale <= 1.0:
                raise RuntimeError("could not read spatial_scale from the latent upsampler "
                                   f"({names.latent_upscale_models}) — is it an LTX spatial upscaler?")
            logger.info("latent upsampler: x%s — modes are FINAL resolutions, stage 1 renders "
                        "at 1/%s of the mode", self.upsampler_scale, self.upsampler_scale)
            for mode in cfg.modes:
                self._stage1_dims(mode)  # validates divisibility, raises with guidance
        self.model_s1, self.model_s2 = model_s1, model_s2
        self.clip, self.vae, self.audio_vae = clip, vae, audio_vae
        self.upscale_model = upscale_model
        self.stg_guider_cls = find_node_class(nodes, "STGGuiderAdvanced")
        # Vendored under custom_nodes_ext/ltxplus and registered by comfy's
        # custom-node scan during boot.
        self.batch_add_guide_cls = find_node_class(nodes, "LTXPlusBatchAddGuide")
        self.img_inplace_at_cls = find_node_class(nodes, "LTXPlusImgInplaceAt")
        logger.info("models loaded in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def _stage1_dims(self, mode: Mode) -> tuple[int, int]:
        """Stage-1 render size for a FINAL-resolution mode.

        Matches the FastVideo server: config modes are the post-upscale
        output resolution; stage 1 runs at mode/scale. With stage 2
        disabled the mode IS the render resolution (also FastVideo's
        semantics for its stage-1-only switch).
        """
        if not self.cfg.stage2_enabled:
            return mode.width, mode.height
        s = self.upsampler_scale
        w, h = mode.width / s, mode.height / s
        if w != int(w) or h != int(h) or int(w) % 32 or int(h) % 32:
            hint = 96 if abs(s - 1.5) < 1e-6 else 64 if abs(s - 2.0) < 1e-6 else f"{32 * s:g}"
            raise ValueError(f"mode {mode.width}x{mode.height}: stage 1 would render at "
                             f"{w:g}x{h:g}, which must be integral and divisible by 32 for the "
                             f"x{s:g} upsampler — use final dims divisible by {hint} "
                             f"(e.g. 1344x768 for x1.5)")
        return int(w), int(h)

    def generate(self, request: GenerationRequest, mode: Mode) -> dict:
        """One full two-stage generation. Returns raw frames + audio."""
        import torch
        cfg, h = self.cfg, self.h
        n = h.nodes
        lt, lta, ncs = h.nodes_lt, h.nodes_lt_audio, h.nodes_custom_sampler
        t0 = time.perf_counter()
        phases: dict[str, float] = {}
        _last = [t0]

        def mark(name: str) -> None:
            # Honest GPU phase timing: drain queued work before reading the clock.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            phases[name] = phases.get(name, 0.0) + (now - _last[0])
            _last[0] = now

        if cfg.compile:
            # Something in the stack resets dynamo's recompile budget back to
            # the default 8 (silent eager fallback past it) — re-assert
            # before every generation; raise-only, effectively free.
            from .perf import ensure_dynamo_limits
            ensure_dynamo_limits()

        with self._lock, torch.no_grad():
            # --- text conditioning (CLIPTextEncode -> LTXVConditioning) ------
            (pos,) = call_node(n.CLIPTextEncode, clip=self.clip, text=request.prompt)
            negative_text = (request.negative_prompt
                             if request.negative_prompt is not None else cfg.negative_prompt)
            (neg,) = call_node(n.CLIPTextEncode, clip=self.clip, text=negative_text)
            pos, neg = call_node(lt.LTXVConditioning, positive=pos, negative=neg,
                                 frame_rate=float(mode.fps))
            mark("text")

            # --- empty AV latents (stage-1 render size; mode = FINAL size) ---
            s1_width, s1_height = self._stage1_dims(mode)
            (lat_video,) = call_node(lt.EmptyLTXVLatentVideo, width=s1_width,
                                     height=s1_height, length=mode.num_frames,
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
            guide_last = None
            if request.last_frame_path:
                guide_last = load_guide_image(request.last_frame_path, cfg.guide_longer_size)
                if image_crf and image_crf > 0:
                    (guide_last,) = call_node(lt.LTXVPreprocess, image=guide_last,
                                              img_compression=int(image_crf))

            if cfg.stage1_conditioning == "inplace":
                # FastVideo semantics: first frame hard-pinned IN the latent,
                # optional tail partially pinned — no keyframe tokens at all,
                # so none of the keyframe bookkeeping runs in the forwards.
                strengths_note = "inplace first=1.0"
                (lat_video,) = call_node(self.img_inplace_at_cls, vae=self.vae,
                                         image=guide_first[0:1], latent=lat_video,
                                         index=0, strength=1.0)
                if guide_last is not None:
                    strengths_note += f", tail={request.last_frame_strength}"
                    (lat_video,) = call_node(self.img_inplace_at_cls, vae=self.vae,
                                             image=guide_last[0:1], latent=lat_video,
                                             index=-1,
                                             strength=request.last_frame_strength)
            else:
                guide_batches: list[tuple[object, str, float]] = []
                strengths_note = f"strength={cfg.guide_strength}"
                if guide_last is not None:
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
                                                    strength=strength,
                                                    attention_bias=cfg.guide_attention_bias)
            if cfg.stage2_enabled:
                strengths_note += f", stage1 {s1_width}x{s1_height}"

            (av_latent,) = call_node(lt.LTXVConcatAVLatent, video_latent=lat_video,
                                     audio_latent=lat_audio)

            mark("cond")

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
            stg_scales = cfg.resolved_stg_scale_values()
            if any(s != 0.0 for s in stg_scales):
                logger.info("STG perturbed pass ENABLED (scales %s, layers %s): one extra DiT "
                            "forward per step with a non-zero scale", stg_scales,
                            cfg.resolved_stg_layers_indices())
            (noise,) = call_node(ncs.RandomNoise, noise_seed=int(request.seed))
            (sampler1,) = call_node(ncs.KSamplerSelect, sampler_name="euler_ancestral")
            (sigmas1,) = call_node(ncs.ManualSigmas, sigmas=_csv(cfg.stage1_sigmas))
            out1 = call_node(ncs.SamplerCustomAdvanced, noise=noise, guider=guider1,
                             sampler=sampler1, sigmas=sigmas1, latent_image=av_latent)[0]
            mark("s1")
            t_stage1 = time.perf_counter()
            v1, a1 = call_node(lt.LTXVSeparateAVLatent, av_latent=out1)

            # --- stage 2: x1.5 upsample + inplace keyframe + cfg_pp refine ---
            # The appended guide tokens must leave the latent AND the
            # conditioning BEFORE upsampling: comfy >= 0.33 refuses
            # keyframe_idxs recorded at a different spatial resolution than
            # the sampled latent (the original workflow cropped only after
            # stage 2, which older comfy tolerated silently). Stage 2 then
            # runs guide-free — its first-frame anchoring is the
            # LTXVImgToVideoInplace keyframe, exactly as in the workflow.
            if cfg.stage1_conditioning == "guide":
                pos2, neg2, v1_clean = call_node(lt.LTXVCropGuides, positive=pos,
                                                 negative=neg, latent=v1)
            else:
                # inplace mode appends no keyframes — nothing to crop.
                pos2, neg2, v1_clean = pos, neg, v1

            if not cfg.stage2_enabled:
                # Stage-1-only: decode the cropped stage-1 latent directly at
                # the mode resolution (audio likewise comes from stage 1 —
                # the refine pass would otherwise also touch it).
                (image_batch,) = call_node(n.VAEDecode, vae=self.vae, samples=v1_clean)
                mark("vdec")
                (audio_out,) = call_node(lta.LTXVAudioVAEDecode, samples=a1,
                                         audio_vae=self.audio_vae)
                mark("adec")
                return self._package(image_batch, audio_out, mode, request,
                                     strengths_note, t0, t_stage1, t_stage1,
                                     phases, mark)

            (upsampled,) = call_node(h.nodes_lt_upsampler.LTXVLatentUpsampler,
                                     samples=v1_clean, upscale_model=self.upscale_model,
                                     vae=self.vae)
            (inplace,) = call_node(lt.LTXVImgToVideoInplace, vae=self.vae,
                                   image=guide_first[0:1], latent=upsampled,
                                   strength=1.0, bypass=False)
            (av2,) = call_node(lt.LTXVConcatAVLatent, video_latent=inplace,
                               audio_latent=a1)
            (guider2,) = call_node(ncs.CFGGuider, model=self.model_s2, positive=pos2,
                                   negative=neg2, cfg=1.0)
            (sampler2,) = call_node(ncs.KSamplerSelect,
                                    sampler_name="euler_ancestral_cfg_pp")
            (sigmas2,) = call_node(ncs.ManualSigmas, sigmas=_csv(cfg.stage2_sigmas))
            out2 = call_node(ncs.SamplerCustomAdvanced, noise=noise, guider=guider2,
                             sampler=sampler2, sigmas=sigmas2, latent_image=av2)[0]
            mark("s2")
            t_stage2 = time.perf_counter()
            v2, a2 = call_node(lt.LTXVSeparateAVLatent, av_latent=out2)
            # Guides were already cropped before the upsample, so the
            # workflow's final LTXVCropGuides is a no-op here — decode directly.

            # --- decode -------------------------------------------------------
            (image_batch,) = call_node(n.VAEDecode, vae=self.vae, samples=v2)
            mark("vdec")
            (audio_out,) = call_node(lta.LTXVAudioVAEDecode, samples=a2,
                                     audio_vae=self.audio_vae)
            mark("adec")
            return self._package(image_batch, audio_out, mode, request,
                                 strengths_note, t0, t_stage1, t_stage2,
                                 phases, mark)

    def _package(self, image_batch, audio_out, mode: Mode, request: GenerationRequest,
                 strengths_note: str, t0: float, t_stage1: float, t_stage2: float,
                 phases: dict[str, float] | None = None, mark=None) -> dict:
        import torch
        # One batched clamp/cast and ONE device->host transfer for the whole
        # clip (the per-frame loop this replaces issued one transfer per
        # frame — hundreds of small syncing copies).
        array = (image_batch.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        frames = list(array)
        waveform = audio_out["waveform"]
        audio_np = waveform[0].float().cpu().numpy() if waveform is not None else None
        sample_rate = int(audio_out.get("sample_rate", 0)) or None
        if mark is not None:
            mark("pack")
        gen_seconds = time.perf_counter() - t0
        stage2_seconds = t_stage2 - t_stage1
        phase_note = ""
        if phases:
            phase_note = " | " + " ".join(f"{k}={v:.1f}s" for k, v in phases.items())
        logger.info("generated %dx%d f%d seed=%s (%s): stage1 %.1fs, stage2 %s, total %.1fs%s",
                    mode.width, mode.height, mode.num_frames, request.seed, strengths_note,
                    t_stage1 - t0, f"{stage2_seconds:.1f}s" if stage2_seconds > 0 else "skipped",
                    gen_seconds, phase_note)
        return {
            "frames": frames,
            "audio": audio_np,
            "audio_sample_rate": sample_rate,
            "gen_seconds": gen_seconds,
            "phases": dict(phases or {}),
        }
