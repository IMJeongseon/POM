"""SD3 + ControlNet-Depth backend.

Extends SD3Backend with an InstantX SD3-Controlnet-Depth model. The source depth
(Depth-Anything-V2) is the structural scaffold injected at every denoising step,
so structure is preserved without residual decomposition or faithful inversion
(design pivot: prior-work style, depth carries geometry).

[HARDWARE-VERIFY] controlnet + transformer block-residual wiring.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import Conditioning, KVCache, VelocityOutput
from .sd3 import SD3Backend


class SD3DepthBackend(SD3Backend):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet_id: str = "InstantX/SD3-Controlnet-Depth",
        depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf",
        conditioning_scale: float = 0.7,
        guidance_scale: float = 5.0,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
    ):
        super().__init__(model_id=model_id, device=device, dtype=dtype)
        from diffusers import SD3ControlNetModel

        from ..depth import DepthEstimator

        self.controlnet = SD3ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=self.dtype
        ).to(device)
        self.conditioning_scale = conditioning_scale
        self.guidance_scale = guidance_scale  # classifier-free guidance for prompt adherence
        self.depth = DepthEstimator(model_id=depth_model, device=device)
        self._control_latent: Optional[torch.Tensor] = None  # VAE-encoded source depth
        self._null: Optional[Conditioning] = None  # cached empty-prompt embeds for CFG
        self._clip_vision = None  # lazy CLIP-L image encoder (Redux-analog)
        self._clip_proc = None

    def _ensure_clip_vision(self):
        if self._clip_vision is None:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            mid = "openai/clip-vit-large-patch14"
            self._clip_vision = CLIPVisionModelWithProjection.from_pretrained(
                mid, torch_dtype=self.dtype
            ).to(self.device).eval()
            self._clip_proc = CLIPImageProcessor.from_pretrained(mid)

    @torch.no_grad()
    def encode_appearance_embed(self, image: torch.Tensor) -> torch.Tensor:
        """CLIP-L image embedding (768-d, joint CLIP space) of the reference subject."""
        self._ensure_clip_vision()
        x = (image.float().clamp(-1, 1) + 1) / 2  # [0,1], float32 for the processor
        pixel_values = self._clip_proc(images=list(x), return_tensors="pt", do_rescale=False)
        pv = pixel_values["pixel_values"].to(self.device, self.dtype)
        return self._clip_vision(pixel_values=pv).image_embeds.float()  # [B,768]

    def appearance_conditioning(self, cond: Conditioning, app_embed, alpha: float) -> Conditioning:
        if app_embed is None or alpha <= 0:
            return cond
        pooled = cond.payload["pooled"].clone()
        d = app_embed.shape[-1]  # 768: the CLIP-L portion of SD3's [CLIP-L; CLIP-G] pooled
        ae = app_embed.to(pooled.dtype).to(pooled.device)
        pooled[:, :d] = (1 - alpha) * pooled[:, :d] + alpha * ae
        return Conditioning(payload={**cond.payload, "pooled": pooled})

    def set_structure_image(
        self, image: Optional[torch.Tensor], foreground_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Compute + cache the source depth control latent (None clears it).

        With foreground_mask, the background depth is flattened to a constant so the
        ControlNet only constrains the subject's geometry -- the background is then free
        to follow the text prompt (e.g. "snowy winter field").
        """
        if image is None:
            self._control_latent = None
            return
        depth_img = self.depth(image.to(self.device))           # [-1,1] depth, [B,3,H,W]
        if foreground_mask is not None:
            m = foreground_mask.to(depth_img.device, depth_img.dtype)
            if m.dim() == 3:
                m = m.unsqueeze(1)
            if m.shape[-2:] != depth_img.shape[-2:]:
                m = torch.nn.functional.interpolate(m, size=depth_img.shape[-2:], mode="bilinear")
            bg = torch.full_like(depth_img, -1.0)               # flat far plane in background
            depth_img = depth_img * m + bg * (1.0 - m)
        self._control_latent = self.encode_image(depth_img)     # VAE latent (float32)

    def _set_taps(self, capture_layers, inject_kv, inject_scale):
        replace = getattr(self, "_inject_replace", False)
        for li, tap in self._taps.items():
            tap.capture = li in capture_layers
            tap.replace = replace
            if inject_kv is not None and li in inject_kv.layers and inject_scale > 0:
                tap.inject = inject_kv.layers[li]
                tap.inject_scale = inject_scale
            else:
                tap.inject = None

    @torch.no_grad()
    def _forward(self, latent, t, enc, pooled) -> torch.Tensor:
        block_res = None
        if self._control_latent is not None:
            block_res = self.controlnet(
                hidden_states=latent.to(self.dtype),
                controlnet_cond=self._control_latent.to(self.dtype),
                conditioning_scale=self.conditioning_scale,
                encoder_hidden_states=enc,
                pooled_projections=pooled,
                timestep=t,
                return_dict=False,
            )[0]
        out = self.transformer(
            hidden_states=latent.to(self.dtype),
            timestep=t,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            block_controlnet_hidden_states=block_res,
            return_dict=False,
        )[0]
        return out.float()

    @torch.no_grad()
    def velocity(
        self,
        latent: torch.Tensor,
        sigma: float,
        cond: Conditioning,
        *,
        capture_layers: tuple[int, ...] = (),
        inject_kv: Optional[KVCache] = None,
        inject_scale: float = 1.0,
        inject_mask: Optional[torch.Tensor] = None,
    ) -> VelocityOutput:
        t = torch.tensor([sigma * 1000.0], device=self.device, dtype=self.dtype)
        enc = cond.payload["prompt_embeds"].to(self.dtype)
        pooled = cond.payload["pooled"].to(self.dtype)

        # conditional pass (capture/inject active here)
        self._set_taps(capture_layers, inject_kv, inject_scale)
        v_cond = self._forward(latent, t, enc, pooled)

        kv = None
        if capture_layers:
            kv = KVCache()
            for li in capture_layers:
                if li in self._taps and self._taps[li].captured is not None:
                    kv.layers[li] = self._taps[li].captured

        # classifier-free guidance: unconditional pass (no capture/inject) for prompt adherence
        if self.guidance_scale and self.guidance_scale != 1.0:
            if self._null is None:
                self._null = self.encode_prompt("")
            self._set_taps((), None, 0.0)
            v_unc = self._forward(
                latent, t,
                self._null.payload["prompt_embeds"].to(self.dtype),
                self._null.payload["pooled"].to(self.dtype),
            )
            v = v_unc + self.guidance_scale * (v_cond - v_unc)
        else:
            v = v_cond

        return VelocityOutput(velocity=v, kv=kv)
