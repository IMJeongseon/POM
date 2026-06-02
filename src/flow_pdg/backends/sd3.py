"""Stable Diffusion 3 backend (MM-DiT, true CFG).

Built on diffusers' StableDiffusion3Pipeline. SD3 supports a genuine
unconditional pass, so text_residual="cfg" (v_base - v_null) works directly.

HARDWARE-VERIFY: the attention K/V capture/injection (capture_layers / inject_kv)
hooks into MM-DiT joint-attention and must be checked on a real SD3 checkpoint;
the velocity math and CFG path are backbone-standard. Marked with [HW] below.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import Conditioning, KVCache, VelocityBackend, VelocityOutput
from ._mmdit_attn import AttnTap, install_attn_taps


class SD3Backend(VelocityBackend):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        device: str = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
    ):
        from diffusers import StableDiffusion3Pipeline

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(device)
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self._taps: dict[int, AttnTap] = install_attn_taps(self.transformer)  # [HW]

    @property
    def supports_true_cfg(self) -> bool:
        return True

    @property
    def default_inject_layers(self) -> tuple[int, ...]:
        # Early-mid blocks (5-8 on SD3-medium's 24): the vital layers for *clean*
        # appearance transfer found by scripts/vital_layer_scan.py. first/last blocks
        # artifact; mid (8-15) transfers too weakly. Re-scan for other checkpoints.
        n = len(self.transformer.transformer_blocks)
        lo = max(1, round(n * 5 / 24))
        return tuple(range(lo, lo + 4))

    def sigmas(self, num_steps: int) -> torch.Tensor:
        self.scheduler.set_timesteps(num_steps, device=self.device)
        sig = self.scheduler.sigmas.to(self.device)  # [N+1], 1 -> 0
        return sig

    def encode_prompt(self, prompt: str, guidance: Optional[float] = None) -> Conditioning:
        pe, ppe, pool, _ = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        return Conditioning(payload={"prompt_embeds": pe, "pooled": pool})

    def null_conditioning(self) -> Conditioning:
        return self.encode_prompt("")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device, self.dtype)
        lat = self.vae.encode(image).latent_dist.sample()
        lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return lat.float()  # pipeline operates in float32; cast to model dtype inside velocity()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        lat = latent / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.vae.decode(lat.to(self.dtype)).sample.clamp(-1, 1)

    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        return height // self.vae_scale, width // self.vae_scale

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
        # FlowMatch: timestep t = sigma * num_train_timesteps
        t = torch.tensor([sigma * 1000.0], device=self.device, dtype=self.dtype)

        # configure attention taps for this call  [HW]
        for li, tap in self._taps.items():
            tap.capture = li in capture_layers
            if inject_kv is not None and li in inject_kv.layers and inject_scale > 0:
                tap.inject = inject_kv.layers[li]
                tap.inject_scale = inject_scale
                tap.inject_mask = inject_mask
            else:
                tap.inject = None

        model_out = self.transformer(
            hidden_states=latent.to(self.dtype),
            timestep=t,
            encoder_hidden_states=cond.payload["prompt_embeds"].to(self.dtype),
            pooled_projections=cond.payload["pooled"].to(self.dtype),
            return_dict=False,
        )[0]
        # SD3 transformer predicts velocity (flow-matching target).
        v = model_out.float()

        kv = None
        if capture_layers:
            kv = KVCache()
            for li in capture_layers:
                if self._taps[li].captured is not None:
                    kv.layers[li] = self._taps[li].captured
        return VelocityOutput(velocity=v, kv=kv)
