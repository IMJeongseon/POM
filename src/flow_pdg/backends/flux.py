"""FLUX.1-dev backend (MM-DiT, guidance-distilled).

FLUX-dev is guidance-distilled, so there is no clean unconditional pass:
text_residual="cfg" is unreliable. Default to prompt-difference (FlowEdit-style)
or guidance-scalar residual (design §7).

Key FLUX specifics handled here:
  - guidance is a model input (a scalar embedding), not a CFG branch.
  - latents are *packed* into a [B, L, C] token sequence; routing/masks operate in
    unpacked spatial latent space, so we pack only inside velocity().
  - double-stream + single-stream blocks: vital layers are re-scanned, not 18-27.

[HARDWARE-VERIFY] attention taps + pack/unpack + guidance embedding wiring.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import Conditioning, KVCache, VelocityBackend, VelocityOutput
from ._mmdit_attn import AttnTap, install_attn_taps


class FluxBackend(VelocityBackend):
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        default_guidance: float = 3.5,
    ):
        from diffusers import FluxPipeline

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype
        self.default_guidance = default_guidance
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(device)
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # taps on double-stream blocks (cross-modal); hooks lengthen image K/V.
        self._taps: dict[int, AttnTap] = install_attn_taps(self.transformer)  # [HW]
        self.image_seq_len: int | None = None  # set in encode_image, used for dynamic shift mu

    def _shift_mu(self) -> float:
        cfg = self.scheduler.config
        base = getattr(cfg, "base_image_seq_len", 256)
        mx = getattr(cfg, "max_image_seq_len", 4096)
        base_s = getattr(cfg, "base_shift", 0.5)
        max_s = getattr(cfg, "max_shift", 1.16)
        seq = self.image_seq_len or 1024
        m = (max_s - base_s) / (mx - base)
        return seq * m + (base_s - m * base)

    @property
    def supports_true_cfg(self) -> bool:
        return False  # guidance-distilled

    @property
    def supports_kv_injection(self) -> bool:
        return False  # RoPE on keys breaks naive K/V concat; needs RoPE-aware path

    @property
    def default_inject_layers(self) -> tuple[int, ...]:
        # double-stream blocks; re-tune with a Stable-Flow vital-layer scan.
        n = len(self.transformer.transformer_blocks)
        return tuple(range(n // 3, 2 * n // 3))

    def sigmas(self, num_steps: int) -> torch.Tensor:
        kwargs = {}
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            kwargs["mu"] = self._shift_mu()
        self.scheduler.set_timesteps(num_steps, device=self.device, **kwargs)
        return self.scheduler.sigmas.to(self.device)

    def encode_prompt(self, prompt: str, guidance: Optional[float] = None) -> Conditioning:
        pe, pooled, _ = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=self.device, num_images_per_prompt=1,
        )
        g = self.default_guidance if guidance is None else guidance
        return Conditioning(payload={"prompt_embeds": pe, "pooled": pooled, "guidance": g})

    def null_conditioning(self) -> Conditioning:
        # not meaningful on FLUX-dev; provided for interface completeness.
        return self.encode_prompt("")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device, self.dtype)
        lat = self.vae.encode(image).latent_dist.sample()
        lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        _, _, h, w = lat.shape
        self.image_seq_len = (h // 2) * (w // 2)  # packed token count for dynamic-shift mu
        return lat.float()  # spatial [B,C,h,w] float32; packing/cast happens inside velocity()

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
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

        b, c, h, w = latent.shape
        packed = FluxPipeline._pack_latents(latent, b, c, h, w)
        ids = self.pipe._prepare_latent_image_ids(b, h // 2, w // 2, self.device, self.dtype)
        guidance = torch.tensor([cond.payload["guidance"]], device=self.device, dtype=self.dtype)
        t = torch.tensor([sigma], device=self.device, dtype=self.dtype)

        # configure attention taps for this call (hook-based; see _mmdit_attn).
        # Capture only: K/V-concat injection is disabled on FLUX (RoPE breaks concat;
        # see supports_kv_injection). inject_kv is intentionally ignored here.
        for li, tap in self._taps.items():
            tap.capture = li in capture_layers
            tap.inject = None

        model_out = self.transformer(
            hidden_states=packed.to(self.dtype),
            timestep=t,
            guidance=guidance,
            pooled_projections=cond.payload["pooled"].to(self.dtype),
            encoder_hidden_states=cond.payload["prompt_embeds"].to(self.dtype),
            txt_ids=torch.zeros(cond.payload["prompt_embeds"].shape[1], 3, device=self.device, dtype=self.dtype),
            img_ids=ids,
            return_dict=False,
        )[0]
        v = FluxPipeline._unpack_latents(model_out, h * self.vae_scale, w * self.vae_scale, self.vae_scale)

        kv = None
        if capture_layers:
            kv = KVCache()
            for li in capture_layers:
                if li in self._taps and self._taps[li].captured is not None:
                    kv.layers[li] = self._taps[li].captured
        return VelocityOutput(velocity=v.float(), kv=kv)
