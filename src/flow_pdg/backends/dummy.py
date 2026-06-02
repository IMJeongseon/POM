"""Analytic dummy backend for testing the algorithm without model weights or GPU.

The "model" is a hand-crafted, deterministic velocity field whose conditioning,
appearance injection and guidance each move the velocity in a *known direction*.
This lets tests assert that residual decomposition, routing, drift gating and the
solver are wired correctly (shapes, signs, separability) end-to-end on CPU.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch

from .base import Conditioning, KVCache, VelocityBackend, VelocityOutput


def _seed_vec(text: str, dim: int) -> torch.Tensor:
    h = hashlib.sha256(text.encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))
    return torch.randn(dim, generator=g)


class DummyBackend(VelocityBackend):
    """A toy rectified-flow field on a [B, C, h, w] latent.

    Target data x0 is encoded directly from the input image (identity VAE), so the
    "ideal" velocity points from noise toward x0. Each prompt adds a fixed
    per-channel bias direction; appearance injection adds another. This makes
    r_text and r_app analytically separable for assertions.
    """

    def __init__(self, channels: int = 4, downscale: int = 8, device: str = "cpu"):
        self.channels = channels
        self.downscale = downscale
        self.device = device
        self._x0: Optional[torch.Tensor] = None  # set on encode_image (data target)

    # -- capabilities -----------------------------------------------------
    @property
    def supports_true_cfg(self) -> bool:
        return True

    @property
    def default_inject_layers(self) -> tuple[int, ...]:
        return (0, 1)

    # -- schedule ---------------------------------------------------------
    def sigmas(self, num_steps: int) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

    # -- conditioning -----------------------------------------------------
    def encode_prompt(self, prompt: str, guidance: Optional[float] = None) -> Conditioning:
        bias = _seed_vec("prompt:" + prompt, self.channels).to(self.device)
        return Conditioning(payload={"bias": bias, "guidance": guidance or 0.0})

    def null_conditioning(self) -> Conditioning:
        return Conditioning(payload={"bias": torch.zeros(self.channels, device=self.device),
                                     "guidance": 0.0})

    # -- image (identity VAE) --------------------------------------------
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        lat = torch.nn.functional.adaptive_avg_pool2d(image, (h // self.downscale, w // self.downscale))
        lat = lat.mean(dim=1, keepdim=True).repeat(1, self.channels, 1, 1).to(self.device)
        self._x0 = lat
        return lat

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        img = latent.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        return torch.nn.functional.interpolate(
            img, scale_factor=self.downscale, mode="nearest"
        ).clamp(-1, 1)

    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        return height // self.downscale, width // self.downscale

    # -- velocity ---------------------------------------------------------
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
        # rectified-flow target velocity points from current latent toward data x0:
        # for x_t = (1-t) x0 + t*eps, ideal v = eps - x0 ~= (x_t - x0)/t. Use a stable
        # surrogate that drives latent->x0 and is well-defined at all sigma.
        x0 = self._x0 if self._x0 is not None else torch.zeros_like(latent)
        v = latent - x0  # base direction (data-anchored)

        bias = cond.payload["bias"].view(1, -1, 1, 1)
        v = v + bias  # prompt conditioning shifts velocity by a known direction

        guidance = float(cond.payload.get("guidance") or 0.0)
        if guidance:
            v = v + 0.1 * guidance * bias  # guidance-scalar effect

        kv_out = None
        if inject_kv is not None and inject_scale > 0.0:
            # appearance injection: add a direction derived from the reference K
            for _, (k, _vv) in inject_kv.layers.items():
                app_dir = k.mean(dim=0).view(1, -1, 1, 1)
                contrib = inject_scale * 0.5 * app_dir
                if inject_mask is not None:
                    contrib = contrib * inject_mask
                v = v + contrib

        if capture_layers:
            kv_out = KVCache()
            for li in capture_layers:
                # fake per-layer K/V keyed off the latent statistics
                k = latent.flatten(2).mean(-1).T.contiguous()  # [C, B] -> used as [tokens, C]
                kv_out.layers[li] = (k, k.clone())

        return VelocityOutput(velocity=v, kv=kv_out)
