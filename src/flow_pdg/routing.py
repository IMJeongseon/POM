"""Velocity-level region routing (design §6.3).

Background is preserved by routing the *anchor* velocity outside the foreground
mask, instead of blending latents after the solver step (which causes ODE drift).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to_latent_mask(
    mask_pixel: torch.Tensor,
    latent_hw: tuple[int, int],
    feather: float = 2.0,
) -> torch.Tensor:
    """Downsample a pixel-space foreground mask to latent resolution + feather.

    Args:
        mask_pixel: [B,1,H,W] (or [1,1,H,W]) in {0,1} or [0,1].
        latent_hw:  (h, w) latent spatial size.
        feather:    Gaussian blur sigma (in latent pixels) for soft edges.
    Returns:
        [B,1,h,w] soft mask in [0,1].
    """
    if mask_pixel.dim() == 3:
        mask_pixel = mask_pixel.unsqueeze(1)
    h, w = latent_hw
    m = F.interpolate(mask_pixel.float(), size=(h, w), mode="bilinear", align_corners=False)
    if feather and feather > 0:
        m = _gaussian_blur(m, sigma=feather)
    return m.clamp(0.0, 1.0)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    c = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=c)
    x = F.conv2d(x, ky, padding=(radius, 0), groups=c)
    return x


def route_velocity(
    v_ctrl: torch.Tensor,
    v_anchor: torch.Tensor,
    mask_fg: torch.Tensor,
) -> torch.Tensor:
    """v_final = M*v_ctrl + (1-M)*v_anchor.

    `mask_fg` is broadcast over the channel dim. For sequence-shaped latents
    (FLUX packs tokens as [B, L, C]) pass a [B, L, 1] mask instead.
    """
    return mask_fg * v_ctrl + (1.0 - mask_fg) * v_anchor
