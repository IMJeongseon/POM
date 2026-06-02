"""Foreground mask extraction via BiRefNet (cached ZhengPeng7/BiRefNet)."""

from __future__ import annotations

import torch


class BiRefNetMasker:
    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet", device: str = "cuda"):
        from transformers import AutoModelForImageSegmentation

        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        ).to(device).eval()
        self.size = (1024, 1024)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] in [-1,1] -> foreground mask [B,1,H,W] in [0,1]."""
        import torch.nn.functional as F

        b, _, h, w = image.shape
        x = (image.clamp(-1, 1) + 1) / 2  # [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        xr = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        xr = (xr - mean) / std
        pred = self.model(xr.to(self.device))[-1].sigmoid()  # [B,1,h',w']
        mask = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        return mask.clamp(0, 1).to(image.device)


def foreground_token_index(mask_pixel: torch.Tensor, num_tokens: int, thresh: float = 0.5) -> torch.Tensor:
    """Pixel foreground mask -> boolean [num_tokens] over a sqrt(L) x sqrt(L) token grid.

    Used to *purify* reference K/V (§2.2): keep only the subject's tokens so injected
    appearance carries style, not the reference's background/shape.
    """
    import math

    import torch.nn.functional as F

    side = int(round(math.sqrt(num_tokens)))
    m = mask_pixel.float()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    g = F.interpolate(m, size=(side, side), mode="bilinear", align_corners=False)
    return (g.flatten() > thresh)  # [num_tokens] bool


def foreground_token_weights(mask_pixel: torch.Tensor, num_tokens: int, floor: float = 0.0) -> torch.Tensor:
    """Soft per-token foreground weight in [floor, 1] over a sqrt(L) grid.

    For purified Redux (paper §2.2): weight each SigLIP patch token by its foreground
    coverage so background/boundary patches are suppressed -> global shape is disrupted,
    leaving texture-only appearance. `floor` keeps a small residual if desired.
    """
    import math

    import torch.nn.functional as F

    side = int(round(math.sqrt(num_tokens)))
    m = mask_pixel.float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(1)
    g = F.interpolate(m, size=(side, side), mode="bilinear", align_corners=False).flatten()
    if g.numel() != num_tokens:  # handle a possible leading special token
        pad = num_tokens - g.numel()
        if pad > 0:
            g = torch.cat([torch.ones(pad, device=g.device), g])
        else:
            g = g[-num_tokens:]
    return floor + (1.0 - floor) * g.clamp(0, 1)
