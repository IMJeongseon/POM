"""Source depth extraction (Depth-Anything-V2) -> ControlNet conditioning image.

Provides the structural scaffold that, together with Blended Noise Initialization,
preserves source geometry (replacing the residual/inversion-fidelity requirement).
"""

from __future__ import annotations

import torch


class DepthEstimator:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device: str = "cuda",
        dtype: torch.dtype | str = torch.float32,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,3,H,W] in [-1,1] -> depth control image [B,3,H,W] in [-1,1].

        Depth is min-max normalized to [0,1], replicated to 3 channels, then mapped
        to [-1,1] to match the VAE input range (ControlNet conditioning image).
        """
        b, _, h, w = image.shape
        pil = ((image.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        from PIL import Image

        imgs = [Image.fromarray(pil[i]) for i in range(b)]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device, self.dtype)
        pred = self.model(**inputs).predicted_depth  # [B,h',w']
        pred = pred.unsqueeze(1).float()
        pred = torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        # per-image min-max normalize
        flat = pred.flatten(1)
        lo = flat.min(1).values.view(-1, 1, 1, 1)
        hi = flat.max(1).values.view(-1, 1, 1, 1)
        depth = (pred - lo) / (hi - lo).clamp_min(1e-6)  # [0,1]
        depth3 = depth.repeat(1, 3, 1, 1)
        return (depth3 * 2 - 1).to(image.device)  # [-1,1]
