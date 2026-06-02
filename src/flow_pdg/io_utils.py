"""Config loading + image I/O helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from .config import (
    AnchorConfig,
    InjectionConfig,
    PipelineConfig,
)


def _coerce_tuples(d: dict, keys: tuple[str, ...]) -> dict:
    for k in keys:
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return d


def load_config(path: str | Path) -> PipelineConfig:
    import yaml

    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    inj = InjectionConfig(**_coerce_tuples(raw.pop("injection", {}), ("layers", "window")))
    anch = AnchorConfig(**raw.pop("anchor", {}))
    return PipelineConfig(injection=inj, anchor=anch, **raw)


def load_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    """Load an image as [1,3,H,W] in [-1,1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size[1], size[0]))
    t = torch.from_numpy(_to_array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0)


def load_mask(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    """Load a foreground mask as [1,1,H,W] in [0,1]."""
    from PIL import Image

    m = Image.open(path).convert("L").resize((size[1], size[0]))
    t = torch.from_numpy(_to_array(m)).float() / 255.0
    return t.view(1, 1, size[0], size[1])


def save_image(t: torch.Tensor, path: str | Path) -> None:
    from PIL import Image

    arr = ((t[0].clamp(-1, 1) + 1.0) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def _to_array(img):
    import numpy as np

    return np.asarray(img)
