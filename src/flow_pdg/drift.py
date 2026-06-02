"""Drift-gated injection strength (design §6.4).

As the edit-branch latent x_E drifts away from the cached anchor latent x_A,
strong raw K/V injection produces ghosting / texture tearing. We gate injection
strength by the relative drift d_i = ||x_E - x_A|| / ||x_A||.
"""

from __future__ import annotations

from enum import Enum

import torch

from .config import InjectionConfig


class InjectMode(str, Enum):
    STRONG = "strong"   # full K/V injection
    SOFT = "soft"       # masked K/V / attention-bias only
    OFF = "off"         # residual-only, no raw K/V injection


def drift_ratio(x_edit: torch.Tensor, x_anchor: torch.Tensor) -> float:
    """d = ||x_E - x_A|| / ||x_A||  (scalar, averaged over batch)."""
    num = (x_edit - x_anchor).flatten(1).norm(dim=1)
    den = x_anchor.flatten(1).norm(dim=1).clamp_min(1e-12)
    return float((num / den).mean().item())


def inject_mode(d: float, cfg: InjectionConfig) -> InjectMode:
    if d < cfg.drift_strong:
        return InjectMode.STRONG
    if d < cfg.drift_medium:
        return InjectMode.SOFT
    return InjectMode.OFF


def inject_scale(mode: InjectMode) -> float:
    """A scalar multiplier applied to the injected K/V contribution."""
    return {InjectMode.STRONG: 1.0, InjectMode.SOFT: 0.5, InjectMode.OFF: 0.0}[mode]
