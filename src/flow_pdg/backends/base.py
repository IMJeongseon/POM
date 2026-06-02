"""Backend interface: a flow-matching velocity predictor with injection hooks.

A backend abstracts away SD3 vs FLUX vs a dummy test model. The pipeline only
talks to this interface, so the velocity-first PDG algorithm is backbone-agnostic
(design §7: common interface + per-backbone adapters).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class Conditioning:
    """Opaque, backend-specific conditioning bundle (encoded prompt(s), guidance)."""

    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class KVCache:
    """Per-layer key/value tensors captured from the source replay, for injection."""

    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


@dataclass
class VelocityOutput:
    velocity: torch.Tensor
    kv: Optional[KVCache] = None  # populated when capture=True


class VelocityBackend(ABC):
    """Minimal contract every backbone adapter must satisfy."""

    @property
    @abstractmethod
    def supports_true_cfg(self) -> bool:
        """True for SD3 (real CFG), False for guidance-distilled FLUX-dev."""

    @property
    @abstractmethod
    def default_inject_layers(self) -> tuple[int, ...]:
        ...

    def encode_appearance_embed(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Image-prompt embedding for the reference (Redux-analog). None if unsupported."""
        return None

    def appearance_conditioning(
        self, cond: "Conditioning", app_embed: Optional[torch.Tensor], alpha: float
    ) -> "Conditioning":
        """Blend the reference image embedding into the pooled 'tone' channel. Passthrough by default."""
        return cond

    def set_structure_image(
        self, image: Optional[torch.Tensor], foreground_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Provide a source image whose structure (e.g. depth) conditions every step.

        If foreground_mask is given, structure is constrained to the foreground only so
        the background is free to follow the text prompt. No-op for backends without a
        structural control (dummy / plain SD3 / FLUX).
        """
        return None

    @property
    def supports_kv_injection(self) -> bool:
        """True if K/V-concat (Attention Context Expansion) is wired for this backbone.

        SD3 (no RoPE in attention) supports it. FLUX applies rotary embeddings to
        keys with a fixed sequence length, so naive K/V concat breaks RoPE; FLUX
        injection needs a RoPE-aware / replace-mode path (not yet wired) -> False.
        """
        return True

    @abstractmethod
    def sigmas(self, num_steps: int) -> torch.Tensor:
        """Flow-time / sigma grid from 1 (noise) to 0 (data), length num_steps+1."""

    @abstractmethod
    def encode_prompt(self, prompt: str, guidance: Optional[float] = None) -> Conditioning:
        ...

    @abstractmethod
    def null_conditioning(self) -> Conditioning:
        """Unconditional bundle (only meaningful when supports_true_cfg)."""

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel image [B,3,H,W] in [-1,1] -> latent."""

    @abstractmethod
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent -> pixel image [B,3,H,W] in [-1,1]."""

    @abstractmethod
    def latent_hw(self, height: int, width: int) -> tuple[int, int]:
        """Latent spatial size for a given pixel resolution."""

    @abstractmethod
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
        """Predict velocity v(latent, sigma, cond).

        capture_layers : capture K/V at these blocks into the returned KVCache.
        inject_kv      : reference K/V to fuse into the listed blocks.
        inject_scale   : drift-gated multiplier (0 disables injection).
        inject_mask    : optional latent-resolution foreground mask to localize fusion.
        """
