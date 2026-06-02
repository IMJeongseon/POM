"""Configuration dataclasses for the velocity-first PDG pipeline.

Mirrors the knobs discussed in docs/pdg_to_flow_matching_design.md §6-§8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class InjectionConfig:
    """Appearance attention K/V injection + drift gating (§6.4)."""

    # transformer block indices to inject into (Stable-Flow vital layers /
    # ReFlex mid-step robust layers). Empty => backend default set.
    layers: tuple[int, ...] = ()
    # only inject within this flow-time window
    window: tuple[float, float] = (0.05, 0.70)
    # drift gating thresholds on d_i = ||x_E - x_A|| / ||x_A||
    drift_strong: float = 0.15   # below: strong K/V injection
    drift_medium: float = 0.35   # below: masked K/V / attn-bias only; above: residual-only
    # overall multiplier on the injected K/V contribution (appearance strength)
    strength: float = 1.0
    # Redux-analog: blend reference CLIP image embedding into the pooled "tone"
    # channel for the foreground pass (0 = off). Sets overall appearance tone.
    appearance_alpha: float = 0.6
    # how to combine reference K/V with source K/V at the attention level
    mode: Literal["concat", "replace"] = "concat"  # concat = Attention Context Expansion


@dataclass
class AnchorConfig:
    """ODE trajectory cache / inversion (§6.1)."""

    num_steps: int = 50
    # partial replay start (flow-time). edit branch starts integrating from t=replay_k.
    # higher (closer to noise=1.0) => more editability; lower => stronger structure.
    replay_k: float = 0.9
    # inversion solver. "midpoint" (RK2, 2 evals/step) reconstructs best (~34 dB with a
    # faithful source caption); "fireflow" reuses the midpoint velocity (~1 eval/step,
    # less accurate); "euler" is 1st-order and reconstructs poorly (~16 dB) -- avoid.
    inversion: Literal["midpoint", "fireflow", "euler"] = "midpoint"
    # cache the selected-layer K/V from the source replay for injection
    cache_kv: bool = True


@dataclass
class PipelineConfig:
    """Top-level config."""

    backend: Literal["sd3_depth", "sd3", "flux", "dummy"] = "sd3_depth"
    height: int = 1024
    width: int = 1024
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"

    injection: InjectionConfig = field(default_factory=InjectionConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)

    # region routing: feather width on the latent-resolution foreground mask (in latent px)
    mask_feather: float = 2.0
    # foreground-only Blended Noise Init: restart the background from noise (sigma=1) so
    # the text prompt fully drives it. Strong background change but trades off foreground
    # fidelity / is finicky -> off by default; foreground-only depth already frees the bg
    # somewhat. Enable for aggressive background edits (tune replay_k together).
    free_background: bool = False
