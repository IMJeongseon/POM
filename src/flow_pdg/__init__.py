"""flow-pdg: velocity-first redesign of PDG for flow-matching backbones (SD3, FLUX).

Realizes docs/pdg_to_flow_matching_design.md §6:
  - Anchor = deterministic ODE trajectory cache (not stochastic noise maps)
  - per-step velocity-residual decomposition with norm clamping
  - velocity-level region routing (not latent blending)
  - drift-gated attention K/V injection
  - Euler / FireFlow-wrap solver that calls a modified vector field F-tilde
"""

from .config import (
    PipelineConfig,
    InjectionConfig,
    AnchorConfig,
)
from .pipeline import EditInputs, TriConditionalFlowPDG

__all__ = [
    "PipelineConfig",
    "InjectionConfig",
    "AnchorConfig",
    "EditInputs",
    "TriConditionalFlowPDG",
]
