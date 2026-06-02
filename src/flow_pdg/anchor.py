"""ODE trajectory cache = the flow-matching replacement for PDG's noise maps (§6.1).

Build steps:
  1. FireFlow/Euler-invert the source latent x0 -> x_T (noise side).
  2. Deterministically replay (noise -> data) under the *source* prompt, recording
     per-node latent x_i^A, velocity v_i^A, and selected-layer K/V.
The recorded latents are the anchors the edit branch compares against (drift) and
routes background to; the K/V are the appearance/structure features for injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .backends.base import Conditioning, KVCache, VelocityBackend
from .config import AnchorConfig


@dataclass
class AnchorTrajectory:
    sigmas: torch.Tensor                              # [N+1] from 1 -> 0
    latents: list[torch.Tensor] = field(default_factory=list)   # x_i^A at each node
    velocities: list[torch.Tensor] = field(default_factory=list)
    kv: list[Optional[KVCache]] = field(default_factory=list)
    x_T: Optional[torch.Tensor] = None                # noise-side latent

    def node_at_or_below(self, sigma_value: float) -> int:
        """Index of the first node whose sigma <= sigma_value (partial-replay start)."""
        for i, s in enumerate(self.sigmas.tolist()):
            if s <= sigma_value + 1e-6:
                return i
        return 0


def invert(
    backend: VelocityBackend,
    x0: torch.Tensor,
    cond: Conditioning,
    sigmas: torch.Tensor,
    method: str = "fireflow",
    record: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Map data latent x0 (sigma=0) -> noise-side latent x_T (sigma=1).

    Integrate dx/dsigma = v along the reversed sigma grid (0 -> 1).
      - "euler"    : 1st-order forward Euler (fast, inaccurate; naive RF inversion).
      - "midpoint" : 2nd-order RK2 midpoint (2 evals/step, accurate).
      - "fireflow" : midpoint method that *reuses* the previous step's midpoint
                     velocity as the endpoint estimate -> ~1 eval/step, ~2nd order.

    If record=True, also return the list of latents at every sigma node (the anchor
    trajectory, consistent with the inversion solver).
    """
    rev = list(reversed(sigmas.tolist()))  # 0 ... 1
    x = x0
    traj = [x.detach().clone()] if record else None
    v_reuse: Optional[torch.Tensor] = None
    for i in range(len(rev) - 1):
        s, s_next = rev[i], rev[i + 1]
        h = s_next - s
        if method == "euler":
            v = backend.velocity(x, s, cond).velocity
            x = x + h * v
        else:  # midpoint / fireflow
            if method == "fireflow" and v_reuse is not None:
                v1 = v_reuse                       # reuse previous midpoint velocity
            else:
                v1 = backend.velocity(x, s, cond).velocity
            x_mid = x + 0.5 * h * v1
            v_mid = backend.velocity(x_mid, s + 0.5 * h, cond).velocity
            x = x + h * v_mid
            v_reuse = v_mid
        if record:
            traj.append(x.detach().clone())
    if record:
        return x, traj
    return x


def build_anchor(
    backend: VelocityBackend,
    x0_src: torch.Tensor,
    src_cond: Conditioning,
    cfg: AnchorConfig,
    capture_layers: tuple[int, ...] = (),
) -> AnchorTrajectory:
    """Build a faithful anchor trajectory.

    The anchor latents are recorded *during inversion* (consistent with the
    inversion solver) rather than re-generated with a lower-order Euler pass, which
    would not retrace the source (empirically ~16 dB). K/V and velocity are then
    captured by a single forward eval at each recorded anchor state.
    """
    sigmas = backend.sigmas(cfg.num_steps).to(x0_src.device)
    x_T, inv_traj = invert(backend, x0_src, src_cond, sigmas, method=cfg.inversion, record=True)
    # inv_traj is indexed along reversed sigmas (0 -> 1, data -> noise);
    # reverse it to align with `sigmas` (1 -> 0, noise -> data).
    latents = list(reversed(inv_traj))

    traj = AnchorTrajectory(sigmas=sigmas, x_T=x_T, latents=latents)
    for i in range(len(sigmas) - 1):
        s = float(sigmas[i])
        out = backend.velocity(
            latents[i], s, src_cond,
            capture_layers=capture_layers if cfg.cache_kv else (),
        )
        traj.velocities.append(out.velocity.detach().clone())
        traj.kv.append(out.kv)
    return traj
