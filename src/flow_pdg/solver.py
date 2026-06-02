"""Flow-matching ODE step utilities (design §6.5).

FlowMatch convention: sigma runs 1 (noise) -> 0 (data); a generation step is
    x_next = x + (sigma_next - sigma) * v
with sigma_next < sigma. Inversion uses the same relation with sigma increasing.

The edit loop uses Euler (single eval) so per-step probing + K/V injection stay
consistent. FireFlow (2nd-order midpoint reuse) is used only for inversion/replay
where no injection happens, avoiding the high-order/injection conflict.
"""

from __future__ import annotations

import torch


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return x + (sigma_next - sigma) * v
