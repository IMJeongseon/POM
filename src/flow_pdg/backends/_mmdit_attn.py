"""MM-DiT attention taps via robust forward hooks (no attention reimplementation).

Strategy (matches design §6.4 / ReFlex I2I targeting):
  - capture: a forward hook on the image-stream `to_k`/`to_v` linears stores their
    output (the image keys/values) at selected blocks during source replay.
  - inject : the same hooks *concatenate* the reference's keys/values onto the
    image `to_k`/`to_v` output. Because diffusers' joint-attention reshapes K/V by
    a dynamic sequence length, lengthening K/V is transparent to the downstream
    processor: image queries simply attend to their own keys + the reference keys
    (Attention Context Expansion), while text keys/values are untouched.

This avoids depending on a specific diffusers JointAttnProcessor implementation, so
it is robust across versions. (inject_mask is not applied at the attention level in
v1; foreground localization is handled by velocity-level routing in the pipeline.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AttnTap:
    capture: bool = False
    captured: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    inject: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    inject_scale: float = 1.0
    inject_mask: Optional[torch.Tensor] = None  # reserved; not used at attn level in v1
    replace: bool = False                        # Cross-Image style: replace image K/V (not concat)
    _last_k: Optional[torch.Tensor] = None       # scratch for pairing k/v capture


def _make_k_hook(tap: AttnTap):
    def hook(module, inputs, output):
        if tap.capture:
            tap._last_k = output.detach()
        if tap.inject is not None and tap.inject_scale > 0:
            ref_k = _prep_ref(tap.inject[0], output)
            if tap.replace:
                # Cross-Image style: image queries attend ONLY to reference keys, so the
                # source cannot fall back on its own content -> appearance is forced.
                return ref_k
            return torch.cat([output, tap.inject_scale * ref_k], dim=1)
        return output

    return hook


def _prep_ref(ref, output):
    ref = ref.to(output.dtype).to(output.device)
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)
    return ref.expand(output.shape[0], -1, -1)


def _make_v_hook(tap: AttnTap):
    def hook(module, inputs, output):
        if tap.capture and tap._last_k is not None:
            tap.captured = (tap._last_k, output.detach())
            tap._last_k = None
        if tap.inject is not None and tap.inject_scale > 0:
            ref_v = _prep_ref(tap.inject[1], output)
            if tap.replace:
                return ref_v
            return torch.cat([output, tap.inject_scale * ref_v], dim=1)
        return output

    return hook


def install_attn_taps(transformer) -> dict[int, AttnTap]:
    """Register to_k/to_v forward hooks on each block's image-stream attention.

    Returns {block_index: AttnTap}.
    """
    taps: dict[int, AttnTap] = {}
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return taps
    for i, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "to_k") or not hasattr(attn, "to_v"):
            continue
        tap = AttnTap()
        attn.to_k.register_forward_hook(_make_k_hook(tap))
        attn.to_v.register_forward_hook(_make_v_hook(tap))
        taps[i] = tap
    return taps
