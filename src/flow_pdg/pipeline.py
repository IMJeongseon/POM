"""TriConditionalFlowPDG: depth-conditioned tri-conditional edit loop.

Pivoted design (prior-work style; no residual decomposition, no faithful-recon
requirement):

    structure  = source depth (ControlNet, every step) + Blended Noise Init (replay_k)
    appearance = reference K/V injected via Attention Context Expansion (vital layers),
                 localized to the foreground by velocity-level mask routing
    background = target text prompt ("snowy winter") drives the generation

Per step (two forward passes, mask-routed):
    v_bg = v(x, s, c_target | depth)                       # prompt + structure
    v_fg = v(x, s, c_target | depth, inject ref K/V)       # + appearance
    v    = M * v_fg + (1 - M) * v_bg                        # appearance only in foreground
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from typing import Optional

from .anchor import AnchorTrajectory, build_anchor
from .backends.base import KVCache, VelocityBackend
from .config import PipelineConfig
from .drift import drift_ratio, inject_mode, inject_scale
from .masks import foreground_token_index
from .routing import route_velocity, to_latent_mask
from .solver import euler_step


def _filter_kv(kv: Optional[KVCache], layers: Optional[set]) -> Optional[KVCache]:
    """Keep only the chosen layers of a KVCache (for the vital-layer scan)."""
    if kv is None or layers is None:
        return kv
    out = KVCache()
    for li, (k, v) in kv.layers.items():
        if li in layers:
            out.layers[li] = (k, v)
    return out


def _purify_kv(anchor: AnchorTrajectory, appearance_mask: torch.Tensor) -> None:
    """Filter each cached K/V to the reference's foreground tokens (in place)."""
    for kv in anchor.kv:
        if kv is None:
            continue
        for li, (k, v) in list(kv.layers.items()):
            idx = foreground_token_index(appearance_mask, k.shape[1]).to(k.device)
            if idx.any():
                kv.layers[li] = (k[:, idx, :], v[:, idx, :])


@dataclass
class EditInputs:
    source: torch.Tensor          # [B,3,H,W] in [-1,1]
    appearance: torch.Tensor      # [B,3,H,W] in [-1,1]
    fg_mask: torch.Tensor         # [B,1,H,W] in [0,1] (source foreground)
    source_prompt: str            # faithful source caption (for content-prior inversion)
    target_prompt: str            # tri-conditional background/scene ("... in a snowy field")
    appearance_prompt: str = ""   # optional caption for the reference inversion
    appearance_mask: "torch.Tensor | None" = None  # reference foreground -> purify K/V (§2.2)


@dataclass
class Prepared:
    src_anchor: AnchorTrajectory
    app_anchor: AnchorTrajectory
    mask: torch.Tensor
    c_tar: object
    c_tar_fg: object
    sigmas: torch.Tensor
    start: int


class TriConditionalFlowPDG:
    def __init__(self, backend: VelocityBackend, cfg: PipelineConfig):
        self.backend = backend
        self.cfg = cfg
        self.inject_layers = cfg.injection.layers or backend.default_inject_layers

    def prepare(self, inp: EditInputs, capture_layers: Optional[tuple] = None) -> Prepared:
        """Build anchors / conditioning once (layer-independent). Reused across the
        vital-layer scan; capture_layers defaults to the configured inject layers."""
        cfg = self.cfg
        be = self.backend
        capture = tuple(capture_layers) if capture_layers is not None else self.inject_layers

        c_tar = be.encode_prompt(inp.target_prompt)
        c_src = be.encode_prompt(inp.source_prompt)
        c_app = be.encode_prompt(inp.appearance_prompt)
        app_embed = be.encode_appearance_embed(inp.appearance)
        c_tar_fg = be.appearance_conditioning(c_tar, app_embed, cfg.injection.appearance_alpha)

        orig_g = getattr(be, "guidance_scale", None)
        try:
            if orig_g is not None:
                be.guidance_scale = 1.0  # inversion without CFG (unstable otherwise)
            be.set_structure_image(None)
            x0_app = be.encode_image(inp.appearance)
            app_anchor = build_anchor(be, x0_app, c_app, cfg.anchor, capture)
            if inp.appearance_mask is not None:
                _purify_kv(app_anchor, inp.appearance_mask)
            be.set_structure_image(inp.source, foreground_mask=inp.fg_mask)
            x0_src = be.encode_image(inp.source)
            src_anchor = build_anchor(be, x0_src, c_src, cfg.anchor, ())
        finally:
            if orig_g is not None:
                be.guidance_scale = orig_g

        h, w = be.latent_hw(cfg.height, cfg.width)
        mask = to_latent_mask(inp.fg_mask, (h, w), feather=cfg.mask_feather).to(x0_src.device)
        sigmas = src_anchor.sigmas
        start = src_anchor.node_at_or_below(cfg.anchor.replay_k)
        return Prepared(src_anchor, app_anchor, mask, c_tar, c_tar_fg, sigmas, start)

    def edit(self, p: Prepared, inject_layers: Optional[tuple] = None) -> torch.Tensor:
        """Run the edit loop. inject_layers restricts K/V injection to a subset
        (None = use all captured layers)."""
        cfg = self.cfg
        be = self.backend
        layers = set(inject_layers) if inject_layers is not None else None

        setattr(be, "_inject_replace", cfg.injection.mode == "replace")
        free = cfg.free_background
        start = 0 if free else p.start
        x = p.src_anchor.latents[start].clone()
        if free:
            # background restarts from noise at sigma=1 (full trajectory to form a scene);
            # foreground keeps the source content prior via masked replay (below).
            x = p.mask * x + (1.0 - p.mask) * torch.randn_like(x)
        replay_k = cfg.anchor.replay_k
        for i in range(start, len(p.sigmas) - 1):
            s, s_next = float(p.sigmas[i]), float(p.sigmas[i + 1])
            t = s

            if free and s > replay_k:
                # high-noise phase: pin foreground to the source trajectory (Blended Noise
                # Init, foreground-only). Released below replay_k so the foreground edits.
                x = p.mask * p.src_anchor.latents[i] + (1.0 - p.mask) * x

            v_bg = be.velocity(x, s, p.c_tar).velocity

            d = drift_ratio(x, p.src_anchor.latents[i])
            scale = inject_scale(inject_mode(d, cfg.injection)) * cfg.injection.strength
            in_window = cfg.injection.window[0] <= t <= cfg.injection.window[1]
            app_kv = p.app_anchor.kv[i] if i < len(p.app_anchor.kv) else None
            if scale > 0.0 and in_window and be.supports_kv_injection and app_kv is not None:
                kv = _filter_kv(app_kv, layers)
                v_fg = be.velocity(
                    x, s, p.c_tar_fg,
                    inject_kv=kv, inject_scale=scale, inject_mask=p.mask,
                ).velocity
            else:
                v_fg = be.velocity(x, s, p.c_tar_fg).velocity if cfg.injection.appearance_alpha > 0 else v_bg

            v_final = route_velocity(v_fg, v_bg, p.mask)
            x = euler_step(x, v_final, s, s_next)

        return be.decode_latent(x)

    def run(self, inp: EditInputs) -> torch.Tensor:
        return self.edit(self.prepare(inp))
