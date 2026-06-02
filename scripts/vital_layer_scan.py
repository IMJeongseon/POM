#!/usr/bin/env python
"""Vital-layer scan for appearance K/V injection (SD3).

Builds anchors once (capturing K/V at every block), then re-runs only the edit loop
injecting at a single block at a time, scoring each with CLIP image similarity:
  app_gain = cos(out, appearance) - cos(source, appearance)   # appearance pulled in
  struct   = cos(out, source)                                  # structure kept (no collapse)
A good layer has high app_gain while struct stays reasonable (artifacts collapse both).

Redux-analog is disabled (appearance_alpha=0) to isolate each layer's K/V effect.

Run:  conda run -n pdg python scripts/vital_layer_scan.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from flow_pdg.backends import make_backend
from flow_pdg.config import PipelineConfig
from flow_pdg.io_utils import load_image, save_image
from flow_pdg.masks import BiRefNetMasker
from flow_pdg.pipeline import EditInputs, TriConditionalFlowPDG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/animals/animal0.png")
    ap.add_argument("--appearance", default="data/animals/cheetah.png")
    ap.add_argument("--source-prompt", default="a giraffe standing in a field")
    ap.add_argument("--target-prompt", default="a giraffe in a snowy winter field")
    ap.add_argument("--appearance-prompt", default="leopard fur spotted pattern")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--group", type=int, default=1, help="layers per scan group")
    ap.add_argument("--save-dir", default="outputs/scan")
    args = ap.parse_args()

    dev = "cuda"
    be = make_backend("sd3_depth", device=dev, dtype="bfloat16")
    n = len(be.transformer.transformer_blocks)
    all_layers = tuple(range(n))

    src = load_image(args.source, (args.res, args.res)).cuda()
    app = load_image(args.appearance, (args.res, args.res)).cuda()
    masker = BiRefNetMasker(device=dev)
    src_mask, app_mask = masker(src), masker(app)

    # CLIP embeds for scoring (reuse the backend's CLIP-L vision encoder)
    e_app = F.normalize(be.encode_appearance_embed(app), dim=-1)
    e_src = F.normalize(be.encode_appearance_embed(src), dim=-1)
    base_app = (e_src @ e_app.T).item()

    cfg = PipelineConfig(backend="sd3_depth", height=args.res, width=args.res, device=dev)
    cfg.anchor.num_steps = args.steps
    cfg.injection.drift_medium = 100.0  # no drift gate during scan
    cfg.injection.drift_strong = 100.0
    cfg.injection.window = (0.0, 1.0)
    cfg.injection.strength = args.strength
    cfg.injection.appearance_alpha = 0.0  # isolate K/V layer effect (no Redux-analog)

    pipe = TriConditionalFlowPDG(be, cfg)
    inp = EditInputs(src, app, src_mask, args.source_prompt, args.target_prompt,
                     args.appearance_prompt, appearance_mask=app_mask)

    print("building anchors once (capture all layers) ...")
    prepared = pipe.prepare(inp, capture_layers=all_layers)

    groups = [tuple(range(i, min(i + args.group, n))) for i in range(0, n, args.group)]
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"baseline cos(source, appearance) = {base_app:.4f}\n")
    print(f"{'layers':<14}{'app_gain':>10}{'struct':>9}")
    print("-" * 33)
    rows = []
    for g in groups:
        out = pipe.edit(prepared, inject_layers=g)
        e_out = F.normalize(be.encode_appearance_embed(out), dim=-1)
        app_sim = (e_out @ e_app.T).item()
        struct = (e_out @ e_src.T).item()
        gain = app_sim - base_app
        rows.append((g, gain, struct))
        tag = "".join(str(x) for x in g) if len(g) <= 3 else f"{g[0]}-{g[-1]}"
        print(f"{tag:<14}{gain:>+10.4f}{struct:>9.4f}")
        save_image(out.float().cpu(), f"{args.save_dir}/L{tag}.png")

    rows.sort(key=lambda r: r[1], reverse=True)
    print("\nTop layers by appearance gain (struct should stay > ~0.5 to avoid collapse):")
    for g, gain, struct in rows[:8]:
        print(f"  layers={g}  app_gain={gain:+.4f}  struct={struct:.4f}")
    print(f"\nper-layer renders saved -> {args.save_dir}/")


if __name__ == "__main__":
    main()
