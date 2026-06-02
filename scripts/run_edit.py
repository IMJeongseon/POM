#!/usr/bin/env python
"""Tri-conditional flow-PDG edit CLI.

Example:
    python scripts/run_edit.py \
        --config configs/sd3_default.yaml \
        --source data/animals/animal0.png \
        --appearance data/animals/cheetah.png \
        --mask data/animals/animal0_mask.png \
        --source-prompt "a giraffe standing on grass" \
        --target-prompt "a giraffe in a snowy winter field" \
        --out outputs/giraffe_zebra_snow.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--appearance", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--source-prompt", required=True)
    ap.add_argument("--target-prompt", required=True)
    ap.add_argument("--appearance-prompt", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    from flow_pdg.backends import make_backend
    from flow_pdg.io_utils import load_config, load_image, load_mask, save_image
    from flow_pdg.pipeline import EditInputs, TriConditionalFlowPDG

    cfg = load_config(args.config)
    size = (cfg.height, cfg.width)

    backend = make_backend(cfg.backend, device=cfg.device, dtype=cfg.dtype)
    pipe = TriConditionalFlowPDG(backend, cfg)

    dev = cfg.device
    inp = EditInputs(
        source=load_image(args.source, size).to(dev),
        appearance=load_image(args.appearance, size).to(dev),
        fg_mask=load_mask(args.mask, size).to(dev),
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        appearance_prompt=args.appearance_prompt,
    )

    with torch.no_grad():
        out = pipe.run(inp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
