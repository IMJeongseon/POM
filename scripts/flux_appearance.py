#!/usr/bin/env python
"""FLUX appearance-transfer baseline (paper-faithful direction): Redux + structure-lock.

Validated recipe (2026-06-01): structure is locked by Blended Noise Init (img2img from
the source) + a FLUX depth ControlNet, while FLUX.1-Redux carries the reference
appearance. Unlike raw Redux+depth (which reproduces the reference animal), the img2img
init preserves the SOURCE structure, so the output = source shape + reference appearance.

Sweet spot: strength~0.8, redux_scale~1.0, controlnet_conditioning_scale~0.7.
Appearance follows the reference (cheetah->spots, tiger->warm, cow->white patches);
fine pattern geometry (e.g. zebra stripes) is still partial -> purification is future work.

This is the appearance+structure base; TEXT-prompt background control is the next axis.

Run:  conda run -n pdg python scripts/flux_appearance.py --source data/animals/animal0.png \
        --appearance data/animals/cheetah.png --out outputs/transfer.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from PIL import Image

DEPTH_CN = "jasperai/Flux.1-dev-Controlnet-Depth"
BASE = "black-forest-labs/FLUX.1-dev"
REDUX = "black-forest-labs/FLUX.1-Redux-dev"


def build():
    from diffusers import (
        FluxControlNetImg2ImgPipeline,
        FluxControlNetModel,
        FluxPriorReduxPipeline,
    )

    cn = FluxControlNetModel.from_pretrained(DEPTH_CN, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=cn, torch_dtype=torch.bfloat16
    ).to("cuda")
    prior = FluxPriorReduxPipeline.from_pretrained(REDUX, torch_dtype=torch.bfloat16).to("cuda")
    return pipe, prior


def depth_pil(source: Image.Image) -> Image.Image:
    from flow_pdg.depth import DepthEstimator
    import torch as _t

    de = DepthEstimator(device="cuda")
    x = (_t.from_numpy(__import__("numpy").asarray(source)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).cuda()
    d = de(x)
    arr = (((d[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy())
    return Image.fromarray(arr)


def transfer(pipe, prior, source, appearance, *, strength=0.8, redux_scale=1.0,
             cn_scale=0.7, steps=28, seed=0, depth=None):
    if depth is None:
        depth = depth_pil(source)
    rx = prior(image=appearance, prompt_embeds_scale=redux_scale,
               pooled_prompt_embeds_scale=redux_scale)
    return pipe(
        image=source, control_image=depth, strength=strength,
        controlnet_conditioning_scale=cn_scale, guidance_scale=3.5,
        num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(seed), **rx,
    ).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--appearance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--redux-scale", type=float, default=1.0)
    ap.add_argument("--cn-scale", type=float, default=0.7)
    a = ap.parse_args()
    pipe, prior = build()
    src = Image.open(a.source).convert("RGB").resize((a.res, a.res))
    app = Image.open(a.appearance).convert("RGB").resize((a.res, a.res))
    out = transfer(pipe, prior, src, app, strength=a.strength,
                   redux_scale=a.redux_scale, cn_scale=a.cn_scale)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
