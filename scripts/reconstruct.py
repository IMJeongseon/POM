#!/usr/bin/env python
"""Measure source-image reconstruction fidelity of the anchor inversion (design §4.2 ①).

Reconstruction = encode -> invert(data->noise) -> generate(noise->data) -> decode.
If the anchor can't faithfully reconstruct the source, the whole tri-conditional
decomposition is unreliable. We report PSNR(original, reconstruction) and latent MSE
for euler / midpoint / fireflow inversion across step counts.

Run:  conda run -n pdg python scripts/reconstruct.py --backend sd3 --source data/animals/animal0.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from flow_pdg.anchor import invert
from flow_pdg.backends import make_backend
from flow_pdg.io_utils import load_image, save_image

MODELS = {"sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
          "flux": "black-forest-labs/FLUX.1-dev"}


def generate(backend, x_T, cond, sigmas, method):
    """Integrate noise->data (sigmas 1->0) with the same scheme as invert()."""
    s_list = sigmas.tolist()  # 1 ... 0
    x = x_T
    v_reuse = None
    for i in range(len(s_list) - 1):
        s, s_next = s_list[i], s_list[i + 1]
        h = s_next - s
        if method == "euler":
            v = backend.velocity(x, s, cond).velocity
            x = x + h * v
        else:
            v1 = v_reuse if (method == "fireflow" and v_reuse is not None) else backend.velocity(x, s, cond).velocity
            x_mid = x + 0.5 * h * v1
            v_mid = backend.velocity(x_mid, s + 0.5 * h, cond).velocity
            x = x + h * v_mid
            v_reuse = v_mid
    return x


def psnr(a, b):
    a = (a.clamp(-1, 1) + 1) / 2
    b = (b.clamp(-1, 1) + 1) / 2
    mse = (a - b).pow(2).mean().item()
    return 10 * math.log10(1.0 / max(mse, 1e-10)), mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sd3", choices=["sd3", "flux"])
    ap.add_argument("--source", default="data/animals/animal0.png")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--methods", nargs="+", default=["euler", "midpoint", "fireflow"])
    ap.add_argument("--steps", nargs="+", type=int, default=[28, 50])
    ap.add_argument("--save-dir", default="outputs/recon")
    args = ap.parse_args()

    dev = "cuda"
    be = make_backend(args.backend, model_id=MODELS[args.backend], device=dev, dtype="bfloat16")
    img = load_image(args.source, (args.res, args.res)).to(dev)
    cond = be.encode_prompt(args.prompt)
    x0 = be.encode_image(img)

    # VAE round-trip floor (upper bound on achievable PSNR)
    vae_psnr, _ = psnr(img, be.decode_latent(x0))
    print(f"source={args.source} res={args.res} prompt={args.prompt!r}")
    print(f"VAE round-trip PSNR (ceiling): {vae_psnr:.2f} dB\n")
    print(f"{'method':<10}{'steps':>6}{'PSNR(dB)':>12}{'latentMSE':>14}")
    print("-" * 42)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for n in args.steps:
            sig = be.sigmas(n).to(dev)
            x_T = invert(be, x0, cond, sig, method=method)
            x0_rec = generate(be, x_T, cond, sig, method=method)
            rec_img = be.decode_latent(x0_rec)
            p, _ = psnr(img, rec_img)
            lat_mse = (x0_rec - x0).pow(2).mean().item()
            print(f"{method:<10}{n:>6}{p:>12.2f}{lat_mse:>14.4e}")
            save_image(rec_img.float().cpu(), f"{args.save_dir}/{args.backend}_{method}_{n}.png")
    print(f"\nreconstructions saved -> {args.save_dir}/")


if __name__ == "__main__":
    main()
