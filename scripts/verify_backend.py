#!/usr/bin/env python
"""On-GPU smoke verification of a real backend ([HW] checklist, docs/IMPLEMENTATION.md).

Verifies, on a real checkpoint:
  1. pipeline load + attention-tap install
  2. prompt encoding
  3. VAE encode -> decode round-trip
  4. single velocity() forward (+ shape)
  5. capture path (K/V cached) and inject path (Attention Context Expansion)
  6. a short end-to-end TriConditionalFlowPDG.run producing a finite image

Run:  conda run -n pdg python scripts/verify_backend.py --backend sd3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from flow_pdg.backends import make_backend
from flow_pdg.config import PipelineConfig
from flow_pdg.pipeline import EditInputs, TriConditionalFlowPDG

MODELS = {
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.1-dev",
}


def box_mask(h, w):
    m = torch.zeros(1, 1, h, w)
    m[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sd3", choices=["sd3", "flux"])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="outputs/verify_sd3.png")
    args = ap.parse_args()

    dev = "cuda"
    print(f"[1] loading backend={args.backend} ({MODELS[args.backend]}) ...")
    be = make_backend(args.backend, model_id=MODELS[args.backend], device=dev, dtype="bfloat16")
    n_taps = len(getattr(be, "_taps", {}) or {})
    print(f"    OK. attention taps installed: {n_taps}")

    print("[2] encode_prompt ...")
    c = be.encode_prompt("a photo of an animal in a snowy field")
    print("    OK. keys:", list(c.payload.keys()))

    print("[3] VAE encode->decode round-trip ...")
    img = (torch.rand(1, 3, args.res, args.res, device=dev) * 2 - 1)
    lat = be.encode_image(img)
    rec = be.decode_latent(lat)
    print(f"    OK. latent {tuple(lat.shape)} dtype={lat.dtype}; recon {tuple(rec.shape)}")

    print("[4] velocity() forward ...")
    s = float(be.sigmas(args.steps)[0])
    layers = be.default_inject_layers[:2]
    v = be.velocity(lat, s, c).velocity
    assert v.shape == lat.shape and torch.isfinite(v).all()
    print(f"    OK. v {tuple(v.shape)} finite; sigma0={s:.3f}; inject_layers={layers}")

    print("[5] capture + inject path ...")
    cap = be.velocity(lat, s, c, capture_layers=layers)
    assert cap.kv is not None and len(cap.kv.layers) > 0, "capture failed"
    li = next(iter(cap.kv.layers))
    k, vv = cap.kv.layers[li]
    print(f"    capture OK. layer {li} K {tuple(k.shape)} V {tuple(vv.shape)}")
    if be.supports_kv_injection:
        inj = be.velocity(lat, s, c, inject_kv=cap.kv, inject_scale=1.0)
        assert inj.velocity.shape == lat.shape and torch.isfinite(inj.velocity).all()
        delta = (inj.velocity - v).abs().mean().item()
        assert delta > 0, "injection had no effect"
        print(f"    inject OK. mean|v_inj - v_base| = {delta:.4e} (>0 confirms ACE works)")
    else:
        print("    inject SKIPPED (backend.supports_kv_injection=False; RoPE-aware path pending)")

    print("[6] end-to-end short run ...")
    cfg = PipelineConfig(backend=args.backend, height=args.res, width=args.res, device=dev)
    cfg.anchor.num_steps = args.steps
    cfg.residual.text_residual = "cfg" if be.supports_true_cfg else "prompt_diff"
    pipe = TriConditionalFlowPDG(be, cfg)
    inp = EditInputs(
        source=img, appearance=(torch.rand(1, 3, args.res, args.res, device=dev) * 2 - 1),
        fg_mask=box_mask(args.res, args.res).to(dev),
        source_prompt="an animal standing on grass",
        target_prompt="an animal in a snowy winter field",
        appearance_prompt="leopard spotted fur",
    )
    out = pipe.run(inp)
    assert out.shape[-1] == args.res and torch.isfinite(out).all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    from flow_pdg.io_utils import save_image
    save_image(out.float().cpu(), args.out)
    print(f"    OK. end-to-end finite. saved -> {args.out}")
    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
