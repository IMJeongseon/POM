#!/usr/bin/env python
"""Tri-conditional editing on FLUX (working, 2026-06-01): structure + appearance + text bg.

Built on the prior appearance-transfer paper's strategy, with TEXT-PROMPT BACKGROUND
control added (the novel axis). Two stages cleanly disentangle the three conditions:

  Stage 1 (structure + appearance):
    FluxControlNetImg2ImgPipeline -- img2img from the source (Blended-Noise structure
    lock) + FLUX depth ControlNet (structure) + FLUX.1-Redux (reference appearance).
  Stage 2 (text background):
    FluxInpaintPipeline -- inpaint the BACKGROUND (BiRefNet foreground mask inverted)
    with the target text prompt, keeping the appearance-transferred foreground intact.

This avoids the entanglement where the img2img/Redux backgrounds override the text.

Run:  conda run -n pdg python scripts/flux_tri_conditional.py \
        --source data/animals/animal0.png --appearance data/animals/cheetah.png \
        --bg-prompt "a snowy winter field, deep snow, overcast sky" --out outputs/tri.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageFilter

BASE = "black-forest-labs/FLUX.1-dev"
REDUX = "black-forest-labs/FLUX.1-Redux-dev"
DEPTH_CN = "jasperai/Flux.1-dev-Controlnet-Depth"


def build():
    from diffusers import (
        FluxControlNetImg2ImgPipeline,
        FluxControlNetModel,
        FluxInpaintPipeline,
        FluxPriorReduxPipeline,
    )
    from flow_pdg.masks import BiRefNetMasker

    cn = FluxControlNetModel.from_pretrained(DEPTH_CN, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        BASE, controlnet=cn, torch_dtype=torch.bfloat16).to("cuda")
    prior = FluxPriorReduxPipeline.from_pretrained(REDUX, torch_dtype=torch.bfloat16).to("cuda")

    # Purified Redux (paper §2.2): mask-weight SigLIP patch tokens by the reference
    # foreground so background/boundary patches are suppressed -> texture-only, no shape.
    # This lets a LOWER img2img strength carry appearance, preserving fine source features
    # (eyes/face) instead of destroying them (raw Redux needed strength~0.9 -> head artifacts).
    from flow_pdg.masks import foreground_token_weights

    def _purify_hook(module, inp, out):
        m = getattr(prior, "_purify_mask", None)
        if m is not None:
            lhs = out.last_hidden_state
            w = foreground_token_weights(m, lhs.shape[1]).to(lhs.device, lhs.dtype)
            out.last_hidden_state = lhs * w[None, :, None]
        return out

    prior.image_encoder.register_forward_hook(_purify_hook)
    # inpaint reuses the loaded base components (no extra VRAM / download)
    inpaint = FluxInpaintPipeline(
        transformer=pipe.transformer, vae=pipe.vae, text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2, tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2, scheduler=pipe.scheduler)
    return pipe, prior, inpaint, BiRefNetMasker(device="cuda")


def depth_pil(source: Image.Image) -> Image.Image:
    from flow_pdg.depth import DepthEstimator
    de = DepthEstimator(device="cuda")
    x = (torch.from_numpy(np.asarray(source)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).cuda()
    d = de(x)
    return Image.fromarray((((d[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()))


def run(pipe, prior, inpaint, masker, source, appearance, bg_prompt, *,
        strength=0.7, redux_scale=1.0, cn_scale=0.8, bg_strength=0.95, steps=32, seed=0, purify=True):
    import numpy as _np
    depth = depth_pil(source)
    # Stage 1: structure + appearance (purified Redux preserves source fine features)
    if purify:
        at = (torch.from_numpy(_np.asarray(appearance)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).cuda()
        prior._purify_mask = masker(at)
    else:
        prior._purify_mask = None
    rx = prior(image=appearance, prompt_embeds_scale=redux_scale, pooled_prompt_embeds_scale=redux_scale)
    s1 = pipe(image=source, control_image=depth, strength=strength,
              controlnet_conditioning_scale=cn_scale, guidance_scale=3.5, num_inference_steps=steps,
              generator=torch.Generator("cuda").manual_seed(seed), **rx).images[0]
    # Stage 2: text background (inpaint everything outside the foreground)
    t = (torch.from_numpy(np.asarray(s1)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).cuda()
    fg = masker(t)[0, 0].cpu().numpy()
    bg = Image.fromarray(((fg < 0.5).astype("uint8") * 255)).filter(ImageFilter.GaussianBlur(3))
    s2 = inpaint(prompt=bg_prompt, image=s1, mask_image=bg, strength=bg_strength,
                 guidance_scale=3.5, num_inference_steps=steps,
                 generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    return s1, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--appearance", required=True)
    ap.add_argument("--bg-prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--redux-scale", type=float, default=1.0)
    a = ap.parse_args()
    pipe, prior, inpaint, masker = build()
    src = Image.open(a.source).convert("RGB").resize((a.res, a.res))
    app = Image.open(a.appearance).convert("RGB").resize((a.res, a.res))
    _, s2 = run(pipe, prior, inpaint, masker, src, app, a.bg_prompt,
                strength=a.strength, redux_scale=a.redux_scale)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s2.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
