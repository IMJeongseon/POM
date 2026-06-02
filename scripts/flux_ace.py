#!/usr/bin/env python
"""FLUX FireFlow + purified Redux + ACE (all three paper components) -> 1x3 grids.

Appearance is injected at the ATTENTION level (ACE) so it does not require destroying
the latent: FireFlow gives a faithful structure prior (eyes/face preserved) at a LOW
replay_k, while ACE replaces the source's image K/V with the reference's at vital FLUX
layers (RoPE-safe REPLACE -- same token length). Reference K/V are captured per step
from a parallel reference inversion (Cross-Image style).

Saves source|appearance|output 1x3 grids with a text caption to outputs/ACE/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxPriorReduxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from flow_pdg.backends._mmdit_attn import install_attn_taps
from flow_pdg.masks import BiRefNetMasker, foreground_token_weights

R, DEV, DT = 768, "cuda", torch.bfloat16


def calc_mu(seq, base=256, mx=4096, bs=0.5, ms=1.16):
    m = (ms - bs) / (mx - base)
    return seq * m + bs - m * base


def grid(src, app, out_arr, caption, path):
    C, pad, cap = 384, 14, 56
    imgs = [src.resize((C, C)), app.resize((C, C)), Image.fromarray(out_arr).resize((C, C))]
    labels = ["Source", "Appearance", "Output"]
    W, H = 3 * C + 4 * pad, pad + 24 + C + cap
    cv = Image.new("RGB", (W, H), (245, 245, 247)); d = ImageDraw.Draw(cv)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        f = ImageFont.load_default()
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        x = pad + i * (C + pad)
        d.text((x + C / 2 - d.textlength(lb, font=f) / 2, 6), lb, fill=(20, 20, 24), font=f)
        cv.paste(im, (x, pad + 24))
    d.text((pad, pad + 24 + C + 12), f'Text prompt:  "{caption}"', fill=(20, 20, 24), font=f)
    cv.save(path)


def main():
    cn = FluxControlNetModel.from_pretrained("jasperai/Flux.1-dev-Controlnet-Depth", torch_dtype=DT)
    pipe = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=cn, torch_dtype=DT).to(DEV)
    vae, tr, cnet, sched, vsf = pipe.vae, pipe.transformer, pipe.controlnet, pipe.scheduler, pipe.vae_scale_factor
    taps = install_attn_taps(tr)  # double-stream blocks
    n = len(tr.transformer_blocks)
    VITAL = {0, 1, 2, n - 3, n - 2, n - 1}

    prior = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=DT).to(DEV)
    def ph(m, i, o):
        msk = getattr(prior, "_pm", None)
        if msk is not None:
            w = foreground_token_weights(msk, o.last_hidden_state.shape[1]).to(o.last_hidden_state.device, o.last_hidden_state.dtype)
            o.last_hidden_state = o.last_hidden_state * w[None, :, None]
        return o
    prior.image_encoder.register_forward_hook(ph)
    masker = BiRefNetMasker(device=DEV)

    SRC = "data/animals/animal0.png"
    src_pil = Image.open(SRC).convert("RGB").resize((R, R))
    from flow_pdg.depth import DepthEstimator
    _x = (torch.from_numpy(np.asarray(src_pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV)
    _d = DepthEstimator(device=DEV)(_x)
    depth = Image.fromarray((((_d[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()))
    depth.save("outputs/fluxbase_depth.png")

    def to_lat(pil):
        x = (torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV, DT)
        l = vae.encode(x).latent_dist.sample()
        return (l - vae.config.shift_factor) * vae.config.scaling_factor

    x0 = to_lat(src_pil); ctrl = to_lat(depth)
    B, C, Hl, Wl = x0.shape
    pack = lambda l: FluxPipeline._pack_latents(l, B, C, Hl, Wl)
    x0p, ctrlp = pack(x0), pack(ctrl)
    img_ids = pipe._prepare_latent_image_ids(B, Hl // 2, Wl // 2, DEV, DT)
    pe, pooled, tids = pipe.encode_prompt(prompt="", prompt_2="", device=DEV, num_images_per_prompt=1, max_sequence_length=256)
    g = torch.tensor([1.0], device=DEV).expand(B)
    sched.set_timesteps(28, device=DEV, mu=calc_mu((Hl // 2) * (Wl // 2)))
    sig = sched.sigmas.to(DEV)

    def clear_taps():
        for t in taps.values():
            t.capture = False; t.inject = None; t.replace = False

    @torch.no_grad()
    def vel(xp, sigma, use_ctrl=True, capture=False, inject=False):
        sg = torch.tensor([sigma], device=DEV).expand(B)
        clear_taps()
        if capture:
            for li in VITAL:
                taps[li].capture = True
        if inject:
            for li in VITAL:
                if taps[li].captured is not None:
                    taps[li].inject = taps[li].captured; taps[li].replace = True
        bs = sbs = None
        if use_ctrl:
            bs, sbs = cnet(hidden_states=xp, controlnet_cond=ctrlp, controlnet_mode=None, conditioning_scale=0.7,
                           timestep=sg, guidance=g, pooled_projections=pooled, encoder_hidden_states=pe,
                           txt_ids=tids, img_ids=img_ids, return_dict=False)
        return tr(hidden_states=xp, timestep=sg, guidance=g, pooled_projections=pooled, encoder_hidden_states=pe,
                  controlnet_block_samples=bs, controlnet_single_block_samples=sbs, txt_ids=tids, img_ids=img_ids,
                  return_dict=False)[0]

    def invert(start_packed):
        rev = list(reversed(sig.tolist())); x = start_packed.clone(); vr = None; traj = [x.clone()]
        for i in range(len(rev) - 1):
            s, sn = rev[i], rev[i + 1]; h = sn - s
            v1 = vr if vr is not None else vel(x, s)
            vm = vel(x + 0.5 * h * v1, s + 0.5 * h)
            x = x + h * vm; vr = vm; traj.append(x.clone())
        return traj

    src_traj = invert(x0p)

    APPS = {"zebra": "animal10.png", "kangaroo": "animal2.png"}
    REPLAY_K = 0.85
    Path("outputs/ACE").mkdir(parents=True, exist_ok=True)
    for an, af in APPS.items():
        app_pil = Image.open(f"data/animals/{af}").convert("RGB").resize((R, R))
        at = (torch.from_numpy(np.asarray(app_pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV, DT)
        prior._pm = masker(at)
        # reference inverted trajectory (for per-step K/V capture)
        ref_traj = invert(pack(to_lat(app_pil)))
        ks = next(i for i in range(len(sig)) if float(sig[i]) <= REPLAY_K)
        x = src_traj[len(sig) - 1 - ks].clone()
        for i in range(ks, len(sig) - 1):
            s, sn = float(sig[i]), float(sig[i + 1])
            vel(ref_traj[len(sig) - 1 - i], s, use_ctrl=False, capture=True)   # capture ref K/V at this sigma
            v = vel(x, s, use_ctrl=True, inject=True)                          # replace into source (ACE)
            x = x + (sn - s) * v
        rec = FluxPipeline._unpack_latents(x, R, R, vsf)
        im = vae.decode((rec / vae.config.scaling_factor + vae.config.shift_factor).to(DT)).sample.clamp(-1, 1)
        arr = ((im[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        cap = f"a giraffe with {an} appearance (FireFlow + purified Redux + ACE)"
        grid(src_pil, app_pil, arr, cap, f"outputs/ACE/ace_{an}.png")
        print("saved", f"outputs/ACE/ace_{an}.png")


if __name__ == "__main__":
    main()
