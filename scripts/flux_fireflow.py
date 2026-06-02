#!/usr/bin/env python
"""Custom FLUX FireFlow inversion + blended-noise edit (depth ControlNet) -- quality work.

Replaces the SDEdit/img2img start (which destroys fine source features at high strength)
with an ACCURATE rectified-flow inversion: invert the source to noise recording the
trajectory, then edit from a partial-replay node. Reconstruction (invert->regenerate, no
edit) should return the source faithfully -- the test gated here before editing.

Stage A here: validate reconstruction fidelity (PSNR) vs the SDEdit baseline.

Run:  conda run -n pdg python scripts/flux_fireflow.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

R = 768
DEV = "cuda"
DT = torch.bfloat16


def calc_mu(seq_len, base=256, mx=4096, base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (mx - base)
    return seq_len * m + base_shift - m * base


def main():
    cn = FluxControlNetModel.from_pretrained("jasperai/Flux.1-dev-Controlnet-Depth", torch_dtype=DT)
    pipe = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=cn, torch_dtype=DT).to(DEV)
    vae, tr, cnet, sched = pipe.vae, pipe.transformer, pipe.controlnet, pipe.scheduler
    vsf = pipe.vae_scale_factor

    OUTDIR = Path("outputs/flux_fireflow")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    giraffe = Image.open("data/animals/animal0.png").convert("RGB").resize((R, R))
    # compute source depth in-script (self-contained; no reliance on a saved file)
    from flow_pdg.depth import DepthEstimator
    _x = (torch.from_numpy(np.asarray(giraffe)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV)
    _d = DepthEstimator(device=DEV)(_x)
    depth = Image.fromarray((((_d[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()))
    depth.save(OUTDIR / "depth.png")

    def to_lat(pil):
        x = (torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV, DT)
        l = vae.encode(x).latent_dist.sample()
        return (l - vae.config.shift_factor) * vae.config.scaling_factor  # [1,C,Hl,Wl]

    x0 = to_lat(giraffe)
    ctrl = to_lat(depth)
    B, C, Hl, Wl = x0.shape
    packed = lambda l: FluxPipeline._pack_latents(l, B, C, Hl, Wl)
    x0p, ctrlp = packed(x0), packed(ctrl)
    img_ids = pipe._prepare_latent_image_ids(B, Hl // 2, Wl // 2, DEV, DT)
    # invert under a SOURCE-SCENE prompt (no subject word -> no ghost). Generating the
    # background under a TARGET-SCENE prompt then applies the scene delta (prompt-to-prompt
    # editing): the inverted noise reconstructs the source scene, the prompt change adds snow.
    INV_SCENE = "a grassy savanna with distant mountains and trees"
    pe, pooled, text_ids = pipe.encode_prompt(prompt=INV_SCENE, prompt_2=INV_SCENE, device=DEV, num_images_per_prompt=1, max_sequence_length=256)
    g1 = torch.tensor([1.0], device=DEV).expand(B)

    seq = (Hl // 2) * (Wl // 2)
    sched.set_timesteps(28, device=DEV, mu=calc_mu(seq))
    sig = sched.sigmas.to(DEV)  # [N+1], 1 -> 0

    @torch.no_grad()
    def vel(xp, sigma, pe_, pooled_, tids_, guid, use_ctrl=True):
        sg = torch.tensor([sigma], device=DEV).expand(B)
        bs = sbs = None
        if use_ctrl:
            bs, sbs = cnet(hidden_states=xp, controlnet_cond=ctrlp, controlnet_mode=None,
                           conditioning_scale=0.9, timestep=sg, guidance=guid,
                           pooled_projections=pooled_, encoder_hidden_states=pe_,
                           txt_ids=tids_, img_ids=img_ids, return_dict=False)
        return tr(hidden_states=xp, timestep=sg, guidance=guid, pooled_projections=pooled_,
                  encoder_hidden_states=pe_, controlnet_block_samples=bs,
                  controlnet_single_block_samples=sbs, txt_ids=tids_, img_ids=img_ids, return_dict=False)[0]

    def decode(xp, name):
        rec = FluxPipeline._unpack_latents(xp, R, R, vsf)
        im = vae.decode((rec / vae.config.scaling_factor + vae.config.shift_factor).to(DT)).sample.clamp(-1, 1)
        a = ((im[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
        Image.fromarray(a).save(OUTDIR / name)
        return a

    # FireFlow invert: data -> noise (reversed grid, midpoint reuse), under null+depth
    rev = list(reversed(sig.tolist()))
    x = x0p.clone(); v_reuse = None; traj = [x.clone()]
    for i in range(len(rev) - 1):
        s, sn = rev[i], rev[i + 1]; h = sn - s
        v1 = v_reuse if v_reuse is not None else vel(x, s, pe, pooled, text_ids, g1)
        vm = vel(x + 0.5 * h * v1, s + 0.5 * h, pe, pooled, text_ids, g1)
        x = x + h * vm; v_reuse = vm; traj.append(x.clone())

    # reconstruction check (no edit)
    x = traj[-1].clone()
    for i in range(len(sig) - 1):
        s, sn = float(sig[i]), float(sig[i + 1])
        x = x + (sn - s) * vel(x, s, pe, pooled, text_ids, g1)
    arr = decode(x, "ff_recon.png")
    mse = ((torch.from_numpy(np.asarray(giraffe)).float()/255) - (torch.from_numpy(arr).float()/255)).pow(2).mean().item()
    print(f"reconstruction PSNR = {10*math.log10(1/max(mse,1e-10)):.2f} dB  -> {OUTDIR}/ff_recon.png")

    # ---- EDIT: blended-noise init from replay_k + purified Redux appearance ----
    from diffusers import FluxPriorReduxPipeline
    from flow_pdg.masks import BiRefNetMasker, foreground_token_weights
    prior = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=DT).to(DEV)
    def ph(m, i, o):
        msk = getattr(prior, "_pm", None)
        if msk is not None:
            w = foreground_token_weights(msk, o.last_hidden_state.shape[1]).to(o.last_hidden_state.device, o.last_hidden_state.dtype)
            o.last_hidden_state = o.last_hidden_state * w[None, :, None]
        return o
    prior.image_encoder.register_forward_hook(ph)
    zebra = Image.open("data/animals/animal10.png").convert("RGB").resize((R, R))
    zt = (torch.from_numpy(np.asarray(zebra)).permute(2, 0, 1).float()/127.5-1).unsqueeze(0).to(DEV, DT)
    prior._pm = BiRefNetMasker(device=DEV)(zt)
    rx = prior(image=zebra)
    rpe, rpooled = rx["prompt_embeds"].to(DT), rx["pooled_prompt_embeds"].to(DT)
    rtids = torch.zeros(rpe.shape[1], 3, device=DEV, dtype=DT)
    gE = torch.tensor([3.5], device=DEV).expand(B)

    masker = BiRefNetMasker(device=DEV)

    # ---- SINGLE-PASS tri-conditional: region-routed velocity (one denoising loop) ----
    # No fresh-noise background (that caused ghosting + discarded the source scene).
    # The WHOLE image starts from one moderate blended-noise prior (source layout preserved
    # for BOTH regions); per step the velocity is region-routed:
    #   foreground -> depth + Redux appearance ;  background -> text *mood* (depth kept to
    #   preserve the source background layout, so text only shifts atmosphere, not the scene).
    import torch.nn.functional as Fn
    src_fg = masker(_x.to(DEV))
    m_tok = Fn.interpolate(src_fg.float(), size=(Hl // 2, Wl // 2), mode="bilinear", align_corners=False)
    m = m_tok.flatten().view(1, -1, 1).to(DEV, DT)            # [1, L, 1] packed-token foreground weight

    # Region-dependent noise: foreground starts at HIGH noise (erase source coat -> appearance
    # can take), background is PINNED to the source-scene prior (layout preserved). The bg pin
    # uses the source prior (savanna, no animal), so no second-animal ghost (unlike fresh noise).
    REPLAY_FG = 0.9   # foreground edit start (high => strong appearance)
    BG_LOCK = 0.75    # pin background to source prior while sigma>this; lower => layout preserved
                      # but weak mood, higher => more scene-edit range (stronger mood)
    ks = next(i for i in range(len(sig)) if float(sig[i]) <= REPLAY_FG)
    # TARGET-SCENE prompts = source scene + the desired change (prompt-to-prompt edit vs INV_SCENE).
    # Scene-only (no subject) so the foreground giraffe is not duplicated in the background.
    BG_TARGETS = {
        "snow": "a snowy winter field with distant snow-capped mountains and bare trees, deep snow on the ground",
        "autumn": "an autumn savanna with distant mountains, golden and orange fall foliage",
        "sunset": "a grassy savanna with distant mountains at sunset, warm orange dusk sky",
    }
    SCENE_W = 4.0     # amplify (target_scene - source_scene) for the BACKGROUND (FlowEdit-style)
    APP_W = 3.0       # amplify (Redux_appearance - structure) for the FOREGROUND, so the appearance
                      # actually transfers at a moderate REPLAY (no extra noise => no ghost / structure loss)
    epe, epooled, etids = pipe.encode_prompt(prompt="", prompt_2="", device=DEV, num_images_per_prompt=1, max_sequence_length=256)
    for bn, bp in BG_TARGETS.items():
        tpe, tpooled, ttids = pipe.encode_prompt(prompt=bp, prompt_2=bp, device=DEV, num_images_per_prompt=1, max_sequence_length=256)
        x = traj[len(sig) - 1 - ks].clone()  # whole image at high noise (REPLAY_FG)
        for i in range(ks, len(sig) - 1):
            s, sn = float(sig[i]), float(sig[i + 1])
            if s > BG_LOCK:   # pin background to source-scene prior (layout preserved, no 2nd animal)
                x = m * x + (1.0 - m) * traj[len(sig) - 1 - i]
            # foreground: structure base + amplified Redux-appearance direction
            v_struct = vel(x, s, epe, epooled, etids, gE, use_ctrl=True)   # depth only (no appearance)
            v_redux = vel(x, s, rpe, rpooled, rtids, gE, use_ctrl=True)    # depth + Redux appearance
            v_fg = v_struct + APP_W * (v_redux - v_struct)
            # background: source-scene base + amplified target-scene direction (prompt-pair)
            v_src = vel(x, s, pe, pooled, text_ids, gE, use_ctrl=False)
            v_tgt = vel(x, s, tpe, tpooled, ttids, gE, use_ctrl=False)
            v_bg = v_src + SCENE_W * (v_tgt - v_src)
            x = x + (sn - s) * (m * v_fg + (1.0 - m) * v_bg)
        decode(x, f"ff_single_{bn}.png")
        print(f"single-pass (amp appearance W={APP_W} + scene W={SCENE_W}) '{bn}' -> {OUTDIR}/ff_single_{bn}.png")


if __name__ == "__main__":
    main()
