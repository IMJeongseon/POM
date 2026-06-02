#!/usr/bin/env python
"""Single-pass tri-conditional (structure + appearance + text background) over combos.

Generalizes scripts/flux_fireflow.py: models loaded once, run_combo() does FireFlow
inversion (under a source-scene prompt) + region-routed edit:
  foreground (source mask): high-noise start + depth + amplified Redux appearance,
  background: pinned to the source-scene prior (layout) then prompt-pair scene edit.
Region-dependent noise resolves the fg(needs high noise)/bg(needs low noise) conflict;
pinning bg to the SOURCE prior (not fresh noise) avoids the second-object ghost.

Out -> outputs/flux_fireflow/combo_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from diffusers import FluxControlNetModel, FluxControlNetPipeline, FluxPriorReduxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from flow_pdg.masks import BiRefNetMasker, foreground_token_weights
from flow_pdg.depth import DepthEstimator

R, DEV, DT = 768, "cuda", torch.bfloat16
REPLAY_FG, BG_LOCK, APP_W, SCENE_W = 0.9, 0.75, 3.0, 4.0


def calc_mu(seq, base=256, mx=4096, bs=0.5, ms=1.16):
    return seq * (ms - bs) / (mx - base) + bs - (ms - bs) / (mx - base) * base


def main():
    cn = FluxControlNetModel.from_pretrained("jasperai/Flux.1-dev-Controlnet-Depth", torch_dtype=DT)
    pipe = FluxControlNetPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", controlnet=cn, torch_dtype=DT).to(DEV)
    vae, tr, cnet, sched, vsf = pipe.vae, pipe.transformer, pipe.controlnet, pipe.scheduler, pipe.vae_scale_factor
    prior = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=DT).to(DEV)

    def ph(mod, i, o):
        msk = getattr(prior, "_pm", None)
        if msk is not None:
            w = foreground_token_weights(msk, o.last_hidden_state.shape[1]).to(o.last_hidden_state.device, o.last_hidden_state.dtype)
            o.last_hidden_state = o.last_hidden_state * w[None, :, None]
        return o
    prior.image_encoder.register_forward_hook(ph)
    masker = BiRefNetMasker(device=DEV)
    depther = DepthEstimator(device=DEV)
    OUT = Path("outputs/flux_fireflow"); OUT.mkdir(parents=True, exist_ok=True)

    def to_lat(pil):
        x = (torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV, DT)
        l = vae.encode(x).latent_dist.sample()
        return (l - vae.config.shift_factor) * vae.config.scaling_factor

    def enc(p):
        return pipe.encode_prompt(prompt=p, prompt_2=p, device=DEV, num_images_per_prompt=1, max_sequence_length=256)

    def run_combo(source, appearance, inv_scene, bg_targets, prefix):
        src_pil = Image.open(source).convert("RGB").resize((R, R))
        app_pil = Image.open(appearance).convert("RGB").resize((R, R))
        sx = (torch.from_numpy(np.asarray(src_pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV)
        d = depther(sx)
        depth_pil = Image.fromarray((((d[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()))

        x0, ctrl = to_lat(src_pil), to_lat(depth_pil)
        B, C, Hl, Wl = x0.shape
        pack = lambda l: FluxPipeline._pack_latents(l, B, C, Hl, Wl)
        x0p, ctrlp = pack(x0), pack(ctrl)
        img_ids = pipe._prepare_latent_image_ids(B, Hl // 2, Wl // 2, DEV, DT)
        g = torch.tensor([1.0], device=DEV).expand(B)
        sched.set_timesteps(28, device=DEV, mu=calc_mu((Hl // 2) * (Wl // 2)))
        sig = sched.sigmas.to(DEV)

        @torch.no_grad()
        def vel(xp, s, pe, po, ti, use_ctrl=True):
            sg = torch.tensor([s], device=DEV).expand(B)
            bs = sbs = None
            if use_ctrl:
                bs, sbs = cnet(hidden_states=xp, controlnet_cond=ctrlp, controlnet_mode=None, conditioning_scale=0.9,
                               timestep=sg, guidance=g, pooled_projections=po, encoder_hidden_states=pe,
                               txt_ids=ti, img_ids=img_ids, return_dict=False)
            return tr(hidden_states=xp, timestep=sg, guidance=g, pooled_projections=po, encoder_hidden_states=pe,
                      controlnet_block_samples=bs, controlnet_single_block_samples=sbs, txt_ids=ti, img_ids=img_ids,
                      return_dict=False)[0]

        # inversion under source-scene prompt (FireFlow midpoint)
        spe, spo, sti = enc(inv_scene)
        rev = list(reversed(sig.tolist())); x = x0p.clone(); vr = None; traj = [x.clone()]
        for i in range(len(rev) - 1):
            s, sn = rev[i], rev[i + 1]; h = sn - s
            v1 = vr if vr is not None else vel(x, s, spe, spo, sti)
            vm = vel(x + 0.5 * h * v1, s + 0.5 * h, spe, spo, sti); x = x + h * vm; vr = vm; traj.append(x.clone())

        # purified Redux appearance + empty (structure) baseline
        at = (torch.from_numpy(np.asarray(app_pil)).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0).to(DEV, DT)
        prior._pm = masker(at)
        rx = prior(image=app_pil)
        rpe, rpo = rx["prompt_embeds"].to(DT), rx["pooled_prompt_embeds"].to(DT)
        rti = torch.zeros(rpe.shape[1], 3, device=DEV, dtype=DT)
        epe, epo, eti = enc("")

        src_fg = masker(sx.to(DEV))
        m = Fn.interpolate(src_fg.float(), size=(Hl // 2, Wl // 2), mode="bilinear", align_corners=False).flatten().view(1, -1, 1).to(DEV, DT)
        ks = next(i for i in range(len(sig)) if float(sig[i]) <= REPLAY_FG)

        for bn, bp in bg_targets.items():
            tpe, tpo, tti = enc(bp)
            x = traj[len(sig) - 1 - ks].clone()
            for i in range(ks, len(sig) - 1):
                s, sn = float(sig[i]), float(sig[i + 1])
                if s > BG_LOCK:
                    x = m * x + (1.0 - m) * traj[len(sig) - 1 - i]
                v_struct = vel(x, s, epe, epo, eti, use_ctrl=True)
                v_redux = vel(x, s, rpe, rpo, rti, use_ctrl=True)
                v_fg = v_struct + APP_W * (v_redux - v_struct)
                v_src = vel(x, s, spe, spo, sti, use_ctrl=False)
                v_tgt = vel(x, s, tpe, tpo, tti, use_ctrl=False)
                v_bg = v_src + SCENE_W * (v_tgt - v_src)
                x = x + (sn - s) * (m * v_fg + (1.0 - m) * v_bg)
            rec = FluxPipeline._unpack_latents(x, R, R, vsf)
            im = vae.decode((rec / vae.config.scaling_factor + vae.config.shift_factor).to(DT)).sample.clamp(-1, 1)
            arr = ((im[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(arr).save(OUT / f"combo_{prefix}_{bn}.png")
            print(f"-> {OUT}/combo_{prefix}_{bn}.png")

    COMBOS = [
        ("data/animals/animal0.png", "data/animals/cheetah.png", "a grassy savanna with distant mountains and trees",
         {"snow": "a snowy winter field with distant snow-capped mountains and bare trees, deep snow on the ground"}, "giraffe-cheetah"),
        ("data/birds/bird0.png", "data/birds/bird7.png", "a bird perched on a branch with a blurred green forest background",
         {"snow": "a bird perched on a snowy branch, snowy winter forest, falling snow"}, "bird-eagle"),
        ("data/animals/animal0.png", "data/animals/animal2.png", "a grassy savanna with distant mountains and trees",
         {"autumn": "an autumn savanna with distant mountains, golden and orange fall foliage"}, "giraffe-kangaroo"),
    ]
    for src, app, inv, bg, pref in COMBOS:
        print(f"=== combo {pref} ===")
        run_combo(src, app, inv, bg, pref)


if __name__ == "__main__":
    main()
