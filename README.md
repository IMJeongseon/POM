# flow-pdg — Training-free Tri-conditional Image Editing on Flow-Matching Backbones

Training-free **tri-conditional** image control on flow-matching models (FLUX / SD3):
control **structure** (source image), **appearance** (reference image), and **background**
(text prompt) at once — e.g. *giraffe (structure) + zebra (appearance) + "snowy winter" (text)*.
Extends the diffusion-based PDG idea to flow matching (FLUX / SD3), building on
training-free appearance-transfer with DiTs and adding a text-driven background axis.

## Method (single-pass, FLUX)

- **Structure** — FireFlow high-fidelity inversion of the source + depth ControlNet.
- **Appearance** — purified FLUX.1-Redux image embedding (foreground-masked), amplified.
- **Background** — prompt-to-prompt (FlowEdit-style) scene edit, amplified.
- **Region-routed velocity** in one denoising loop: `v = M·v_fg + (1−M)·v_bg`, with
  region-dependent noise (foreground high-noise for appearance, background pinned to the
  source-scene prior for layout) — avoids the second-object ghost.

## Layout

```
src/flow_pdg/      core library (anchor, residual, routing, drift, backends, masks, depth)
scripts/           flux_fireflow.py (main single-pass), flux_tri.py (combos), verify/scan/...
configs/           pipeline configs
tests/             core unit tests (pytest tests/test_core.py)
docs/              design + implementation notes
```

## Setup / run

```bash
conda activate pdg            # torch + diffusers + cuda
python -m pytest tests/test_core.py -q
python scripts/flux_fireflow.py        # single-pass tri-conditional -> outputs/flux_fireflow/
```

## Notes

- `references/` (paper PDFs), `data/` (input images), and `outputs/` are **git-ignored**
  (size / copyright / regenerable). To version your own inputs, remove `data/` from `.gitignore`.
- Models (FLUX.1-dev, FLUX.1-Redux-dev, jasperai depth ControlNet, SD3, BiRefNet,
  Depth-Anything-V2) are loaded from the HuggingFace cache.
