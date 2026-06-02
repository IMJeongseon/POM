#!/usr/bin/env python
"""Compose a 1x3 grid (source | appearance | output) with the text prompt as caption.

Run:  conda run -n pdg python scripts/make_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

CELL = 384
GAP = 10
MARGIN = 14
LABEL_H = 30
CAP_H = 64
BG = (245, 245, 247)
FG = (20, 20, 24)


def _font(size: int):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def grid(source, appearance, output, prompt, out_path):
    labels = ["Source", "Appearance", "Output"]
    imgs = [Image.open(p).convert("RGB").resize((CELL, CELL)) for p in (source, appearance, output)]
    W = 3 * CELL + 2 * GAP + 2 * MARGIN
    H = MARGIN + LABEL_H + CELL + CAP_H + MARGIN
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    lf, cf = _font(20), _font(22)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        x = MARGIN + i * (CELL + GAP)
        tw = d.textlength(lab, font=lf)
        d.text((x + (CELL - tw) / 2, MARGIN + 4), lab, fill=FG, font=lf)
        canvas.paste(im, (x, MARGIN + LABEL_H))
    # caption (prompt) below
    cap_y = MARGIN + LABEL_H + CELL + 8
    lines = _wrap(d, f'Text prompt:  "{prompt}"', cf, W - 2 * MARGIN)
    for j, ln in enumerate(lines):
        d.text((MARGIN, cap_y + j * 26), ln, fill=FG, font=cf)
    canvas.save(out_path)
    print("saved", out_path)


JOBS = [
    ("animals/animal0.png", "animals/cheetah.png", "outputs/final_L5678_a0.3_s1.8.png",
     "a giraffe in a snowy winter field", "outputs/grid_giraffe.png"),
    ("birds/bird0.png", "birds/bird7.png", "outputs/dom_birds.png",
     "a bird perched on a branch in a snowy winter forest", "outputs/grid_birds.png"),
    ("cars/car0.png", "cars/car6.png", "outputs/dom_cars.png",
     "a vintage car on a snowy winter mountain road", "outputs/grid_cars.png"),
    ("fish/fish0.png", "fish/fish5.png", "outputs/dom_fish.png",
     "a fish swimming over a colorful coral reef", "outputs/grid_fish.png"),
]


def main():
    for s, a, o, p, out in JOBS:
        grid(ROOT / "data" / s, ROOT / "data" / a, ROOT / o, p, ROOT / out)
    # stacked sheet
    rows = [Image.open(ROOT / out).convert("RGB") for *_, out in JOBS]
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 10 * (len(rows) - 1)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10
    sheet.save(ROOT / "outputs/grid_all.png")
    print("saved", ROOT / "outputs/grid_all.png")


if __name__ == "__main__":
    main()
