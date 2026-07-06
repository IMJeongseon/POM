#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image with Qwen-Image from a text prompt."
    )
    parser.add_argument("prompt", help="Text prompt for image generation.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Hugging Face model id.")
    parser.add_argument("--out", default="outputs/qwen_output.png", help="Output image path.")
    parser.add_argument(
        "--aspect",
        default="1:1",
        choices=sorted(ASPECT_RATIOS),
        help="Preset output aspect ratio.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override output height.")
    parser.add_argument("--width", type=int, default=None, help="Override output width.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument("--negative-prompt", default=" ", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids to use, e.g. '0,1'.")
    parser.add_argument("--device-map", default=None, choices=["balanced", "cuda", "cpu"], help="Multi-GPU placement strategy. Defaults to 'balanced' when --gpus has multiple ids.")
    parser.add_argument("--max-memory", default=None, help="Per-device memory limit, e.g. '0:20GiB,1:22GiB,cpu:96GiB'.")
    parser.add_argument("--print-device-map", action="store_true", help="Print accelerate device maps after model loading.")
    parser.add_argument(
        "--no-magic",
        action="store_true",
        help="Do not append Qwen's recommended quality suffix.",
    )
    return parser.parse_args()


def parse_gpu_ids(gpus):
    if not gpus:
        return []
    return [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]


def parse_max_memory(max_memory):
    if not max_memory:
        return None
    parsed = {}
    for item in max_memory.split(","):
        key, value = item.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = value.strip()
    return parsed


def main():
    args = parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from diffusers import DiffusionPipeline

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"
        print("CUDA를 찾지 못해 CPU로 실행합니다. 매우 느릴 수 있습니다.")

    width, height = ASPECT_RATIOS[args.aspect]
    width = args.width if args.width is not None else width
    height = args.height if args.height is not None else height

    prompt = args.prompt
    if not args.no_magic:
        prompt += ", Ultra HD, 4K, cinematic composition."

    device_map = args.device_map
    if device_map is None and len(gpu_ids) > 1:
        device_map = "balanced"

    load_kwargs = {"torch_dtype": dtype}
    parsed_max_memory = parse_max_memory(args.max_memory)
    if device_map and torch.cuda.is_available():
        load_kwargs["device_map"] = device_map
    if parsed_max_memory:
        load_kwargs["max_memory"] = parsed_max_memory

    pipe = DiffusionPipeline.from_pretrained(args.model, **load_kwargs)
    if args.print_device_map:
        for name in ["transformer", "text_encoder", "text_encoder_2", "vae"]:
            module = getattr(pipe, name, None)
            device_map_info = getattr(module, "hf_device_map", None)
            if device_map_info is not None:
                print(f"{name} device_map: {device_map_info}")
    if not device_map:
        pipe = pipe.to(device)

    generator = torch.Generator("cpu").manual_seed(args.seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=generator,
    ).images[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"Saved image: {out_path}")
    print(f"Model: {args.model}")
    if gpu_ids:
        print(f"GPUs: {', '.join(gpu_ids)}")
    if device_map:
        print(f"Device map: {device_map}")
    print(f"Size: {width}x{height}, steps: {args.steps}, true_cfg_scale: {args.true_cfg_scale}, seed: {args.seed}")


if __name__ == "__main__":
    main()
