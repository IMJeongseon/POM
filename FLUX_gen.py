#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image with FLUX from a text prompt."
    )
    parser.add_argument("prompt", help="Text prompt for image generation.")
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-dev",
        help="Hugging Face model id. Default is FLUX.1-dev. Use FLUX.1-schnell for faster tests.",
    )
    parser.add_argument("--out", default="outputs/flux_output.png", help="Output image path.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids to use, e.g. '0,1'.")
    parser.add_argument("--device-map", default=None, choices=["balanced", "cuda", "cpu"], help="Multi-GPU placement strategy. Defaults to 'balanced' when --gpus has multiple ids.")
    parser.add_argument("--max-memory", default=None, help="Per-device memory limit, e.g. '0:20GiB,1:22GiB,cpu:96GiB'.")
    parser.add_argument("--print-device-map", action="store_true", help="Print accelerate device maps after model loading.")
    parser.add_argument("--split-transformer", action=argparse.BooleanOptionalAction, default=True, help="Load FLUX transformer separately with device_map so its internal layers can be split across GPUs.")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload model parts to CPU to reduce VRAM usage.",
    )
    return parser.parse_args()


def default_steps(model_id):
    return 4 if "schnell" in model_id.lower() else 50


def default_guidance_scale(model_id):
    return 0.0 if "schnell" in model_id.lower() else 3.5


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
    from diffusers import FluxPipeline, FluxTransformer2DModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. torch.cuda.is_available()가 False입니다.")

    steps = args.steps if args.steps is not None else default_steps(args.model)
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else default_guidance_scale(args.model)
    )

    device_map = args.device_map
    if device_map is None and len(gpu_ids) > 1:
        device_map = "balanced"

    load_kwargs = {"torch_dtype": torch.bfloat16}
    parsed_max_memory = parse_max_memory(args.max_memory)
    if device_map:
        load_kwargs["device_map"] = device_map
    if parsed_max_memory:
        load_kwargs["max_memory"] = parsed_max_memory

    if args.split_transformer and device_map:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            **load_kwargs,
        )
        pipe_kwargs = {"torch_dtype": torch.bfloat16, "transformer": transformer}
        if device_map:
            pipe_kwargs["device_map"] = device_map
        if parsed_max_memory:
            pipe_kwargs["max_memory"] = parsed_max_memory
        pipe = FluxPipeline.from_pretrained(args.model, **pipe_kwargs)
    else:
        pipe = FluxPipeline.from_pretrained(args.model, **load_kwargs)
    if args.print_device_map:
        for name in ["transformer", "text_encoder", "text_encoder_2", "vae"]:
            module = getattr(pipe, name, None)
            device_map_info = getattr(module, "hf_device_map", None)
            if device_map_info is not None:
                print(f"{name} device_map: {device_map_info}")

    if device_map:
        pass
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    generator = torch.Generator("cpu").manual_seed(args.seed)

    call_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    if "schnell" in args.model.lower():
        call_kwargs["max_sequence_length"] = 256

    image = pipe(**call_kwargs).images[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    print(f"Saved image: {out_path}")
    print(f"Model: {args.model}")
    if gpu_ids:
        print(f"GPUs: {', '.join(gpu_ids)}")
    if device_map:
        print(f"Device map: {device_map}")
    print(f"Steps: {steps}, guidance_scale: {guidance_scale}, seed: {args.seed}")


if __name__ == "__main__":
    main()
