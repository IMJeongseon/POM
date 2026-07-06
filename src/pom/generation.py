from pathlib import Path
import os


def default_steps(model_id: str) -> int:
    return 4 if "schnell" in model_id.lower() else 50


def default_guidance_scale(model_id: str) -> float:
    return 0.0 if "schnell" in model_id.lower() else 3.5


def parse_gpu_ids(gpus: str | None) -> list[str]:
    if not gpus:
        return []
    return [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]


def parse_max_memory(max_memory: str | None) -> dict | None:
    if not max_memory:
        return None
    parsed = {}
    for item in max_memory.split(","):
        key, value = item.split(":", 1)
        key = key.strip()
        parsed[int(key) if key.isdigit() else key] = value.strip()
    return parsed


def validate_visible_cuda_devices(torch, gpu_ids: list[str], device_map: str | None) -> None:
    visible_count = torch.cuda.device_count()
    if gpu_ids and visible_count < len(gpu_ids):
        raise RuntimeError(
            "Requested GPUs are not all visible to PyTorch/CUDA. "
            f"--gpus={','.join(gpu_ids)} requested {len(gpu_ids)} device(s), "
            f"but torch sees {visible_count}. "
            "Avoid unhealthy GPUs and remember that CUDA_VISIBLE_DEVICES remaps selected "
            "physical GPUs to logical ids 0..N-1."
        )
    if device_map == "balanced" and visible_count < 2:
        raise RuntimeError(
            "device_map='balanced' needs at least two visible CUDA devices, "
            f"but torch sees {visible_count}. Use a valid multi-GPU set or switch to "
            "single-GPU/CPU offload settings."
        )


class FluxGenerator:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        cpu_offload: bool = True,
        gpus: str | None = None,
        device_map: str | None = None,
        max_memory: str | None = None,
        verbose_device_map: bool = False,
        split_transformer: bool = False,
    ) -> None:
        gpu_ids = parse_gpu_ids(gpus)
        if gpu_ids:
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다. torch.cuda.is_available()가 False입니다.")

        self.model_id = model_id
        if device_map is None and len(gpu_ids) > 1:
            device_map = "balanced"
        validate_visible_cuda_devices(torch, gpu_ids, device_map)

        parsed_max_memory = parse_max_memory(max_memory)
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if device_map:
            load_kwargs["device_map"] = device_map
        if parsed_max_memory:
            load_kwargs["max_memory"] = parsed_max_memory

        if split_transformer and device_map:
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                **load_kwargs,
            )
            pipe_kwargs = {"torch_dtype": torch.bfloat16, "transformer": transformer}
            if device_map:
                pipe_kwargs["device_map"] = device_map
            if parsed_max_memory:
                pipe_kwargs["max_memory"] = parsed_max_memory
            self.pipe = FluxPipeline.from_pretrained(model_id, **pipe_kwargs)
        else:
            self.pipe = FluxPipeline.from_pretrained(model_id, **load_kwargs)
        self.device_map = device_map
        self.gpu_ids = gpu_ids
        if verbose_device_map:
            for name in ["transformer", "text_encoder", "text_encoder_2", "vae"]:
                module = getattr(self.pipe, name, None)
                device_map_info = getattr(module, "hf_device_map", None)
                if device_map_info is not None:
                    print(f"{name} device_map: {device_map_info}")

        if device_map:
            self.device = "cuda"
        elif cpu_offload:
            self.pipe.enable_model_cpu_offload()
            self.device = "cpu"
        else:
            self.pipe.to("cuda")
            self.device = "cuda"

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        *,
        seed: int,
        image_path: str | Path | None = None,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Path:
        import torch

        steps = steps if steps is not None else default_steps(self.model_id)
        guidance_scale = (
            guidance_scale
            if guidance_scale is not None
            else default_guidance_scale(self.model_id)
        )

        generator = torch.Generator("cpu").manual_seed(seed)
        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if "schnell" in self.model_id.lower():
            kwargs["max_sequence_length"] = 256

        image = self.pipe(**kwargs).images[0]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path

