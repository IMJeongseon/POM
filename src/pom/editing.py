from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from .generation import parse_gpu_ids, parse_max_memory


def default_edit_steps(model_id: str) -> int:
    if "kontext" in model_id.lower():
        return 28
    return 50


def default_edit_guidance_scale(model_id: str) -> float | None:
    if "kontext" in model_id.lower():
        return 3.5
    return None


def default_true_cfg_scale(model_id: str) -> float:
    if "qwen" in model_id.lower():
        return 4.0
    return 1.0


class ImageEditGenerator:
    def __init__(
        self,
        model_id: str,
        pipeline_name: str,
        *,
        gpus: str | None = None,
        device_map: str | None = None,
        max_memory: str | None = None,
        cpu_offload: bool = True,
        verbose_device_map: bool = False,
    ) -> None:
        gpu_ids = parse_gpu_ids(gpus)
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch
        from diffusers import FluxKontextPipeline, QwenImageEditPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다. torch.cuda.is_available()가 False입니다.")

        pipelines = {
            "flux-kontext": FluxKontextPipeline,
            "qwen-edit": QwenImageEditPipeline,
        }
        if pipeline_name not in pipelines:
            raise ValueError(f"unknown edit pipeline: {pipeline_name}")

        if device_map is None and len(gpu_ids) > 1:
            device_map = "balanced"

        load_kwargs = {"torch_dtype": torch.bfloat16}
        parsed_max_memory = parse_max_memory(max_memory)
        if device_map:
            load_kwargs["device_map"] = device_map
        if parsed_max_memory:
            load_kwargs["max_memory"] = parsed_max_memory

        self.model_id = model_id
        self.pipeline_name = pipeline_name
        self.pipe = pipelines[pipeline_name].from_pretrained(model_id, **load_kwargs)

        if verbose_device_map:
            for name in ["transformer", "text_encoder", "text_encoder_2", "vae"]:
                module = getattr(self.pipe, name, None)
                device_map_info = getattr(module, "hf_device_map", None)
                if device_map_info is not None:
                    print(f"{name} device_map: {device_map_info}")

        if not device_map:
            if cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")

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
        true_cfg_scale: float | None = None,
    ) -> Path:
        import torch

        image = Image.open(image_path).convert("RGB") if image_path else None
        steps = steps if steps is not None else default_edit_steps(self.model_id)
        guidance_scale = (
            guidance_scale
            if guidance_scale is not None
            else default_edit_guidance_scale(self.model_id)
        )
        true_cfg_scale = (
            true_cfg_scale
            if true_cfg_scale is not None
            else default_true_cfg_scale(self.model_id)
        )

        kwargs = {
            "image": image,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "generator": torch.Generator("cpu").manual_seed(seed),
        }
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if true_cfg_scale is not None:
            kwargs["true_cfg_scale"] = true_cfg_scale

        result = self.pipe(**kwargs).images[0]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        return out_path


def default_edit_model(pipeline_name: str) -> str:
    if pipeline_name == "flux-kontext":
        return "black-forest-labs/FLUX.1-Kontext-dev"
    if pipeline_name == "qwen-edit":
        return "Qwen/Qwen-Image-Edit"
    raise ValueError(f"unknown edit pipeline: {pipeline_name}")
