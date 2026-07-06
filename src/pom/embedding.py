from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class VisionEmbedder:
    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            embedding = outputs.last_hidden_state[:, 0][0]

        embedding = embedding.float().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def save_embedding(self, image_path: str | Path, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, self.embed_image(image_path))
        return out_path


def load_embedding(path: str | Path) -> np.ndarray:
    embedding = np.load(path).astype(np.float32)
    norm = np.linalg.norm(embedding) + 1e-8
    return embedding / norm
