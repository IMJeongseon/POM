from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .memory import MemoryBank


@dataclass
class ObjectScore:
    object_id: str
    similarity: float
    reference_crop: str
    candidate_crop: str


@dataclass
class CandidateScore:
    image_path: str
    mean_identity: float
    object_scores: list[ObjectScore]


def _image_histogram(path: str | Path, bins: int = 24) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    hist = np.concatenate(hist_parts)
    norm = np.linalg.norm(hist) + 1e-8
    return hist / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def score_candidate(
    image_path: str | Path,
    bank: MemoryBank,
    candidate_crops: dict[str, str],
) -> CandidateScore:
    object_scores = []
    for object_id, candidate_crop in candidate_crops.items():
        memory = bank.get(object_id)
        ref = _image_histogram(memory.crop_path)
        cand = _image_histogram(candidate_crop)
        object_scores.append(
            ObjectScore(
                object_id=object_id,
                similarity=cosine_similarity(ref, cand),
                reference_crop=memory.crop_path,
                candidate_crop=str(candidate_crop),
            )
        )

    mean_identity = (
        float(np.mean([score.similarity for score in object_scores])) if object_scores else 0.0
    )
    return CandidateScore(
        image_path=str(image_path),
        mean_identity=mean_identity,
        object_scores=object_scores,
    )

