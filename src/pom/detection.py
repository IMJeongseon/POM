from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_with_manual_boxes(
    image_path: str | Path,
    boxes_by_object: dict[str, list[int]],
    out_dir: str | Path,
) -> dict[str, str]:
    image = Image.open(image_path).convert("RGB")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = {}
    for object_id, bbox in boxes_by_object.items():
        crop_path = out_dir / f"{object_id}.png"
        image.crop(tuple(bbox)).save(crop_path)
        crops[object_id] = str(crop_path)
    return crops


class ZeroShotObjectDetector:
    def __init__(
        self,
        model: str = "google/owlvit-base-patch32",
        threshold: float = 0.08,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers가 필요합니다: pip install transformers") from exc

        self.detector = pipeline(
            task="zero-shot-object-detection",
            model=model,
            device=0,
        )
        self.threshold = threshold

    def detect_best_box(self, image_path: str | Path, label: str) -> list[int] | None:
        image = Image.open(image_path).convert("RGB")
        results = self.detector(image, candidate_labels=[label])
        results = [r for r in results if r["score"] >= self.threshold]
        if not results:
            return None
        best = max(results, key=lambda item: item["score"])
        box = best["box"]
        return [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]

    def crop_objects(
        self,
        image_path: str | Path,
        object_labels: dict[str, str],
        out_dir: str | Path,
        require_all: bool = True,
    ) -> dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        crops = {}
        missing = []

        for object_id, label in object_labels.items():
            bbox = self.detect_best_box(image_path, label)
            if bbox is None:
                missing.append(f"{object_id}({label})")
                continue
            crop_path = out_dir / f"{object_id}.png"
            image.crop(tuple(bbox)).save(crop_path)
            crops[object_id] = str(crop_path)

        if missing and require_all:
            raise RuntimeError("객체 탐지 실패: " + ", ".join(missing))
        if missing:
            crops["__missing__"] = missing
        return crops

