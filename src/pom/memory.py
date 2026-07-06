from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


BBox = tuple[int, int, int, int]


@dataclass
class ObjectMemory:
    object_id: str
    label: str
    source_image: str
    crop_path: str
    bbox_xyxy: BBox
    description: str = ""
    embedding_path: str | None = None


class MemoryBank:
    def __init__(self) -> None:
        self.objects: dict[str, ObjectMemory] = {}

    def add(self, memory: ObjectMemory) -> None:
        self.objects[memory.object_id] = memory

    def get(self, object_id: str) -> ObjectMemory:
        return self.objects[object_id]

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"objects": [asdict(obj) for obj in self.objects.values()]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "MemoryBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bank = cls()
        for item in payload["objects"]:
            item["bbox_xyxy"] = tuple(item["bbox_xyxy"])
            bank.add(ObjectMemory(**item))
        return bank


def crop_object(image_path: str | Path, bbox_xyxy: BBox, out_path: str | Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    crop = image.crop(bbox_xyxy)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def build_memory_from_bboxes(
    image_path: str | Path,
    object_specs: list[dict],
    out_dir: str | Path,
    embedder=None,
) -> MemoryBank:
    bank = MemoryBank()
    out_dir = Path(out_dir)
    for spec in object_specs:
        object_id = spec["id"]
        bbox = tuple(spec["bbox_xyxy"])
        crop_path = crop_object(image_path, bbox, out_dir / f"{object_id}.png")
        embedding_path = None
        if embedder is not None:
            embedding_path = embedder.save_embedding(
                crop_path,
                out_dir / "embeddings" / f"{object_id}.npy",
            )
        bank.add(
            ObjectMemory(
                object_id=object_id,
                label=spec["label"],
                source_image=str(image_path),
                crop_path=str(crop_path),
                bbox_xyxy=bbox,
                description=spec.get("description", ""),
                embedding_path=str(embedding_path) if embedding_path else None,
            )
        )
    return bank

