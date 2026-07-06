#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pom.detection import ZeroShotObjectDetector, crop_with_manual_boxes
from pom.generation import FluxGenerator
from pom.memory import build_memory_from_bboxes
from pom.scoring import score_candidate


def parse_args():
    parser = argparse.ArgumentParser(description="Run training-free POM-Rerank.")
    parser.add_argument("--config", default="configs/dog_sequence.json")
    parser.add_argument("--run-dir", default="outputs/memory_guided/pom_rerank_dog")
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-detector", action="store_true")
    parser.add_argument("--detector-model", default="google/owlvit-base-patch32")
    parser.add_argument("--detector-threshold", type=float, default=0.02)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True, help="Reuse existing generated images when present.")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids to use, e.g. '0,1'.")
    parser.add_argument("--device-map", default=None, choices=["balanced", "cuda", "cpu"], help="Multi-GPU placement strategy. Defaults to 'balanced' when --gpus has multiple ids.")
    parser.add_argument("--max-memory", default=None, help="Per-device memory limit, e.g. '0:20GiB,1:22GiB,cpu:96GiB'.")
    parser.add_argument("--print-device-map", action="store_true", help="Print accelerate device maps after model loading.")
    parser.add_argument("--split-transformer", action=argparse.BooleanOptionalAction, default=True, help="Load FLUX transformer separately with device_map so its internal layers can be split across GPUs.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rewrite_prompt(prompt: str, objects: list[dict]) -> str:
    memory_text = []
    for obj in objects:
        description = obj.get("description", "")
        if description:
            memory_text.append(f"{obj['label']}: {description}")
    if not memory_text:
        return prompt
    return prompt + " Keep object identity consistent: " + "; ".join(memory_text) + "."


def candidate_seed(base_seed: int, step_index: int, candidate_index: int) -> int:
    return base_seed + step_index * 1000 + candidate_index


def main():
    args = parse_args()
    cfg = load_config(args.config)

    run_dir = Path(args.run_dir)
    images_dir = run_dir / "images"
    memory_dir = run_dir / "memory"
    candidates_dir = run_dir / "candidates"
    selected_dir = run_dir / "selected"
    for path in [images_dir, memory_dir, candidates_dir, selected_dir]:
        path.mkdir(parents=True, exist_ok=True)

    model_id = args.model or cfg["model"]
    base_seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    num_candidates = args.candidates if args.candidates is not None else cfg.get("candidates", 4)

    generator = FluxGenerator(
        model_id=model_id,
        cpu_offload=cfg.get("cpu_offload", True),
        gpus=args.gpus,
        device_map=args.device_map,
        max_memory=args.max_memory,
        verbose_device_map=args.print_device_map,
        split_transformer=args.split_transformer,
    )
    detector = ZeroShotObjectDetector(model=args.detector_model, threshold=args.detector_threshold) if args.use_detector else None
    prompts = cfg["prompts"]


    first = prompts[0]
    first_path = images_dir / "step_01.png"
    if args.reuse_existing and first_path.exists():
        print(f"기존 첫 이미지 재사용: {first_path}")
    else:
        generator.generate(
            first["prompt"],
            first_path,
            seed=base_seed,
            width=cfg.get("width", 1024),
            height=cfg.get("height", 1024),
            steps=cfg.get("steps"),
            guidance_scale=cfg.get("guidance_scale"),
        )

    object_specs = cfg["objects"]
    if detector:
        detected_specs = []
        for spec in cfg["objects"]:
            bbox = detector.detect_best_box(first_path, spec.get("detect_label", spec["label"]))
            if bbox is None:
                if "bbox_xyxy" not in spec:
                    raise RuntimeError(f"첫 이미지에서 객체 탐지 실패: {spec['id']}({spec['label']})")
                bbox = spec["bbox_xyxy"]
                print(f"탐지 실패, config bbox 사용: {spec['id']} -> {bbox}")
            detected = dict(spec)
            detected["bbox_xyxy"] = bbox
            detected_specs.append(detected)
        object_specs = detected_specs

    bank = build_memory_from_bboxes(first_path, object_specs, memory_dir / "objects")
    bank_path = memory_dir / "memory_bank.json"
    bank.to_json(bank_path)

    selected = [{"step": 1, "prompt": first["prompt"], "image": str(first_path)}]
    all_scores = []

    for step_index, step_cfg in enumerate(prompts[1:], start=2):
        prompt = rewrite_prompt(step_cfg["prompt"], cfg["objects"])
        target_objects = step_cfg.get("reappearing_objects", [])
        candidate_scores = []

        for cand_idx in range(num_candidates):
            seed = candidate_seed(base_seed, step_index, cand_idx)
            cand_path = candidates_dir / f"step_{step_index:02d}" / f"cand_{cand_idx:02d}.png"
            generator.generate(
                prompt,
                cand_path,
                seed=seed,
                width=cfg.get("width", 1024),
                height=cfg.get("height", 1024),
                steps=cfg.get("steps"),
                guidance_scale=cfg.get("guidance_scale"),
            )

            if not target_objects:
                candidate_scores.append(
                    {
                        "image_path": str(cand_path),
                        "mean_identity": 0.0,
                        "object_scores": [],
                        "seed": seed,
                    }
                )
                continue

            crop_dir = candidates_dir / f"step_{step_index:02d}" / f"cand_{cand_idx:02d}_crops"
            if detector:
                labels = {obj_id: next((obj.get("detect_label", obj["label"]) for obj in cfg["objects"] if obj["id"] == obj_id), bank.get(obj_id).label) for obj_id in target_objects}
                try:
                    crops = detector.crop_objects(cand_path, labels, crop_dir)
                except RuntimeError as exc:
                    print(f"후보 탐지 실패, identity 0 처리: {cand_path} ({exc})")
                    candidate_scores.append({
                        "image_path": str(cand_path),
                        "mean_identity": 0.0,
                        "object_scores": [],
                        "seed": seed,
                        "detector_error": str(exc),
                    })
                    continue
            else:
                manual = step_cfg.get("candidate_bboxes", {}).get(str(cand_idx))
                if manual is None:
                    raise RuntimeError(
                        "재등장 객체 rerank에는 --use-detector 또는 candidate_bboxes가 필요합니다."
                    )
                crops = crop_with_manual_boxes(cand_path, manual, crop_dir)

            score = score_candidate(cand_path, bank, crops)
            payload = asdict(score)
            payload["seed"] = seed
            candidate_scores.append(payload)

        best = max(candidate_scores, key=lambda item: item["mean_identity"])
        selected_path = selected_dir / f"step_{step_index:02d}.png"
        shutil.copyfile(best["image_path"], selected_path)
        selected.append(
            {
                "step": step_index,
                "prompt": step_cfg["prompt"],
                "rewritten_prompt": prompt,
                "image": str(selected_path),
                "best_score": best,
            }
        )
        all_scores.append({"step": step_index, "scores": candidate_scores})

    write_json(run_dir / "selected.json", {"selected": selected})
    write_json(run_dir / "scores.json", {"scores": all_scores})
    print(f"완료: {run_dir}")
    print(f"선택 결과: {run_dir / 'selected.json'}")
    print(f"점수 결과: {run_dir / 'scores.json'}")


if __name__ == "__main__":
    main()

