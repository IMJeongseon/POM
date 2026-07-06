# POM-Rerank: Training-Free Persistent Object Memory

## Goal

POM-Rerank reduces reappearing object identity inconsistency without training.

The method keeps an explicit object memory from the first generated image, then uses that memory
to select the most identity-consistent candidate when an object reappears.

## Pipeline

1. Generate the initial image from the first prompt.
2. Extract object crops for persistent objects such as dog, ball, and dog house.
3. Store the crops and object descriptions in a memory bank.
4. For each later prompt, rewrite the prompt with memory descriptions.
5. Generate multiple candidate images with different seeds.
6. If target objects reappear, detect/crop them in each candidate.
7. Score each candidate by object-level visual similarity against the memory bank.
8. Select the candidate with the highest identity score.

## Current Implementation

Main script:

```bash
python3 scripts/pom_rerank.py --config configs/dog_sequence.json --use-detector
```

Default model:

```text
black-forest-labs/FLUX.1-dev
```

Default output:

```text
outputs/memory_guided/pom_rerank_dog/
```

Important files:

```text
src/pom/generation.py   # FLUX generation
src/pom/memory.py       # object memory bank
src/pom/detection.py    # zero-shot object detection and cropping
src/pom/scoring.py      # training-free identity score
scripts/pom_rerank.py   # full pipeline
configs/dog_sequence.json
```

## Notes

The first version uses a simple RGB histogram similarity score for object crops. This is a stable
starting point with minimal dependencies, but it should later be replaced or complemented with
DINOv2, SigLIP, or CLIP visual embeddings.

The detector path uses a frozen zero-shot detector through `transformers`. If detection fails,
manual candidate bounding boxes can be added to the config for controlled experiments.

## Next Improvements

1. Replace histogram scoring with DINOv2 object embedding similarity.
2. Add candidate grid visualization.
3. Add text-only and prompt-rewrite baselines.
4. Add object-memory update rules for visible objects.
5. Add optional test-time guidance after reranking is validated.
