# POM: Persistent Object Memory

## Problem Definition: Training-Free Persistent Object Identity in Sequential Image Generation

## 1. Motivation

Text-to-image models can generate visually realistic single images from detailed prompts.
However, when a scene is described as a sequence of events, the same object often loses its
visual identity after it disappears and later reappears.

For example:

1. "A white dog is standing in a yard with a white dog chew, a toy ball and a dog's house."
2. "The dog grabs the ball, goes into the house, and is no longer visible."
3. "The dog comes out of the house carrying the ball in its mouth."

Although the phrases "the dog" and "the ball" refer to the previously introduced objects,
current image generation models often regenerate them as new instances. The reappearing dog may
change breed, size, fur texture, facial appearance, or collar details. The ball may also change
color, shape, or scale.

This reveals a gap between linguistic coreference and visual object persistence.

## 2. Core Problem

We define the problem as **Reappearing Object Identity Inconsistency**:

> Given a sequence of text prompts describing the same scene over time, a text-to-image model
> fails to preserve the visual identity of an object when the object becomes occluded, leaves the
> visible scene, or is temporarily absent, then reappears in a later image.

The key failure is not simply poor image quality. The model may produce plausible images at each
step, but the generated sequence is inconsistent because object identity is not preserved.

## 3. Research Setting

This project studies the problem under a **training-free** setting.

The method must not require:

- Fine-tuning the image generation model
- Training a new adapter
- Training a temporal consistency model
- Updating model weights

The method may use:

- Frozen text-to-image models such as FLUX or Qwen-Image
- Frozen vision encoders such as CLIP, DINOv2, or SigLIP
- Segmentation or detection models
- Prompt rewriting
- Reference crops or object masks
- Test-time guidance
- Attention or feature manipulation during inference
- External object memory constructed from generated images

## 4. Input and Output

### Input

A sequence of prompts:

```text
P_1, P_2, ..., P_T
```

where each prompt describes a scene state or event. Some noun phrases refer to objects introduced
in earlier prompts.

Optionally, the first generated image or user-provided reference image can be used to initialize
object memory.

### Output

A sequence of generated images:

```text
I_1, I_2, ..., I_T
```

The images should satisfy:

1. Each image matches its corresponding prompt.
2. The global scene remains coherent across the sequence.
3. Reappearing objects preserve their visual identity.
4. Temporarily invisible objects can be remembered without being forced to remain visible.

## 5. Object Identity Definition

For this study, an object identity is defined by visual attributes that should remain stable unless
the prompt explicitly changes them.

Examples:

- Dog: breed, body shape, fur color, fur texture, face structure, collar, size
- Ball: color pattern, shape, approximate size
- Dog house: structure, color, sign, position, scale

An object may change pose, location, visibility, or interaction state while preserving identity.

For example, the same dog may:

- Stand in the yard
- Enter the dog house and become invisible
- Come out of the dog house
- Carry the same ball in its mouth

These are state changes, not identity changes.

## 6. Failure Modes

This project focuses on the following failure modes.

### 6.1 Reappearance Drift

An object disappears and later reappears with a different visual identity.

Example:

- A fluffy white dog reappears as a smaller terrier-like dog.

### 6.2 Attribute Drift

Important visual attributes change across images.

Example:

- The ball changes from a blue-orange toy ball to a red ball.

### 6.3 Object Replacement

The model generates a plausible object of the same category, but not the same instance.

Example:

- "The dog" is interpreted as any dog rather than the previously generated dog.

### 6.4 Memory Collapse During Absence

When an object is not visible in an intermediate image, its visual identity is not retained for
later generation.

Example:

- The dog enters the house in image 2, then image 3 generates a new dog because no visible dog was
  available in image 2.

### 6.5 Scene-Object Binding Failure

The object is preserved partially, but its relation to the scene becomes inconsistent.

Example:

- The dog reappears, but the dog house changes position or the ball is no longer the same object.

## 7. Training-Free Research Hypothesis

The central hypothesis is:

> Reappearing object inconsistency can be reduced without training by constructing an explicit
> instance-level visual memory from earlier generated images and reusing that memory during later
> image generation.

The memory may include:

- Object crops
- Object masks
- CLIP/DINO/SigLIP embeddings
- Diffusion or flow model internal features
- Cross-attention maps
- Text-to-object bindings

Instead of relying only on textual expressions such as "the dog" or "the ball", the generation
process should bind these expressions to stored visual instances.

## 8. Candidate Method Family

This project will investigate training-free methods such as:

1. Prompt rewriting with explicit object descriptions
2. Reference crop conditioning
3. Instance memory bank construction
4. Object-level visual similarity guidance
5. Attention-level reuse of object features
6. Masked region guidance for reappearing objects
7. Occlusion-aware memory retention

The initial baseline should compare:

- Text-only sequential generation
- Prompt rewriting with object attributes
- Reference-image or crop-guided generation
- Memory-guided generation

## 9. Evaluation Goals

The evaluation should measure both prompt alignment and identity consistency.

Possible metrics:

- CLIP text-image similarity for prompt alignment
- DINOv2 or SigLIP similarity between original and reappearing object crops
- Object attribute consistency score
- Segmentation-based object localization consistency
- Human preference study

Important comparisons:

- Same object before disappearance vs after reappearance
- Object visible continuously vs object absent in the middle
- Text-only baseline vs training-free memory-guided method

## 10. Scope

This project focuses on image sequences generated from text prompts, not full video generation.

The target scenario is a short sequence of independent images that should share persistent object
identity. Temporal smoothness between adjacent frames is not the main goal. The main goal is
instance-level consistency under disappearance and reappearance.

## 11. Initial Example

Reference prompt sequence:

```text
P1: A white dog is standing in a yard with a white dog chew, a toy ball and a dog's house.
P2: The dog grabs the ball, goes into the house, and is no longer visible.
P3: The dog comes out of the house carrying the ball in its mouth.
```

Expected behavior:

- The dog in P3 should be the same dog introduced in P1.
- The ball in P3 should be the same ball introduced in P1.
- The dog house should remain the same object and location across images.
- The dog and ball may be invisible or partially visible in P2 without losing identity.

Observed failure:

- The dog reappears with a different breed or appearance.
- The ball changes color or size.
- The dog house sign, shape, or layout may drift.

This example will serve as the first benchmark case for the project.
