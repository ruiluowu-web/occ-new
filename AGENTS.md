# AGENTS.md

## Overview
This is the **inference-only** release of OcclusionFormer (ICML 2026). No training code, no tests, no CI.

## Entrypoints
- `demo_occlusionformer.py` — Streamlit UI (recommended for interactive use)
- `inference_occlusionformer.py` — CLI batch/individual inference

## Environment
- Python 3.11, conda env: `conda create -n OcclusionFormer python=3.11 -y`
- Install: `pip install --upgrade -r requirements.txt`
- Backbone model: `black-forest-labs/FLUX.1-dev` (downloaded automatically by diffusers)
- Device: CUDA GPU expected; bfloat16 dtype

## Checkpoint setup
Download from HuggingFace (`FudanCVL/OcclusionFormer`) into `./ckpt/`. The directory must contain:
- `lora.safetensors` — LoRA adapter weights
- `occ.pth` — layout/occlusion-specific state dict

Alternatively, a merged `bundle_weights.safetensors` is supported (see `checkpoint_utils.py`).

## Source layout
```
src/
  occlusionformer/
    transformer.py    — OcclusionFormerFluxTransformer2DModel (extends FLUX DiT)
    tools.py          — Layout class, attention processors, encoding helpers
    inference.py      — Custom inference loop (monkey-patches FluxPipeline)
    config.py         — Block indices, mask predictor hyperparams
    checkpoint_utils.py — Bundle checkpoint loading (merged safetensors)
  utils.py            — Bbox conversion (xyxy↔xywh), palette generator
  transformer_utils.py — Custom FeedForward (used by transformer.py)
```

## Layout JSON format
```json
{
  "prompt": "...",
  "height": 1024, "width": 1024,
  "annos": [
    {"bbox": [x1,y1,x2,y2], "caption": "...", "category_name": "...", "occludes": "1,3"}
  ]
}
```
- `bbox` is xyxy, pixel coordinates. Both xyxy and xywh are supported at parse time (inferred from `x2>x1` check).
- `occludes` is 1-based comma-separated instance IDs this object covers (controls Z-order).
- `caption` per instance is required; `category_name` is optional.

## Key architectural notes
- The transformer is loaded with `block_type="occlusion"` to activate the occlusion-aware blocks.
- `grounding_ratio` (default 0.3) controls how many denoising steps use layout conditioning; after that threshold, layout is disabled for the remaining steps.
- The `inference()` function takes over `FluxPipeline.__call__` semantics — it uses `self = pipeline` internally to reuse pipeline internals.
- Occlusion ordering is modeled via `layout.occluder` (per-instance list of objects that occlude it) and `layout.bbox_masks`.
- Do **not** use `--enable_layout` and `--disable_layout` together on CLI; they are mutually exclusive.
