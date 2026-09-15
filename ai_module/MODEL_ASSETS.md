# Reusable model assets

This file documents model locations, asset requirements, loading entrypoints,
and recorded versions only. It is not an architecture authority. The current
AI Module architecture is defined only in
[`ARCHITECTURE_FINAL.md`](ARCHITECTURE_FINAL.md).

## Canonical manifest

`configs/model_assets.json` is the canonical relative-path manifest. Paths are
resolved from the `ai_module` directory. Model weights are intentionally not
tracked by Git.

## Asset locations

| Component | Required local asset | Loader / worker |
|---|---|---|
| MASt3R | `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | live model chain |
| Qwen3-VL | `checkpoints/qwen3vl/Qwen3-VL-8B-Instruct/` | `integrations.qwen3vl.worker_server` |
| SAM2 | `checkpoints/sam2/sam2.1_hiera_base_plus.pt` | `integrations.perception.sam2_worker_server` |
| SAM2 config | `third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml` | SAM2 worker |
| YOLO-World | `checkpoints/yolo_world/x_stage1-62b674ad.pth` | `integrations.perception.yolo_world_worker.py` |
| YOLO text encoder | `checkpoints/text_encoders/clip-vit-base-patch32/` | YOLO-World worker |
| Offline wheels | `vendor/wheels/` | container/offline installation |

Vendored sources are located under:

- `third_party/Grounded-SAM-2/`
- `third_party/YOLO-World/`

Each vendored project retains its own license and version metadata. In
particular, review YOLO-World's GPL-3.0 distribution obligations before
publishing a combined deliverable.

## Loading

Start the persistent Qwen3-VL worker from `ai_module`:

```bash
python3 -m integrations.qwen3vl.worker_server \
  --checkpoint checkpoints/qwen3vl/Qwen3-VL-8B-Instruct \
  --endpoint @mast3r_qwen3vl \
  --quantization int8 \
  --max-pixels 1048576
```

Probe that worker:

```bash
python3 -m integrations.qwen3vl.worker_probe \
  --endpoint @mast3r_qwen3vl \
  --timeout 120
```

Start the persistent SAM2 worker:

```bash
python3 -m integrations.perception.sam2_worker_server \
  --checkpoint checkpoints/sam2/sam2.1_hiera_base_plus.pt \
  --config third_party/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --endpoint @mast3r_sam2
```

YOLO-World and GroundingDINO assets remain available to the configured
perception ensemble. Their enablement and thresholds come only from
`configs/model_assets.json`; this document does not override runtime config.

## Recorded identifiers and versions

- Manifest schema: `2.1`
- Qwen model ID: `Qwen/Qwen3-VL-8B-Instruct`
- MASt3R checkpoint family:
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
- SAM2 checkpoint family: `sam2.1_hiera_base_plus`
- YOLO-World checkpoint: `x_stage1-62b674ad.pth`

`configs/model_assets.json` and the license/version files inside each vendored
source tree are the authoritative records for exact configured paths and source
revisions.
