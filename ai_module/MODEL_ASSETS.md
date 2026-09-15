# Rebuild assets

| Model | Local path | Inference configuration |
|---|---|---|
| IDEA-Research/grounding-dino-tiny | `checkpoints/rebuild/grounding_dino_tiny` | native Transformers detection; custom CUDA kernels disabled; one prompt per image batch; box/text threshold 0.25 |
| SAM2.1 base+ | `checkpoints/rebuild/sam21/sam2.1_hiera_base_plus.pt` | image predictor; boxes supplied by GroundingDINO; one image encoding per view |
| Qwen/Qwen3-VL-4B-Instruct | `checkpoints/rebuild/qwen4b` | Transformers, BF16 weights and compute, no quantization; 262144 image pixels; 1536 parse / 192 verification output tokens |
| depth-anything/DA3METRIC-LARGE | `checkpoints/rebuild/da3metric` | one perspective view on demand, canonical depth converted once with processed focal length, then LiDAR Z calibration |

Qwen remains GPU-resident in BF16. GroundingDINO-Tiny plus SAM2.1 and DA3 are mutually exclusive auxiliary GPU groups; each model is initialized once, and phase changes move the inactive group to CPU before returning the required group to the configured device. Switching away from SAM2.1 clears its image cache. Completed RTX 5090 runs measured an AI process peak around 11.2 GiB; per-case timings and task outcomes are recorded separately in the development report.

The BF16 choice is based on the four-question compiler diagnostic in `runs/rebuild/compiler_bf16_diagnostic.json`, where the compiler output retained the expected relation structure and order across the four questions. The target 16 GB Laptop's timing and shared resource behavior remain unverified.

Meta access was denied. Florence-2 produced full-image photo detections in captured project views. A Qwen4B 2D prototype then emitted repeated invalid coordinates after movement. Both proposal paths are removed from runtime. The selected detector is IDEA-Research/grounding-dino-tiny, followed by the existing SAM2.1 base+ image predictor. Qwen remains the sole local language/verification model. `photo`, `tv cabinet`, and `tv` are queried as `picture frame`, `tv stand`, and `television`; stored semantic labels retain the task meaning. Detector vocabulary comes from the executable AST.

Qwen weights were copied out of the existing local Hugging Face cache into the explicit asset directory. DA3 is downloaded into that directory. Runtime uses local paths with offline Hub/Transformers settings and has no personal cache or login dependency. `python3 -m rebuild.prepare_assets --help` describes asset preparation; Qwen4B, GroundingDINO and DA3 use public repositories, and SAM2.1 uses its official public download server. Do not put tokens in source, build arguments, logs or chat.

The image uses `elias1012/cmu-vln-2026:latest` for Python 3.12, ROS Jazzy and CUDA-enabled PyTorch. `rebuild/requirements.txt` lists additional packages; Docker installs official DA3 source and the base image SAM2.1 package. Model licenses are retained with upstream packages/assets. No model, source or Docker hash verification is performed.

SAM3.1's published Object Multiplex improvements concern video multi-object tracking. Its H100/128-object speedup is not evidence of faster or better single-image segmentation here. Neither SAM3 nor SAM3.1 is part of this runtime. [Official release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

From `ai_module/`, prepare all four assets with:

```bash
python3 -m rebuild.prepare_assets --output-dir checkpoints/rebuild
```

SAM2.1 base+ uses the public checkpoint URL listed in the [official download script](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh), without a gated Hub repository. Existing assets are reused; this command does not perform hash verification. The local weights remain outside Git.

Asset preparation can run in the existing base image without installing packages on the host. From the repository root:

```bash
mkdir -p ai_module/checkpoints/rebuild
docker run --rm --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace -e HF_HOME=/tmp/cmu-rebuild-hf \
  -v "$PWD/ai_module/rebuild:/workspace/rebuild:ro" \
  -v "$PWD/ai_module/checkpoints/rebuild:/assets" \
  --entrypoint python3 elias1012/cmu-vln-2026:latest \
  -m rebuild.prepare_assets --output-dir /assets
```

Category verification now uses the same Qwen instance with left-padded batches of four current-region crops, bounded to 384 × 384 each. The generic physical-category prompt distinguishes object identity from depicted content; there are no category-specific examples or scene rules. Every proposed region is checked before memory fusion. Unconfirmed/negative proposals remain recorded separately.

Preparation is the network operation; the runtime image sets Hub/Transformers offline. Runtime directly imports its installed SAM2.1 and DA3 packages and does not search legacy or developer source directories.
