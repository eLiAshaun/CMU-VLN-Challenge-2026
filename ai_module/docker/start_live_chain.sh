#!/usr/bin/env bash
set -Eeuo pipefail

AI_ROOT="${MAST3R_AI_ROOT:-/home/docker/ai_module}"
QWEN_ENDPOINT="${MAST3R_QWEN_ENDPOINT:-@mast3r_qwen3vl}"
SAM_ENDPOINT="${MAST3R_SAM_ENDPOINT:-@mast3r_sam2}"
QWEN_CHECKPOINT="${MAST3R_QWEN_CHECKPOINT:-$AI_ROOT/checkpoints/qwen3vl/Qwen3-VL-8B-Instruct}"
SAM_CHECKPOINT="${MAST3R_SAM_CHECKPOINT:-$AI_ROOT/checkpoints/sam2/sam2.1_hiera_base_plus.pt}"
# build_sam2 resolves this name through Hydra's pkg://sam2 search path.  An
# absolute source-tree path is not a valid Hydra config name.
SAM_CONFIG="${MAST3R_SAM_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
SAM_CONFIG_FILE="$AI_ROOT/third_party/Grounded-SAM-2/sam2/$SAM_CONFIG"
OUTPUT_ROOT="${MAST3R_OUTPUT_ROOT:-$AI_ROOT/runs/live_robot}"
QWEN_QUANTIZATION="${MAST3R_QWEN_QUANTIZATION:-int8}"
QWEN_MAX_PIXELS="${MAST3R_QWEN_MAX_PIXELS:-1048576}"

export PYTHONPATH="$AI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AI_ROOT"

for required in "$QWEN_CHECKPOINT" "$SAM_CHECKPOINT" "$SAM_CONFIG_FILE" \
  "$AI_ROOT/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"; do
  if [[ ! -e "$required" ]]; then
    printf 'ERROR: required new-chain asset is missing: %s\n' "$required" >&2
    exit 1
  fi
done

children=()
cleanup() {
  trap - TERM INT EXIT
  if ((${#children[@]})); then
    kill "${children[@]}" >/dev/null 2>&1 || true
    wait "${children[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup TERM INT EXIT

python3 -m integrations.qwen3vl.worker_server \
  --checkpoint "$QWEN_CHECKPOINT" \
  --endpoint "$QWEN_ENDPOINT" \
  --quantization "$QWEN_QUANTIZATION" \
  --max-pixels "$QWEN_MAX_PIXELS" &
children+=("$!")
qwen_pid="$!"

python3 -m integrations.perception.sam2_worker_server \
  --checkpoint "$SAM_CHECKPOINT" \
  --config "$SAM_CONFIG" \
  --endpoint "$SAM_ENDPOINT" \
  --device cuda &
children+=("$!")
sam_pid="$!"

python3 -m integrations.qwen3vl.worker_probe \
  --endpoint "$QWEN_ENDPOINT" --timeout 180 --pid "$qwen_pid" &
qwen_probe_pid="$!"
python3 -m integrations.perception.sam2_worker_probe \
  --endpoint "$SAM_ENDPOINT" --timeout 180 --pid "$sam_pid" &
sam_probe_pid="$!"
wait "$qwen_probe_pid"
wait "$sam_probe_pid"

python3 -m integrations.ros.live_task_probe \
  --ai-root "$AI_ROOT" --output-root "$OUTPUT_ROOT" &
children+=("$!")

# The container is unhealthy as soon as any required persistent process exits.
wait -n "${children[@]}"
exit 1
