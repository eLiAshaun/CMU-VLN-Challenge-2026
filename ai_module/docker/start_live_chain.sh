#!/usr/bin/env bash
set -Eeuo pipefail

AI_ROOT="${MAST3R_AI_ROOT:-/home/docker/ai_module}"
QWEN_ENDPOINT="${MAST3R_QWEN_ENDPOINT:-@mast3r_qwen3vl}"
SAM_ENDPOINT="${MAST3R_SAM_ENDPOINT:-@mast3r_sam2}"
QWEN_CHECKPOINT="${MAST3R_QWEN_CHECKPOINT:-$AI_ROOT/checkpoints/qwen3vl/Qwen3-VL-8B-Instruct}"
SAM_CHECKPOINT="${MAST3R_SAM_CHECKPOINT:-$AI_ROOT/checkpoints/sam2/sam2.1_hiera_base_plus.pt}"
SAM_CONFIG="${MAST3R_SAM_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
SAM_CONFIG_FILE="$AI_ROOT/third_party/Grounded-SAM-2/sam2/$SAM_CONFIG"
OUTPUT_ROOT="${MAST3R_OUTPUT_ROOT:-$AI_ROOT/runs/live_robot}"
QWEN_QUANTIZATION="${MAST3R_QWEN_QUANTIZATION:-int8}"
QWEN_MAX_PIXELS="${MAST3R_QWEN_MAX_PIXELS:-1048576}"
QWEN_BATCH_MAX_NEW_TOKENS="${MAST3R_QWEN_BATCH_MAX_NEW_TOKENS:-256}"
YOLO_ENDPOINT="${MAST3R_YOLO_ENDPOINT:-@mast3r_yolo}"
case "${MAST3R_YOLO_ENABLED:-true}" in
  1|true|TRUE|yes|YES|on|ON) YOLO_ENABLED=true ;;
  *) YOLO_ENABLED=false ;;
esac
# Export the normalized value so the config loader and the worker launcher see
# the same switch. YOLO is now enabled by default as primary proposal generator.
export MAST3R_YOLO_ENABLED="$YOLO_ENABLED"

# Integrity/checksum gates are deliberately disabled for this competition
# runtime.  Assets are accepted when they exist, are readable, and their
# persistent worker can actually load and answer a health probe.
export MAST3R_CHECKSUM_VERIFICATION=disabled
export MAST3R_MODEL_HASH_VERIFICATION=disabled
export MAST3R_SOURCE_HASH_VERIFICATION=disabled
export MAST3R_RELEASE_MANIFEST_GENERATION=disabled

export PYTHONPATH="$AI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AI_ROOT"

for required in "$QWEN_CHECKPOINT" "$SAM_CHECKPOINT" "$SAM_CONFIG_FILE"; do
  if [[ ! -r "$required" ]]; then
    printf 'ERROR: required lean-chain asset is missing or unreadable: %s\n' "$required" >&2
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

# Qwen and SAM start concurrently.  Qwen is the primary task-conditioned
# proposal generator; SAM converts those boxes into masks.
python3 -m integrations.qwen3vl.worker_server \
  --checkpoint "$QWEN_CHECKPOINT" \
  --endpoint "$QWEN_ENDPOINT" \
  --quantization "$QWEN_QUANTIZATION" \
  --max-pixels "$QWEN_MAX_PIXELS" \
  --batch-max-new-tokens "$QWEN_BATCH_MAX_NEW_TOKENS" &
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

# YOLO-World is optional supplementation.  Its absence must never make the
# proposal stream empty because Qwen remains the primary proposal backend.
if [[ "$YOLO_ENABLED" == "true" ]]; then
  python3 -m integrations.perception.yolo_world_server \
    --model-config "$AI_ROOT/third_party/YOLO-World/configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py" \
    --checkpoint "$AI_ROOT/checkpoints/yolo_world/x_stage1-62b674ad.pth" \
    --text-model-path "$AI_ROOT/checkpoints/text_encoders/clip-vit-base-patch32" \
    --endpoint "$YOLO_ENDPOINT" \
    --device cuda &
  children+=("$!")
  yolo_pid="$!"
  python3 -m integrations.perception.yolo_world_probe \
    --endpoint "$YOLO_ENDPOINT" --timeout 300 --pid "$yolo_pid" &
  yolo_probe_pid="$!"
  wait "$yolo_probe_pid"
fi

python3 -m integrations.ros.live_task_probe \
  --ai-root "$AI_ROOT" --output-root "$OUTPUT_ROOT" &
children+=("$!")

# A required persistent process exiting makes the AI service unhealthy.
wait -n "${children[@]}"
exit 1
