#!/bin/bash
set -euo pipefail

# MLX REAP pruning experiment wrapper.
# Usage: bash experiments/mlx-pruning.sh [MODEL_NAME] [DATASET] [PRUNE_METHOD] [COMPRESSION_RATIO] [SEED]

MODEL=${1:-"LiquidAI/LFM2.5-8B-A1B-MLX-4bit"}
DATASET=${2:-"theblackcat102/evol-codealpaca-v1"}
METHOD=${3:-"reap"}
RATIO=${4:-"0.25"}
SEED=${5:-"42"}
MAX_SAMPLES=${MAX_SAMPLES:-8}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-"artifacts/mlx/$(basename "$MODEL")/${METHOD}-${RATIO}-seed-${SEED}"}

echo "=== REAP MLX Pruning ==="
echo "Model:          $MODEL"
echo "Dataset:        $DATASET"
echo "Method:         $METHOD"
echo "Ratio:          $RATIO"
echo "Seed:           $SEED"
echo "Max samples:    $MAX_SAMPLES"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Output:         $OUTPUT_DIR"

python -m reap.backends.mlx.entrypoint \
    --model-name "$MODEL" \
    --dataset-name "$DATASET" \
    --prune-method "$METHOD" \
    --compression-ratio "$RATIO" \
    --seed "$SEED" \
    --max-samples "$MAX_SAMPLES" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --output-dir "$OUTPUT_DIR" \
    --verbose
