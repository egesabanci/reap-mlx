#!/bin/bash


export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
server_log_file_name="calibrate-${FIRST_DEVICE}.log"

model=${2}
dataset=${3}
seed=${4:-11}
compression_ratio=0.5

echo "Running calibration with model: $model, dataset: $dataset, compression ratio: $compression_ratio, seed: $seed on devices: $CUDA_VISIBLE_DEVICES"

python src/reap/prune.py \
    --model-name $model \
    --dataset-name $dataset \
    --compression-ratio $compression_ratio \
    --prune-method frequency \
    --profile true \
    --vllm_port $port \
    --server-log-file-name $server_log_file_name \
    --do-eval false \
    --distance_measure cosine \
    --seed $seed \
    --run_observer_only true \
    --output_file_name "observations_1024_cosine-seed_${seed}.pt"