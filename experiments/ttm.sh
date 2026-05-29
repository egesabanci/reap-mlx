#!/bin/bash
model_name=${1}
dataset_name=${2}
compression_ratio=${3}
server_log_file_name=${4}
port=${5}
run_lm_eval=${6:-true}
run_evalplus=${7:-true}
run_livecodebench=${8:-true}
echo "Running TTM: $model_name, dataset: $dataset_name, compression ratio ${compression_ratio}"

python src/reap/main.py \
    --profile false \
    --compression-ratio $compression_ratio \
    --dataset-name $dataset_name \
    --model-name ${model_name} \
    --server_log_file_name $server_log_file_name \
    --vllm-port $port \
    --expert-sim ttm \
    --linkage-method average \
    --frequency-penalty false \
    --softmax-temperature 1.0 \
    --skip-first true \
    --skip-last true \
    --permute wm \
    --distance_measure cosine \
    --merge-method frequency_weighted_average \
    --merged-model-dir-name "ttm-${compression_ratio}" \
    --cluster-description "ttm-no-freq-penalty" \
    --do-eval false


short_model_name=$(echo $model_name | cut -d'/' -f2)
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
merged_model="artifacts/${short_model_name}/${short_dataset_name}/non_uniform_merged_models/ttm-${compression_ratio}/ttm-no-freq-penalty"
bash eval.sh ${merged_model} ${model_name} ${port} ${server_log_file_name} ${run_lm_eval} ${run_evalplus} ${run_livecodebench}
echo "Finished SubMoE for model: ${model_name}, dataset: ${dataset_name}, compression ratio: ${compression_ratio}"
# echo "Removing safetensor files from ${merged_model}"
# rm ${merged_model}/*.safetensors