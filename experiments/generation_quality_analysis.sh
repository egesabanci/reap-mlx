#!/bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3"

# run compression to create checkpoints: 

model_name=Qwen/Qwen3-30B-A3B
dataset_name=theblackcat102/evol-codealpaca-v1
compression_ratio=0.50
pruning_method="reap"

# WMEAN
echo "running WMEAN: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
python src/reap/prune.py \
                    --model-name $model_name \
                    --dataset-name $dataset_name \
                    --compression-ratio $compression_ratio \
                    --prune-method $pruning_method \
                    --profile false \
                    --do-eval false \
                    --distance_measure cosine


echo "running M-SMoE: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
python src/reap/main.py \
    --compression-ratio ${compression_ratio} \
    --model-name ${model_name} \
    --dataset-name ${dataset_name} \
    --merge-method frequency_weighted_average \
    --permute wm \
    --cluster-method mc_smoe \
    --profile false \
    --expert-sim router_logits \
    --distance_measure cosine \
    --frequency-penalty false \
    --merged-model-dir-name "m_smoe-${compression_ratio}" \
    --cluster-description "m_smoe" \
    --do-eval false


echo "Running HC-SMoE: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
python src/reap/main.py \
    --compression_ratio ${compression_ratio} \
    --model-name ${model_name} \
    --dataset-name ${dataset_name} \
    --merge-method frequency_weighted_average \
    --profile false \
    --expert-sim characteristic_activation \
    --distance_measure euclidean \
    --linkage-method average \
    --frequency-penalty false \
    --skip-first false \
    --skip-last false \
    --merged-model-dir-name "hc_smoe-${compression_ratio}" \
    --cluster-description "hc_smoe" \
    --do-eval false

python scripts/generation_quality_analysis.py \
    --base_model_name "Qwen/Qwen3-30B-A3B" \
    --wmean_model_dir artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/pruned_models/reap-0.50 \
    --m_smoe_model_dir artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/non_uniform_merged_models/m_smoe-0.50/m_smoe \
    --hc_smoe_model_dir artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/merged_models/hc_smoe-0.50/hc_smoe \
    --output_dir "artifacts/fig/generation_quality_analysis/qwen-generation_quality_analysis-all-0.50" \
    # --plot_only  # can be used to regenerate plots if data already collected
