#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
port=8100
server_log_file_name="sft-resume.log"
WORKING_DIR=$(pwd)
# model_dir="artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/non_uniform_merged_models/submoe-0.50/submoe"
model_dir="artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/pruned_models/reap-0.50"
dataset_name="theblackcat102/evol-codealpaca-v1"
# short_model_name=$(echo $model_name | cut -d'/' -f2)
short_model_name=Qwen3-30B-A3B
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
compression_ratio=0.50
# run_name=${short_model_name}-${short_dataset_name}-hc_smoe-${compression_ratio}-sft
# echo $run_name


# no lora
accelerate launch \
    --config_file config/fsdp_v2_4-proc.yaml \
    src/reap/sft.py \
    --model_name_or_path ${model_dir} \
    --dataset_name ${dataset_name} \
    --assistant_only_loss true \
    --learning_rate 5.0e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --packing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 2048 \
    --save_strategy steps \
    --save_steps 1000 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --save_total_limit 1 \
    --output_dir ${model_dir}/sft \
    --overwrite_output_dir \
    --run_name ${short_model_name}-${short_dataset_name}-w-EAN-mean-${compression_ratio}-sft \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 1000 \
    --report_to wandb \
    --mixed_precision 'no' \
    --gradient_checkpointing false \
    --num_train_epochs 1 \
    --resume_from_checkpoint ${model_dir}/sft/checkpoint-4000
    # --use_peft 
    # --lora_r 32 \
    # --lora_alpha 16 \
    # --max-steps 5 \

# eval
sft_dir="${model_dir}/sft"
bash eval.sh ${sft_dir} ${model_name} ${port} ${server_log_file_name}
