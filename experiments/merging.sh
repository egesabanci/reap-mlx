export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name=${2:-"Qwen/Qwen3-30B-A3B"}
seed=${3:-42}
singleton_super_experts=${4:false}
singleton_outlier_experts=${5:false}

server_log_file_name="merging-${FIRST_DEVICE}-seed_${seed}.log"
run_lm_eval=true
run_evalplus=true
run_livecodebench=true

models=(
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # "zai-org/GLM-4.5-Air"
    # "baidu/ERNIE-4.5-21B-A3B-PT"
    "Qwen/Qwen3-30B-A3B"
)
datasets=(
    "theblackcat102/evol-codealpaca-v1"
    # "allenai/c4"
    # "euclaise/WritingPrompts_curated"
)
compression_ratios=(
    0.50
    # 0.25
)

for dataset_name in "${datasets[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
        echo "Running with model: $model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
        # bash hc_smoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${seed} ${singleton_super_experts} ${singleton_outlier_experts}
        bash m_smoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${seed} ${singleton_super_experts} ${singleton_outlier_experts}
        # bash submoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${seed}
    done
done

