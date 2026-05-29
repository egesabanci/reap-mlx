export CUDA_VISIBLE_DEVICES=0
server_log_file_name="merging_restricted_clusters_0.log"
port=8000

WORKING_DIR=$(pwd)

models=(
    "Qwen/Qwen3-30B-A3B"
    # "baidu/ERNIE-4.5-21B-A3B-PT"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
datasets=(
    # "allenai/c4"
    "theblackcat102/evol-codealpaca-v1"
)
compression_ratios=(
    0.50
    0.25
)
max_cluster_sizes=(
    2
    4
    8
    16
    32
    64
)

for model_name in "${models[@]}"; do
    for dataset_name in "${datasets[@]}"; do
        for compression_ratio in "${compression_ratios[@]}"; do
            for max_cluster_size in "${max_cluster_sizes[@]}"; do
                python src/reap/main.py \
                    --compression_ratio ${compression_ratio} \
                    --model-name ${model_name} \
                    --dataset-name ${dataset_name} \
                    --merge-method frequency_weighted_average \
                    --profile false \
                    --server_log_file_name $server_log_file_name \
                    --vllm-port $port \
                    --expert-sim characteristic_activation \
                    --distance_measure euclidean \
                    --linkage-method average \
                    --frequency-penalty false \
                    --skip-first false \
                    --skip-last false \
                    --merged-model-dir-name "hc_smoe-${compression_ratio}" \
                    --cluster-description "hc_smoe-restricted_${max_cluster_size}" \
                    --do-eval true \
                    --max-cluster-size ${max_cluster_size}

                short_model_name=$(echo $model_name | cut -d'/' -f2)
                short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
                model_dir="artifacts/${short_model_name}/${short_dataset_name}/merged_models/hc_smoe-${compression_ratio}/hc_smoe-restricted_${max_cluster_size}"


                echo "Running evaluation for merged model: ${model_dir}"
                python src/reap/eval.py \
                    --model-name $model_dir \
                    --vllm_port $port \
                    --server_log_file_name $server_log_file_name \
                    --run-lm-eval true \
                    --run-evalplus true \
                    --results_dir $model_dir/eval
                echo "Finished evaluating merged model: ${model_dir}"


                echo "Removing safetensor files from ${model_dir}"
                rm ${model_dir}/*.safetensors
                echo "Finished HC-SMoE for model: ${model_name}, dataset: ${dataset_name}, compression ratio: ${compression_ratio}"
            done
        done
    done
done