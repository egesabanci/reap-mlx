
#!/bin/bash

model_dir=${1}
seed=${2:-42}
port=${3}
server_log_file_name=${4}

# qa
run_lm_eval=${5:-true}
# coding
run_evalplus=${6:-true}
run_livecodebench=${7:-true}
# math
run_math=${8:-false}
# wildbench
run_wildbench=${9:-false}



WORKING_DIR=$(pwd)

echo "Running evaluation for model: ${model_dir}"
echo "Seed: ${seed}, Port: ${port}, Server log file: ${server_log_file_name}"
echo "Run lm-eval: ${run_lm_eval}, Run eval-plus: ${run_evalplus}, Run livecodebench: ${run_livecodebench}, Run math: ${run_math}, Run wildbench: ${run_wildbench}"

python src/reap/eval.py \
    --model-name $model_dir \
    --vllm_port $port \
    --server_log_file_name $server_log_file_name \
    --run-lm-eval $run_lm_eval \
    --run-evalplus $run_evalplus \
    --run-livecodebench $run_livecodebench \
    --run-wildbench $run_wildbench \
    --run-math $run_math \
    --results_dir $model_dir/eval \
    --seed $seed
echo "Finished evaluating merged model: ${model_dir}"
