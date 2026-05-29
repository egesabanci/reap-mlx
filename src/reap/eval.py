import logging
from typing import Tuple
import pathlib
import os
import json
import shutil
import subprocess
import time
import requests

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from lm_eval import evaluator
from lm_eval.utils import make_table
from evalplus.evaluate import evaluate as evalplus_evaluator
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import uvloop

from reap.args import ReapArgs, ModelArgs, EvalArgs
from reap.model_util import patched_model_map, MODEL_ATTRS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_original_model_name(model_name: str) -> Tuple[str, bool]:
    original_model_name_map = {
        "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
        "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
        "DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "GLM-4.5-Air": "zai-org/GLM-4.5-Air",
        "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    }

    original_model = None
    for key, value in original_model_name_map.items():
        if key in model_name:
            original_model = value
            break
    uncompressed_model = False
    if original_model is None:
        # it's an uncompressed model or bad path
        if model_name in original_model_name_map.values():
            original_model = model_name
            uncompressed_model = True
        else:
            logger.warning(
                f"Could not find original model for {model_name}, using model_name as original_model"
            )
            original_model = model_name
    return original_model, uncompressed_model


def wait_for_server(base_url, timeout=1200, check_interval=5):
    """Wait for the server to be ready by checking the health endpoint."""
    health_url = f"{base_url}/health"
    start_time = time.time()

    logger.info(f"Waiting for server to be ready at {health_url}")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.info(f"Server not ready yet, waiting {check_interval} seconds...")
        time.sleep(check_interval)

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def start_server(model_name, model_args, eval_args, seed, log_file, port):
    """Starts a VLLM server for the specified model."""

    num_gpus = torch.cuda.device_count()
    logger.info("Running on %d GPUs", num_gpus)

    override_generation_config = {}
    if not eval_args.greedy:
        override_generation_config = {
            "temperature": eval_args.temperature,
            "top_p": eval_args.top_p,
            "top_k": eval_args.top_k,
            "min_p": eval_args.min_p,
        }
        logger.info(
            "Using sampling with temperature=%s, top_p=%s, top_k=%s, min_p=%s",
            eval_args.temperature,
            eval_args.top_p,
            eval_args.top_k,
            eval_args.min_p,
        )
    else:
        logger.info("Using greedy decoding")
    override_generation_config_str = json.dumps(override_generation_config)

    hf_overrides = {}
    max_num_seqs = 32
    max_model_len = 32768
    gpu_memory_utilization = 0.90
    if model_args.num_experts_per_tok_override is not None:
        logger.info(
            f"Overriding number of experts per token to {model_args.num_experts_per_tok_override}"
        )
        key = "num_experts_per_tok"
        if "ernie" in model_name.lower():
            key = "moe_k"
        hf_overrides = {key: model_args.num_experts_per_tok_override}
    hf_overrides_str = json.dumps(hf_overrides)
    max_num_batched_tokens = max_model_len

    original_model_name, _ = get_original_model_name(model_name)

    # TODO: once  limit_mm_per_prompt={"image": 1 if use_image else 0}, is stable, use
    # in place of patching vllm.model_executor.models.registry for Llama4.

    server_command = [
        "vllm",
        "serve",
        model_name,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(num_gpus),
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--enable-expert-parallel",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--override-generation-config",
        override_generation_config_str,
        "--trust-remote-code",
        "--hf-overrides",
        hf_overrides_str,
        "--served-model-name",  # for HELM eval.
        original_model_name,
        model_name,
    ]

    logger.info(f"Starting VLLM OpenAI API server for {model_name} on port {port}")
    logger.info(
        f"Using {num_gpus} GPUs with {gpu_memory_utilization} memory utilization"
    )
    logger.info(f"Command: {' '.join(server_command)}")

    # Start the server process with log redirection
    with open(log_file, "w") as log_file:
        process = subprocess.Popen(
            server_command, stdout=log_file, stderr=subprocess.STDOUT
        )

    base_url = f"http://0.0.0.0:{port}"

    wait_for_server(base_url)

    return base_url, process


def run_evaluate(model_args, results_dir, eval_args, seed):
    model_name = model_args.model_name
    if isinstance(model_name, pathlib.Path):
        model_name = model_name.__str__()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    use_server = eval_args.use_server
    if results_dir is None:
        model_short_name = model_name.split("/")[-1]
        if model_args.num_experts_per_tok_override is not None:
            model_short_name += (
                f"-num_experts_per_tok_{model_args.num_experts_per_tok_override}"
            )
        results_dir = pathlib.Path.cwd() / "artifacts" / "eval" / model_short_name
    if isinstance(results_dir, str):
        results_dir = pathlib.Path(results_dir)
    if not eval_args.greedy:
        results_dir = results_dir.parent / f"{results_dir.name}_sampling"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    num_gpus = torch.cuda.device_count()
    model_name = patched_model_map(model_name)
    if use_server:
        server_endpoint, process = start_server(
            model_name,
            model_args,
            eval_args,
            seed,
            log_file=eval_args.server_log_file_name,
            port=eval_args.vllm_port,
        )

    if eval_args.run_lm_eval:
        results_file_base_name = results_dir / "lm_eval_results"
        model_args = {
            "pretrained": model_name,
            "tensor_parallel_size": num_gpus,
            "gpu_memory_utilization": 0.85,
            "num_concurrent": 32,
            "timeout": 1200,
            "max_retries": 10,
            "trust_remote_code": True,
        }
        if "baidu" in model_name.lower():
            logger.warning("Using slow tokenizer for Ernie-4.5")
            model_args["use_fast_tokenizer"] = False
        if use_server:
            model_args["base_url"] = f"{server_endpoint}/v1/completions"
            model_args["tokenized_requests"] = False
        logger.info(f"Running lm-eval on tasks {eval_args.lm_eval_tasks}")
        is_ernie = "ernie" in model_name.lower()
        logger.warning(f"Is Ernie: {is_ernie}, using batch size 1")
        if use_server:
            results = evaluator.simple_evaluate(
                model="local-completions",
                model_args=model_args,
                tasks=eval_args.lm_eval_tasks,
                num_fewshot=0,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                batch_size=eval_args.parallel_tasks if not is_ernie else 1,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
            )
        else:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=eval_args.lm_eval_tasks,
                num_fewshot=0,
                batch_size="auto",
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
            )
        try:
            with open(f"{results_file_base_name}_table.txt", "w") as f:
                print(make_table(results))
                print(make_table(results), file=f)
                if "groups" in results:
                    print(make_table(results, "groups"))
                    print(make_table(results, "groups"), file=f)
            with open(f"{results_file_base_name}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            pass
        logger.info(f"Finished evaluating lm-eval")

    try:
        if eval_args.run_evalplus:
            enable_thinking = True
            if "qwen" in model_name.lower() or "glm-4.5" in model_name.lower():
                logger.info("Disabling thinking for Qwen/GLM models")
                enable_thinking = False
            for task in eval_args.evalplus_tasks:
                logger.info(f"Running evalplus on task {task}")
                output_file = results_dir / f"{task}.json"
                # evalplus fork
                if use_server:
                    evalplus_evaluator(
                        model=model_name,
                        root=results_dir / "evalplus_results",
                        dataset=task,
                        backend="openai",
                        attn_implementation="flash_attention_2",
                        greedy=eval_args.greedy,
                        output_file=output_file,
                        base_url=f"{server_endpoint}/v1",
                        temperature=eval_args.temperature
                        if not eval_args.greedy
                        else 0.0,
                        enable_thinking=enable_thinking,
                        parallel_tasks=eval_args.parallel_tasks,
                    )
                else:
                    evalplus_evaluator(
                        model=model_name,
                        root=results_dir / "evalplus_results",
                        dataset=task,
                        backend="hf",
                        attn_implementation="flash_attention_2",
                        greedy=eval_args.greedy,
                        output_file=output_file,
                        temperature=eval_args.temperature
                        if not eval_args.greedy
                        else 0.0,
                        enable_thinking=enable_thinking,
                    )
    except Exception as e:
        logger.error(f"An error occurred during evalplus: {e}")
        raise e
        pass
    try:
        if eval_args.run_livecodebench:
            if not use_server:
                raise ValueError(
                    "Current LCB ReapBase model style implementation requries a vLLM server to be running"
                )
            from lcb_runner.runner.main import main as lcb_main
            from lcb_runner.runner.main import get_args_dict

            original_model, uncompressed_model = get_original_model_name(model_name)

            lcb_args = get_args_dict(
                model=original_model,
                n=1,
                output_path=results_dir,
                enable_thinking=False,
                base_url=f"{server_endpoint}/v1",
                start_date="2025-01-01",
                end_date="2025-07-31",
                evaluate=True,
                timeout=120,
                local_model_path=model_name if not uncompressed_model else None,
                max_tokens=16384,
            )
            logger.info(f"Running LiveCodeBench with args: {lcb_args}")
            lcb_main(lcb_args)
            logger.info(f"Finished evaluating LiveCodeBench")
    except Exception as e:
        logger.error(f"An error occurred during livecodebench: {e}")
        pass
    try:
        if eval_args.run_wildbench:
            from helm.benchmark.run import helm_run, create_helm_run_args
            from helm.common.hierarchical_logger import setup_default_logging

            original_model, uncompressed_model = get_original_model_name(model_name)

            # move config to model dir
            config_src = (
                pathlib.Path(__file__).parent.parent.parent
                / "config"
                / f"wildbench_prod_env_{eval_args.vllm_port}"
            )
            local_path = f"{model_name}/wildbench_prod_env_{eval_args.vllm_port}"
            shutil.copytree(config_src, local_path, dirs_exist_ok=True)

            suite = "test"
            run_entries = [f"wildbench:subset=v2,model={original_model}"]
            helm_args = create_helm_run_args(
                suite=suite,
                local_path=local_path,
                run_entries=run_entries,
                output_path=f"{results_dir}/wildbench",
                cache_instances=True,
                disable_cache=False,
            )
            logger.info(f"Running WildBench with args: {helm_args}")
            setup_default_logging()
            helm_run(helm_args)
            logger.info(f"Finished evaluating WildBench")
    except Exception as e:
        logger.error(f"An error occurred during wildbench: {e}")
        pass
    if eval_args.run_math:
        try:
            from evalscope.run import run_task, TaskConfig

            task_config = TaskConfig(
                model=model_name,
                generation_config={
                    "do_sample": False,
                    "max_new_tokens": 16384,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                datasets=[
                    "gsm8k",
                    "math_500",
                ],
                api_url=f"{server_endpoint}/v1",
                api_key="EMPTY",
                timeout=3600,
                work_dir=results_dir / "evalscope_results",
                dataset_args={
                    "gsm8k": {
                        "few_shot_num": 0,
                    }
                },
                eval_batch_size=32,
                eval_type="service",
            )
            logger.info(f"Running evalscope math with config: {task_config}")
            run_task(task_config)
            logger.info(f"Finished evaluating evalscope math benchmarks")
        except Exception as e:
            logger.error(f"An error occurred during math evaluation: {e}")
            pass

    if use_server:
        process.terminate()
    if use_server and "process" in locals():
        process.terminate()


if __name__ == "__main__":
    parser = HfArgumentParser((ReapArgs, ModelArgs, EvalArgs))
    reap_args, model_args, eval_args = parser.parse_args_into_dataclasses()
    run_evaluate(
        model_args=model_args,
        results_dir=eval_args.results_dir,
        eval_args=eval_args,
        seed=reap_args.seed,
    )
