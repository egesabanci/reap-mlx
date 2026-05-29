import argparse
import sys
from pathlib import Path

from report_evals import process_eval_directory


def generate_report(model_directory_str: str):
    """Generates evaluation results report for a model."""
    model_dir = Path(model_directory_str)
    if not model_dir.is_dir():
        print(
            f"Error: Model directory not found at {model_dir}",
            file=sys.stderr,
        )
        return

    output_csv_path = model_dir / "results_summary.csv"

    header = [
        "Model",
        "calib_dataset",
        "compression_technique",
        "compression_method",
        "recovery_method",
        "subdir",
        "compression_ratio",
        "seed",
        "perserve experts",
        "HumanEval (pass@1)",
        "HumanEval+ (pass@1)",
        "MBPP",
        "MBPP+",
        "evalplus",
        "arc_c (acc norm)",
        "arc_e (acc norm)",
        "boolq",
        "hellaswag (acc norm)",
        "mmlu",
        "openbookqa (acc norm)",
        "rte",
        "winogrande",
        "mc_average",
        "livecodebench_pass@1",
        "Wildbench_creative_writing_score_rescaled",
        "gsm8k",
        "MATH-500",
        "AIME-25",
        "math_average",
    ]

    results_to_print = []

    model_name = model_dir.name

    for calib_dataset_dir in model_dir.iterdir():
        if not calib_dataset_dir.is_dir():
            continue
        calib_dataset = calib_dataset_dir.name

        for tech_dir in calib_dataset_dir.iterdir():
            if not tech_dir.is_dir() or tech_dir.name not in [
                "pruned_models",
                "merged_models",
                "non_uniform_merged_models",
            ]:
                continue

            compression_technique = (
                "pruning" if "pruned" in tech_dir.name else "merging"
            )

            for eval_parent_dir in tech_dir.iterdir():
                if not eval_parent_dir.is_dir():
                    continue

                eval_dir = None
                if compression_technique == "merging":
                    # Merged models have an extra directory layer, e.g., m_smoe-0.50/m_smoe/eval
                    for subdir in eval_parent_dir.iterdir():
                        if subdir.is_dir() and (subdir / "eval").is_dir():
                            eval_dir = subdir / "eval"
                            break
                else:
                    # Pruned models have eval directly inside, e.g., l1_unstructured-0.50/eval
                    if (eval_parent_dir / "eval").is_dir():
                        eval_dir = eval_parent_dir / "eval"

                if not eval_dir or not eval_dir.is_dir():
                    continue

                try:
                    result_row = process_eval_directory(eval_dir, tech_dir)
                    if result_row:
                        (
                            output_name,
                            compression_method_full,
                            compression_ratio,
                            seed,
                            _,  # old perserve_super_experts, now unused
                            *metrics,
                        ) = result_row

                        compression_method = compression_method_full.split("-")[0]

                        perserve_experts = "N/A"
                        if (
                            "perserve_outlier" in output_name
                            or "outlier_expert_singletons" in output_name
                        ):
                            perserve_experts = "outlier"
                        elif (
                            "perserve_super" in output_name
                            or "super_expert_singletons" in output_name
                        ):
                            perserve_experts = "super"

                        subdir_path = (
                            Path(model_name)
                            / calib_dataset
                            / tech_dir.name
                            / output_name
                        )

                        full_row = [
                            model_name,
                            calib_dataset,
                            compression_technique,
                            compression_method,
                            "N/A",  # recovery_method
                            str(subdir_path),
                            compression_ratio,
                            seed,
                            perserve_experts,
                            *metrics,
                        ]
                        results_to_print.append(full_row)
                except Exception as e:
                    print(
                        f"Warning: Error processing directory {eval_dir}: {e}",
                        file=sys.stderr,
                    )

    with open(output_csv_path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for result in results_to_print:
            f.write(",".join(map(str, result)) + "\n")

    print(f"Results summary saved to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results report for a model."
    )
    parser.add_argument(
        "model_directory",
        type=str,
        help="The root directory for a model (e.g., 'artifacts/Qwen3-30B-A3B').",
    )
    args = parser.parse_args()
    generate_report(args.model_directory)


if __name__ == "__main__":
    main()