import pandas as pd
from pathlib import Path
from report_results import generate_report


def main():
    """to
    Runs the report generation for specified models and consolidates the results.
    """
    base_path = Path("artifacts")
    model_dirs = [
        "ERNIE-4.5-21B-A3B-PT",
        "GLM-4.5-Air",
        "Llama-4-Scout-17B-16E-Instruct",
        "Mixtral-8x7B-Instruct-v0.1",
        "Qwen3-30B-A3B",
        "Qwen3-Coder-480B-A35B-Instruct-FP8",
    ]

    all_results_dfs = []

    for model_dir_name in model_dirs:
        model_path = base_path / model_dir_name
        print(f"Processing {model_path}...")
        generate_report(str(model_path))

        results_csv = model_path / "results_summary.csv"
        if results_csv.exists():
            print(f"Found results at {results_csv}")
            all_results_dfs.append(pd.read_csv(results_csv))
        else:
            print(f"Warning: No results summary found for {model_dir_name}")

    baselines_csv = base_path / "baselines.csv"
    if baselines_csv.exists():
        print(f"Found baselines at {baselines_csv}")
        baselines_df = pd.read_csv(baselines_csv)
        all_results_dfs.append(baselines_df)
    else:
        print(f"Warning: {baselines_csv} not found.")

    if all_results_dfs:
        combined_df = pd.concat(all_results_dfs, ignore_index=True)
        output_path = base_path / "all_results.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"All results saved to {output_path}")
    else:
        print("No results were generated or found. No combined file created.")


if __name__ == "__main__":
    main()
