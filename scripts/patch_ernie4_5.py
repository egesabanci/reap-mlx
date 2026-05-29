import os
import shutil
from huggingface_hub import snapshot_download

def main():
    # Locate this script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to local new files
    new_modeling = os.path.normpath(
        os.path.join(script_dir, os.pardir, "src", "reap", "models", "modeling_ernie4_5_moe.py")
    )
    new_tokenizer = os.path.normpath(
        os.path.join(script_dir, os.pardir, "src", "reap", "models", "tokenization_ernie4_5.py")
    )

    # Download the full model into artifacts/models/<model_name>
    model_name = "ERNIE-4.5-21B-A3B-PT"
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, model_name)

    snapshot_download(
        repo_id="baidu/ERNIE-4.5-21B-A3B-PT",
        repo_type="model",
        revision="23c6cb79af127ec966c9c03506dc80a3ff9fa953",
        local_dir=model_dir,
    )

    # Patch files in the downloaded model directory
    shutil.copy2(new_modeling, os.path.join(model_dir, "modeling_ernie4_5.py"))
    shutil.copy2(new_tokenizer, os.path.join(model_dir, "tokenization_ernie4_5.py"))

    print(f"Replaced modeling and tokenizer in {model_dir}")

if __name__ == "__main__":
    main()
