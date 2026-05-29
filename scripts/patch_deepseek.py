import os
import shutil
from huggingface_hub import snapshot_download

def main():
    # Locate this script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the local new modeling_deepseek.py
    new_file = os.path.normpath(
        os.path.join(script_dir, os.pardir, "src", "reap", "models", "modeling_deepseek.py")
    )
    # Download the full model into artifacts/models/<model_name>
    model_name = "DeepSeek-V2-Lite-Chat"
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, model_name)
    snapshot_download(
        repo_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        repo_type="model",
        local_dir=model_dir,
    )
    # Patch modeling_deepseek.py in the downloaded model directory
    cached_file = os.path.join(model_dir, "modeling_deepseek.py")
    shutil.copy2(new_file, cached_file)
    print(f"Replaced {cached_file} with {new_file}")

if __name__ == "__main__":
    main()
