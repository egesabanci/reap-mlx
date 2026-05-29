#!/bin/bash
git submodule init
git submodule update
uv venv .venv --seed --python 3.12
uv pip install --upgrade pip
uv pip install setuptools wheel  # --seed not working in some cases
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -vv --torch-backend auto

# For Ernie4-5, uncomment the below:
# .venv/bin/python scripts/patch_ernie4_5.py

# for Llama4 add this to vllm.model_executor.models.registry:_TEXT_GENERATION_MODELS in alphabetical order:
# "Llama4ForCausalLM": ("llama4", "Llama4ForCausalLM"),