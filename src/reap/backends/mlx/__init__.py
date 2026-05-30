"""Import-safe MLX backend package boundary.

This module intentionally avoids importing MLX, PyTorch, CUDA, vLLM, or the
existing PyTorch-first REAP entrypoints at module import time.
"""
