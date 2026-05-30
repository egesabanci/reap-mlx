"""MLX backend namespace skeleton.

Phase 0 exposes no runtime functionality. Importing this module must remain
safe on machines without Torch, vLLM, MLX, or mlx-lm installed.
"""

BACKEND_NAME = "mlx"

__all__: tuple[str, ...] = ("BACKEND_NAME",)
