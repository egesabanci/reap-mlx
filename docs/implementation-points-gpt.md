# REAP MLX Backend Implementation Points

This document captures the main engineering points for making the current REAP
(Router-weighted Expert Activation Pruning) implementation work well on MLX
backends, especially MLX-LM on Apple Silicon.

## Alignment Notice - 2026-05-31

The production and official experimentation workflow remains the original
PyTorch/CUDA REAP implementation. The MLX backend is a parallel Apple Silicon
experimentation path.

The MLX goal is adapter-driven support for compatible MoE weights, not a
one-model port. Qwen3-MoE is the bootstrap/reference adapter only; model-specific
routing, expert layout, shared-expert behavior, and config updates must live
behind explicit MLX adapter contracts.

## 1. Split CUDA-only Dependencies

The current codebase imports PyTorch, vLLM, DeepSpeed, and CUDA-oriented
evaluation paths too eagerly. The MLX path should be importable on a machine
without CUDA or PyTorch.

- Move `vllm` imports in `data.py` and `eval.py` behind lazy import boundaries.
- Avoid importing CUDA evaluation code from pruning entrypoints.
- Consider backend-specific extras such as `reap[torch]` and `reap[mlx]`.
- Keep the existing PyTorch/CUDA path intact while adding a separate MLX path.

## 2. Add a Real MLX Backend

MLX should not be treated as a drop-in tensor shim for PyTorch. The execution
model, module layout, memory behavior, and MoE implementation are different.

Recommended shape:

```text
src/reap/backends/
  torch/
    observer.py
    prune.py
    metrics.py
  mlx/
    observer.py
    router.py
    prune.py
    model_util.py
    metrics.py
```

Shared concepts should remain backend-neutral:

- calibration batch format
- pruning metric names and schemas
- pruning decisions
- saved model metadata
- smoke-test interface

## 3. Use MLX-LM MoE Structure Directly

MLX-LM MoE models usually store experts as stacked tensors inside `switch_mlp`,
not as PyTorch `ModuleList`s. Pruning should slice these tensors directly.

Typical tensors to slice on the first dimension:

- `switch_mlp.gate_proj.weight`
- `switch_mlp.up_proj.weight`
- `switch_mlp.down_proj.weight`
- router/gate `weight`
- router/gate `bias`, if present
- correction bias tensors, where present

After slicing, update model/config fields such as:

- `num_experts`
- `num_local_experts`
- `n_routed_experts`
- `moe_num_experts`
- `top_k`
- `num_experts_per_tok`
- `moe_k`

If the retained expert count is smaller than the configured top-k value, clamp
top-k to the retained expert count.

## 4. Rewrite the Observer Around MLX Layerwise Replay

The existing PyTorch observer depends on `register_forward_hook`, which is not
the right abstraction for MLX. The MLX implementation should use explicit
layerwise replay.

The current `LayerwiseMoEObserver` is the closest conceptual match, but the MLX
version should:

- replay model layers explicitly
- collect hidden states layer by layer
- process one or a few layers at a time
- call `mx.eval(...)` deliberately
- avoid retaining large lazy graphs across calibration batches
- use MLX memory APIs instead of CUDA memory APIs

## 5. Avoid All-Expert Activation Materialization

The current PyTorch REAP path often computes all expert activations with shape:

```text
(num_experts, tokens, hidden)
```

That is expensive and should not be the default MLX strategy.

For pruning-only REAP, compute and aggregate only the selected expert outputs
that the router actually used. MLX-LM already executes routed experts through
`SwitchGLU`, so the MLX observer should reuse that path.

Needed per batch:

- selected expert indices
- selected routing weights
- selected expert output norms
- token counts per expert
- weighted activation-norm sums
- max activation norms

This is likely the largest performance and memory improvement for Apple Silicon.

## 6. Preserve Exact Router Semantics

Router semantics differ across supported MoE architectures. A generic
`topk(softmax(router_logits))` implementation is not always correct.

The MLX backend should provide a router adapter that returns:

- raw router logits, when available
- selected expert indices
- selected routing weights
- optional router metadata

Architecture-specific details to preserve:

- Qwen: full softmax, top-k selection, optional top-k renormalization
- Mixtral: top-k over raw logits, softmax over selected scores
- DeepSeek: group-limited routing, scaling factors, optional shared experts
- GLM: sigmoid routing, correction bias, group expert selection, scaling
- ERNIE: softmax/sigmoid gate behavior and selected-score normalization

For MLX correctness, prefer the model's actual routing decisions over a
backend-independent approximation. If parity with the existing PyTorch REAP
behavior is required, add an explicit compatibility mode.

## 7. Keep REAP Accumulators Small and CPU-Side

MLX does not expose every PyTorch reduction helper used by the current code,
such as `bincount` and scatter-style accumulation. Since REAP's accumulated
statistics are small, keep them as NumPy/CPU arrays where practical.

Recommended flow:

1. Run model/router/expert computation in MLX.
2. Compute compact per-token or per-expert values in MLX.
3. Call `mx.eval(...)`.
4. Transfer compact arrays to CPU.
5. Update NumPy accumulators.

This avoids graph retention, reduces memory pressure, and keeps metric code
simple.

## 8. Use MLX Memory Controls

Replace CUDA-specific memory handling with MLX equivalents.

Useful APIs:

- `mx.eval(...)`
- `mx.metal.clear_cache()`
- `mx.metal.get_active_memory()`
- `mx.metal.get_peak_memory()`
- `mx.metal.reset_peak_memory()`
- `mx.metal.set_memory_limit(...)`
- `mx.metal.set_cache_limit(...)`

The MLX implementation should be explicit about evaluation boundaries because
MLX is lazy by default.

## 9. Separate MLX Save and Evaluation Paths

The existing evaluation stack is vLLM/CUDA-oriented. The first MLX milestone
should be:

1. load an MLX-LM MoE model
2. run calibration
3. compute REAP saliency
4. prune experts
5. save with MLX-LM utilities
6. reload the saved model
7. run a local generation smoke test

Full benchmark parity through vLLM, lm-eval, EvalPlus, or LiveCodeBench can be a
later compatibility layer.

## 10. Handle Quantized Experts Deliberately

MLX-LM supports quantized layers, including quantized expert projections.
Pruning quantized expert tensors may require slicing associated metadata, not
only the main weight tensor.

For quantized `SwitchLinear` variants, inspect and slice consistently:

- `weight`
- `scales`
- `biases`
- `bias`, if present

If direct quantized slicing is risky, the first implementation can dequantize,
prune, save, and optionally requantize as a separate step.

## 11. Add MLX-Specific Tests

The MLX backend should have focused tests independent from the CUDA test suite.

Recommended tests:

- tiny synthetic Qwen3-MoE MLX model pruning
- tiny synthetic Mixtral MLX model pruning
- router adapter output shape and semantic checks
- REAP metric accumulation compared against a NumPy brute-force reference
- expert tensor slicing shape checks
- save/reload/generation smoke test

The existing PyTorch tests should remain as regression coverage for the current
backend.

## 12. Suggested Implementation Order

1. Make imports backend-safe.
2. Add MLX model registry and model loading helpers.
3. Implement MLX router adapters.
4. Implement selected-expert REAP metric accumulation.
5. Implement MLX layerwise observer.
6. Implement MLX pruning mutation for stacked experts.
7. Add save/reload smoke test.
8. Add architecture-specific coverage incrementally.

The key design principle is to optimize for MLX-LM's native execution model
instead of emulating PyTorch hooks and all-expert activation tensors.
