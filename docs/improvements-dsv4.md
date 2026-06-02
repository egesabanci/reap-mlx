# REAP MLX — Deep System View v4: Findings & Improvements

> **Scope**: Comprehensive analysis of the REAP MLX codebase (`reap-mlx`).  
> **Context**: macOS-only (Apple Silicon), MLX runtime, MoE model pruning.  
> **Date**: 2026-06-01

---

## Table of Contents

1. [Architecture Summary](#architecture-summary)
2. [Design Strengths](#design-strengths)
3. [Confirmed Issues & Recommendations](#confirmed-issues--recommendations)
   - [3.1 Qwen3 Observer: Redundant Attention Mask Computation](#31-qwen3-observer-redundant-attention-mask-computation)
   - [3.2 Observer MoE Layer Code Duplication](#32-observer-moe-layer-code-duplication)
   - [3.3 Router Epsilon Asymmetry](#33-router-epsilon-asymmetry)
   - [3.4 Uniform Expert Count Constraint](#34-uniform-expert-count-constraint)
   - [3.5 Hardcoded Smoke Test Parameters](#35-hardcoded-smoke-test-parameters)
   - [3.6 Config Mutation Pattern Documentation](#36-config-mutation-pattern-documentation)
4. [Performance Deep-Dive](#performance-deep-dive)
   - [4.1 Qwen3 Attention Mask Hoisting](#41-qwen3-attention-mask-hoisting)
   - [4.2 MLX Graph Compilation Potential](#42-mlx-graph-compilation-potential)
   - [4.3 Eval Frequency Tuning](#43-eval-frequency-tuning)
   - [4.4 Python Loop Overhead in Layer Replay](#44-python-loop-overhead-in-layer-replay)
   - [4.5 NumPy Scatter Operations Scaling](#45-numpy-scatter-operations-scaling)
   - [4.6 LFM2 Mask Computation (Already Optimized)](#46-lfm2-mask-computation-already-optimized)
   - [4.7 MLX Type Conversion Overhead](#47-mlx-type-conversion-overhead)
   - [4.8 PruningState.report() Array Copying](#48-pruningstatereport-array-copying)
5. [Code Quality & Maintenance Observations](#code-quality--maintenance-observations)
   - [5.1 Import-Safety Enforcement](#51-import-safety-enforcement)
   - [5.2 Adapter Pattern Extensibility](#52-adapter-pattern-extensibility)
   - [5.3 Injectable Dependencies for Testability](#53-injectable-dependencies-for-testability)
   - [5.4 Save-Reload Validation Thoroughness](#54-save-reload-validation-thoroughness)
   - [5.5 Router Code Duplication](#55-router-code-duplication)
   - [5.6 Observer Code Duplication](#56-observer-code-duplication)
   - [5.7 Test Coverage Gaps](#57-test-coverage-gaps)
6. [Potential Future Concerns](#potential-future-concerns)
   - [6.1 Non-Uniform Expert Architectures](#61-non-uniform-expert-architectures)
   - [6.2 Tokenizer Edge Cases](#62-tokenizer-edge-cases)
   - [6.3 Pairwise Expert Frequency Scaling](#63-pairwise-expert-frequency-scaling)
   - [6.4 Calibration Data Size & Memory](#64-calibration-data-size--memory)
7. [Confirmed Non-Issues](#confirmed-non-issues)
8. [Summary of Recommendations](#summary-of-recommendations)

---

## Architecture Summary

REAP MLX implements Router-weighted Expert Activation Pruning (REAP) for MLX-LM Mixture-of-Experts models on Apple Silicon. The pipeline is intentionally linear and inspectable:

```
Load → Calibrate → Observe → Prune → Save → Validate → Smoke
```

**Module Map:**

| Module | Responsibility | Heavy Imports |
|---|---|---|
| `reap/__init__.py` | Lightweight package root | None |
| `reap/data.py` | HF datasets calibration loading | `datasets` (lazy) |
| `reap/model_adapters.py` | Adapter pattern: Qwen3-MoE, LFM2-MoE | `mlx_lm` (lazy, mask helpers only) |
| `reap/router.py` | Router logic matching MLX-LM semantics | `mlx.core` (lazy) |
| `reap/metrics.py` | `PruningState` — NumPy accumulation | None (pure NumPy) |
| `reap/observer.py` | Layerwise MLX replay with eval boundaries | `mlx.core` (lazy) |
| `reap/prune.py` | In-place expert weight slicing | None (NumPy only) |
| `reap/save.py` | Save/reload/validate artifacts | `mlx_lm` (lazy) |
| `reap/entrypoint.py` | CLI pipeline orchestration | All heavy imports deferred |
| `reap/validation_metrics.py` | Structured telemetry (`RunMetrics`) | `mlx.core` (lazy, memory sampling only) |

**Supported Models:**

| Adapter | Model Family | Tested On |
|---|---|---|
| `qwen3_moe` | Qwen3-MoE MLX-LM models | Unit tests |
| `lfm2_moe` | Liquid LFM2.5 MoE MLX-LM models | `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` |

**Pruning Methods:**

| Method Key | Description |
|---|---|
| `reap` | Weighted expert activation norm / expert frequency |
| `expert_frequency` / `frequency` | Count of router-selected assignments |
| `weighted_expert_frequency_sum` / `weighted_frequency_sum` | Sum of router scores |
| `ean_sum` | Sum of expert output norms |
| `ean_mean` | Mean expert output norm |
| `weighted_ean_sum` | Router-score-weighted sum of output norms |
| `max_activations` | Maximum expert output activation |

---

## Design Strengths

### 1. **Import-Light Package Boundary (Verified by Tests)**

Every single module is verified via subprocess-based tests to never import `mlx`, `mlx_lm`, `datasets`, `torch`, or `vllm` at import time. The test spins up a subprocess with a custom `sys.meta_path` import blocker, imports the target module, and asserts no blocked modules were loaded. This is not aspirational — it's enforced.

```
tests/test_mlx_no_torch_import.py       — tests reap.__init__
tests/test_mlx_data.py                  — tests reap.data
tests/test_mlx_metrics.py               — tests reap.metrics
tests/test_mlx_router.py                — tests reap.router
tests/test_mlx_model_adapters.py        — tests reap.model_adapters
tests/test_mlx_observer.py              — tests reap.observer
tests/test_mlx_prune.py                 — tests reap.prune
tests/test_mlx_save.py                  — tests reap.save
tests/test_mlx_validation_metrics.py    — tests reap.validation_metrics
tests/test_mlx_cli.py                   — tests reap.entrypoint
```

**Why it matters**: `import reap` is always cheap. Users can inspect the package, run `--help`, or import it as a dependency without pulling in the entire MLX runtime stack until execution begins.

### 2. **Adapter Pattern for Model Families**

`Qwen3MoeModelAdapter` and `Lfm2MoeModelAdapter` cleanly isolate architecture-specific differences:

- Layer discovery (`model.layers` vs `model.model.layers`)
- MoE module location (`layer.mlp` vs `layer.feed_forward`)
- Config attribute names (`num_experts_per_tok` vs `top_k`)
- Expert bias support (LFM2 only)
- Config update after pruning

Adding a new model family requires implementing a single adapter class with ~5 methods.

### 3. **Explicit MLX Evaluation Boundaries**

The observer calls `mx.eval(h)` after every single layer:

```python
for layer_idx, layer in enumerate(layers):
    # ... forward pass ...
    eval_fn(h)  # mx.eval by default
```

Without this, MLX's lazy computation graph would accumulate across all layers and sequences, causing memory to blow up. This is critical for Apple Silicon where unified memory is shared between CPU and GPU.

### 4. **Injectable Dependencies for Testability**

Every pipeline function accepts optional injectable callbacks:

```python
def main(
    argv=None,
    *,
    load_model_fn: Callable | None = None,
    load_calibration_sequences_fn: Callable | None = None,
    observe_model_fn: Callable | None = None,
    prune_experts_fn: Callable | None = None,
    save_pruned_model_fn: Callable | None = None,
    smoke_fn: Callable | None = None,
    print_fn: Callable = print,
) -> int:
```

The CLI tests (`test_mlx_cli.py`) verify the entire pipeline end-to-end with injected mock functions — no real MLX, no real models, no network access.

### 5. **Comprehensive Save-Reload Validation**

After saving the pruned model, `save_pruned_model()` reloads it and validates:

- `config.json` exists and contains correct `num_experts`
- Every MoE layer's switch projections (`gate_proj`, `up_proj`, `down_proj`) have correct first-dimension shapes
- Gate weight shapes match retained expert count
- LFM2 expert bias shapes are validated when present
- Optional generation smoke test runs on the reloaded model

### 6. **Structured Telemetry**

`RunMetrics` captures everything in a single `validation-metrics.json`:

- Run configuration and runtime metadata
- Model architecture facts (layers, expert counts, shapes)
- Calibration statistics (samples, tokens, distributions)
- Per-phase timings with derived percentages and throughput
- Per-layer observer summaries with saliency statistics
- Pruning decisions (retained/removed per layer)
- Save/reload artifact sizes
- MLX memory samples at 5 pipeline checkpoints
- Process memory (max RSS)
- Smoke test results

### 7. **Resilient Pipeline Error Handling**

The main pipeline catches `KeyboardInterrupt` (returns exit code 130) and all exceptions (writes failure metrics before re-raising). Failed runs produce usable telemetry for debugging.

---

## Confirmed Issues & Recommendations

### 3.1 Qwen3 Observer: Redundant Attention Mask Computation

**Location**: `src/reap/observer.py`, `_observe_qwen3_model()`, lines ~85-92

**Current code:**

```python
for sequence in calibration_sequences:
    tokens = _batch_tokens(mx, sequence)
    h = embed_tokens(tokens)

    for layer_idx, layer in enumerate(layers):
        mask = _attention_mask(                    # ← Computed every layer
            h,
            sequence_length=tokens.shape[-1],
            mask_fn=mask_fn,
        )
        h = _run_attention(layer, h, mask)
        # ...
```

When `mask_fn` is `None` (the default), `_attention_mask()` calls `make_attention_mask(hidden_states, cache=None)`. Since `h` maintains the same shape `[1, seq_len, hidden]` across all decoder layers, the causal attention mask is **identical for every layer**. This mask is recomputed `num_layers × num_sequences` times instead of `num_sequences` times.

**Severity**: Low-medium. For a 32-layer model with 8 sequences of 1024 tokens, this creates 256 identical attention masks instead of 8. The mask itself is small (1 × 1024 × 1024 × 4 bytes ≈ 4MB in float32), so the cost is primarily graph node overhead, not memory.

**How LFM2 already handles this correctly:**

```python
for sequence in calibration_sequences:
    tokens = _batch_tokens(mx, sequence)
    h = embed_tokens(tokens)

    attn_mask, conv_mask = _lfm2_masks(           # ← Computed once per sequence
        h, sequence_length=tokens.shape[-1], mask_fn=mask_fn
    )

    for layer_idx, layer in enumerate(layers):
        operator_mask = attn_mask if _is_lfm2_attention_layer(layer) else conv_mask
        # ...
```

**Recommended fix:**

```python
for sequence in calibration_sequences:
    tokens = _batch_tokens(mx, sequence)
    h = embed_tokens(tokens)
    mask = _attention_mask(                        # ← Hoisted out of loop
        h,
        sequence_length=tokens.shape[-1],
        mask_fn=mask_fn,
    )

    for layer_idx, layer in enumerate(layers):
        h = _run_attention(layer, h, mask)         # ← Reuse mask
        # ...
```

**Effort**: 2-line change. Move line 88-92 outside the layer loop to line 86.

---

### 3.2 Observer MoE Layer Code Duplication

**Location**: `src/reap/observer.py`, `_observe_moe_layer()` and `_observe_lfm2_moe_layer()`

**Current code — two identical functions:**

```python
# _observe_moe_layer (Qwen3):
def _observe_moe_layer(layer, moe_input, state, *, adapter, config):
    mx = _require_mlx_core()
    moe = adapter.get_moe(layer)
    routing = Qwen3MoeRouter(moe, config)(moe_input)    # ← Only difference
    switch_mlp = getattr(moe, "switch_mlp", None)
    if not callable(switch_mlp):
        raise ValueError("MoE layer does not expose a callable switch_mlp module.")
    selected_outputs = switch_mlp(moe_input, routing.indices)
    state.accumulate(
        indices=routing.indices,
        scores=routing.scores.astype(mx.float32),
        selected_outputs=selected_outputs.astype(mx.float32),
    )
    moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)
    shared_expert = get_shared_expert(moe)
    if shared_expert is not None:
        moe_out = moe_out + shared_expert(moe_input)
    return moe_out

# _observe_lfm2_moe_layer (LFM2):
def _observe_lfm2_moe_layer(layer, moe_input, state, *, adapter, config):
    mx = _require_mlx_core()
    moe = adapter.get_moe(layer)
    routing = Lfm2MoeRouter(moe, config)(moe_input)     # ← Only difference
    # ... identical from here ...
```

These two functions are **100% identical** except for the router class name. Both routers produce `RouterResult` with `.indices` and `.scores`, and the rest of the logic is architecture-agnostic.

**Severity**: Medium. Any bug fix or improvement must be manually replicated. If a third model family is added, a third copy would appear.

**Recommended fix**: Unify into a single parameterized function:

```python
def _observe_moe_layer_impl(
    layer: Any,
    moe_input: Any,
    state: PruningState,
    *,
    adapter: Any,
    config: Mapping[str, Any],
    router_cls: type,
) -> Any:
    mx = _require_mlx_core()
    moe = adapter.get_moe(layer)
    routing = router_cls(moe, config)(moe_input)
    switch_mlp = getattr(moe, "switch_mlp", None)
    if not callable(switch_mlp):
        raise ValueError("MoE layer does not expose a callable switch_mlp module.")
    selected_outputs = switch_mlp(moe_input, routing.indices)
    state.accumulate(
        indices=routing.indices,
        scores=routing.scores.astype(mx.float32),
        selected_outputs=selected_outputs.astype(mx.float32),
    )
    moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)
    shared_expert = get_shared_expert(moe)
    if shared_expert is not None:
        moe_out = moe_out + shared_expert(moe_input)
    return moe_out
```

Then in the caller:

```python
# In _observe_qwen3_model:
h = h + _observe_moe_layer_impl(
    layer, moe_input, accumulators[layer_idx],
    adapter=adapter, config=config, router_cls=Qwen3MoeRouter,
)

# In _observe_lfm2_model:
h = h_mid + _observe_moe_layer_impl(
    layer, ffn_input, accumulators[layer_idx],
    adapter=adapter, config=config, router_cls=Lfm2MoeRouter,
)
```

Delete the two old functions. Keep backward-compatible thin wrappers if needed for external callers.

**Effort**: ~15-line change.

---

### 3.3 Router Epsilon Asymmetry

**Location**: `src/reap/router.py`

**Current state:**

| Router | Normalization | Epsilon? |
|---|---|---|
| `Qwen3MoeRouter` | `scores / scores.sum()` | ❌ No |
| `Lfm2MoeRouter` | `scores / (scores.sum() + 1e-20)` | ✅ Yes |

**Analysis**: The Qwen3 router uses `mx.softmax(logits, axis=-1, precise=True)`. With `precise=True`, MLX uses a numerically stable softmax algorithm with higher internal precision. The selected top-k softmax values are always > 0 (exp(x) > 0 for any finite x), and their sum is always > 0. Therefore, the Qwen3 path **cannot divide by zero** — the epsilon is not needed.

The LFM2 router uses `mx.softmax(logits, axis=-1)` without `precise` and supports `expert_bias` which could theoretically push some gate values toward zero, so the `1e-20` epsilon is defensive coding. However, since softmax output is still positive, the sum is still > 0.

**Severity**: Very low (cosmetic). Both paths are correct. But the asymmetry is confusing and suggests one might be wrong when neither is.

**Recommendation**: Either add epsilon to Qwen3 for consistency, or remove it from LFM2 for cleanliness. Since both are safe, alignment in either direction is fine. I recommend adding `+ 1e-20` to Qwen3 to match LFM2's defensive style.

**Also noted**: `metrics.py` uses `_FLOAT_EPS = np.finfo(np.float64).eps` (~2.2e-16). If a unified epsilon constant is desired across the codebase, define it once rather than using different values in different modules.

**Effort**: 1-line change.

---

### 3.4 Uniform Expert Count Constraint

**Location**: `src/reap/prune.py`, `prune_experts()`, lines ~70-78

**Current behavior:**

```python
if config_num_experts is None:
    config_num_experts = retained_count       # First MoE layer sets the expectation
    config_top_k = new_top_k
elif config_num_experts != retained_count:
    raise ValueError(
        "MLX MoE config update requires all pruned layers to retain "
        f"the same expert count. Layer {layer_idx} retained "
        f"{retained_count}, expected {config_num_experts}."
    )
```

This enforces that **all MoE layers must prune to the same number of experts**. For the currently supported models (LFM2.5, Qwen3-MoE) this is fine — they have uniform expert counts. But it explicitly **blocks**:

- Models where different layers have different numbers of experts
- Layer-specific compression ratios (e.g., prune earlier layers more aggressively)
- Hybrid architectures with varying MoE configurations per layer

**Severity**: Low for current use cases. Would become a blocker if a model with non-uniform expert counts needs support.

**Recommendation**: Document this constraint clearly in the README's Supported Models section. If non-uniform pruning becomes a requirement, the fix would involve storing per-layer expert counts in the config (e.g., `config["num_experts_per_layer"] = {0: 4, 1: 6, ...}`) instead of a single global `config["num_experts"]`.

**Effort**: Documentation-only for now. Implementation would require ~30 lines if needed later.

---

### 3.5 Hardcoded Smoke Test Parameters

**Location**: `src/reap/save.py`, `generation_smoke()` and `src/reap/entrypoint.py`

**Current state:**

```python
# generation_smoke defaults:
prompt: str = "What is your name?",
max_tokens: int = 16,
```

These are not exposed via CLI flags. The smoke test is meant to validate that the pruned model generates coherent text — but a 16-token generation from "What is your name?" is a minimal signal. Users might want to customize the prompt (to match the model's domain) or increase tokens for a more thorough validation.

**Severity**: Low. The smoke test works fine as-is for basic validation. But for a configurable pipeline, this stands out.

**Recommendation**: Add `--smoke-prompt` and `--smoke-max-tokens` CLI arguments:

```python
parser.add_argument("--smoke-prompt", default="What is your name?")
parser.add_argument("--smoke-max-tokens", type=int, default=16)
```

Pass them through `main()` to `save_pruned_model()` → `generation_smoke()`. The `generation_smoke` function already accepts `prompt` and `max_tokens` as keyword arguments, so only the CLI layer needs changes.

**Effort**: ~10 lines in `entrypoint.py`, 2 lines in the call chain.

---

### 3.6 Config Mutation Pattern Documentation

**Location**: `src/reap/prune.py`, `prune_experts()`

**Current behavior:** `prune_experts()` mutates `config` in-place:

```python
def prune_experts(model, config, observer_data, prune_method, compression_ratio, *, adapter=None):
    # ...
    for layer_idx in adapter.identify_moe_layers(model):
        # ... prunes layers ...
    # After loop:
    update_config(config, num_experts=config_num_experts, top_k=config_top_k)
    return keep_by_layer
```

The metrics code in `entrypoint.py` correctly handles this:

```python
config_before_prune = dict(config)          # Snapshot before
with metrics.phase("prune"):
    keep_by_layer = prune_experts_fn(...)   # Mutates config
metrics.record_pruning(keep_by_layer, config_before=config_before_prune, config_after=config, ...)
```

This works correctly in the single-pass pipeline. However:

- If `prune_experts()` is called multiple times on the same config dict (e.g., iterative pruning experiments), the "before" snapshot would be stale after the first call
- Callers might not realize the config is mutated and try to reuse it

**Severity**: Low. The current pipeline calls it once. But undocumented mutation is a latent risk.

**Recommendation**: Add a docstring note to `prune_experts()`:

```
Note: This function mutates `config` in-place. The caller should copy the config
before calling if the original values are needed after pruning.
```

**Effort**: 2-line docstring addition.

---

## Performance Deep-Dive

### 4.1 Qwen3 Attention Mask Hoisting

Already covered in [Section 3.1](#31-qwen3-observer-redundant-attention-mask-computation). This is the most concrete, low-effort win.

**Estimated impact**: Reduces graph node creation by `(num_layers - 1) × num_sequences` attention masks. For 32-layer × 8-sequence runs: ~248 fewer mask allocations.

### 4.2 MLX Graph Compilation Potential

**Location**: `src/reap/observer.py`, `_observe_qwen3_model()` and `_observe_lfm2_model()`

The observer repeatedly executes the same forward pass graph for each layer of each sequence. This is an ideal candidate for MLX's `mx.compile()`:

```python
# Current:
for sequence in calibration_sequences:
    for layer_idx, layer in enumerate(layers):
        h = embed_tokens(tokens)
        mask = ...  # after hoisting
        h = _run_attention(layer, h, mask)
        # ... MoE forward ...
        eval_fn(h)

# Potential:
@mx.compile  # or mx.compile the per-layer steps
def compiled_layer_forward(layer, h, mask, ...):
    # ...
```

**Caveats**:
- `mx.compile()` works best with static shapes. The observer uses varying sequence lengths (up to `max_seq_length`), so compilation would need to handle dynamic shapes or pad sequences.
- MLX's compile is relatively new and may not support all operations used in the pipeline.
- The benefit would be largest for models with many layers (reduced graph submission overhead).

**Recommendation**: Experiment with `mx.compile()` on the per-layer forward pass. Start with a fixed sequence length (pad all calibration sequences to `max_seq_length`) and measure throughput improvement. If the gain is significant (>20%), consider adding a `--compile` flag.

**Effort**: Experiment required. Implementation would be ~20 lines if viable.

### 4.3 Eval Frequency Tuning

**Location**: `src/reap/observer.py`, `eval_fn(h)` calls in both observer paths

**Current**: `mx.eval(h)` after every single layer. This is the safest approach for memory but forces a GPU synchronization at each layer boundary.

**Trade-off**: Evaluating every N layers instead of every layer:

| Eval Frequency | Peak Memory | Throughput |
|---|---|---|
| Every layer (current) | Lowest | Lowest |
| Every 2 layers | ~2× intermediate activations | Moderate gain |
| Every 4 layers | ~4× intermediate activations | Higher gain |
| Per sequence only | Highest (all layers' activations) | Highest |

For Apple Silicon's unified memory, peak memory is critical — the model weights already consume significant memory, and activations accumulate on top. An 8B model with 4-bit quantization is ~4GB. Adding 32 layers of float16 activations at 1024 tokens × 4096 hidden is ~256MB per layer, so evaluating every 4 layers would add ~1GB peak — likely safe.

**Recommendation**: Make eval frequency configurable via `--eval-frequency` flag (default `1` for safety). Users with abundant memory can increase it for faster calibration.

**Effort**: ~15 lines (add parameter to observer functions, CLI flag, pass through).

### 4.4 Python Loop Overhead in Layer Replay

**Location**: `src/reap/observer.py`, all observer paths

Each iteration of the layer loop involves:
1. Python function call overhead
2. MLX graph submission for attention/operator
3. MLX graph submission for norm
4. MLX graph submission for MoE forward
5. MLX eval (graph execution + synchronization)
6. Python back to loop header

For a 32-layer model × 8 sequences: 256 separate Python→MLX→Python round trips. With batch-size-1 sequences (e.g., 1024 tokens × 4096 hidden = 4M elements), the GPU work per layer is sub-millisecond, making Python overhead potentially significant.

**Potential improvements:**
- **Padded batching**: Pad calibration sequences to uniform length and batch multiple sequences together. A batch of 4 sequences would reduce Python round trips by 4× but increase peak memory proportionally.
- **Sequence-level batching**: Group short sequences together within the same forward pass.

**Recommendation**: Low priority for current calibration sizes (8 samples). Would become more important if calibration is scaled to 100+ samples. Document as a future optimization path.

### 4.5 NumPy Scatter Operations Scaling

**Location**: `src/reap/metrics.py`, `PruningState.accumulate()`

```python
np.add.at(self.ean_sum, flat_indices, flat_norms)
np.add.at(self.weighted_ean_sum, flat_indices, flat_norms * flat_scores)
np.add.at(self.weighted_expert_frequency_sum, flat_indices, flat_scores)
np.maximum.at(self.max_activations, flat_indices, flat_maxes)
```

These unbuffered in-place operations are O(tokens × top_k). For current defaults (8 sequences × 1024 tokens × 4 top_k = 32K scatter ops per layer), this is fast. But scaling linearly with calibration data:

| Calibration | Tokens | Top-K | Scatter Ops/Layer |
|---|---|---|---|
| Default (8×1024×4) | 32K | 4 | 128K |
| Large (128×2048×8) | 2M | 8 | 16M |
| Extreme (512×4096×16) | 33M | 16 | 528M |

At 16M+ scatter ops, CPU time could become measurable (tens of milliseconds per layer).

**Recommendation**: Fine for current defaults. If scaling to larger calibration sets, consider batching the scatter operations or moving accumulation to GPU (MLX has scatter-add operations that run on GPU).

### 4.6 LFM2 Mask Computation (Already Optimized)

**Location**: `src/reap/observer.py`, `_observe_lfm2_model()`

The LFM2 observer already computes masks once per sequence (outside the layer loop):

```python
attn_mask, conv_mask = _lfm2_masks(
    h, sequence_length=tokens.shape[-1], mask_fn=mask_fn
)
for layer_idx, layer in enumerate(layers):
    operator_mask = attn_mask if _is_lfm2_attention_layer(layer) else conv_mask
```

✅ This is correct and efficient. No change needed.

### 4.7 MLX Type Conversion Overhead

**Location**: `src/reap/observer.py`, both `_observe_moe_layer` variants

```python
scores=routing.scores.astype(mx.float32),
selected_outputs=selected_outputs.astype(mx.float32),
```

These `.astype()` calls create MLX graph nodes for type conversion. If the source tensors are already `float32`, MLX likely optimizes these away (identity op), but they still add to the graph before compilation/optimization.

**Recommendation**: Check if conversion is actually needed. If `routing.scores` and `selected_outputs` are already `float32` (which they likely are in most MLX-LM models), remove the `.astype()` calls. This is a micro-optimization.

### 4.8 PruningState.report() Array Copying

**Location**: `src/reap/metrics.py`, `PruningState.report()`

```python
return {
    "expert_frequency": self.expert_frequency.copy(),
    "pairwise_expert_frequency": self.pairwise_expert_frequency.copy(),
    "ean_sum": self.ean_sum.copy(),
    "weighted_ean_sum": self.weighted_ean_sum.copy(),
    ...
}
```

Each `.copy()` duplicates a NumPy array. Called once per MoE layer at the end of observation. For current expert counts (32-64), these are trivial (few KB each). The copies are immediately serialized to JSON in `RunMetrics.write()` and then garbage collected.

**Recommendation**: No action needed for current scale. This is documented for completeness.

---

## Code Quality & Maintenance Observations

### 5.1 Import-Safety Enforcement

**Status**: ✅ Exemplary

Every module is verified via subprocess-based tests. The test pattern is:

1. Spawn a subprocess with `sys.meta_path` import blocker
2. Import the target module
3. Assert no blocked modules (`torch`, `vllm`, `mlx`, `mlx_lm`, `datasets`) were loaded
4. Verify basic functionality (construct objects, call `--help`, etc.)

This is a genuine best practice for ML-framework-wrapping packages. Most libraries claim "import-safe" but don't verify it. REAP MLX does.

### 5.2 Adapter Pattern Extensibility

**Status**: ✅ Good

The adapter interface requires ~5 methods per implementation:

| Method | Purpose |
|---|---|
| `layers(model)` | Return decoder layers |
| `identify_moe_layers(model)` | Return indices of MoE layers |
| `is_moe_layer(layer)` | Boolean check |
| `get_moe(layer)` | Return MoE module |
| `get_dense_mlp(layer)` | Return dense MLP module |
| `get_layer_config(layer, config)` | Return `MoeLayerConfig` |

Adding a new model family is scoped to implementing this interface and adding a router class if routing differs. The `infer_model_adapter()` function auto-detects based on config and model layout.

**Improvement opportunity**: The adapter could expose a `update_config(config, num_experts, top_k)` method instead of having separate `update_qwen3_moe_config()` and `update_lfm2_moe_config()` functions. Currently `prune.py` calls these by name.

### 5.3 Injectable Dependencies for Testability

**Status**: ✅ Excellent

The `main()` function signature accepts `load_model_fn`, `load_calibration_sequences_fn`, `observe_model_fn`, `prune_experts_fn`, `save_pruned_model_fn`, `smoke_fn`, and `print_fn`. Every test uses these to verify pipeline logic without loading real models. Production code uses the defaults (real MLX functions).

### 5.4 Save-Reload Validation Thoroughness

**Status**: ✅ Excellent

`save_pruned_model()` validates (in order):
1. Output directory is a directory (not a file)
2. `config.json` exists after save
3. Weight files (`.safetensors` or `.npz`) exist after save
4. Reloaded config `num_experts` matches expected count
5. Reloaded model has adapter-visible MoE layers
6. Every MoE layer's switch projections have correct first dimensions
7. Gate weight shapes match retained expert count
8. LFM2 expert bias shape is validated when present

The tests cover:
- Config mismatch detection
- Shape mismatch detection
- Missing artifact detection
- Invalid reload return format
- Expert bias shape mismatch
- Output path that is a file (not a directory)

### 5.5 Router Code Duplication

**Status**: ⚠️ Opportunity for consolidation

`Qwen3MoeRouter` and `Lfm2MoeRouter` share ~80% of their logic. Differences:

| Feature | Qwen3 | LFM2 |
|---|---|---|
| Hidden state handling | Flattens to 2D before gate | Passes full shape |
| Softmax | `precise=True` | Default |
| Expert bias | ❌ | ✅ |
| Norm epsilon | ❌ | ✅ (1e-20) |

**Recommendation**: Extract a base class with shared logic and make the differences explicit via template method pattern or configuration flags. This would make the differences obvious and reduce the surface area for bugs.

**Effort**: ~40-line refactor. Moderate effort, good maintainability win.

### 5.6 Observer Code Duplication

**Status**: ⚠️ Needs consolidation

`_observe_qwen3_model()` and `_observe_lfm2_model()` share significant structure:
- Token embedding
- Layer iteration
- Mask computation (already covered)
- Eval calls
- Accumulator initialization

The differences are:
- Mask computation (single vs dual masks)
- Layer forward pass (attention + post-norm vs operator + ffn-norm)
- Router class in the MoE observer

These could potentially be unified, but the differences are more structural than the MoE layer functions. The priority is fixing the MoE layer functions first (Section 3.2).

### 5.7 Test Coverage Gaps

The test suite is comprehensive (10 test files), but a few paths are untested:

| Gap | Priority | Notes |
|---|---|---|
| `debug_memory=True` in observer | Low | Memory logging path untested |
| `mask_fn` returning `None` for `seq_len == 1` | Low | Edge case in observer |
| `--verbose` / logging output behavior | Low | CLI flag behavior |
| `_log_memory` debug function | Low | Internal helper |
| Integration test with real model | Future | Would require downloading GB models |
| `mask_fn` returning different mask per layer | Low | Custom mask function support |

All critical paths (import safety, routing, pruning, save/reload, pipeline orchestration, metrics) are well-covered.

---

## Potential Future Concerns

### 6.1 Non-Uniform Expert Architectures

If a model with varying expert counts per layer needs support:
1. The uniform constraint in `prune_experts()` would need to be relaxed
2. Config would need per-layer expert counts (e.g., `config["num_experts"]` → `config["per_layer_experts"]`)
3. `save_pruned_model()` validation would need to check per-layer shapes against per-layer expected counts
4. `RunMetrics` would need to track per-layer before/after counts

### 6.2 Tokenizer Edge Cases

`extract_text()` handles many common dataset formats (messages, conversations, instruction/input/output, raw text). However, multimodal content (images, audio) in the `content` field would be JSON-serialized rather than extracted as text. This is correct behavior (no text to extract) but could produce unexpected JSON strings in calibration text.

### 6.3 Pairwise Expert Frequency Scaling

The `pairwise_expert_frequency` matrix is O(num_experts²). For current models (32 experts = 1024 elements), this is negligible. For models with 256 experts, it becomes 65,536 int64 elements per layer — still manageable (~512KB per layer). For 1024 experts (future MoE models), it would be 1M elements (~8MB) per layer, which could become a memory concern.

### 6.4 Calibration Data Size & Memory

The entire pipeline loads calibration sequences into memory as a list of numpy arrays. Each sequence at 2048 tokens consumes ~8KB (int32). 128 sequences × 8KB = ~1MB — trivial. But if someone runs with 10,000 sequences at 4096 tokens, that's ~160MB of calibration data in memory. The pipeline doesn't stream from disk.

---

## Confirmed Non-Issues

These were initially flagged in analysis but confirmed as non-issues after deeper examination:

| Claim | Resolution |
|---|---|
| Qwen3 router division by zero | Softmax with `precise=True` guarantees positive selected scores — sum is always > 0 |
| `ru_maxrss` platform-dependent units | macOS-only codebase — `ru_maxrss` is always bytes on macOS |
| `_batch_tokens()` with `None` input_ids | All calibration sequences come from `load_calibration_sequences()` which always includes `"input_ids"` |
| `compute_keep_indices()` with all-NaN scores | Observer accumulator always produces finite scores (zero-initialized + finite additions from model outputs) |
| Expert bias Python list indexing | Trivial for current expert counts (32-64), and MLX handles Python list indexing correctly |
| `_validate_args` → `parser.error` flow | Standard argparse error handling pattern that works correctly |
| `PruningState.report()` array copying | Called once per layer, copies are KB-sized, immediately serialized to JSON and GC'd |
| Config mutation correctness | `main()` snapshots config with `dict(config)` before pruning — correct for single-pass pipeline |

---

## Summary of Recommendations

### Immediate (Low Effort, Clear Benefit)

| # | Recommendation | Effort | File(s) | Impact |
|---|---|---|---|---|
| 1 | **Hoist Qwen3 attention mask** out of layer loop | 2 lines | `observer.py:85-92` | Reduces ~250 redundant mask creations |
| 2 | **Unify MoE layer observer functions** into one parameterized by router class | ~15 lines | `observer.py:280-320` | Eliminates code duplication, simplifies future additions |
| 3 | **Add epsilon to Qwen3 norm** or remove from LFM2 | 1 line | `router.py:118` or `router.py:212` | Consistency |
| 4 | **Document uniform expert count constraint** in README | 1 line | `README.md` | Clarity |
| 5 | **Document config mutation** in `prune_experts()` docstring | 2 lines | `prune.py:30` | Prevents future misuse |

### Short-Term (Moderate Effort, User-Facing)

| # | Recommendation | Effort | File(s) | Impact |
|---|---|---|---|---|
| 6 | **Expose smoke test CLI flags** (`--smoke-prompt`, `--smoke-max-tokens`) | ~12 lines | `entrypoint.py`, `save.py` | User configurability |
| 7 | **Add `--eval-frequency` flag** for tunable memory/throughput | ~15 lines | `observer.py`, `entrypoint.py` | Performance tuning for power users |

### Future (Requires Experimentation)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 8 | **Investigate `mx.compile()`** for observer forward pass | Experiment | Potential 20-50% throughput improvement |
| 9 | **Extract shared router base class** | ~40 lines | Code quality and maintainability |
| 10 | **Consider padded batching** for large calibration sets | ~50 lines | 2-4× throughput for many-sequence calibration |
| 11 | **Unify Qwen3/LFM2 observer paths** (structural consolidation) | ~60 lines | Code quality |

### Non-Actionable (Accepted Design Decisions)

| # | Design Decision | Rationale |
|---|---|---|
| — | Batch-size-1 calibration | Simpler mask and accumulator logic, lower memory |
| — | No streaming calibration loading | Full dataset fits in memory for default configs |
| — | Per-layer eval | Safest memory strategy for Apple Silicon unified memory |
| — | Uniform expert count constraint | Currently supported models are uniform |

---

*Generated by deep codebase analysis of `reap-mlx` at commit snapshot 2026-06-01.*
