# REAP MLX Implementation Plan - GPT v0.1

Date: 2026-05-30

This is a sharper implementation plan after comparing:

- `docs/implementation-points-gpt.md`
- `docs/implementation-points-dsv4.md`
- `docs/implementation-plan-synthesized.md`
- the current REAP codebase
- the installed `mlx` / `mlx_lm` APIs on this machine

The goal is not to make every REAP feature backend-agnostic immediately. The
first goal is to get pruning-only REAP working well on MLX-LM models, with a
small and defensible implementation surface.

## Core Conclusion

Do not begin with a broad `AbstractBackend` that wraps tensor primitives.

That would force MLX through a PyTorch-shaped algorithm and encourage an
all-expert activation implementation, which is exactly the expensive path we
should avoid on Apple Silicon.

The better first implementation is a separate MLX pruning pipeline:

```text
load MLX-LM model
prepare calibration tokens
replay layers explicitly
observe actual routed expert outputs
accumulate REAP pruning stats on CPU
prune stacked expert tensors in-place
save with MLX-LM utilities
reload and smoke test
```

The existing PyTorch/CUDA path should remain stable while this path is added.

## Important Corrections From Further Investigation

### MLX API Details

On this machine:

- `mlx` version: `0.31.2`
- `mlx_lm` version: `0.31.3`
- `mx.bincount` is not available.
- `mx.scatter_add` is not available.
- `mx.topk` returns values only, not `(values, indices)`.
- `mx.argpartition` and `mx.take_along_axis` are available and are what MLX-LM
  itself uses for router selection.
- `mx.get_active_memory`, `mx.get_peak_memory`, `mx.get_cache_memory`,
  `mx.set_cache_limit`, and `mx.set_memory_limit` are available.
- `mx.metal.get_active_memory` exists but is deprecated in favor of the top-level
  memory APIs.

### MLX-LM Model Details

MLX-LM does have mutable `nn.Module` objects. We do not need to operate only on
raw weight dictionaries.

Useful module APIs:

- `parameters()`
- `children()`
- `named_modules()`
- `modules()`
- `update(...)`
- `update_modules(...)`
- `save_weights(...)`
- `load_weights(...)`

MoE experts are usually stacked in `switch_mlp`:

```text
switch_mlp.gate_proj.weight
switch_mlp.up_proj.weight
switch_mlp.down_proj.weight
```

For quantized switch layers, associated tensors also exist:

```text
weight
scales
biases
bias
```

### Attention Mask Detail

MLX-LM model forwards generally create a causal mask internally and do not accept
Hugging Face-style `attention_mask`.

That means the first MLX calibration path should avoid padded multi-sample
batches. Otherwise, pad tokens become part of the context and the observed MoE
traffic will differ from the PyTorch path.

Initial recommendation:

- use batch size `1`, unpadded sequences, or already packed calibration streams
- keep a valid-token mask only for metrics if padded batches are later supported
- treat full HF-style attention-mask support as a later feature

### Save Behavior Detail

`mlx_lm.utils.save(...)` writes weights, config, tokenizer, copied Python files,
generation config, and a model card. In the installed version, it calls
`save_model(..., donate_model=True)` internally, regardless of the exposed
`donate_model` argument.

Practical implication:

- run any in-memory smoke generation before `utils.save(...)`, or
- save, then reload the saved model for the smoke test

The latter is the better validation.

### Notes After Reading The Synthesized Plan

The synthesized plan agrees with the main direction: separate MLX path,
selected-only pruning metrics, CPU-side accumulators, and deferred evaluation.
There are two implementation details to keep precise:

1. The final weighted MoE output is not enough for REAP attribution. REAP needs
   per-selected-expert outputs before the top-k weighted sum, otherwise the
   activation norm cannot be assigned to the individual routed expert. In MLX-LM
   this means using `moe.switch_mlp(x, indices)` directly and preserving its
   `[batch, seq, top_k, hidden]` output before summing over `top_k`.

2. The current `data.py` is not framework-agnostic yet. It imports `torch`,
   imports `vllm.TokensPrompt` at module import time, and builds padded
   `BatchEncoding` objects. The MLX path should either use a minimal MLX
   calibration loader first or refactor `data.py` to support NumPy/MLX outputs
   and lazy `vllm` imports.

Also, exact saliency parity with the PyTorch backend should not be the default
success criterion across all architectures. For models with different router
semantics, the MLX default should use actual MLX-LM routing behavior. A
compatibility mode can be added where strict PyTorch-style metric parity is
needed.

## Non-Goals For v0.1

The first MLX implementation should not attempt:

- full merge support
- token-to-token matching metrics
- characteristic activation distance metrics
- permutation support
- vLLM-compatible evaluation
- broad tensor-operation backend abstraction
- padded batched calibration with exact HF attention-mask parity

Those can come later once pruning-only REAP is correct and reloadable.

## Proposed File Layout

Add an MLX-specific implementation area:

```text
src/reap/backends/
  __init__.py
  mlx/
    __init__.py
    data.py
    entrypoint.py
    metrics.py
    model_util.py
    observer.py
    prune.py
    router.py
    save.py
```

Possible CLI entrypoint:

```text
python -m reap.backends.mlx.entrypoint ...
```

Do not route through `src/reap/main.py` at first, because `main.py` imports
PyTorch, vLLM evaluation, CUDA-centric utilities, and the existing observer
stack. A shared `--backend` flag can be added later after the MLX path works.

## Data Model

### Router Result

Router adapters should return a small architecture-neutral object:

```python
@dataclass
class MlxRouterResult:
    logits: mx.array | None
    indices: mx.array      # [batch, seq, top_k]
    scores: mx.array       # [batch, seq, top_k]
    score_mode: str        # "actual", "compat_softmax_logits", etc.
```

Default behavior should use the model's actual routing scores.

Compatibility with the current PyTorch REAP behavior can be added explicitly via
`score_mode="compat_softmax_logits"` where possible.

### Selected Expert Batch

The observer should not materialize all expert activations.

Instead, it should produce:

```python
@dataclass
class SelectedExpertBatch:
    indices: np.ndarray        # [valid_tokens, top_k]
    scores: np.ndarray         # [valid_tokens, top_k]
    output_norms: np.ndarray   # [valid_tokens, top_k]
    output_maxes: np.ndarray   # [valid_tokens, top_k]
    num_valid_tokens: int
```

The selected outputs come from:

```python
selected_outputs = moe.switch_mlp(moe_input, router.indices)
```

with shape:

```text
[batch, seq, top_k, hidden]
```

Then:

```python
output_norms = mx.linalg.norm(selected_outputs, axis=-1)
output_maxes = selected_outputs.max(axis=-1)
```

`output_maxes` intentionally follows the current PyTorch behavior, which tracks
the maximum raw activation value, not the absolute maximum.

## REAP Metric Accumulation

Use CPU/NumPy accumulators for v0.1.

Reason:

- REAP state is small.
- MLX lacks `bincount` in this environment.
- Avoiding scatter-heavy MLX code reduces graph complexity.
- NumPy accumulation is easy to validate against current tests.

Suggested state:

```python
@dataclass
class MlxPruningState:
    total_tokens: int
    expert_frequency: np.ndarray
    pairwise_expert_frequency: np.ndarray
    ean_sum: np.ndarray
    weighted_ean_sum: np.ndarray
    weighted_expert_frequency_sum: np.ndarray
    max_activations: np.ndarray
```

Derived values at report time:

```python
expert_proba = expert_frequency / total_tokens
ean_mean = ean_sum / expert_frequency
reap = weighted_ean_sum / expert_frequency
```

Use zero where `expert_frequency == 0`.

For each observed batch:

```python
flat_experts = indices.reshape(-1)
flat_scores = scores.reshape(-1)
flat_norms = output_norms.reshape(-1)
flat_maxes = output_maxes.reshape(-1)

freq = np.bincount(flat_experts, minlength=num_experts)
state.total_tokens += num_valid_tokens
state.expert_frequency += freq
state.pairwise_expert_frequency += freq[:, None] + freq[None, :]

np.add.at(state.ean_sum, flat_experts, flat_norms)
np.add.at(state.weighted_ean_sum, flat_experts, flat_norms * flat_scores)
np.add.at(state.weighted_expert_frequency_sum, flat_experts, flat_scores)

for expert_id in np.unique(flat_experts):
    state.max_activations[expert_id] = max(
        state.max_activations[expert_id],
        flat_maxes[flat_experts == expert_id].max(),
    )
```

This reproduces the pruning metrics needed by `prune_method` values:

- `frequency`
- `ean_sum`
- `ean_mean`
- `weighted_ean_sum`
- `reap`
- `max_activations`

`weighted_frequency_sum` should be normalized to the existing key naming. The
current code uses `weighted_expert_frequency_sum`, while `PruneArgs` lists
`weighted_frequency_sum`; this mismatch should be fixed or aliased.

## Layer Replay Strategy

MLX has no PyTorch-style forward hooks. The observer should replay layers
explicitly.

Most MLX-LM MoE decoder layers follow this shape:

```python
r = layer.self_attn(layer.input_layernorm(h), mask, cache)
h_attn = h + r
moe_input = layer.post_attention_layernorm(h_attn)
mlp_out = layer.mlp(moe_input) or layer.block_sparse_moe(moe_input)
h = h_attn + mlp_out
```

The MLX observer should implement this as a small set of architecture adapters,
not as a fully generic reflection system.

For each layer:

1. Run attention normally.
2. Compute the post-attention MLP/MoE input.
3. If the block is dense, run it and continue.
4. If the block is MoE:
   - call the router adapter
   - call `switch_mlp(moe_input, indices)`
   - aggregate selected-output metrics
   - compute the normal weighted MoE output
   - add shared expert output if the architecture has shared experts
5. Call `mx.eval(h)` at controlled boundaries.

Recommended initial implementation:

- support batch size `1`
- no KV cache during calibration
- no generation cache
- full prompt forward only
- one model resident in unified memory
- process layers sequentially

## Router Adapters

The router adapter is the most important correctness boundary.

### Qwen3-MoE

MLX-LM behavior:

```python
logits = moe.gate(x)
gates = mx.softmax(logits, axis=-1, precise=True)
indices = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
scores = mx.take_along_axis(gates, indices, axis=-1)
if moe.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
```

Use this exact behavior for v0.1.

### Mixtral

MLX-LM behavior:

```python
logits = moe.gate(x)
indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
selected_logits = mx.take_along_axis(logits, indices, axis=-1)
scores = mx.softmax(selected_logits, axis=-1, precise=True)
```

This differs from generic `softmax(logits)` followed by top-k.

### DeepSeek-V2

MLX-LM gate already implements:

- full softmax
- optional group-limited greedy selection
- routed scaling factor

For v0.1 after Qwen/Mixtral, prefer calling `moe.gate(x)` and separately compute
raw logits as `x @ moe.gate.weight.T` only if needed for compatibility metadata.

### GLM4-MoE

MLX-LM gate uses:

- sigmoid scores
- correction bias
- group selection
- original-score selection
- optional top-k normalization
- routed scaling factor

This should not be approximated by a generic softmax adapter.

### ERNIE 4.5 MoE

MLX-LM behavior:

- `gate_act` is softmax or sigmoid depending on config
- top-k over activated gates
- selected scores are normalized by selected-score sum
- optional shared experts

## Pruning Strategy

Pruning should operate on live MLX-LM modules.

### Expert Selection

Use NumPy for choosing experts:

```python
experts_to_prune = np.argsort(saliency)[:n_experts_to_prune]
keep = np.array([i for i in range(num_experts) if i not in experts_to_prune])
```

Preserve the existing behavior:

- lower saliency means prune first
- `frequency` maps to `expert_frequency`
- `expert_proba` can be derived
- super-expert preservation can be added after the baseline pruning works

### Switch Layer Slicing

For each `SwitchLinear` or `QuantizedSwitchLinear` inside `switch_mlp`, slice
expert dimension `0`:

```python
linear.weight = linear.weight[keep]
linear.scales = linear.scales[keep]      # if present
linear.biases = linear.biases[keep]      # if present and not None
linear.bias = linear.bias[keep]          # if present
```

The same applies to:

- `switch_mlp.gate_proj`
- `switch_mlp.up_proj`
- `switch_mlp.down_proj`

### Router Slicing

Router/gate weights are sliced on output expert dimension:

```python
gate.weight = gate.weight[keep]
gate.bias = gate.bias[keep]              # if present
gate.e_score_correction_bias = gate.e_score_correction_bias[keep]  # if present
```

For quantized router linear layers, also slice `scales` and `biases`.

### Config and Attribute Updates

Update both runtime module attributes and saved config dict.

Architecture-specific fields:

| Architecture | Expert Count Field | Top-K Field | MoE Attribute |
|---|---|---|---|
| Qwen3-MoE | `num_experts` | `num_experts_per_tok` / `top_k` | `mlp` |
| Mixtral | `num_local_experts` | `num_experts_per_tok` | `block_sparse_moe` |
| DeepSeek-V2 | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| GLM4-MoE | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| ERNIE 4.5 MoE | `moe_num_experts` / `moe_capacity` | `moe_k` | `mlp` |

Always clamp top-k:

```python
new_top_k = min(old_top_k, num_retained_experts)
```

## Import and Dependency Plan

Current blockers:

- `src/reap/data.py` imports `vllm.TokensPrompt` at module import time.
- `src/reap/main.py`, `src/reap/prune.py`, and `src/reap/layerwise_prune.py` import
  `reap.eval`, which imports vLLM and uses CUDA.
- `pyproject.toml` has CUDA-heavy packages in required dependencies.

v0.1 should avoid the existing entrypoints instead of refactoring them first.

Minimum import-safe work:

- keep `src/reap/__init__.py` light
- add MLX modules that do not import PyTorch or vLLM
- add lazy import guards around `vllm` in `data.py` when touched
- eventually split dependencies into extras:

```text
reap[torch]
reap[mlx]
reap[eval]
```

## Calibration Data Plan

v0.1 can use a minimal MLX calibration loader:

- load tokenizer through `mlx_lm.load(..., return_config=True)` or HF tokenizer
- tokenize text into NumPy arrays or Python lists
- convert each unpadded sequence to `mx.array`
- process one sequence at a time

Avoid using the current `BaseDatasetProcessor` directly until it can produce
non-PyTorch, non-vLLM outputs.

Later, adapt `data.py` to support:

```python
return_tensors="np"
return_format="mlx"
pack_samples=True
batch_size=1
```

## Save and Smoke Test

Load:

```python
from mlx_lm import load
model, tokenizer, config = load(model_path, return_config=True)
```

Save:

```python
from mlx_lm import utils
utils.save(
    dst_path=output_dir,
    src_path_or_repo=model_path,
    model=model,
    tokenizer=tokenizer,
    config=updated_config,
)
```

Validation:

1. Save pruned model.
2. Reload from `output_dir`.
3. Run a short generation or forward pass.
4. Verify all pruned MoE layers report the retained expert count.

## Test Plan

Add MLX tests that skip cleanly when `mlx` / `mlx_lm` is unavailable.

### Unit Tests

1. `test_mlx_qwen_router_adapter_matches_model_code`
   - construct tiny Qwen3-MoE MLX block
   - compare adapter indices/scores with the block's own logic

2. `test_mlx_mixtral_router_adapter_matches_model_code`
   - verify Mixtral selected-softmax behavior

3. `test_mlx_reap_selected_metrics_match_numpy_reference`
   - fixed synthetic `indices`, `scores`, `selected_outputs`
   - compare state fields to a hand-computed NumPy reference

4. `test_mlx_prune_switch_mlp_slices_expected_shapes`
   - tiny Qwen3-MoE model
   - prune one expert
   - assert `switch_mlp` and router shapes

5. `test_mlx_pruned_tiny_model_forward`
   - run forward before and after pruning
   - assert output shape is unchanged

### Integration Tests

1. `test_mlx_qwen_prune_save_reload_smoke`
   - small local synthetic model if possible
   - otherwise mark as slow/model-required

2. `test_mlx_no_torch_import_for_mlx_backend`
   - import MLX backend in an environment without importing `torch`
   - this protects the most important portability boundary

## Implementation Phases

### Phase 0: Groundwork

- Add this plan.
- Add `src/reap/backends/mlx/` package skeleton.
- Make sure importing the MLX backend does not import PyTorch or vLLM.

### Phase 1: Qwen3-MoE Pruning Prototype

- Implement Qwen3-MoE model adapter.
- Implement Qwen router adapter.
- Implement selected-output REAP metric accumulation.
- Implement explicit Qwen layer replay.
- Implement Qwen stacked-expert pruning.
- Add tiny-model unit tests.

Success criteria:

- tiny Qwen3-MoE MLX model can be observed
- REAP stats are produced
- one expert can be pruned
- forward still works

### Phase 2: Save/Reload Path

- Update config dict after pruning.
- Save with MLX-LM utilities.
- Reload saved model.
- Run forward/generation smoke test.

Success criteria:

- saved model reloads with retained expert count
- no missing safetensor keys
- no stale config expert count

### Phase 3: Calibration Loader

- Add simple unpadded calibration sequence loader.
- Support composite dataset specs later.
- Avoid padded batch attention mismatch in v0.1.

Success criteria:

- can run a small real calibration set with batch size `1`
- no PyTorch tensor creation required

### Phase 4: Mixtral

- Add Mixtral adapter.
- Add Mixtral tests.
- Validate selected-softmax routing behavior.

### Phase 5: DeepSeek, GLM, ERNIE

- Add architecture adapters one at a time.
- Prefer calling the architecture's own gate implementation.
- Preserve shared expert behavior in forward replay.

### Phase 6: Optional Expansion

- Add merge metrics if needed.
- Add MLX-LM server evaluation.
- Add common `--backend` CLI only after both paths are stable.
- Consider backend-neutral result serialization.

## Main Risks

### Router Semantics Drift

Risk: pruning decisions differ because routing is approximated.

Mitigation:

- use architecture-specific adapters
- compare adapter outputs against MLX-LM model code
- add compatibility mode only explicitly

### Padded Batch Mismatch

Risk: MLX calibration sees pad tokens as real context.

Mitigation:

- v0.1 uses unpadded batch size `1`
- add padded attention support only after baseline works

### Quantized Tensor Slicing

Risk: slicing `weight` but not `scales` / `biases` corrupts quantized experts.

Mitigation:

- centralize slicing helper
- test quantized `SwitchLinear`
- inspect every module for optional quantization fields

### Stale Config

Risk: saved model reloads with old expert count.

Mitigation:

- update runtime attrs and config dict together
- reload after save as required validation

### Over-Refactoring

Risk: trying to make all current code backend-agnostic delays the actual MLX
pruning path.

Mitigation:

- create separate MLX path first
- refactor shared code only after behavior is working

## Recommended First PR Shape

The first meaningful PR should be intentionally small:

```text
docs/reap-mlx-implementation-plan-gpt-v0.1.md
src/reap/backends/__init__.py
src/reap/backends/mlx/__init__.py
src/reap/backends/mlx/router.py
src/reap/backends/mlx/metrics.py
tests/test_mlx_router.py
tests/test_mlx_pruning_metrics.py
```

It should not yet touch the PyTorch observer, PyTorch prune path, merge path, or
evaluation path.

The second PR can add:

```text
src/reap/backends/mlx/model_util.py
src/reap/backends/mlx/observer.py
src/reap/backends/mlx/prune.py
tests/test_mlx_prune_qwen.py
```

This keeps the work reviewable and avoids mixing import cleanup, algorithm
changes, model mutation, and save/reload behavior in one large patch.

## v0.1 Decision Summary

- Separate MLX pruning path first.
- Qwen3-MoE first, Mixtral second.
- Use selected expert outputs, not all expert outputs.
- Use NumPy accumulators.
- Use exact MLX-LM router semantics.
- Avoid padded batches initially.
- Prune live MLX modules.
- Save and reload with MLX-LM.
- Defer merge/eval/backend-unification until pruning works.
