# REAP MLX Implementation Plan - GPT v0.2

Date: 2026-05-30

This document is a polished synthesis of every current planning document in
`docs/`:

- `implementation-points-gpt.md`
- `implementation-points-dsv4.md`
- `implementation-plan-synthesized.md`
- `implementation-plan-v0.1.md`
- `implementation-plan-v0.2.md`
- `reap-mlx-implementation-plan-gpt-v0.1.md`

It keeps the useful ideas, resolves contradictions, and turns the plan into a
smaller implementation contract. This is still a planning document only.

## Alignment Notice - 2026-05-31

The production and official experimentation workflow remains the original
PyTorch/CUDA REAP implementation. The MLX backend is a parallel Apple Silicon
experimentation path.

The MLX goal is adapter-driven support for compatible MoE weights, not a
one-model port. Qwen3-MoE is the bootstrap/reference adapter only; model-specific
routing, expert layout, shared-expert behavior, and config updates must live
behind explicit MLX adapter contracts.

## Executive Decision

Build a separate MLX pruning path first. Do not begin by refactoring the whole
repo into a generic backend abstraction.

The first working target is:

```text
MLX-LM Qwen3-MoE -> selected-output REAP stats -> prune experts -> save -> reload
```

The PyTorch/CUDA implementation should remain stable while this path is added.
Once the MLX path works end to end, shared schemas and command dispatch can be
pulled upward.

## Resolved Disagreements

| Topic | v0.2 Decision |
|---|---|
| Backend shape | Add a parallel `src/reap/mlx/` path. Do not add a broad `AbstractBackend` first. |
| Pruning target | Pruning-only REAP first. Defer merging, TTM, CA metrics, permutation, and vLLM eval. |
| Expert activations | Use selected expert outputs only. Never materialize `(num_experts, tokens, hidden)` for pruning. |
| Router logic | Use architecture-specific MLX-LM semantics. No generic `topk(softmax(logits))`. |
| Metric state | Use NumPy CPU accumulators. Avoid MLX scatter/count work for running stats. |
| Model mutation | Prune live MLX-LM modules first, then save via MLX-LM utilities. Do not make raw weight-dict surgery the primary path. |
| Calibration data | Start with unpadded batch-size-1 sequences. Do not assume current `data.py` is MLX-safe. |
| Evaluation | Save/reload/forward or short generation smoke test only. Full benchmark eval is later. |

## Facts To Keep In Mind

The local environment currently has:

```text
mlx     0.31.2
mlx_lm  0.31.3
```

Important API facts:

- `mx.bincount` is not available.
- `mx.scatter_add` is not available.
- `mx.topk` returns values only, not `(values, indices)`.
- `mx.argpartition` and `mx.take_along_axis` are available.
- `mx.allclose` is available in this environment.
- Top-level memory APIs exist: `mx.get_active_memory`, `mx.get_peak_memory`,
  `mx.get_cache_memory`, `mx.set_cache_limit`, `mx.set_memory_limit`.
- `mx.metal.get_active_memory` exists but is deprecated.
- MLX-LM models are mutable `nn.Module` objects with `parameters`, `modules`,
  `named_modules`, `update`, `update_modules`, `save_weights`, and
  `load_weights`.

### Corrections After Reading Peer v0.2

The peer `implementation-plan-v0.2.md` mostly aligns with this plan, but these
details should be corrected during implementation:

- `total_tokens` must count valid input tokens, not flattened top-k routes.
  `expert_frequency` counts routes and can sum to `total_tokens * top_k`.
- Explicit full-sequence replay must use a causal mask, not `mask=None`, unless
  sequence length is one. Use the same mask construction MLX-LM uses for normal
  model forward.
- Quantized expert handling cannot be fully deferred if the first real target is
  a 4-bit MLX model. At minimum, the pruning helper must slice `weight`,
  `scales`, `biases`, and `bias` consistently for quantized switch layers.
- MLX-LM models generally expose `args`; the config dict comes from
  `mlx_lm.load(..., return_config=True)`. Do not assume `model.config` exists.
- Router adapter tests should compare adapter routing against architecture
  reference logic or native gate output. `switch_mlp` only consumes indices; it
  does not itself validate router selection.

## First Implementation Package

Use this package layout:

```text
src/reap/mlx/
  __init__.py
  accumulator.py
  data.py
  model_adapters.py
  observer.py
  prune.py
  router.py
  save.py
  smoke_test.py
  cli.py
```

Rationale:

- `src/reap/mlx/` makes it explicit that this is not a generic backend shim.
- Existing root modules can stay PyTorch-oriented for now.
- The MLX package can be imported without `torch`, `vllm`, or CUDA packages.
- A later `reap.entry` or `--backend` dispatcher can call into this package once
  both paths are stable.

Avoid importing these from the MLX package:

```text
reap.main
reap.prune
reap.layerwise_prune
reap.eval
reap.observer
reap.layerwise_observer
```

Those modules currently pull in PyTorch, vLLM, CUDA evaluation, or hook-based
observer code.

## First CLI Shape

Use an independent MLX entrypoint:

```bash
python -m reap.mlx.cli \
  --model-name mlx-community/Qwen3-30B-A3B-4bit-DWQ \
  --dataset-name theblackcat102/evol-codealpaca-v1 \
  --prune-method reap \
  --compression-ratio 0.25 \
  --max-samples 128 \
  --output-dir artifacts/mlx/qwen3-reap-0.25
```

For v0.2 implementation, this CLI should not share `HfArgumentParser` with the
existing CUDA entrypoints unless doing so remains import-safe.

## Observer Data Contract

The MLX observer should report the same pruning keys that the current pruner
expects:

```python
observer_data[layer_idx] = {
    "total_tokens": int,
    "expert_frequency": np.ndarray,                 # int64 [E]
    "pairwise_expert_frequency": np.ndarray,        # int64 [E, E]
    "expert_proba": np.ndarray,                     # float64 [E]
    "ean_sum": np.ndarray,                          # float64 [E]
    "ean_mean": np.ndarray,                         # float32/float64 [E]
    "weighted_ean_sum": np.ndarray,                 # float64 [E]
    "weighted_expert_frequency_sum": np.ndarray,    # float64 [E]
    "reap": np.ndarray,                             # float32/float64 [E]
    "max_activations": np.ndarray,                  # float32 [E]
}
```

Do not support these in the first MLX pruning path:

```text
ean_ca
ttm_similarity_matrix
characteristic_activation
routed_characteristic_activation
online_characteristic_activation_dist
router_logit_similiarity
```

Those require merging-style all-expert information or additional activation
state.

Also fix or alias the naming mismatch:

```text
PruneArgs choice:  weighted_frequency_sum
Actual key:        weighted_expert_frequency_sum
```

The MLX pruner should accept both names.

## Selected Expert Batch

REAP attribution needs per-selected-expert outputs before the top-k weighted
sum. The final MoE output is not enough.

Use this internal batch object:

```python
@dataclass
class SelectedExpertBatch:
    indices: np.ndarray        # [tokens, top_k]
    scores: np.ndarray         # [tokens, top_k]
    output_norms: np.ndarray   # [tokens, top_k]
    output_maxes: np.ndarray   # [tokens, top_k]
    num_tokens: int
```

For a routed MoE block:

```python
router = route(moe, moe_input)
selected_outputs = moe.switch_mlp(moe_input, router.indices)
output_norms = mx.linalg.norm(selected_outputs, axis=-1)
output_maxes = selected_outputs.max(axis=-1)
moe_out = (selected_outputs * router.scores[..., None]).sum(axis=-2)
```

Then add any shared expert output if the architecture has shared experts.

## Metric Accumulation

Use NumPy for running state. REAP state is tiny compared with model activations,
and this avoids MLX lazy graph retention.

Per batch:

```python
flat_experts = indices.reshape(-1)
flat_scores = scores.reshape(-1)
flat_norms = output_norms.reshape(-1)
flat_maxes = output_maxes.reshape(-1)

freq = np.bincount(flat_experts, minlength=num_experts)

state.total_tokens += num_tokens
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

At report time:

```python
expert_proba = expert_frequency / max(total_tokens, 1)
ean_mean = safe_divide(ean_sum, expert_frequency)
reap = safe_divide(weighted_ean_sum, expert_frequency)
```

This matches the effective behavior of the existing `OnlineStatsTracker` path:
batch means are weighted by routed expert counts, so the final result equals
sum divided by count.

Important compatibility detail:

- `total_tokens` counts valid tokens.
- `expert_frequency` counts routes. With top-k > 1, `expert_frequency.sum()` can
  exceed `total_tokens`.
- This mirrors current PyTorch behavior.

## Layer Replay

MLX has no PyTorch forward hooks. The observer should explicitly replay decoder
layers.

Most supported MLX-LM decoder layers follow this shape:

```python
r = layer.self_attn(layer.input_layernorm(h), mask, cache=None)
h_attn = h + r
moe_input = layer.post_attention_layernorm(h_attn)
mlp_out = run_dense_or_moe(layer, moe_input)
h = h_attn + mlp_out
```

The v0.2 implementation should use model adapters rather than one universal
reflection function.

Adapter responsibilities:

```python
class MlxModelAdapter:
    def layers(self, model): ...
    def make_attention_mask(self, h): ...
    def is_moe_layer(self, layer): ...
    def get_moe(self, layer): ...
    def run_attention(self, layer, h, mask): ...
    def run_dense_mlp(self, layer, moe_input): ...
    def run_moe_with_observation(self, layer, moe_input, state): ...
    def update_config_after_prune(self, config, retained_experts): ...
```

Start with Qwen3-MoE. Add Mixtral second.

Call `mx.eval(h)` after each layer or after a small bounded group of layers.
During development, log `mx.get_active_memory()` and `mx.get_peak_memory()`.

## Router Adapters

Router adapters are correctness-critical.

### Qwen3-MoE

Match MLX-LM code:

```python
logits = moe.gate(x)
gates = mx.softmax(logits, axis=-1, precise=True)
k = moe.top_k
indices = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
scores = mx.take_along_axis(gates, indices, axis=-1)
if moe.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
```

Do not use `mx.topk` for indices.

### Mixtral

Match MLX-LM code:

```python
logits = moe.gate(x)
k = moe.num_experts_per_tok
indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
selected_logits = mx.take_along_axis(logits, indices, axis=-1)
scores = mx.softmax(selected_logits, axis=-1, precise=True)
```

This is not equivalent to full-softmax top-k.

### DeepSeek-V2

Prefer calling `moe.gate(x)`.

The gate handles:

- full softmax
- group-limited greedy routing
- routed scaling factor

Raw logits can be computed as `x @ moe.gate.weight.T` for metadata if needed.

### GLM4-MoE

Prefer calling `moe.gate(x)`.

The gate handles:

- sigmoid scores
- correction bias
- group expert selection
- original-score selection
- optional top-k normalization
- routed scaling factor

### ERNIE 4.5 MoE

Match its gate activation:

- softmax or sigmoid depending on config
- top-k over activated scores
- selected-score normalization
- optional shared experts

## Pruning Live MLX Modules

The primary implementation should prune live MLX-LM modules, not raw
pre-sanitize weight dictionaries.

Reason:

- `mlx_lm.load(...)` returns a mutable model.
- Loaded MoE experts are already represented as `switch_mlp` stacked tensors.
- `mlx_lm.utils.save(...)` saves from `model.parameters()`.
- Runtime module attributes and saved parameters stay in sync.

Central slicing helper:

```python
def slice_first_dim(module, keep):
    for name in ("weight", "scales", "biases", "bias"):
        value = module.get(name) if hasattr(module, "get") else getattr(module, name, None)
        if value is not None:
            setattr(module, name, value[keep])
```

Apply to:

```text
moe.switch_mlp.gate_proj
moe.switch_mlp.up_proj
moe.switch_mlp.down_proj
```

Router slicing:

```python
moe.gate.weight = moe.gate.weight[keep]
moe.gate.bias = moe.gate.bias[keep]                         # if present
moe.gate.e_score_correction_bias = ...[keep]                # if present
```

Then update runtime attributes:

```text
Qwen:    moe.num_experts, moe.top_k
Mixtral: moe.num_experts, moe.num_experts_per_tok
DeepSeek/GLM: gate.n_routed_experts, gate.top_k, moe.num_experts_per_tok
ERNIE:   moe.k, args.moe_num_experts, args.moe_capacity
```

Always clamp top-k:

```python
new_top_k = min(old_top_k, retained_expert_count)
```

## Config Update

Update both the live model arguments and the config dict returned by
`mlx_lm.load(..., return_config=True)`.

Architecture-specific config fields:

| Architecture | Expert Count | Top-K |
|---|---|---|
| Qwen3-MoE | `num_experts` | `num_experts_per_tok` |
| Mixtral | `num_local_experts` | `num_experts_per_tok` |
| DeepSeek-V2 | `n_routed_experts` | `num_experts_per_tok` |
| GLM4-MoE | `n_routed_experts` | `num_experts_per_tok` |
| ERNIE 4.5 | `moe_num_experts`, `moe_capacity` | `moe_k` |

Reload validation must check that the saved config and the actual loaded module
shapes agree.

## Calibration Data

The current `src/reap/data.py` is not MLX-safe yet:

- imports `torch`
- imports `vllm.TokensPrompt` at module import time
- emits PyTorch `BatchEncoding`
- pads batched samples with `attention_mask`

The first MLX path should use a minimal loader:

```text
dataset rows -> tokenizer -> list[int] -> mx.array([tokens])
```

Initial constraints:

- batch size `1`
- no padding
- no HF-style attention mask
- optional truncation to `max_length`
- simple sample cap such as `--max-samples`

Packed full-length streams are also acceptable because all tokens are real
tokens. Padded multi-sample batches are not part of v0.2.

Later, `data.py` can be refactored to support:

```text
return_tensors="np"
return_format="mlx"
lazy vllm import
optional explicit padding-aware MLX attention masks
```

## Save And Reload

Use MLX-LM utilities:

```python
from mlx_lm import load
from mlx_lm import utils

model, tokenizer, config = load(model_name, return_config=True)

utils.save(
    dst_path=output_dir,
    src_path_or_repo=model_name,
    model=model,
    tokenizer=tokenizer,
    config=updated_config,
)
```

Because the installed `utils.save(...)` donates/clears model parameters via
`save_model(..., donate_model=True)`, validation should be:

1. Save.
2. Reload from `output_dir`.
3. Run a short forward or generation smoke test.
4. Verify retained expert counts after reload.

## First Implementation Phases

### Phase 1: Pure MLX Units

Add:

```text
src/reap/mlx/__init__.py
src/reap/mlx/router.py
src/reap/mlx/accumulator.py
tests/test_mlx_router.py
tests/test_mlx_accumulator.py
```

Success criteria:

- importing `reap.mlx` does not import `torch` or `vllm`
- Qwen router adapter matches MLX-LM Qwen code
- Mixtral router adapter matches MLX-LM Mixtral code
- accumulator matches a hand-written NumPy reference

### Phase 2: Qwen Observation And Pruning

Add:

```text
src/reap/mlx/model_adapters.py
src/reap/mlx/observer.py
src/reap/mlx/prune.py
tests/test_mlx_qwen_observer.py
tests/test_mlx_qwen_prune.py
```

Success criteria:

- tiny Qwen3-MoE MLX model observes at least one MoE layer
- observer reports required pruning keys
- one expert can be pruned
- forward output shape is unchanged after pruning

### Phase 3: Save/Reload

Add:

```text
src/reap/mlx/save.py
src/reap/mlx/smoke_test.py
tests/test_mlx_save_reload.py
```

Success criteria:

- pruned tiny model saves
- reloaded model has retained expert count
- reloaded model runs a forward pass

### Phase 4: Minimal Real Pipeline

Add:

```text
src/reap/mlx/data.py
src/reap/mlx/cli.py
```

Success criteria:

- can run Qwen3-MoE MLX pruning on a small real calibration sample set
- output directory contains reloadable MLX-LM model
- no CUDA/vLLM/PyTorch import required for the MLX CLI

### Phase 5: Mixtral

Add Mixtral model adapter, pruning config updates, and tests.

### Phase 6: DeepSeek, GLM, ERNIE

Add one architecture at a time. Prefer calling native gate modules where routing
logic is complex.

## Test Matrix

Unit tests should not download large models.

Recommended tests:

- `test_mlx_import_does_not_import_torch_or_vllm`
- `test_qwen_router_adapter_matches_reference_code`
- `test_mixtral_router_adapter_matches_reference_code`
- `test_accumulator_counts_topk_routes`
- `test_accumulator_report_derives_ean_mean_and_reap`
- `test_qwen_observer_reports_pruning_keys`
- `test_prune_slices_switch_mlp_and_router`
- `test_prune_slices_quantized_switch_linear_metadata`
- `test_save_reload_tiny_qwen`

Slow/model-required tests can be separately marked:

- real `mlx-community/Qwen3-30B-A3B-*` calibration smoke
- real save/reload/generate smoke

## Risk Register

| Risk | Mitigation |
|---|---|
| Router semantics drift | Test adapters against MLX-LM source behavior; call native gates for complex routers. |
| Padded calibration mismatch | v0.2 only uses unpadded batch-size-1 or packed streams. |
| Quantized metadata corruption | Centralize first-dim slicing and test `weight`, `scales`, `biases`, `bias`. |
| Stale config after save | Update runtime args and config dict; reload every saved model in tests. |
| Over-refactoring | Keep MLX package independent; defer shared CLI/backend cleanup. |
| Memory graph growth | Use `mx.eval` boundaries and NumPy accumulators. |
| Saliency parity confusion | Default to actual MLX-LM routing; add explicit compatibility mode only later. |

## Deferred Work

- Full `--backend` integration in existing entrypoints
- Dependency split into `reap[torch]`, `reap[mlx]`, `reap[eval]`
- Refactor `data.py` into torch/np/mlx output formats
- Merging metrics on MLX
- All-expert activation mode
- vLLM-compatible or lm-eval-compatible MLX evaluation
- Hugging Face PyTorch to MLX conversion workflow
- Strict PyTorch saliency parity mode

## The First PR Should Not Do Too Much

The first implementation PR should stop at router and accumulator units:

```text
src/reap/mlx/__init__.py
src/reap/mlx/router.py
src/reap/mlx/accumulator.py
tests/test_mlx_router.py
tests/test_mlx_accumulator.py
```

That gives the collaboration a clean foundation and prevents unrelated debates
about CLI, datasets, save formats, and pruning mutation from blocking the core
algorithm.

## v0.2 Summary

The MLX version of REAP should be MLX-native:

- selected expert outputs
- exact MLX-LM routing
- NumPy accumulators
- explicit layer replay
- live module pruning
- save/reload validation

The first useful milestone is not "the repo has a backend abstraction." The
first useful milestone is "a tiny MLX Qwen3-MoE model can be observed, pruned,
saved, reloaded, and forwarded without importing CUDA tooling."
