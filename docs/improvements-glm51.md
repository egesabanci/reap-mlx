# REAP MLX — Verified Gap Analysis & Improvements

> **Scope**: Deep analysis of the REAP MLX codebase, cross-verified against both
> independent review and source code. Every finding was traced to specific lines
> and validated by reading the implementation.
> **Date**: 2026-06-01

---

## Table of Contents

1. [Methodology](#methodology)
2. [Confirmed Issues & Recommendations](#confirmed-issues--recommendations)
   - [2.1 Qwen3 Observer: Redundant Attention Mask Computation](#21-qwen3-observer-redundant-attention-mask-computation)
   - [2.2 Observer MoE Layer Code Duplication](#22-observer-moe-layer-code-duplication)
   - [2.3 Router Epsilon Asymmetry](#23-router-epsilon-asymmetry)
   - [2.4 Uniform Expert Count Constraint](#24-uniform-expert-count-constraint)
   - [2.5 Hardcoded Smoke Test Parameters](#25-hardcoded-smoke-test-parameters)
   - [2.6 Config Mutation Without Documentation](#26-config-mutation-without-documentation)
   - [2.7 Pairwise Expert Frequency Computed But Never Used for Pruning](#27-pairwise-expert-frequency-computed-but-never-used-for-pruning)
   - [2.8 Per-Layer mx.eval() Barrier Frequency](#28-per-layer-mxeval-barrier-frequency)
3. [New Findings (Not in Prior Analyses)](#new-findings-not-in-prior-analyses)
   - [3.1 LFM2 Expert Bias Changes REAP Metric Semantics](#31-lfm2-expert-bias-changes-reap-metric-semantics)
   - [3.2 Qwen3 argpartition Does Not Guarantee Sorted Output](#32-qwen3-argpartition-does-not-guarantee-sorted-output)
   - [3.3 _keep_list Creates Python Lists for MLX Indexing](#33-_keep_list-creates-python-lists-for-mlx-indexing)
   - [3.4 extract_text() Produces Garbage for Multimodal Records](#34-extract_text-produces-garbage-for-multimodal-records)
   - [3.5 generation_smoke() Can Crash on Broken Chat Templates](#35-generation_smoke-can-crash-on-broken-chat-templates)
4. [Partially Confirmed / Downgraded Issues](#partially-confirmed--downgraded-issues)
   - [4.1 MLX .astype(float32) Conversion Overhead — Downgraded](#41-mlx-astypefloat32-conversion-overhead--downgraded)
   - [4.2 PruningState.report() Array Copying — Non-Issue at Current Scale](#42-pruningstatereport-array-copying--non-issue-at-current-scale)
5. [Disproved Issues](#disproved-issues)
6. [Confirmed Non-Issues (Prior Analysis Verified Correct)](#confirmed-non-issues-prior-analysis-verified-correct)
7. [Prioritized Recommendations](#prioritized-recommendations)

---

## Methodology

This analysis was produced by reading every source file, test file, and
configuration file in the repository, then cross-referencing findings against
the existing `docs/improvements-dsv4.md` document. Each claim was traced to
specific line numbers and verified against the actual implementation. Findings
are classified as:

- ✅ **Confirmed** — verified directly in code; issue is real
- ⚠️ **Partially confirmed** — issue is real but severity or nuance differs
- ❌ **Not confirmed** — claim is incorrect or misleading after verification
- 🆕 **New finding** — not identified in either prior analysis

---

## Confirmed Issues & Recommendations

### 2.1 Qwen3 Observer: Redundant Attention Mask Computation

**Location**: `src/reap/observer.py`, lines 87–96

**Status**: ✅ Confirmed

**Current code:**

```python
for sequence in calibration_sequences:
    tokens = _batch_tokens(mx, sequence)
    h = embed_tokens(tokens)

    for layer_idx, layer in enumerate(layers):
        mask = _attention_mask(                    # ← Inside layer loop
            h,
            sequence_length=tokens.shape[-1],
            mask_fn=mask_fn,
        )
        h = _run_attention(layer, h, mask)
        ...
```

**Problem**: When `mask_fn is None` (the default), the causal attention mask
depends only on `h.shape[-1]` (sequence length), which does not change across
layers. The mask is recomputed `num_layers × num_sequences` times instead of
`num_sequences` times.

The LFM2 path already handles this correctly — masks are computed once per
sequence outside the layer loop (lines 142–146).

**Nuance**: When `mask_fn is not None`, the mask may depend on hidden state
values and MUST stay inside the layer loop. The fix must only hoist the
default causal mask path.

**Recommended fix** (5 lines):

```python
for sequence in calibration_sequences:
    tokens = _batch_tokens(mx, sequence)
    h = embed_tokens(tokens)
    # Compute default causal mask once per sequence (not per layer).
    default_mask = _attention_mask(
        h, sequence_length=tokens.shape[-1], mask_fn=mask_fn
    ) if mask_fn is None else None

    for layer_idx, layer in enumerate(layers):
        mask = mask_fn(h, cache=None) if mask_fn is not None else default_mask
        h = _run_attention(layer, h, mask)
        ...
```

**Impact**: For a 32-layer model with 8 calibration sequences at 1024 tokens,
this eliminates ~256 redundant mask allocations (~4MB each in float32). The
primary benefit is reduced MLX graph node overhead, not memory savings.

---

### 2.2 Observer MoE Layer Code Duplication

**Location**: `src/reap/observer.py`, lines 323–346 vs 349–372

**Status**: ✅ Confirmed

`_observe_moe_layer` and `_observe_lfm2_moe_layer` are byte-for-byte identical
except for the router class (`Qwen3MoeRouter` vs `Lfm2MoeRouter`). Both:

1. Call `_require_mlx_core()`
2. Get the MoE module via `adapter.get_moe(layer)`
3. Create a router and route `moe_input`
4. Call `switch_mlp(moe_input, routing.indices)`
5. Accumulate `indices`, `scores`, `selected_outputs` into `PruningState`
6. Compute `moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)`
7. Add shared expert output if present
8. Return `moe_out`

**Recommended fix** (~20 lines):

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

Then callers become:
```python
# Qwen3:
h = h + _observe_moe_layer_impl(layer, moe_input, accumulators[layer_idx],
    adapter=adapter, config=config, router_cls=Qwen3MoeRouter)

# LFM2:
h = h_mid + _observe_moe_layer_impl(layer, ffn_input, accumulators[layer_idx],
    adapter=adapter, config=config, router_cls=Lfm2MoeRouter)
```

**Impact**: Eliminates ~25 lines of duplication. Any future bug fix or metric
addition only needs to be made once. Adding a third model family only requires
a new router class, not a new observe function.

---

### 2.3 Router Epsilon Asymmetry

**Location**: `src/reap/router.py`

**Status**: ✅ Confirmed

| Router | Line | Normalization | Epsilon |
|---|---|---|---|
| `Qwen3MoeRouter` | 127 | `flat_scores / flat_scores.sum(...)` | None |
| `Lfm2MoeRouter` | 215 | `scores / (mx.sum(...) + 1e-20)` | `1e-20` |

Neither router can produce a division-by-zero because softmax always outputs
positive values. The epsilon is purely defensive.

However, the codebase uses three different epsilon values for the same concept:
- `1e-20` in LFM2 norm_topk_prob normalization
- Nothing in Qwen3 norm_topk_prob normalization
- `_FLOAT_EPS` (~2.2e-16) in `metrics.py` for zero-frequency denominators

**Recommended fix** (1 line + documentation):

Either add epsilon to Qwen3 for consistency:
```python
# Line 127, Qwen3MoeRouter:
flat_scores = flat_scores / (flat_scores.sum(axis=-1, keepdims=True) + 1e-20)
```

Or remove from LFM2 if confident (not recommended — defensive coding is good).

Define a module-level constant if unification is desired:
```python
_NORM_EPSILON = 1e-20
```

**Impact**: Cosmetic. No correctness impact. But the asymmetry is confusing
during code review and suggests one path might be wrong when neither is.

---

### 2.4 Uniform Expert Count Constraint

**Location**: `src/reap/prune.py`, lines ~70–78

**Status**: ✅ Confirmed

```python
if config_num_experts != retained_count:
    raise ValueError(
        "MLX MoE config update requires all pruned layers to retain "
        f"the same expert count."
    )
```

This enforces that all MoE layers prune to the same number of experts. For
currently supported models (LFM2.5, Qwen3) which have uniform expert counts,
this is fine. But it blocks:

1. Models where different layers have different numbers of experts
2. Per-layer compression ratios (e.g., prune earlier layers more aggressively)
3. Any iterative multi-pass pruning strategy

**Recommended fix**: Documentation-only for now. Add a note to the README's
Supported Models section and a comment in `prune_experts()`. If non-uniform
pruning becomes needed, the fix would involve per-layer config entries
(`config["num_experts_per_layer"]`) instead of a single global `num_experts`.

**Impact**: Low for current use cases. Would become a blocker for heterogeneous
architectures.

---

### 2.5 Hardcoded Smoke Test Parameters

**Location**: `src/reap/save.py`, `generation_smoke()`:

```python
prompt: str = "What is your name?",
max_tokens: int = 16,
```

**Status**: ✅ Confirmed

These are not exposed via CLI flags. The `generation_smoke` function already
accepts `prompt` and `max_tokens` as keyword arguments, so only the CLI layer
needs changes.

**Recommended fix** (~12 lines in `entrypoint.py`):

```python
parser.add_argument("--smoke-prompt", default="What is your name?")
parser.add_argument("--smoke-max-tokens", type=int, default=16)
```

Pass them through `main()` → `save_pruned_model()` → `generation_smoke()`.

**Impact**: User configurability. Allows domain-specific smoke tests (e.g.,
code generation models could use a coding prompt).

---

### 2.6 Config Mutation Without Documentation

**Location**: `src/reap/prune.py`, `prune_experts()`

**Status**: ✅ Confirmed

`prune_experts()` mutates `config` in-place via `update_qwen3_moe_config()` /
`update_lfm2_moe_config()`. The caller in `entrypoint.py` correctly snapshots
with `config_before_prune = dict(config)` before calling, but this mutation
behavior is undocumented.

**Recommended fix** (2 lines):

```python
def prune_experts(model, config, observer_data, prune_method, compression_ratio, *, adapter=None):
    """Prune adapter-discovered MLX MoE experts in place.

    Note: This function mutates `config` in-place. The caller should
    copy the config before calling if the original values are needed
    after pruning.

    Returns a mapping from layer index to ascending retained expert indices.
    """
```

**Impact**: Prevents future misuse. Low effort.

---

### 2.7 Pairwise Expert Frequency Computed But Never Used for Pruning

**Location**: `src/reap/metrics.py`, line 118

```python
self.pairwise_expert_frequency += (
    batch_frequency[:, None] + batch_frequency[None, :]
)
```

**Status**: ✅ Confirmed

This runs on every `accumulate()` call. It creates an `O(num_experts²)` matrix
every time. Grepping the entire codebase shows:

- **`metrics.py`**: Declared, populated, reported
- **`test_mlx_metrics.py`**: Tested
- **`test_mlx_observer.py`**: Checked in expected keys
- **`validation_metrics.py`**: NOT consumed (zero hits)
- **`prune.py`**: NOT used by any pruning method

The value `batch_frequency[:, None] + batch_frequency[None, :]` is also
semantically questionable: it's the **sum** of per-expert frequencies
(`freq[i] + freq[j]`), not a true pairwise co-occurrence matrix (`freq[i, j]`
meaning "experts i and j were both selected for the same token"). The actual
pairwise co-occurrence would require knowing which experts were selected
simultaneously per token, which the current accumulator doesn't track.

**Recommended fix** (~10 lines):

Add an optional parameter to `PruningState.initialize()`:

```python
@classmethod
def initialize(cls, num_experts: int, *, track_pairwise: bool = False) -> "PruningState":
    ...
    pairwise_expert_frequency=np.zeros((num_experts, num_experts), dtype=np.int64) if track_pairwise else None,
    ...
```

And in `accumulate()`, skip the pairwise update when `self.pairwise_expert_frequency is None`.

**Impact**: Eliminates O(num_experts²) per-token overhead on the hot path during
observation. For 32 experts, this is 1024 int64 additions per token. For
128+ experts, it would be 16K+ additions per token.

---

### 2.8 Per-Layer mx.eval() Barrier Frequency

**Location**: `src/reap/observer.py`, lines 112 and 167

```python
eval_fn(h)  # Called after every layer
```

**Status**: ✅ Confirmed as correct behavior

Calling `mx.eval(h)` after every layer is the safest memory strategy for
Apple Silicon's unified memory. Without this, MLX's lazy computation graph
accumulates across all layers and sequences, causing memory blow-up.

However, for users with abundant memory (e.g., M3 Ultra with 192GB), evaluating
every N layers instead of every layer would reduce GPU→CPU synchronization
barrriers and improve throughput.

**Trade-off**:

| Eval Frequency | Peak Memory | Throughput |
|---|---|---|
| Every layer (current) | Lowest | Lowest |
| Every 2 layers | ~2× intermediate activations | Moderate gain |
| Every 4 layers | ~4× intermediate activations | Higher gain |
| Per sequence only | Highest | Highest |

**Recommended fix** (~15 lines): Add `--eval-frequency` CLI flag (default `1`),
pass through to observer functions, and change `eval_fn(h)` to only evaluate
every N layers.

**Impact**: Power users with large-memory machines can tune for throughput.
Default behavior remains safest.

---

## New Findings (Not in Prior Analyses)

### 3.1 LFM2 Expert Bias Changes REAP Metric Semantics

**Location**: `src/reap/router.py`, lines 211–213

```python
gates = mx.softmax(logits, axis=-1)
if self.use_expert_bias:
    gates = gates + self.expert_bias
```

**Status**: 🆕 New finding

When `use_expert_bias=True` (LFM2 models), the router scores stored in
`RouterResult.scores` are no longer probabilities — they can exceed 1.0 or be
negative. These biased scores then flow into the observer's `PruningState`:

```python
state.accumulate(
    indices=routing.indices,
    scores=routing.scores.astype(mx.float32),
    selected_outputs=selected_outputs.astype(mx.float32),
)
```

This means `weighted_ean_sum` and `weighted_expert_frequency_sum` reflect the
biased scores, not pure router probabilities. The REAP metric
(`weighted_ean_sum / count_denominator`) then incorporates the expert bias in
its weighting.

This is **correct behavior** for matching the actual model output — the MoE
output computation uses these same biased scores:
```python
moe_out = (selected_outputs * routing.scores[..., None]).sum(axis=-2)
```

However, it's an important documentation gap: the REAP paper's formula for
weighted activation norm assumes pure softmax probabilities as weights. With
expert bias, the weighting includes the bias, which changes the semantic
interpretation of the REAP score.

**Recommended action**: Add a note in the README's Pruning Methods section and
in `router.py` docstrings explaining that `use_expert_bias=True` causes REAP
scores to incorporate expert bias, which may differ from the original paper's
formulation.

**Impact**: Documentation correctness. Not a bug, but a semantic nuance that
users and researchers should be aware of.

---

### 3.2 Qwen3 argpartition Does Not Guarantee Sorted Output

**Location**: `src/reap/router.py`, lines 122–123

```python
flat_indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
flat_scores = mx.take_along_axis(gates, flat_indices, axis=-1)
```

**Status**: 🆕 New finding (not a bug, just an invariant to document)

`mx.argpartition` guarantees that the top-k elements are on the right side of
the partition, but does NOT guarantee they are sorted within that partition.
The returned `indices` and `scores` in `RouterResult` are in partition order,
not descending probability order.

This is **correct behavior** — the observer and pruner only need the selected
expert indices and their corresponding scores, not sorted scores. The
accumulation in `PruningState.accumulate()` uses `np.add.at()` and
`np.maximum.at()` which are order-independent.

However, anyone consuming `RouterResult.scores` and assuming
`scores[..., 0]` is the best expert would be wrong. The scores are the actual
gate values of the selected experts, just not in any particular order.

**Impact**: None for current code. Worth a brief comment in RouterResult's
docstring.

---

### 3.3 _keep_list Creates Python Lists for MLX Indexing

**Location**: `src/reap/prune.py`

```python
def _keep_list(keep_indices: np.ndarray) -> list[int]:
    return [int(idx) for idx in np.asarray(keep_indices, dtype=np.int64).tolist()]
```

Used in `slice_first_dim`:
```python
_set_module_value(module, field_name, value[keep_list])
```

**Status**: 🆕 New finding (not a bug, a design boundary issue)

When `value` is an MLX array, `value[keep_list]` triggers MLX's advanced
indexing with a Python list. This works correctly but:

1. Creates a Python list of ints from a numpy array, then passes it to MLX
2. MLX must convert the Python list back to an internal index array
3. The numpy→Python→MLX round-trip adds overhead for large expert counts

For current models (32–64 experts), this is trivial. For 256+ experts or
iterative pruning experiments, using MLX-native indexing (`mx.array(keep_indices)`)
would be cleaner.

**Impact**: Negligible for current scale. Document as a future optimization
path.

---

### 3.4 extract_text() Produces Garbage for Multimodal Records

**Location**: `src/reap/data.py`, `_normalize_content()`

```python
if isinstance(item, Mapping):
    if item.get("type") == "text" and isinstance(item.get("text"), str):
        parts.append(item["text"])
    else:
        parts.append(json.dumps(dict(item), sort_keys=True))
```

**Status**: 🆕 New finding

When `item` is a mapping that isn't a text block (e.g., `{"type": "image",
"url": "https://..."}`), the code JSON-serializes it as a string. This string
then gets tokenized and fed into calibration, producing nonsensical calibration
text like `{"type": "image", "url": "https://..."}`.

The code should skip non-text content blocks instead of JSON-serializing them.

**Recommended fix** (~5 lines):

```python
if isinstance(item, Mapping):
    if item.get("type") == "text" and isinstance(item.get("text"), str):
        parts.append(item["text"])
    # Skip non-text content blocks (images, audio, etc.)
    elif item.get("type") not in (None, "text"):
        continue
    else:
        parts.append(json.dumps(dict(item), sort_keys=True))
```

**Impact**: Prevents noisy calibration data for multimodal datasets. Low
severity — most calibration datasets are text-only.

---

### 3.5 generation_smoke() Can Crash on Broken Chat Templates

**Location**: `src/reap/save.py`, lines ~135–142

```python
if getattr(tokenizer, "chat_template", None) and hasattr(
    tokenizer, "apply_chat_template"
):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
```

**Status**: 🆕 New finding

If a tokenizer has a `chat_template` attribute and `apply_chat_template`
method, but the template itself is broken (e.g., references undefined variables,
has Jinja2 syntax errors), the `apply_chat_template` call will raise an
exception that crashes the entire save/reload validation pipeline.

Since `generation_smoke` is called inside the `save_pruned_model` pipeline,
this exception would prevent the metrics file from being written, losing all
telemetry for the run.

**Recommended fix** (~5 lines):

```python
try:
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
except Exception:
    logger.debug("Chat template application failed, using raw prompt")
```

**Impact**: Prevents a broken chat template from crashing the validation
pipeline. The raw prompt is a reasonable fallback for smoke testing.

---

## Partially Confirmed / Downgraded Issues

### 4.1 MLX .astype(float32) Conversion Overhead — Downgraded

**Location**: `src/reap/observer.py`, lines 341–343 and 366–368

```python
scores=routing.scores.astype(mx.float32),
selected_outputs=selected_outputs.astype(mx.float32),
```

The prior analysis (`improvements-dsv4.md` Section 4.7) suggested these might
be unnecessary overhead. However:

- For 4-bit quantized models (the default `LFM2.5-8B-A1B-MLX-4bit`), model
  outputs may be in `bfloat16` or `float16`. The `.astype(mx.float32)` ensures
  consistent dtype for the accumulation operations.
- For `float32` models, MLX likely optimizes identity `.astype()` calls away.
- Removing these casts could silently lose precision on mixed-precision models.

**Verdict**: These are **defensive casts**, not pure overhead. Low priority.
Removing them would require verifying that all MLX-LM model dtypes produce
float32 outputs, which is not guaranteed.

---

### 4.2 PruningState.report() Array Copying — Non-Issue at Current Scale

**Location**: `src/reap/metrics.py`

The prior analysis (`improvements-dsv4.md` Section 4.8) correctly identifies
that `report()` copies all arrays. For current expert counts (32–64), these
copies are <1KB each. They're immediately consumed by JSON serialization and
garbage collected.

**Verdict**: Non-actionable at current scale. Would only matter if
`pairwise_expert_frequency` (Section 2.7) were kept for very large expert counts.

---

## Disproved Issues

### ❌ Qwen3 Router Division by Zero Risk

**Claim**: The Qwen3 router's normalization
`flat_scores / flat_scores.sum(...)` could divide by zero.

**Disproven**: `mx.softmax(logits, axis=-1, precise=True)` guarantees all
output values are positive (since `exp(x) > 0` for all finite x). The sum of
positive values is always > 0. Division by zero is impossible.

### ❌ moe_input Not Evaluated Before Dense MLP (My Initial Analysis)

**Claim**: The intermediate `moe_input` (layernorm output) is not evaluated
independently, which could cause memory accumulation.

**Disproven**: `moe_input` is part of the same MLX computation graph as `h`.
The `eval_fn(h)` call at the end of each layer forces evaluation of the entire
graph, including all intermediates. The per-layer eval is the correct
granularity — there's no need for separate intermediate evaluations.

---

## Confirmed Non-Issues (Prior Analysis Verified Correct)

The following items from `improvements-dsv4.md` were verified as correctly
identified non-issues:

| Claim | Verification |
|---|---|
| `ru_maxrss` is always bytes on macOS | ✅ Correct — macOS-only codebase |
| `_batch_tokens()` with None input_ids | ✅ Correct — all sequences come from `load_calibration_sequences()` which always includes `"input_ids"` |
| `compute_keep_indices()` with all-NaN scores | ✅ Correct — `PruningState` accumulator always produces finite scores |
| Config mutation correctness in entrypoint | ✅ Correct — `dict(config)` snapshot is shallow but sufficient |
| Import-safety enforcement | ✅ Exemplary — subprocess-based tests with `sys.meta_path` blockers |
| Adapter pattern extensibility | ✅ Good design — ~5 methods per adapter |
| Injectable dependencies | ✅ Excellent — every pipeline function accepts callbacks |
| Save-reload validation thoroughness | ✅ Comprehensive — 8 validation checks |

---

## Prioritized Recommendations

### P1 — Immediate (Low Effort, Clear Benefit)

| # | Recommendation | File(s) | Effort | Impact |
|---|---|---|---|---|
| 1 | **Hoist Qwen3 attention mask** out of layer loop (with `mask_fn` guard) | `observer.py:87-96` | 5 lines | Eliminates ~250 redundant mask ops per run |
| 2 | **Unify MoE layer observer** into single `router_cls`-parameterized function | `observer.py:323-372` | ~20 lines | Eliminates code duplication, reduces bug surface |
| 3 | **Make pairwise_expert_frequency conditional** (skip unless requested) | `metrics.py:118` | ~10 lines | Hot-path O(experts²) elimination |

### P2 — Short-Term (Moderate Effort, User-Facing)

| # | Recommendation | File(s) | Effort | Impact |
|---|---|---|---|---|
| 4 | **Expose `--smoke-prompt` and `--smoke-max-tokens`** flags | `entrypoint.py`, `save.py` | ~12 lines | User configurability |
| 5 | **Align router epsilon** (add `1e-20` to Qwen3 or define shared constant) | `router.py:127` | 1 line + docs | Code clarity |
| 6 | **Document config mutation** in `prune_experts()` docstring | `prune.py:30` | 2 lines | Prevents misuse |
| 7 | **Document expert bias effect on REAP semantics** | `router.py` or README | 3-5 lines | Research correctness |
| 8 | **Fix extract_text() for multimodal content** — skip non-text blocks | `data.py` | ~5 lines | Prevents garbage calibration |
| 9 | **Add try/except around apply_chat_template** in smoke test | `save.py` | ~5 lines | Robustness |

### P3 — Medium-Term (Performance or Architecture)

| # | Recommendation | File(s) | Effort | Impact |
|---|---|---|---|---|
| 10 | **Add `--eval-frequency` flag** for tunable memory/throughput | `observer.py`, `entrypoint.py` | ~15 lines | Power-user throughput tuning |
| 11 | **Document uniform expert count constraint** in README | `README.md` | 1 line | Clarity |
| 12 | **Document argpartition's unsorted output** in RouterResult docstring | `router.py` | 2 lines | API clarity |

### P4 — Future (Requires Experimentation or Significant Refactoring)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| 13 | **Investigate `mx.compile()`** for observer forward pass | Experiment | Potential 20-50% throughput |
| 14 | **Extract shared router base class** | ~40 lines | Maintainability |
| 15 | **Consider padded batching** for large calibration sets | ~50 lines | 2-4× throughput for many sequences |

---

*Analysis produced by deep source-level verification of `reap-mlx` at commit
snapshot 2026-06-01. Every finding was traced to specific lines and validated
against the implementation. Cross-referenced against `docs/improvements-dsv4.md`.*