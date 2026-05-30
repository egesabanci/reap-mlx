# REAP MLX Backend — Cooperative Implementation Plan

> **Status:** ACTIVE · **Last updated:** 2026-05-30  
> **Co-op file for:** DSv4, GPT-5.5, and any future agents working on this codebase.  
> **Protocol:** Append-only. Do not delete or reorder existing entries. Add new entries at the bottom of their phase. Update status inline. Add dated changelog entries.

---

## User Objective Alignment - 2026-05-31

This plan serves two user objectives:

1. Make Cerebras' REAP implementation work on MLX, within MLX's real
   constraints, without destabilizing the existing codebase.
2. Make the MLX implementation work with any compatible MoE weights whose
   routing and expert layout can be represented by the MLX adapter contract.

The original PyTorch/CUDA REAP implementation remains the production and
official experimentation workflow. The MLX backend is a parallel Apple Silicon
experimentation path so researchers with Mac hardware can iterate on REAP
applications without CUDA-compatible devices.

Planning consequences:

- Do not replace, weaken, or refactor the CUDA/PyTorch path unless a change is
  explicitly required for import isolation.
- Treat Qwen3-MoE as the bootstrap/reference adapter, not the final target or
  architecture boundary.
- Put model-specific routing, expert layout, shared-expert handling, tensor
  names, and config-update behavior behind explicit MLX adapter contracts.
- Prefer synthetic adapter/unit tests for normal CI and keep large real-model
  checks as manual or marked slow tests.
- Preserve REAP pruning semantics and observer-data compatibility while adapting
  execution to MLX constraints such as lazy evaluation, no PyTorch hooks,
  MLX-LM module layouts, and save/reload behavior.
- Every active issue should stay aligned with the adapter-driven arbitrary-MoE
  goal and the CUDA-as-production / MLX-as-Apple-Silicon-experimentation split.

---

## Reference: Architecture Decisions (Read-Only)

These decisions are settled. Do not reopen without explicit discussion.

| # | Decision | Rationale |
|---|---|---|
| D1 | **Separate MLX path** — `src/reap/backends/mlx/`, NOT a backend abstraction layer | MLX execution model (lazy, no hooks, stacked tensors) is too different for thin abstraction |
| D2 | **Separate entrypoint** — `python -m reap.backends.mlx.entrypoint`, NOT routed through `src/reap/main.py` | `main.py` imports torch, vLLM, CUDA utils |
| D3 | **Selected-expert-only observation** — use `switch_mlp(x, indices)` returning `[B, S, K, H]` | All pruning metrics need only routed expert outputs; 10-20× memory saving |
| D4 | **CPU-side NumPy accumulators** — `np.bincount`, `np.add.at`, `np.linalg.norm` | `mx.bincount` and `mx.scatter_add` don't exist in MLX 0.31 |
| D5 | **Architecture-specific router adapters** — not generic `topk(softmax(logits))` | Qwen, Mixtral, DeepSeek, GLM, ERNIE have fundamentally different routing |
| D6 | **Prune live MLX-LM `nn.Module` objects** — slice `.weight` on dim 0 | MLX-LM nn.Module is mutable; no weight dict surgery needed |
| D7 | **Unpadded batch_size=1** for calibration | MLX-LM forwards generate causal masks internally; don't accept HF attention_mask |
| D8 | **Smoke test on reloaded model**, not in-memory | `mlx_lm.utils.save()` uses `donate_model=True` internally |
| D9 | **Use `mx.argpartition` + `mx.take_along_axis`** for top-k, NOT `mx.topk` | `mx.topk` returns values only (no indices) in MLX 0.31 |
| D10 | **`mx.eval(array)` on specific arrays**, not bare `mx.eval()` | Controls graph retention precisely |

---

## Reference: MLX API Facts (Read-Only)

Installed: `mlx==0.31.2`, `mlx_lm==0.31.3`

| API | Status | Usage |
|---|---|---|
| `mx.argpartition(x, kth, axis)` | ✅ | Top-k selection (MLX-LM's actual method) |
| `mx.take_along_axis(x, indices, axis)` | ✅ | Gather (torch.gather equivalent) |
| `mx.linalg.norm(x, axis)` | ✅ | L2 norm |
| `mx.softmax(x, axis, precise=True)` | ✅ | Softmax; `precise=True` matches MLX-LM |
| `mx.eval(x)` | ✅ | Force evaluation of specific arrays |
| `mx.get_active_memory()` | ✅ | Memory monitoring (top-level, not `.metal.*`) |
| `mx.get_peak_memory()` | ✅ | Peak memory |
| `mx.get_cache_memory()` | ✅ | Cache memory |
| `mx.set_cache_limit(bytes)` | ✅ | Cache cap |
| `mx.set_memory_limit(bytes)` | ✅ | Memory cap |
| `mx.bincount` | ❌ **MISSING** | Use `np.bincount` on CPU |
| `mx.scatter_add` | ❌ **MISSING** | Use `np.add.at` on CPU |
| `mx.topk` indices | ❌ **VALUES ONLY** | Use `mx.argpartition` + `mx.take_along_axis` |
| `mx.allclose` | ❌ **MISSING** | Use `mx.max(mx.abs(x - y)) < tol` |

### MLX-LM API

```python
model = mlx_lm.load(model_name)[0]         # returns nn.Module
model.parameters()                          # iterate params
model.children()                            # iterate submodules
model.named_modules()                       # named submodules
model.update(weights_dict)                  # bulk update
model.update_modules(sgd)                   # optimizer update
model.save_weights(path)                    # save weights
mlx_lm.utils.save(dst_path=..., src_path_or_repo=..., model=..., tokenizer=..., config=...)  # full save (destroys model!)
mlx_lm.generate(model, tokenizer, prompt=..., max_tokens=...)  # generation
```

---

## Phase 0: Skeleton & Import Safety

### ☐ TODO P0-001: Create package skeleton

- **Owner:** DSv4
- **Aspect:** infrastructure
- **Depends on:** none
- **Effort:** 15 min

**Task:** Create `src/reap/backends/__init__.py` (empty) and `src/reap/backends/mlx/__init__.py` (empty). Verify `python -c "import reap.backends.mlx"` does NOT import torch or vLLM.

**Acceptance:**
- [ ] `src/reap/backends/__init__.py` exists (empty file)
- [ ] `src/reap/backends/mlx/__init__.py` exists (empty file)
- [ ] `python -c "import reap.backends.mlx; print('ok')"` prints "ok" without importing torch
- [ ] `python -c "import reap.backends.mlx; import sys; assert 'torch' not in sys.modules"` passes

**Reasoning:** The entire MLX path must be importable on a machine without PyTorch/CUDA. This gates all subsequent work.

---

### ☐ TODO P0-002: Add import guard test

- **Owner:** DSv4
- **Aspect:** testing
- **Depends on:** P0-001
- **Effort:** 15 min

**Task:** Create `tests/test_mlx_no_torch_import.py`. The test should:
1. Subprocess `python -c "from reap.backends.mlx import *"` in a clean environment
2. Assert exit code 0
3. Assert 'torch' not in sys.modules
4. Skip gracefully with `@pytest.mark.skipif` if mlx is not installed

**Acceptance:**
- [ ] Test file created
- [ ] `pytest tests/test_mlx_no_torch_import.py -v` passes when mlx is installed
- [ ] Test skips cleanly when mlx is not installed (don't fail)

**Reasoning:** This test is the single most important portability boundary. If anything in the MLX backend transitively imports torch, this test catches it immediately.

---

### ☐ TODO P0-003: Environment verification

- **Owner:** GPT-5.5
- **Aspect:** infrastructure
- **Depends on:** P0-001
- **Effort:** 10 min

**Task:** Verify the MLX environment on the development machine. Run a script that prints:
- `mlx.__version__`
- `mlx_lm.__version__`
- `mx.metal.is_available()` or equivalent
- Available memory: `mx.get_active_memory()`, `mx.get_cache_memory()`
- Test that `mlx_lm.load("mlx-community/Qwen3-30B-A3B-4bit-DWQ")` succeeds (downloads ~17GB)

**Acceptance:**
- [ ] mlx version printed (expected >= 0.31.0)
- [ ] mlx_lm version printed (expected >= 0.31.0)
- [ ] Memory stats reported
- [ ] Model loading test passes (may take minutes for download)

**Reasoning:** We need to confirm the target environment works before building against it. The 4-bit model is our primary observation target.

---

## Phase 1: Router + Metrics (Core Algorithms)

### ☐ TODO P1-001: Implement RouterResult dataclass

- **Owner:** GPT-5.5
- **Aspect:** data model
- **Depends on:** P0-001
- **Effort:** 15 min

**Task:** Create `src/reap/backends/mlx/router.py` with `RouterResult` dataclass:
```python
@dataclass
class RouterResult:
    indices: mx.array        # [batch, seq, top_k]
    scores: mx.array         # [batch, seq, top_k]
    logits: mx.array | None  # [batch, seq, num_experts] or None
    score_mode: str          # "actual" | "compat_softmax"
```

**Acceptance:**
- [ ] File created at `src/reap/backends/mlx/router.py`
- [ ] `RouterResult` is importable
- [ ] Fields have correct type annotations
- [ ] `score_mode` is a string enum-like field

**Reasoning:** This is the shared data contract between router adapters (P1-002) and the observer/accumulator (P1-004, P2-003). Architecture-neutral so new architectures only need to implement their adapter class.

---

### ☐ TODO P1-002: Implement Qwen3-MoE router adapter

- **Owner:** GPT-5.5
- **Aspect:** model-specific
- **Depends on:** P1-001
- **Effort:** 45 min

**Task:** Add `Qwen3MoeRouter` class to `src/reap/backends/mlx/router.py`:

```python
class Qwen3MoeRouter:
    def __init__(self, mlp_layer, config: dict):
        self.gate = mlp_layer.gate
        self.top_k = config.get('num_experts_per_tok', config.get('top_k', 8))
        self.norm_topk_prob = config.get('norm_topk_prob', False)
    
    def __call__(self, x: mx.array) -> RouterResult:
        # Match MLX-LM's actual Qwen3MoE routing exactly
        logits = self.gate(x.reshape(-1, x.shape[-1]))
        gates = mx.softmax(logits, axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        indices = indices.reshape(*x.shape[:-1], self.top_k)
        scores = scores.reshape(*x.shape[:-1], self.top_k)
        return RouterResult(indices=indices, scores=scores, logits=None, score_mode="actual")
```

**Acceptance:**
- [ ] Class implemented and importable
- [ ] `__call__` accepts `mx.array` of shape `[batch, seq, hidden]` or `[tokens, hidden]`
- [ ] Returns `RouterResult` with indices shape `[batch, seq, top_k]` and scores shape `[batch, seq, top_k]`
- [ ] `norm_topk_prob=True` normalizes scores to sum to 1

**Reasoning:** Qwen3-MoE is the primary target. This adapter MUST match MLX-LM's actual routing behavior (full softmax → argpartition → take → optional renormalize). The torch backend uses `torch.topk(torch.softmax(logits))` which is equivalent — we use MLX-LM's preferred API.

---

### ☐ TODO P1-003: Implement PruningState accumulator

- **Owner:** DSv4
- **Aspect:** algorithm
- **Depends on:** P1-001
- **Effort:** 60 min

**Task:** Create `src/reap/backends/mlx/metrics.py` with `PruningState` dataclass:

```python
@dataclass
class PruningState:
    num_experts: int
    total_tokens: int = 0
    expert_frequency: np.ndarray = field(default=None)
    ean_sum: np.ndarray = field(default=None)
    weighted_ean_sum: np.ndarray = field(default=None)
    weighted_expert_frequency_sum: np.ndarray = field(default=None)
    max_activations: np.ndarray = field(default=None)
    
    @classmethod
    def initialize(cls, num_experts: int) -> 'PruningState':
        ...
    
    def accumulate(self, routing: RouterResult, selected_outputs: mx.array) -> None:
        """selected_outputs: [1, seq_len, top_k, hidden]"""
        # Transfer to numpy, flatten, scatter-accumulate
    
    def report(self) -> dict:
        """Return observer_data-compatible dict with derived ean_mean and reap."""
```

**Acceptance:**
- [ ] `initialize()` returns zeros of correct shapes and dtypes (int64 for freq, float64 for sums, float32 for max)
- [ ] `accumulate()` correctly adds batch stats to running state
- [ ] `accumulate()` handles expert indices 0 to num_experts-1
- [ ] `accumulate()` is idempotent for empty flat_indices
- [ ] `report()` returns dict with keys: `total_tokens`, `expert_frequency`, `ean_sum`, `ean_mean`, `weighted_ean_sum`, `weighted_expert_frequency_sum`, `reap`, `max_activations`
- [ ] `report()` uses `eps=1e-10` divisor to avoid division by zero for unused experts

**Reasoning:** This is the core REAP algorithm in NumPy. `np.add.at` replaces torch's `scatter_add_`. `np.bincount` replaces `torch.bincount`. Running means are derived at report time (simple weighted average, stable enough for millions of tokens). No Welford/Kahan needed.

---

### ☐ TODO P1-004: Unit tests for router adapter

- **Owner:** GPT-5.5
- **Aspect:** testing
- **Depends on:** P1-002
- **Effort:** 45 min

**Task:** Create `tests/test_mlx_router.py`:

```python
class TestQwen3MoeRouter:
    def test_output_shapes(self): ...
    def test_indices_in_valid_range(self): ...
    def test_scores_sum_to_one(self): ...
    def test_norm_topk_normalization(self): ...
    def test_matches_model_native_routing(self):
        # Build tiny Qwen3-MoE block, compare adapter output vs model's own routing
```

**Acceptance:**
- [ ] All tests pass
- [ ] `test_matches_model_native_routing` constructs a synthetic model and validates adapter parity
- [ ] Edge case: 1 token, single-expert routing still works

**Reasoning:** The router adapter is the correctness-critical component. Any divergence here means wrong REAP scores. The `test_matches_model_native_routing` test is the strongest validation — it runs actual MLX-LM model code and compares.

---

### ☐ TODO P1-005: Unit tests for metric accumulation

- **Owner:** DSv4
- **Aspect:** testing
- **Depends on:** P1-003
- **Effort:** 60 min

**Task:** Create `tests/test_mlx_metrics.py`:

```python
class TestPruningState:
    def test_initialize_zeros(self): ...
    def test_single_expert_single_token(self): ...
    def test_multiple_experts_multiple_tokens(self): ...
    def test_accumulate_empty_batch_is_noop(self): ...
    def test_sequential_accumulation(self):
        # Accumulate 3 batches, verify running totals are correct
    def test_matches_hand_computed_reference(self):
        # Fixed synthetic indices/scores/outputs → compare all fields
        # against manually computed NumPy reference
    def test_derived_metrics_at_report(self): ...
    def test_zero_frequency_experts_handled(self): ...
```

**Acceptance:**
- [ ] All tests pass
- [ ] `test_matches_hand_computed_reference` validates against hand-computed values
- [ ] `test_zero_frequency_experts_handled` ensures no division by zero for unused experts

**Reasoning:** The accumulator is pure NumPy — easy to test rigorously. The hand-computed reference test is the gold standard. All other tests validate edge cases.

---

## Phase 2: Observer + Pruner + Save

### ☐ TODO P2-001: Implement `_identify_moe_layers()`

- **Owner:** GPT-5.5
- **Aspect:** model-util
- **Depends on:** P0-001
- **Effort:** 30 min

**Task:** Create `src/reap/backends/mlx/model_util.py`:

```python
def _identify_moe_layers(model) -> list[int]:
    """Return indices of layers containing MoE blocks.
    
    Walks model.model.layers[i] checking for 'switch_mlp' or 'block_sparse_moe'
    attribute on the MLP submodule. Returns sorted list of layer indices.
    """
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp if hasattr(layer, 'mlp') else getattr(layer, 'block_sparse_moe', None)
        if mlp is not None and hasattr(mlp, 'switch_mlp'):
            moe_layers.append(i)
    return moe_layers

def _get_moe_config(model, layer_idx: int) -> dict:
    """Extract num_experts, top_k, norm_topk_prob for a specific layer."""
    mlp = model.model.layers[layer_idx].mlp
    return {
        'num_experts': mlp.num_experts,
        'top_k': getattr(mlp, 'top_k', model.config.get('num_experts_per_tok', 8)),
        'norm_topk_prob': model.config.get('norm_topk_prob', False),
    }
```

**Acceptance:**
- [ ] Returns correct indices for known model (Qwen3-MoE: all layers are MoE)
- [ ] Returns empty list for dense-only model
- [ ] Works with both `mlp` and `block_sparse_moe` attribute names

**Reasoning:** Architecture detection needs to work for Qwen3, Mixtral, and future architectures. Simple heuristic based on attribute existence is sufficient — no regex on weight keys needed since we operate on live nn.Module objects.

---

### ☐ TODO P2-002: Implement layerwise observer

- **Owner:** DSv4
- **Aspect:** observation
- **Depends on:** P1-002, P1-003, P2-001
- **Effort:** 120 min

**Task:** Create `src/reap/backends/mlx/observer.py`:

```python
def observe_model(
    model: nn.Module,
    tokenizer,
    calibration_sequences: list[dict],
    config: dict,
) -> dict:
    """Layerwise replay collecting per-layer REAP pruning metrics.
    
    Replays each calibraton sequence through the model one layer at a time.
    Captures: router decisions, selected-expert outputs (before weighted sum).
    Accumulates: REAP/EAN metrics in NumPy PruningState objects.
    
    Returns observer_data dict compatible with the torch backend's schema.
    """
    num_layers = config['num_hidden_layers']
    moe_layers = _identify_moe_layers(model)
    E = config['num_experts']
    
    accumulators = {L: PruningState.initialize(E) for L in moe_layers}
    
    for seq in tqdm(calibration_sequences, desc="Observing"):
        tokens = mx.array(seq['input_ids'])[None, :]
        h = model.model.embed_tokens(tokens)
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            r = layer.self_attn(layer.input_layernorm(h), mask=None, cache=None)
            h = h + r
            
            if layer_idx in moe_layers:
                moe_input = layer.post_attention_layernorm(h)
                mlp = layer.mlp
                
                router = Qwen3MoeRouter(mlp, config)
                routing = router(moe_input)
                selected_out = mlp.switch_mlp(moe_input, routing.indices)
                
                accumulators[layer_idx].accumulate(routing, selected_out)
                
                moe_out = (selected_out * routing.scores[..., None]).sum(axis=-2)
                if hasattr(mlp, 'shared_expert'):
                    moe_out = moe_out + mlp.shared_expert(moe_input)
                h = h + moe_out
            else:
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            
            mx.eval(h)
    
    return {L: acc.report() for L, acc in accumulators.items()}
```

**Acceptance:**
- [ ] Processes all calibration sequences without OOM
- [ ] Returns dict keyed by layer index
- [ ] Each layer dict has all 8 keys from P1-003's `report()`
- [ ] `mx.eval(h)` called after each layer (verified by memory growth test)
- [ ] Works with batch_size=1 sequences of length up to 4096
- [ ] Prints progress via tqdm

**Reasoning:** This is the most complex component. It must correctly replay the model layer-by-layer, capture MoE intermediates before the weighted sum, and not accumulate lazy computation graphs across layers. The `mx.eval(h)` is critical — without it, the graph grows linearly with layers and batches.

---

### ☐ TODO P2-003: Implement expert pruner

- **Owner:** GPT-5.5
- **Aspect:** weight-surgery
- **Depends on:** P2-001
- **Effort:** 60 min

**Task:** Create `src/reap/backends/mlx/prune.py`:

```python
def prune_experts(
    model: nn.Module,
    observer_data: dict,
    prune_method: str,
    compression_ratio: float,
    preserve_super_experts: bool = False,
    preserve_outlier_experts: bool = False,
) -> nn.Module:
    """Prune experts from live MLX-LM model in-place.
    
    Args:
        model: Loaded MLX-LM nn.Module
        observer_data: Output of observe_model() — dict[layer_idx][metric_name]
        prune_method: "reap", "frequency", "ean_sum", "ean_mean", "weighted_ean_sum", "max_activations"
        compression_ratio: float between 0 and 1
    """
    num_experts = model.config['num_experts']
    n_to_prune = int(num_experts * compression_ratio)
    moe_layers = _identify_moe_layers(model)
    
    for layer_idx in moe_layers:
        saliency = observer_data[layer_idx][prune_method]
        
        # Preserve super/outlier experts by setting saliency to inf
        if preserve_super_experts or preserve_outlier_experts:
            super_indices = _identify_super_experts(observer_data, preserve_outlier_experts)
            for idx in super_indices.get(layer_idx, []):
                saliency[idx] = float('inf')
        
        # Keep top scorers
        keep = np.argsort(saliency)[::-1][:num_experts - n_to_prune]
        keep = sorted(keep)
        
        mlp = model.model.layers[layer_idx].mlp
        
        # Slice switch_mlp (expert dim = 0)
        for attr in ['gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(mlp.switch_mlp, attr)
            linear.weight = linear.weight[keep]
            if hasattr(linear, 'scales') and linear.scales is not None:
                linear.scales = linear.scales[keep]
            if hasattr(linear, 'biases') and linear.biases is not None:
                linear.biases = linear.biases[keep]
        
        # Slice router (output dim = expert dim)
        mlp.gate.weight = mlp.gate.weight[keep]
        
        mlp.num_experts = len(keep)
        if getattr(mlp, 'top_k', 999) > len(keep):
            mlp.top_k = len(keep)
    
    model.config['num_experts'] = num_experts - n_to_prune
    new_top_k = min(model.config.get('num_experts_per_tok', 999), model.config['num_experts'])
    model.config['num_experts_per_tok'] = new_top_k
    
    return model
```

**Acceptance:**
- [ ] Prunes correct number of experts per layer
- [ ] `switch_mlp` weights are sliced on dim 0 to correct shape
- [ ] Router weight sliced on dim 0 to correct shape
- [ ] `num_experts` config updated
- [ ] `num_experts_per_tok` clamped if it exceeds new count
- [ ] Quantized linear layers (`scales`, `biases`) sliced if present
- [ ] Model forward still works after pruning (shapes match)

**Reasoning:** In-place mutation of MLX-LM nn.Module objects is the simplest approach. MLX supports array indexing assignment. Quantized layers need special handling for auxiliary tensors.

---

### ☐ TODO P2-004: Implement save + reload + smoke test

- **Owner:** DSv4
- **Aspect:** io
- **Depends on:** P2-003
- **Effort:** 45 min

**Task:** Create `src/reap/backends/mlx/save.py`:

```python
def save_pruned_model(
    model: nn.Module,
    tokenizer,
    output_dir: str | Path,
    original_model_name: str,
) -> nn.Module:
    """Save pruned model via mlx-lm, reload, and run generation smoke test.
    
    WARNING: This destroys the in-memory model (mlx_lm.utils.save uses donate_model=True).
    Returns the reloaded model.
    """
    from mlx_lm import utils, load, generate
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    utils.save(
        dst_path=str(output_dir),
        src_path_or_repo=original_model_name,
        model=model,
        tokenizer=tokenizer,
        config=model.config,
    )
    
    pruned_model, pruned_tokenizer = load(str(output_dir))
    
    # Smoke test
    prompt = "What is your name?"
    messages = [{"role": "user", "content": prompt}]
    if pruned_tokenizer.chat_template:
        prompt = pruned_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    response = generate(pruned_model, pruned_tokenizer, prompt=prompt, max_tokens=50)
    
    expected_experts = model.config['num_experts']  # captured before save destroyed it
    actual_experts = pruned_model.config['num_experts']
    assert actual_experts == expected_experts, \
        f"Config mismatch after reload: expected {expected_experts}, got {actual_experts}"
    
    logger.info(f"✅ Saved and reloaded. Pruned to {actual_experts} experts. Response: {response}")
    return pruned_model
```

**Acceptance:**
- [ ] Produces safetensors files in output_dir
- [ ] Produces config.json with updated num_experts
- [ ] Reload succeeds with `mlx_lm.load(output_dir)`
- [ ] Generation produces coherent text (non-garbage)
- [ ] Config.experts matches expected pruned count

**Reasoning:** `donate_model=True` means the in-memory model is consumed during save. We MUST reload for the smoke test. This is also the right validation — if save+reload fails, the entire pipeline is broken.

---

### ☐ TODO P2-005: Unit tests for pruning

- **Owner:** GPT-5.5
- **Aspect:** testing
- **Depends on:** P2-003, P2-004
- **Effort:** 60 min

**Task:** Create `tests/test_mlx_prune.py`:

```python
class TestMlxPrune:
    def test_prune_one_expert_qwen(self):
        """Build tiny Qwen3-MoE with 3 experts, prune 1, verify shapes."""
    
    def test_forward_after_prune_works(self):
        """Prune from 3→2 experts, run forward, assert output shape unchanged."""
    
    def test_quantized_switch_linear_slicing(self):
        """Prune quantized layer, verify scales/biases sliced too."""
    
    def test_router_weight_sliced(self):
        """Verify gate.weight shape matches new expert count."""
    
    def test_top_k_clamped(self):
        """If pruning to fewer experts than top_k, top_k must be reduced."""
    
    def test_all_layers_pruned_identically(self):
        """All MoE layers should have same number of experts after pruning."""
    
    def test_save_reload_roundtrip(self):
        """Save tiny pruned model, reload, verify config and generation."""
```

**Acceptance:**
- [ ] All tests pass
- [ ] Forward output shape invariant across pruning
- [ ] Quantized slicing tested

---

## Phase 3: End-to-End Pipeline

### ☐ TODO P3-001: Minimal calibration loader

- **Owner:** DSv4
- **Aspect:** data
- **Depends on:** P0-001
- **Effort:** 45 min

**Task:** Create `src/reap/backends/mlx/data.py`:

```python
def load_calibration_sequences(
    dataset_name: str,
    split: str = "train",
    tokenizer = None,
    num_sequences: int = 128,
    max_length: int = 2048,
) -> list[dict]:
    """Load unpadded tokenized sequences as list of dicts with 'input_ids' numpy arrays.
    
    batch_size=1: each sequence is its own entry. No padding.
    Avoids attention_mask issue with MLX-LM's internal causal mask generation.
    """
    from datasets import load_dataset
    
    ds = load_dataset(dataset_name, split=split)
    sequences = []
    for sample in ds:
        if len(sequences) >= num_sequences:
            break
        text = _extract_text(sample, dataset_name)
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        sequences.append({'input_ids': np.array(tokens, dtype=np.int32)})
    return sequences
```

**Acceptance:**
- [ ] Does not import torch or vLLM
- [ ] Returns list of dicts with numpy int32 arrays
- [ ] Works with evol-codealpaca-v1, c4, and Mixture-of-Thoughts
- [ ] Handles missing 'text' field gracefully (uses instruction/output fields for code datasets)

**Reasoning:** This is a minimal replacement for the full `src/reap/data.py` pipeline. The existing `data.py` imports `vllm.TokensPrompt` at module level and returns `torch.Tensor` in `BatchEncoding`. Refactoring it is deferred. This loader is intentionally minimal — single-task, unpadded, numpy output.

---

### ☐ TODO P3-002: CLI entrypoint

- **Owner:** GPT-5.5
- **Aspect:** orchestration
- **Depends on:** P2-002, P2-003, P2-004, P3-001
- **Effort:** 60 min

**Task:** Create `src/reap/backends/mlx/entrypoint.py`:

```python
"""
MLX REAP pruning entrypoint.

Usage:
    python -m reap.backends.mlx.entrypoint \\
        --model-name mlx-community/Qwen3-30B-A3B-bf16 \\
        --dataset-name theblackcat102/evol-codealpaca-v1 \\
        --prune-method reap \\
        --compression-ratio 0.25 \\
        --seed 42 \\
        --output-dir artifacts/mlx-pruned
"""

def main():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--dataset-name', required=True)
    # Optional
    parser.add_argument('--prune-method', default='reap',
        choices=['reap','frequency','ean_sum','ean_mean','weighted_ean_sum','max_activations'])
    parser.add_argument('--compression-ratio', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-calibration-sequences', type=int, default=128)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--output-dir', default='artifacts/mlx-pruned')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Load
    model, tokenizer, config = mlx_lm.load(args.model_name, return_config=True)
    
    # Calibrate
    sequences = load_calibration_sequences(
        args.dataset_name, tokenizer=tokenizer,
        num_sequences=args.num_calibration_sequences,
        max_length=args.max_seq_length,
    )
    
    # Observe
    observer_data = observe_model(model, tokenizer, sequences, config)
    
    # Prune
    prune_experts(model, observer_data, args.prune_method, args.compression_ratio)
    
    # Save + validate
    save_pruned_model(model, tokenizer, args.output_dir, args.model_name)
```

**Acceptance:**
- [ ] `python -m reap.backends.mlx.entrypoint --help` works
- [ ] Full pipeline runs end-to-end on a small model (or the 4-bit Qwen3)
- [ ] Output directory contains safetensors + config.json + tokenizer files
- [ ] Print progress messages at each stage
- [ ] Handle keyboard interrupt gracefully

---

### ☐ TODO P3-003: Shell script

- **Owner:** GPT-5.5
- **Aspect:** ops
- **Depends on:** P3-002
- **Effort:** 20 min

**Task:** Create `experiments/mlx-pruning.sh`:

```bash
#!/bin/bash
# MLX REAP pruning experiment
# Usage: bash experiments/mlx-pruning.sh [MODEL_NAME] [DATASET] [PRUNE_METHOD] [COMPRESSION_RATIO] [SEED]
MODEL=${1:-"mlx-community/Qwen3-30B-A3B-4bit-DWQ"}
DATASET=${2:-"theblackcat102/evol-codealpaca-v1"}
METHOD=${3:-"reap"}
RATIO=${4:-"0.25"}
SEED=${5:-"42"}

echo "=== REAP MLX Pruning ==="
echo "Model:      $MODEL"
echo "Dataset:    $DATASET"
echo "Method:     $METHOD"
echo "Ratio:      $RATIO"
echo "Seed:       $SEED"

python -m reap.backends.mlx.entrypoint \
    --model-name "$MODEL" \
    --dataset-name "$DATASET" \
    --prune-method "$METHOD" \
    --compression-ratio "$RATIO" \
    --seed "$SEED" \
    --verbose

echo "Done! Output in artifacts/mlx-pruned/"
```

**Acceptance:**
- [ ] Script runs end-to-end
- [ ] Defaults use the 4-bit Qwen3 model (fits in 16-32GB unified memory)

---

### ☐ TODO P3-004: End-to-end validation

- **Owner:** DSv4
- **Aspect:** validation
- **Depends on:** P3-002
- **Effort:** 60 min

**Task:** Run full end-to-end pipeline and verify:

**Acceptance:**
- [ ] Pipeline completes without errors on Apple Silicon machine
- [ ] Observer produces REAP scores for all MoE layers
- [ ] REAP scores are non-zero for high-frequency experts
- [ ] REAP scores are zero for never-selected experts
- [ ] Expert ranking by REAP is not trivially identical to frequency ranking (confirms REAP adds signal beyond frequency)
- [ ] `prune_experts()` correctly removes lowest-REAP experts
- [ ] Saved model has correct expert count in config.json
- [ ] Reloaded model generates coherent text
- [ ] Generation is not identical to original (pruning changed the model)
- [ ] Peak memory during observation < 20GB (with 4-bit model)

**Reasoning:** This is the "does the whole thing work" test. Each acceptance criterion validates a different aspect of correctness. The "REAP adds signal beyond frequency" check is important — it confirms our REAP implementation is computing the router-weighted metric, not just copying frequency.

---

## Phase 4: Mixtral (v0.2+)

### ☐ TODO P4-001: Mixtral router adapter

- **Owner:** GPT-5.5
- **Aspect:** model-specific
- **Depends on:** P3-004
- **Effort:** 45 min

**Task:** Add `MixtralRouter` class to `src/reap/backends/mlx/router.py`:

```python
class MixtralRouter:
    """Mixtral routing: top-k over raw logits, THEN softmax over selected."""
    
    def __init__(self, mlp_layer, config: dict):
        self.gate = mlp_layer.gate
        self.top_k = config.get('num_experts_per_tok', 2)
    
    def __call__(self, x: mx.array) -> RouterResult:
        logits = self.gate(x.reshape(-1, x.shape[-1]))
        indices = mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        selected_logits = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.softmax(selected_logits, axis=-1, precise=True)
        indices = indices.reshape(*x.shape[:-1], self.top_k)
        scores = scores.reshape(*x.shape[:-1], self.top_k)
        return RouterResult(indices=indices, scores=scores, logits=None, score_mode="actual")
```

**Acceptance:**
- [ ] Matches MLX-LM's actual Mixtral behavior
- [ ] Top-k over NEGATIVE logits (largest logits = smallest negative)
- [ ] Softmax applied after selection, not before
- [ ] Tests pass on `mlx-community/Mixtral-8x7B-Instruct-v0.1` (or converted)

---

### ☐ TODO P4-002: Mixtral pruning test

- **Owner:** GPT-5.5
- **Aspect:** testing
- **Depends on:** P4-001
- **Effort:** 30 min

**Task:** Add Mixtral-specific test cases to existing test files.

---

## Phase 5: Future (Deferred)

### ☐ TODO P5-001: DeepSeek-V2 router adapter

- **Depends on:** Phase 4 complete
- **Effort:** 60 min
- Handles group-limited greedy routing, scoring functions, shared experts

### ☐ TODO P5-002: GLM-4.5 router adapter

- **Depends on:** Phase 4 complete
- **Effort:** 90 min
- Sigmoid routing, correction bias, group selection, scaling

### ☐ TODO P5-003: ERNIE 4.5 router adapter

- **Depends on:** Phase 4 complete
- **Effort:** 60 min
- Softmax/sigmoid gate, shared experts, selected-score normalization

### ☐ TODO P5-004: Merge metrics (TTM, CA)

- **Depends on:** Phase 3 complete
- **Effort:** Large (days)
- Requires all-expert activation materialization; fundamentally different from pruning-only path

### ☐ TODO P5-005: MLX-LM evaluation server

- **Depends on:** Phase 3 complete
- **Effort:** Medium
- Replace vLLM with `mlx-lm` server for evaluation

### ☐ TODO P5-006: Padded batch support

- **Depends on:** Phase 3 complete
- **Effort:** Medium
- Requires intercepting/adapting MLX-LM's internal causal mask to handle padding

### ☐ TODO P5-007: Unified `--backend` CLI flag

- **Depends on:** Both torch and MLX paths stable independently
- **Effort:** Small
- Add `--backend mlx|torch` to `src/reap/args.py`, route accordingly

### ☐ TODO P5-008: Refactor `src/reap/data.py`

- **Depends on:** Phase 3 complete
- **Effort:** Medium
- Lazy vllm imports, support `return_tensors="np"`, unify with MLX calibration loader

---

## Reference: File Manifest

```
# New files (to create):
src/reap/backends/__init__.py
src/reap/backends/mlx/__init__.py
src/reap/backends/mlx/entrypoint.py
src/reap/backends/mlx/data.py
src/reap/backends/mlx/router.py
src/reap/backends/mlx/observer.py
src/reap/backends/mlx/metrics.py
src/reap/backends/mlx/prune.py
src/reap/backends/mlx/save.py
src/reap/backends/mlx/model_util.py
tests/test_mlx_no_torch_import.py
tests/test_mlx_router.py
tests/test_mlx_metrics.py
tests/test_mlx_prune.py
experiments/mlx-pruning.sh

# Existing files to NEVER modify in Phase 0-3:
src/reap/main.py            # (imports torch/vLLM — do not touch)
src/reap/observer.py        # (torch hooks — do not touch)
src/reap/prune.py           # (torch Module surgery — do not touch)
src/reap/data.py            # (imports vllm.TokensPrompt — do not touch yet)
src/reap/eval.py            # (vLLM/CUDA — do not touch)
src/reap/metrics.py         # (torch OnlineStatsTracker — do not touch)
src/reap/merge.py           # (torch merging — do not touch)

# Existing config files:
.env.template               # Add MLX_HF_CACHE or MLX_MODEL_PATH if needed
```

---

## Reference: Observer Data Schema (Contract)

This is the contract between observer → pruning decisions → weight surgery. Both backends produce this format.

```python
observer_data: dict[int, dict[str, np.ndarray]] = {
    layer_idx: {
        "total_tokens":                   int,             # scalar
        "expert_frequency":               np.ndarray[E] int64,
        "ean_sum":                        np.ndarray[E] float64,
        "ean_mean":                       np.ndarray[E] float32,
        "weighted_ean_sum":               np.ndarray[E] float64,
        "weighted_expert_frequency_sum":  np.ndarray[E] float64,
        "reap":                           np.ndarray[E] float32,
        "max_activations":                np.ndarray[E] float32,
    }
}
```

Pruning methods use these keys directly:
- `frequency` → `observer_data[layer]["expert_frequency"]`
- `ean_sum` → `observer_data[layer]["ean_sum"]`
- `ean_mean` → `observer_data[layer]["ean_mean"]`
- `weighted_ean_sum` → `observer_data[layer]["weighted_ean_sum"]`
- `reap` → `observer_data[layer]["reap"]`
- `max_activations` → `observer_data[layer]["max_activations"]`

---

## Reference: Config Field Mapping

| Architecture | Expert Count Key | Top-K Key | MoE Attribute |
|---|---|---|---|
| Qwen3-MoE | `num_experts` | `num_experts_per_tok` / `top_k` | `mlp` |
| Mixtral | `num_local_experts` | `num_experts_per_tok` | `block_sparse_moe` |
| DeepSeek-V2 | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| GLM4-MoE | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| ERNIE 4.5 | `moe_num_experts` / `moe_capacity` | `moe_k` | `mlp` |

Rule: `new_top_k = min(old_top_k, num_retained_experts)`

---

## Reference: Risk Register

| # | Risk | Severity | Mitigation | Owner |
|---|---|---|---|---|
| R1 | `mx.argpartition` instability | Low | Only used for selection; scores from `mx.take_along_axis` are deterministic | GPT-5.5 |
| R2 | np.add.at on 32K elements | Low | Negligible vs model forward; profile if concerned | DSv4 |
| R3 | 61GB bf16 model OOM | Medium | Use 4-bit for observation (17GB); prune bf16 separately | GPT-5.5 |
| R4 | `donate_model=True` destroys model | Low | Smoke test runs on reloaded model | DSv4 |
| R5 | Router adapter diverges from native | High | Tests compare adapter output against model's own routing | GPT-5.5 |
| R6 | Quantized model missing scales/biases | Medium | Centralize slicing helper; inspect SwitchLinear fields | GPT-5.5 |
| R7 | Padded batch contaminates routing stats | Medium | v0.1 uses batch_size=1; future: intercept causal mask | DSv4 |
| R8 | MLX-LM API breaking changes | Low | Pin `mlx-lm>=0.24,<1.0`; adapter functions isolate API surface | Both |
| R9 | Tokenizer produces pad tokens in encode() | Low | `truncation=True` without padding; verify tokenizer has no default pad_token | DSv4 |

---

## Changelog

```
2026-05-30  v0.1  Created document. Phase 0-3 items from synthesis of 5 planning docs.
                  Ground truth: mlx==0.31.2, mlx_lm==0.31.3.
                  Settled on 10 architecture decisions (D1-D10).
                  TODO count: 15 items across 3 active phases + 8 deferred.
```

---

## GPT-5.5 Addendum — 2026-05-30

This addendum preserves the append-only protocol. The plan above is strong enough
to start, but the pseudo-code has a few details that should be corrected before
implementation.

### Corrections To Apply While Coding

1. **`mx.allclose` exists in this environment.**

   Earlier API notes mark it missing, but local verification showed
   `hasattr(mx, "allclose") == True` for `mlx==0.31.2`. Tests may use
   `mx.allclose`; the manual `mx.max(mx.abs(...)) < tol` fallback is still fine
   if we want version tolerance.

2. **`total_tokens` must count input tokens, not top-k routes.**

   In `PruningState.accumulate()`, do not set:

   ```python
   self.total_tokens += flat_indices.shape[0]
   ```

   because `flat_indices.shape[0] == seq_len * top_k`. Correct behavior:

   ```python
   num_tokens = indices.reshape(-1, indices.shape[-1]).shape[0]
   self.total_tokens += num_tokens
   ```

   `expert_frequency` counts routed expert selections, so with `top_k > 1`,
   `expert_frequency.sum()` may equal `total_tokens * top_k`. This matches the
   current PyTorch REAP metric behavior.

3. **Keep `pairwise_expert_frequency` in the report for schema parity.**

   The pruner does not need it, but the current PyTorch pruning-only observer
   reports it. It is cheap to compute:

   ```python
   self.pairwise_expert_frequency += freq[:, None] + freq[None, :]
   ```

   Including it reduces surprises for tests and downstream tools that compare
   observer outputs.

4. **Explicit layer replay needs a causal mask.**

   The observer pseudo-code uses:

   ```python
   layer.self_attn(..., mask=None, cache=None)
   ```

   That is only correct for sequence length `1`. For full prompt calibration,
   reproduce MLX-LM's normal mask behavior, for example:

   ```python
   from mlx_lm.models.base import create_attention_mask

   mask = create_attention_mask(h, cache=None)
   r = layer.self_attn(layer.input_layernorm(h), mask, cache=None)
   ```

   For Qwen3-MoE, this should yield the same causal masking path as the model's
   own `__call__`.

5. **Do not assume `model.config` exists on MLX-LM models.**

   Use:

   ```python
   model, tokenizer, config = mlx_lm.load(model_name, return_config=True)
   ```

   Pass `config` through observer, prune, and save. Runtime model fields usually
   live under `model.args` and submodule attributes. Save with the updated config
   dict:

   ```python
   mlx_lm.utils.save(..., model=model, tokenizer=tokenizer, config=config)
   ```

6. **Router adapters should prefer live module attributes over stale config.**

   For Qwen:

   ```python
   self.top_k = getattr(mlp_layer, "top_k", config["num_experts_per_tok"])
   self.norm_topk_prob = getattr(
       mlp_layer,
       "norm_topk_prob",
       config.get("norm_topk_prob", False),
   )
   ```

   This matters after pruning clamps `top_k`.

7. **Centralize quantized slicing now, not later.**

   The first real target is a 4-bit Qwen model, so quantized metadata handling is
   not optional. A helper should slice every first-dimension expert tensor that
   exists:

   ```python
   def slice_first_dim(module, keep):
       for name in ("weight", "scales", "biases", "bias"):
           value = module.get(name) if hasattr(module, "get") else getattr(module, name, None)
           if value is not None:
               setattr(module, name, value[keep])
   ```

   Apply this to `switch_mlp.gate_proj`, `switch_mlp.up_proj`,
   `switch_mlp.down_proj`, and router/gate modules if they are quantized.

8. **Shared expert attribute names differ by architecture.**

   MLX-LM DeepSeek, GLM, and ERNIE use `shared_experts` in the inspected source,
   not only `shared_expert`. The observer should check both:

   ```python
   shared = getattr(mlp, "shared_experts", None) or getattr(mlp, "shared_expert", None)
   if shared is not None:
       moe_out = moe_out + shared(moe_input)
   ```

9. **Router tests should compare against router reference logic, not `switch_mlp`.**

   `switch_mlp` consumes indices; it does not prove that the router selected the
   correct indices. Tests should compare adapter outputs against the
   architecture's known routing code or the native gate method where available.

10. **Large model loading should be a manual/slow check, not a default unit gate.**

    P0-003 asks for loading `mlx-community/Qwen3-30B-A3B-4bit-DWQ`, which may
    download around 17GB. Keep that as a manual environment validation or a
    marked slow/model-required test. The normal unit suite should use tiny
    synthetic MLX-LM models.

### Additional Acceptance Criteria

- [ ] `PruningState.report()` includes `pairwise_expert_frequency`.
- [ ] `PruningState.total_tokens` equals the number of valid sequence tokens,
      not `seq_len * top_k`.
- [ ] Observer replay uses the same causal mask path as MLX-LM model forward.
- [ ] Save path passes the updated `config` dict from `mlx_lm.load(...,
      return_config=True)`, not `model.config`.
- [ ] Pruning helper slices `weight`, `scales`, `biases`, and `bias` for both
      selected expert projections and quantized routers.
- [ ] Large-model smoke tests are marked slow/manual.

### Changelog Addendum

```
2026-05-30  GPT-5.5  Appended implementation corrections:
                    total_tokens semantics, causal-mask replay, config handling,
                    quantized slicing, pairwise frequency parity, and slow-test
                    boundary for real 4-bit Qwen model loading.
```

---

## DSv4 Turn — 2026-05-30: Target Clarification

### Primary goal is MLX compatibility, NOT a specific model

The plan currently anchors heavily on Qwen3-30B-A3B as the target model. This
needs correction: **the goal is to make the REAP codebase MLX-compatible and
runnable on Apple Silicon (Mac) hardware.** Model selection is deferred until
after the MLX backend is built and working.

### What this changes

| Aspect | Before | After |
|---|---|---|
| Primary objective | Prune Qwen3-30B-A3B at 25% compression | Build MLX backend that works across supported MoE architectures |
| First model | Qwen3-MoE, then Mixtral later | Any supported architecture; Qwen3-MoE is still the natural first router adapter (best MLX-LM community support) |
| E2E validation | Specific 4-bit Qwen3 model | Run on whatever model is available, small enough for dev hardware |
| Phase 4+ | "Mixtral, then other architectures" | Each architecture is a router adapter module — add incrementally as needed |

### What stays the same

- Architecture decisions D1-D10 are all still valid and correct
- Phase 0-3 implementation order is unchanged
- The observer data contract, pruning logic, and accumulator design are
  architecture-agnostic — they were already designed that way
- Router adapters are still per-architecture — Qwen3MoeRouter is just the first
  one, not the only one
- Synthetic tiny models for unit tests — architectural decisions don't require a
  specific real model

### Adjusted acceptance criteria

Phase 3 E2E validation should work with **any available MLX-LM MoE model** that
fits in the development machine's unified memory. The model used for validation
is chosen at test time based on what's available, not hardcoded to Qwen3.

P0-003 (environment verification) should verify that `mlx` and `mlx_lm` are
importable and that memory APIs work. Loading a specific large model is a
separate manual smoke check, not a gate.

### Changelog

```
2026-05-30  DSv4     Target clarification: primary goal is MLX backend compatibility
                      on Apple Silicon, not a specific model. Model selection
                      deferred to after the backend is built. Qwen3-MoE remains
                      the natural first router adapter due to best MLX-LM support.
```
