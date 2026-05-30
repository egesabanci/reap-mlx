# REAP MLX Backend — Implementation Plan v0.2 (Final)

> Synthesis of 5 documents, API ground-truthing against installed `mlx==0.31.2` / `mlx_lm==0.31.3`,
> and deep data-flow tracing of the REAP codebase. This is the authoritative plan.

---

## 0. Executive Summary

**What:** Implement REAP expert pruning on MLX/Apple Silicon for Qwen3-30B-A3B.

**How:** A separate `src/reap/backends/mlx/` pipeline that shares only data schemas and pruning-decision logic with the existing PyTorch code — not a 1:1 port and NOT a backend abstraction layer. The MLX path is architecturally different because MLX has no hooks, uses lazy evaluation, and MLX-LM stores experts as stacked tensors in `switch_mlp`.

**First deliverable:** Prune Qwen3-30B-A3B at 25% compression, save, reload, smoke-test — all on Apple Silicon.

**Non-goals:** Evaluation, merging, quantized expert handling, multi-architecture support beyond Qwen3 (v0.2+).

---

## 1. MLX API Ground Truth

Installed versions: `mlx==0.31.2`, `mlx_lm==0.31.3`. Key API facts discovered (not assumed):

### Available

| API | Usage |
|---|---|
| `mx.argpartition(x, kth, axis)` | Partition array; MLX-LM uses this for top-k selection |
| `mx.take_along_axis(x, indices, axis)` | Gather (torch.gather equivalent) |
| `mx.linalg.norm(x, axis)` | L2 norm |
| `mx.softmax(x, axis, precise=True)` | Softmax; `precise=True` matches MLX-LM defaults |
| `mx.eval(x)` | Force evaluation of specific arrays |
| `mx.get_active_memory()`, `mx.get_peak_memory()`, `mx.get_cache_memory()` | Memory monitoring (note: top-level, not `.metal.*`) |
| `mx.set_cache_limit(bytes)`, `mx.set_memory_limit(bytes)` | Memory caps |
| `mx.savez()`, `mx.savez_compressed()` | Simple array serialization |

### NOT Available (critical)

| Missing API | Workaround |
|---|---|
| `mx.bincount` | Use `np.bincount` on CPU after `mx.eval()` + transfer |
| `mx.scatter_add` | Use `np.add.at` on CPU |
| `mx.topk` returning indices | Use `mx.argpartition` + `mx.take_along_axis` |
| `mx.allclose` | Use `mx.max(mx.abs(x - y)) < tol` |

### MLX-LM nn.Module (mutable, not just weight dicts)

```python
model = mlx_lm.load("mlx-community/Qwen3-30B-A3B-bf16")[0]
# model has: .parameters(), .children(), .named_modules(), .modules()
# model supports: .update(weights_dict), .update_modules(sgd)
# model supports: .save_weights(path)
# switch_mlp.gate_proj.weight[keep] slices experts on dim 0
```

### Attention Mask Limitation

MLX-LM model forwards generate **causal masks internally** and do NOT accept HuggingFace-style `attention_mask`. Padded batches will treat pad tokens as real context, skewing MoE routing statistics. **v0.1 must use batch_size=1 with unpadded sequences.**

### Save Behavior

`mlx_lm.utils.save()` internally calls `donate_model=True`, which mutates/destroys the in-memory model. **Smoke test must happen after save+reload**, not on the in-memory pruned model.

---

## 2. Data Flow (Framework-Agnostic Core)

### Stage 1 → Stage 2: Observation produces per-layer dict

```python
observer_data[layer] = {
    # All pruning methods use a subset of:
    "total_tokens":     int,                # tokens processed through this layer
    "expert_frequency": array[E] int64,     # tokens routed to each expert
    "ean_sum":          array[E] float64,   # sum of ||expert_output||₂
    "ean_mean":         array[E] float32,   # running mean (ean_sum / expert_frequency)
    "weighted_ean_sum": array[E] float64,   # sum of router_weight × ||output||
    "weighted_expert_frequency_sum": array[E] float64,  # sum of router weights
    "reap":             array[E] float32,   # router-weighted activation norm mean
    "max_activations":  array[E] float32,   # per-expert max activation
}
```

**Validation:** All pruning metrics only require data from tokens that actually selected a given expert. In the torch code, even `max_activations` is computed only over selected tokens (see `pruning_metrics.py:192-194`). Therefore, **selected-expert-only observation is 100% information-equivalent to all-expert observation for pruning.**

### Stage 2: Pruning decisions (pure logic, framework-agnostic)

```python
saliency = observer_data[layer][prune_method]       # e.g., "reap"
n_to_prune = int(num_experts * compression_ratio)
keep = sorted(np.argsort(saliency)[::-1][:num_experts - n_to_prune])
```

Supported prune methods in v0.1: `frequency`, `ean_sum`, `ean_mean`, `weighted_ean_sum`, `reap`, `max_activations`.

`ean_ca` requires `routed_characteristic_activation` (merging-only metric) — not supported in selected-only mode.

### Stage 3: Weight surgery (MLX-LM specific)

Prune live `nn.Module` objects by slicing stacked tensors on dim 0, updating router weights, and updating config fields.

---

## 3. Architecture Decisions

### Decision 1: Separate MLX path, separate entrypoint

**Do NOT route through `src/reap/main.py`** — it imports PyTorch, vLLM, CUDA utilities, and the existing observer stack. Instead:

```
python -m reap.backends.mlx.entrypoint --model-name ... --dataset-name ... --compression-ratio 0.25
```

A shared `--backend` flag can be added later once both paths stabilize independently.

### Decision 2: Selected-expert-only observation

Use `moe.switch_mlp(hidden_states, indices)` which returns `[batch, seq, top_k, hidden]` — per-token outputs for only the selected experts. Compute norms on this tensor before the weighted sum. This avoids the 10-20× memory blowup of computing all `E` expert outputs.

**Critical detail:** REAP needs per-selected-expert outputs BEFORE the top-k weighted sum. The final MoE output is `sum(selected_outputs * scores, axis=top_k_dim)`. REAP attribution requires `||selected_outputs[t, k, :]||₂` for each `(token, selected_expert)` pair.

### Decision 3: CPU-side NumPy accumulators

Rationale: `mx.bincount` and `mx.scatter_add` don't exist in MLX 0.31.2. REAP state is small (`E ≈ 128`, few float64 arrays). Keeping accumulators on CPU avoids lazy graph retention across batches.

```
Per-batch: MLX compute → mx.eval(stats) → np.array(stats) → NumPy accumulate
```

No `OnlineStatsTracker` port needed. Running means computed via weighted average over millions of tokens.

### Decision 4: Architecture-specific router adapters

Each architecture has different routing. A `topk(softmax(logits))` approximation is WRONG for Mixtral, DeepSeek, GLM, ERNIE. Use MLX-LM's actual routing behavior.

Qwen3-MoE routing (MLX-LM's actual code path):
```python
logits = gate(x)                                          # [tokens, E]
gates = mx.softmax(logits, axis=-1, precise=True)
indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
scores = mx.take_along_axis(gates, indices, axis=-1)
if norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
```

Mixtral routing (MLX-LM's actual code path):
```python
logits = gate(x)
indices = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., :top_k]
selected_logits = mx.take_along_axis(logits, indices, axis=-1)
scores = mx.softmax(selected_logits, axis=-1, precise=True)
```
Note: this differs from Qwen — top-k over raw logits, THEN softmax over selected.

### Decision 5: Prune live MLX-LM nn.Module objects

MLX-LM provides mutable `nn.Module` objects. Weight surgery operates on these directly:

```python
for layer in moe_layers:
    mlp = model.layers[layer].mlp
    # Slice stacked expert tensors
    mlp.switch_mlp.gate_proj.weight = mlp.switch_mlp.gate_proj.weight[keep]
    mlp.switch_mlp.up_proj.weight = mlp.switch_mlp.up_proj.weight[keep]
    mlp.switch_mlp.down_proj.weight = mlp.switch_mlp.down_proj.weight[keep]
    # Slice router
    mlp.gate.weight = mlp.gate.weight[keep]
    # Update config
    mlp.num_experts = len(keep)
```

### Decision 6: Unpadded calibration (batch_size=1)

MLX-LM forwards don't accept `attention_mask`. Pad tokens would be treated as real context, contaminating routing statistics. v0.1 uses single unpadded sequences. Batched support with padding can be added later.

---

## 4. Component Designs

### 4.1 Calibration Loader (`src/reap/backends/mlx/data.py`)

Minimal loader that avoids importing torch/vLLM:

```python
def load_calibration_sequences(dataset_name, split, tokenizer, num_sequences, max_length):
    """Load tokenized sequences as list of numpy arrays / Python lists.
    
    Uses HuggingFace datasets + tokenizer. Returns list of dicts with 'input_ids' as numpy.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    
    sequences = []
    for i, sample in enumerate(ds):
        if len(sequences) >= num_sequences:
            break
        text = _extract_text(sample, dataset_name)  # per-dataset text extraction
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        sequences.append({'input_ids': np.array(tokens, dtype=np.int32)})
    return sequences
```

**Note:** The existing `src/reap/data.py` imports `vllm.TokensPrompt` at module level and returns `torch.Tensor` in `BatchEncoding`. Refactoring it is deferred. For v0.1, use this standalone loader.

### 4.2 Router Adapters (`src/reap/backends/mlx/router.py`)

```python
@dataclass
class RouterResult:
    """Architecture-neutral routing output."""
    indices: mx.array       # [batch, seq, top_k]
    scores: mx.array        # [batch, seq, top_k] — actual routing weights
    logits: mx.array | None # [batch, seq, num_experts] — raw logits if available
    score_mode: str         # "actual" | "compat_softmax"

class Qwen3MoeRouter:
    """Captures Qwen3-MoE routing decisions using MLX-LM's actual behavior."""
    
    def __init__(self, mlp_layer, config):
        self.gate = mlp_layer.gate
        self.top_k = config['num_experts_per_tok']
        self.norm_topk_prob = config.get('norm_topk_prob', False)
    
    def __call__(self, x: mx.array) -> RouterResult:
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

### 4.3 Observer (`src/reap/backends/mlx/observer.py`)

Replays each calibration sequence through the model layer-by-layer, capturing MoE routing and selected-expert outputs.

```python
def observe_model(model, tokenizer, calibration_sequences, config) -> dict:
    """Layerwise replay collecting per-layer REAP metrics."""
    num_layers = config['num_hidden_layers']
    E = config['num_experts']
    moe_layers = _identify_moe_layers(model)  # check which layers have switch_mlp
    
    accumulators = {L: PruningState.initialize(E) for L in moe_layers}
    
    for seq in calibration_sequences:
        tokens = mx.array(seq['input_ids'])[None, :]  # [1, seq_len]
        h = model.model.embed_tokens(tokens)
        
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            
            # Attention
            r = layer.self_attn(layer.input_layernorm(h), mask=None, cache=None)
            h = h + r
            
            if layer_idx in moe_layers:
                moe_input = layer.post_attention_layernorm(h)
                mlp = layer.mlp
                
                # Router
                router = Qwen3MoeRouter(mlp, config)
                routing = router(moe_input)
                
                # Selected-expert outputs (BEFORE weighted sum)
                selected_out = mlp.switch_mlp(moe_input, routing.indices)
                # Shape: [1, seq_len, top_k, hidden]
                
                # Accumulate REAP metrics
                _accumulate(accumulators[layer_idx], routing, selected_out)
                
                # Compute final MoE output for next layer
                moe_out = (selected_out * routing.scores[..., None]).sum(axis=-2)
                
                # Shared experts (if architecture has them)
                if hasattr(mlp, 'shared_expert'):
                    moe_out = moe_out + mlp.shared_expert(moe_input)
                
                h = h + moe_out
            else:
                # Dense FFN layer
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            
            mx.eval(h)  # Release per-layer graph
    
    # Convert accumulators to observer_data dict (same schema as torch)
    return {L: acc.report() for L, acc in accumulators.items()}
```

### 4.4 Accumulator (`src/reap/backends/mlx/metrics.py`)

```python
@dataclass
class PruningState:
    total_tokens: int = 0
    expert_frequency: np.ndarray = None       # [E] int64
    ean_sum: np.ndarray = None                # [E] float64
    weighted_ean_sum: np.ndarray = None       # [E] float64
    weighted_expert_frequency_sum: np.ndarray = None  # [E] float64
    max_activations: np.ndarray = None        # [E] float32
    
    @classmethod
    def initialize(cls, num_experts):
        return cls(
            expert_frequency=np.zeros(num_experts, dtype=np.int64),
            ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_ean_sum=np.zeros(num_experts, dtype=np.float64),
            weighted_expert_frequency_sum=np.zeros(num_experts, dtype=np.float64),
            max_activations=np.zeros(num_experts, dtype=np.float32),
        )
    
    def accumulate(self, routing: RouterResult, selected_outputs: mx.array):
        """selected_outputs: [1, seq_len, top_k, hidden]"""
        # Transfer to CPU
        indices = np.array(routing.indices)         # [1, seq_len, top_k]
        scores = np.array(routing.scores)           # [1, seq_len, top_k]
        sel_out = np.array(selected_outputs)        # [1, seq_len, top_k, hidden]
        
        # Flatten
        flat_indices = indices.reshape(-1)          # [seq_len * top_k]
        flat_scores = scores.reshape(-1)            # [seq_len * top_k]
        flat_norms = np.linalg.norm(sel_out, axis=-1).reshape(-1)   # [seq_len * top_k]
        flat_maxes = sel_out.reshape(-1, sel_out.shape[-1]).max(axis=-1)  # [seq_len * top_k]
        
        n = flat_indices.shape[0]
        self.total_tokens += n
        
        # Frequency (np.bincount works on CPU)
        freq = np.bincount(flat_indices, minlength=len(self.expert_frequency))
        self.expert_frequency += freq
        
        # Weighted sums (np.add.at = scatter_add)
        np.add.at(self.ean_sum, flat_indices, flat_norms)
        np.add.at(self.weighted_ean_sum, flat_indices, flat_norms * flat_scores)
        np.add.at(self.weighted_expert_frequency_sum, flat_indices, flat_scores)
        
        # Max activations per expert
        for e in np.unique(flat_indices):
            self.max_activations[e] = max(self.max_activations[e],
                                          flat_maxes[flat_indices == e].max())
    
    def report(self) -> dict:
        """Return observer_data-compatible dict with derived metrics."""
        eps = 1e-10
        freq = self.expert_frequency.astype(np.float64) + eps
        return {
            'total_tokens': self.total_tokens,
            'expert_frequency': self.expert_frequency,
            'ean_sum': self.ean_sum,
            'ean_mean': (self.ean_sum / freq).astype(np.float32),
            'weighted_ean_sum': self.weighted_ean_sum,
            'weighted_expert_frequency_sum': self.weighted_expert_frequency_sum,
            'reap': (self.weighted_ean_sum / freq).astype(np.float32),
            'max_activations': self.max_activations,
        }
```

### 4.5 Pruner (`src/reap/backends/mlx/prune.py`)

```python
def prune_experts(model, observer_data, prune_method, compression_ratio):
    """Prune experts from a live MLX-LM model in-place."""
    num_experts = model.config['num_experts']
    n_to_prune = int(num_experts * compression_ratio)
    moe_layers = _identify_moe_layers(model)
    
    for layer_idx in moe_layers:
        saliency = observer_data[layer_idx][prune_method]
        keep = np.argsort(saliency)[::-1][:num_experts - n_to_prune]
        keep = sorted(keep)  # stable ordering
        
        mlp = model.model.layers[layer_idx].mlp
        
        # Slice switch_mlp stacked tensors (dim 0 = expert dim)
        for attr in ['gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(mlp.switch_mlp, attr)
            linear.weight = linear.weight[keep]
            if hasattr(linear, 'scales') and linear.scales is not None:
                linear.scales = linear.scales[keep]
            if hasattr(linear, 'biases') and linear.biases is not None:
                linear.biases = linear.biases[keep]
        
        # Slice router weight (output dim = expert dim)
        mlp.gate.weight = mlp.gate.weight[keep]
        
        # Update config
        mlp.num_experts = len(keep)
        if mlp.top_k > len(keep):
            mlp.top_k = len(keep)
    
    # Update model-level config
    model.config['num_experts'] = num_experts - n_to_prune
    new_top_k = min(model.config.get('num_experts_per_tok', 999), model.config['num_experts'])
    model.config['num_experts_per_tok'] = new_top_k
    
    return model
```

### 4.6 Save & Smoke (`src/reap/backends/mlx/save.py`)

```python
def save_and_validate(model, tokenizer, output_dir, original_model_path):
    """Save pruned model, reload, run generation smoke test."""
    from mlx_lm import utils, load
    
    # Save (note: this destroys the in-memory model due to donate_model=True)
    utils.save(
        dst_path=output_dir,
        src_path_or_repo=original_model_path,
        model=model,
        tokenizer=tokenizer,
        config=model.config,
    )
    
    # Reload
    pruned_model, pruned_tokenizer = load(output_dir)
    
    # Smoke test: generation
    prompt = "What is your name?"
    messages = [{"role": "user", "content": prompt}]
    if pruned_tokenizer.chat_template:
        prompt = pruned_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    from mlx_lm import generate
    response = generate(pruned_model, pruned_tokenizer, prompt=prompt, max_tokens=50)
    
    # Verify config
    assert pruned_model.config['num_experts'] == num_experts - n_to_prune, \
        f"Config mismatch: {pruned_model.config['num_experts']}"
    
    logger.info(f"Smoke test passed. Response: {response}")
    return pruned_model, pruned_tokenizer
```

---

## 5. File Structure

```
src/reap/backends/
├── __init__.py                          # Empty
└── mlx/
    ├── __init__.py                      # Empty
    ├── entrypoint.py                    # ~80 lines — CLI, orchestration
    ├── data.py                          # ~60 lines — minimal calibration loader
    ├── router.py                        # ~100 lines — Qwen3MoeRouter, RouterResult
    ├── observer.py                      # ~200 lines — layerwise replay
    ├── metrics.py                       # ~120 lines — PruningState accumulator
    ├── prune.py                         # ~100 lines — expert tensor slicing
    ├── save.py                          # ~50 lines — save + reload + smoke
    └── model_util.py                    # ~40 lines — _identify_moe_layers, config helpers

tests/
├── test_mlx_router.py                   # Router adapter correctness
├── test_mlx_metrics.py                  # REAP metric equivalence
├── test_mlx_prune.py                    # Slice shapes, forward after pruning
└── test_mlx_no_torch_import.py          # Import guard

experiments/
└── mlx-pruning.sh                       # Shell script entry point
```

Total new code: ~750 lines. First PR: router + metrics only (~220 lines). Second PR: observer + prune + save (~350 lines). Third PR: entrypoint + CLI (~180 lines).

---

## 6. Implementation Phases

### Phase 0: Skeleton (do NOT import torch/vLLM)
- [ ] Create `src/reap/backends/mlx/` with empty `__init__.py`
- [ ] Add `test_mlx_no_torch_import.py`: assert importing the backend doesn't pull torch
- [ ] Verify: `python -c "import reap.backends.mlx; print('ok')"` works without `torch` in env

### Phase 1: Router + Metrics (PR #1)
- [ ] `router.py`: `RouterResult`, `Qwen3MoeRouter`
- [ ] `metrics.py`: `PruningState`
- [ ] `test_mlx_router.py`: tiny synthetic Qwen3-MoE block, compare adapter indices against model's own routing
- [ ] `test_mlx_metrics.py`: fixed synthetic `indices/scores/outputs` → compare PruningState against hand-computed NumPy reference

### Phase 2: Observer + Prune + Save (PR #2)
- [ ] `model_util.py`: `_identify_moe_layers()`, architecture detection
- [ ] `observer.py`: `observe_model()` with layerwise replay
- [ ] `prune.py`: `prune_experts()`
- [ ] `save.py`: `save_and_validate()`
- [ ] `test_mlx_prune.py`: tiny Qwen3-MoE, prune 1 expert, assert shapes, assert forward works

### Phase 3: Entrypoint + CLI (PR #3)
- [ ] `data.py`: minimal calibration loader
- [ ] `entrypoint.py`: full pipeline orchestration
- [ ] `experiments/mlx-pruning.sh`
- [ ] End-to-end: Qwen3-30B-A3B @ 25% compression, save, reload, smoke test

### Phase 4+: Mixtral (post-v0.2)
- [ ] Mixtral router adapter (selected-softmax behavior)
- [ ] Tests

### Phase 5+: Other architectures + merge metrics (future)
- DeepSeek, GLM, ERNIE router adapters
- Merge metrics (TTM, CA) — requires all-expert activations
- MLX-LM server evaluation
- Shared `--backend` CLI flag

---

## 7. Config Field Mapping (per Architecture)

| Architecture | Expert Count Field | Top-K Field | MoE Attribute |
|---|---|---|---|
| Qwen3-MoE | `num_experts` | `num_experts_per_tok` / `top_k` | `mlp` |
| Mixtral | `num_local_experts` | `num_experts_per_tok` | `block_sparse_moe` |
| DeepSeek-V2 | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| GLM4-MoE | `n_routed_experts` | `num_experts_per_tok` | `mlp` |
| ERNIE 4.5 | `moe_num_experts` / `moe_capacity` | `moe_k` | `mlp` |

Always clamp: `new_top_k = min(old_top_k, num_retained_experts)`.

---

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| `mx.argpartition` not fully stable across batches | Only used for routing; scores come from `mx.take_along_axis` which is deterministic |
| NP.add.at performance for 128 experts × 4096 tokens × 8 top-k | ~32K elements per batch; negligible vs model forward |
| 61GB bf16 model exceeds unified memory | Use 4-bit quantized model for observation (17GB); prune full-precision separately |
| MLX-LM API changes between versions | Pin `mlx-lm>=0.24,<1.0`; isolate in adapter functions |
| `donate_model=True` destroys model after save | Smoke test runs on reloaded model |
| Padded batch contaminates routing stats | v0.1 uses batch_size=1, unpadded; future: intercept causal mask |
| Router adapter diverges from MLX-LM's actual routing | Tests compare adapter output against model's own `switch_mlp` call |
| Pruning quantized model misses `scales/biases` | Centralize slicing helper; inspect every `SwitchLinear` for quantization fields |

---

## 9. What We Explicitly Defer To v0.2+

- **Merge support** (requires all-expert activation materialization, TTM/CA distance metrics)
- **Evaluation** (vLLM replacement via `mlx-lm` server)
- **Padded batched calibration** (requires attention mask support in MLX-LM forward)
- **Shared `--backend` CLI flag** (after both paths stabilize independently)
- **Architectures beyond Qwen3-MoE** (Mixtral, DeepSeek, GLM, ERNIE)
- **Quantized expert handling** (dequantize → prune → requantize as fallback when needed)
- **`src/reap/data.py` refactoring** (lazy vllm imports, numpy output support)

---

## 10. Relationship to Prior Documents

This document supersedes:

- `implementation-points-gpt.md` — initial insight doc (GPT-5.5)
- `implementation-points-dsv4.md` — deep architecture analysis (DSv4)
- `implementation-plan-synthesized.md` — first synthesis attempt
- `implementation-plan-v0.1.md` — second synthesis with data flow tracing
- `reap-mlx-implementation-plan-gpt-v0.1.md` — GPT-5.5's v0.1 with API checks

v0.2 incorporates the MLX API ground truth discovered by GPT-5.5 (no `mx.bincount`, no `mx.topk` indices, `mx.argpartition` usage, mutable nn.Module, attention mask limitation), plus the data flow validation and selected-only correctness proof from DSv4. It corrects API inaccuracies in earlier docs and provides the simplest possible implementation surface.
