# REAP MLX Backend — Implementation Plan v0.1

> Deep re-examination of the PyTorch/CUDA REAP codebase and cross-analysis of dsv4 + gpt implementation points.
> This document replaces all prior docs with a single, rigorous, question-every-assumption implementation plan.

---

## Alignment Notice - 2026-05-31

Read model-specific language in this historical draft through the current
objective: the PyTorch/CUDA REAP path remains production/official, while MLX is
a parallel Apple Silicon experimentation backend.

Qwen3-MoE is the bootstrap/reference adapter, not the final target boundary. The
MLX implementation must be adapter-driven so any compatible MoE weights can be
represented through explicit routing, expert-layout, shared-expert, and config
contracts.

---

## 0. Executive Summary

**Goal:** Implement REAP expert pruning on MLX/Apple Silicon for Qwen3-30B-A3B (primary target), with a design that extends to Mixtral, Llama-4, DeepSeek V2, GLM-4.5, and ERNIE 4.5 as MLX models become available.

**Strategy:** Parallel implementation in `src/reap/mlx/` sharing only data schemas and pruning-decision logic with the existing torch code. The MLX path is architecturally different — not a 1:1 port — because MLX-LM has no hooks, uses stacked expert tensors, runs on unified memory, and has lazy evaluation.

**Non-goal for v0.1:** Evaluation (keep `--do-eval false`), merging (pruning only), quantized expert handling (dequantize-as-fallback).

---

## 1. Data Flow Trace (Framework-Agnostic Core)

Understanding exactly what data crosses stage boundaries reveals the abstraction seams.

### Stage 1 → Stage 2: Observation output

Per MoE layer `L`, after processing all calibration batches:

```python
observer_data[L] = {
    # --- Required for all pruning methods ---
    "total_tokens":     int,              # total tokens that passed through layer L
    "expert_frequency": array[E],         # how many tokens selected each expert (int)
    "ean_sum":          array[E],         # sum of ||expert_output||₂ (float64)
    "ean_mean":         array[E],         # running mean via OnlineStatsTracker (float32)
    "weighted_ean_sum": array[E],         # sum of router_weight × ||output|| (float64)
    "weighted_expert_frequency_sum": array[E],  # sum of router weights per expert (float64)
    "reap":             array[E],         # router-weighted activation norm mean (float32)
    "max_activations":  array[E],         # per-expert max activation magnitude (float32)

    # --- Pruning-only metrics: used by specific --prune-method values ---
    "expert_proba":     array[E],         # expert_frequency / total_tokens (optional)

    # --- Merging-only metrics (NOT computed in pruning-only mode) ---
    "pairwise_expert_frequency":  array[E,E],   # required for TTM normalization
    "ttm_similarity_matrix":      array[E,E],   # token-to-token matching distances
    "routed_characteristic_activation":  array[E,H],  # EAN per expert (used by ean_ca)
    "characteristic_activation":  array[E,H],   # HC-SMoE merging
    "online_characteristic_activation_dist": array[E,E],  # SubMoE merging
    "router_logit_similiarity":   array[E,E],   # MC-SMoE merging
}
```

**Key insight:** When `record_pruning_metrics_only=True` (which should be the default for the MLX pruning path), only the pruning metrics are computed. The merging metrics require all-expert activations and are skipped.

**Validation of selected-only approach:** Every pruning metric above only requires data from tokens that actually selected a given expert. We never need to know expert `j`'s output on a token that didn't route to expert `j`. The `max_activations` field in the torch code also only computes max over selected-token activations (see `update_pruning_state` lines 192-194). Therefore, **selected-expert-only observation is 100% information-equivalent for pruning**.

### Stage 2: Pruning decisions (framework-agnostic, pure logic)

```python
# Given observer_data and --prune-method and --compression-ratio:
for layer in range(num_layers):
    saliency = observer_data[layer][prune_method]  # e.g., observer_data[layer]["reap"]
    n_to_prune = int(num_experts * compression_ratio)
    experts_to_keep = argsort(saliency)[n_to_prune:]  # keep top scorers
    # Or: if preserving super experts, set their saliency to float('inf') first
```

Supported prune methods and their data dependencies:

| --prune-method | Uses observer_data key | Requires merging metrics? |
|---|---|---|
| `frequency` | `expert_frequency` | No ✓ |
| `ean_sum` | `ean_sum` | No ✓ |
| `ean_mean` | `ean_mean` | No ✓ |
| `weighted_frequency_sum` | `weighted_expert_frequency_sum` | No ✓ |
| `weighted_ean_sum` | `weighted_ean_sum` | No ✓ |
| `weighted_ean_sum_l2` | `weighted_ean_sum_l2` (derived) | No ✓ |
| `reap` | `reap` | No ✓ |
| `reap_l2` | `reap_l2` (derived) | No ✓ |
| `max_activations` | `max_activations` | No ✓ |
| `ean_ca` | `routed_characteristic_activation` | **Yes** ✗ |

The `ean_ca` method is the only pruning method that requires merging-only data. For v0.1 MLX, we can either skip `ean_ca` support or compute `routed_characteristic_activation` in a separate (slower) pass.

### Stage 3: Weight surgery

- **PyTorch:** Modify `nn.ModuleList` (remove expert modules), slice router linear layer, update `num_experts` config.
- **MLX:** Slice stacked tensors or delete individual expert weight keys. Update `config.json`.

---

## 2. MLX-LM Model Structure (Discovered vs Assumed)

MLX-LM stores models as safetensors files with flat key-value weight dictionaries loaded as `mx.array` values. The `mlx_lm.load()` function returns `(model, tokenizer)` where `model` is an `nn.Module`-like object.

### Confirmed: Qwen3-MoE MLX models exist on HuggingFace

```
mlx-community/Qwen3-30B-A3B-bf16           # 61GB, bf16, 13 shards
mlx-community/Qwen3-30B-A3B-4bit-DWQ       # 17GB, 4-bit quantized, 4 shards
mlx-community/Qwen3-30B-A3B-mixed-3-4bit   # 14GB, mixed 3/4-bit, 3 shards
```

These were converted by `mlx-lm` v0.24+ and use the `qwen3_moe` architecture.

### MLX-LM MoE weight storage format

Based on MLX community models and the MLX framework conventions, there are two possible formats:

**Format A — Individual expert keys (more common for MLX-LM converted models):**
```
model.layers.0.mlp.experts.0.gate_proj.weight   → [hidden, intermediate]
model.layers.0.mlp.experts.0.up_proj.weight      → [hidden, intermediate]
model.layers.0.mlp.experts.0.down_proj.weight    → [intermediate, hidden]
model.layers.0.mlp.experts.1.gate_proj.weight   → [hidden, intermediate]
...
model.layers.0.mlp.gate.weight                    → [num_experts, hidden]  (router)
```

**Format B — Stacked expert tensors (common for MLX-LM native models):**
```
model.layers.0.mlp.switch_mlp.gate_proj.weight   → [num_experts, hidden, intermediate]
model.layers.0.mlp.switch_mlp.up_proj.weight      → [num_experts, hidden, intermediate]
model.layers.0.mlp.switch_mlp.down_proj.weight    → [num_experts, intermediate, hidden]
model.layers.0.mlp.gate.weight                     → [num_experts, hidden]
```

**Our approach:** The MLX backend must auto-detect the format by inspecting the weight keys of the loaded model. The observer and pruner then use the appropriate strategy.

### Config location

MLX-LM models store config in HuggingFace `config.json` format, with keys like:
```json
{
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "top_k": 8,
  "hidden_size": 2048,
  "intermediate_size": 1536,
  "num_hidden_layers": 48,
  "moe_intermediate_size": 768,
  ...
}
```

The pruning step must update `num_experts` (and clamp `num_experts_per_tok` if it exceeds the new count) in the saved model's config.

---

## 3. Architectural Decisions

### Decision 1: Separate implementation, shared schema

**Rejected:** Heavy `AbstractBackend` ABC with 30+ methods. MLX's execution model (lazy, functional, no hooks) is too fundamentally different from PyTorch's (eager, OOP, hooks) for a thin abstraction.

**Accepted:** The MLX path lives in `src/reap/mlx/` as a parallel implementation. It shares:
- `src/reap/data.py` — dataset loading & tokenization (HF, framework-agnostic)
- `src/reap/cluster.py` — scipy clustering (framework-agnostic) — though unused for pruning
- Data schema: the `observer_data` dictionary format
- Pruning decision logic: select top-k lowest saliency scores
- Arg parsing: `src/reap/args.py` with added `--backend mlx|torch` flag

The torch code stays in `src/reap/` (root) unchanged — no refactoring required.

### Decision 2: Selected-expert-only observation (always for pruning)

**Rationale:** The pruning metrics require NO information from non-selected experts. Computing all `E` experts' outputs is a 10-20× waste. The torch code does this because hooks are coarse-grained (they run on the entire MoE block output) and the code needs all-expert activations for merging metrics. But with `record_pruning_metrics_only=True`, only pruning metrics are needed.

**Implementation:** Instead of a hook that intercepts the MoE block, the MLX observer explicitly calls the MoE forward function and captures only the selected experts' outputs. This requires the observer to understand the MoE layer's internal routing, which leads to...

### Decision 3: Per-architecture router adapters (not generic routing)

**Rationale:** Different architectures have fundamentally different routing algorithms:

| Architecture | Routing | MLX-LM availability |
|---|---|---|
| Qwen3-MoE | Full softmax, top-k, optional renormalization | ✅ bf16 + 4-bit |
| Mixtral | Top-k over raw logits, softmax over selected | ✅ Likely available |
| Llama 4 Scout | Fused MoE, `gate_up_proj` stacked | ❓ No community models yet |
| DeepSeek V2 | Group-limited greedy, scoring function | ❓ Complex MLA attention |
| GLM-4.5 Air | Sigmoid gate, correction bias, group selection | ❓ |
| ERNIE 4.5 | Softmax/sigmoid, shared experts | ❓ |

**Implementation:** For each supported architecture, implement a `RouterAdapter` that:
1. Takes hidden states and the model's MoE layer
2. Returns `{selected_experts, router_weights, router_logits, per_expert_outputs}`
3. Uses the model's native routing weights and logic

For Qwen3-MoE specifically, the adapter can either:
- **Option A:** Call the model's native `Qwen3MoeSparseMoeBlock.forward()` and instrument it to return intermediates (best correctness, requires understanding MLX-LM internals)
- **Option B:** Reimplement the routing using the loaded router weights (simple, risk of divergence)
- **Option C:** Monkey-patch the MoE layer to capture router decisions

**Decision for v0.1:** Start with Option B (reimplement routing from weights) and validate against Option A by comparing REAP scores. If divergence > 1e-4, switch to Option A.

### Decision 4: CPU-side NumPy accumulators (not OnlineStatsTracker)

**Rationale:** MLX's lazy evaluation means that keeping accumulators inside the MLX computation graph would cause unbounded growth across batches. Instead:

```
Per-batch flow:
  MLX: compute router_logits, selected_experts, router_weights, per_expert_outputs
  MLX: compute per-batch stats: freq, ean_norms, weighted norms
  mx.eval(stats)  ← force evaluation, release graph
  CPU: np.array(stats)  ← transfer compact arrays
  CPU: accumulate into running totals via NumPy
```

The `OnlineStatsTracker` (Welford's algorithm with Kahan summation) is not needed because:
- The running means (`ean_mean`, `reap`) are computed across millions of tokens; simple weighted averaging is numerically stable enough
- If exact numerical parity with the torch backend is needed, we can implement a simple NumPy Welford tracker (trivial compared to scattering into MLX)

### Decision 5: Pruning = weight dict mutation

**Rationale:** MLX-LM models are weight dicts. Pruning involves:
1. Load weights as `Dict[str, mx.array]`
2. For each MoE layer, slice/drop expert weights
3. Slice router weights
4. Update `config.json` entries
5. Save via `mx.save_safetensors()` (sharded) or `mlx_lm` utilities

No in-place modification of live `nn.Module` objects. The pruned model is constructed from scratch or by mutating the weight dict.

---

## 4. Detailed Component Designs

### 4.1 Model Loader (`src/reap/mlx/model_loader.py`)

```python
def load_model(model_name: str) -> tuple[nn.Module, dict, PreTrainedTokenizer]:
    """Load MLX-LM model. Returns (model_obj, weights_dict, tokenizer)."""
    model, tokenizer = mlx_lm.load(model_name)
    weights = mlx_lm.utils.load_weights(model_name)  # flat dict: str → mx.array
    return model, weights, tokenizer

def identify_moe_layers(weights: dict) -> list[int]:
    """Find which layers have MoE blocks by inspecting weight keys.
    
    Returns list of layer indices that contain MoE blocks.
    For Qwen3-MoE, all layers are MoE (returns range(num_hidden_layers)).
    For DeepSeek, some layers are dense FFN (returns subset).
    """
    # Pattern: look for keys matching model.layers.{idx}.mlp.experts.*
    moe_layers = set()
    for key in weights:
        if 'mlp.experts' in key or 'mlp.gate' in key:
            match = re.search(r'layers\.(\d+)', key)
            if match:
                moe_layers.add(int(match.group(1)))
    return sorted(moe_layers)

def detect_expert_format(weights: dict, layer_idx: int) -> str:
    """Detect whether experts are stored as individual keys or stacked tensors.
    
    Returns 'individual' or 'stacked'.
    """
    # Check for stacked format
    keys_with_layer = [k for k in weights if f'layers.{layer_idx}.mlp' in k]
    if any('switch_mlp' in k for k in keys_with_layer):
        return 'stacked'
    # Check for individual format
    if any(re.search(rf'experts\.\d+\.', k) for k in keys_with_layer):
        return 'individual'
    raise ValueError(f"Cannot detect expert format for layer {layer_idx}")
```

### 4.2 Router Adapter (`src/reap/mlx/router.py`)

```python
class Qwen3MoeRouterAdapter:
    """Captures routing decisions from Qwen3-MoE MLX model."""
    
    def __init__(self, weights: dict, layer_idx: int, config: dict):
        self.num_experts = config['num_experts']
        self.top_k = config['num_experts_per_tok']
        self.norm_topk_prob = config.get('norm_topk_prob', False)
        
        # Load router weight (flat key format)
        gate_key = f'model.layers.{layer_idx}.mlp.gate.weight'
        self.gate_weight = mx.array(weights[gate_key])  # [E, hidden]
    
    def __call__(self, hidden_states: mx.array) -> dict:
        """Returns routing decisions for a batch of hidden states.
        
        Args:
            hidden_states: [batch, seq, hidden] or [tokens, hidden]
        
        Returns:
            {
                'selected_indices': [tokens, top_k],
                'selected_weights': [tokens, top_k],
                'router_logits': [tokens, num_experts],
            }
        """
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        
        # Qwen3 routing: full softmax, top-k selection
        router_logits = hidden_states @ self.gate_weight.T  # [tokens, E]
        router_probs = mx.softmax(router_logits, axis=-1)
        selected_weights, selected_indices = mx.topk(router_probs, self.top_k, axis=-1)
        
        # Optional top-k renormalization
        if self.norm_topk_prob:
            selected_weights = selected_weights / selected_weights.sum(axis=-1, keepdims=True)
        
        return {
            'selected_indices': selected_indices,
            'selected_weights': selected_weights,
            'router_logits': router_logits,
        }
```

### 4.3 Observer (`src/reap/mlx/observer.py`)

The observer runs calibration batches through the model, layer by layer, collecting REAP metrics.

```python
def observe_model(
    model, weights, tokenizer, calibration_batches, config
) -> dict[int, dict]:
    """Run calibration and collect per-layer REAP metrics.
    
    Returns observer_data dict matching the schema in Section 1.
    """
    num_layers = config['num_hidden_layers']
    moe_layers = identify_moe_layers(weights)
    observer_data = {}
    
    # Initialize per-layer accumulators (NumPy, on CPU)
    for layer_idx in moe_layers:
        observer_data[layer_idx] = _init_layer_accumulators(config['num_experts'])
    
    # Process each calibration batch
    for batch in tqdm(calibration_batches):
        # Convert tokens to MLX arrays
        input_ids = mx.array(batch['input_ids'].numpy())  # [batch, seq]
        attention_mask = mx.array(batch['attention_mask'].numpy())
        
        # Forward through embedding + all layers (layerwise to capture MoE data)
        hidden_states = _embedding_forward(model, weights, input_ids)
        
        for layer_idx in range(num_layers):
            if layer_idx in moe_layers:
                # MoE layer: capture routing + per-expert outputs
                router = Qwen3MoeRouterAdapter(weights, layer_idx, config)
                routing = router(hidden_states)
                
                # Compute per-expert outputs for SELECTED experts only
                expert_outputs = _compute_selected_expert_outputs(
                    weights, layer_idx, config, hidden_states, routing['selected_indices']
                )
                
                # Accumulate stats
                _accumulate_batch_stats(
                    observer_data[layer_idx],
                    routing,
                    expert_outputs,
                    attention_mask,
                    config['num_experts'],
                )
                
                # Compute MoE output and add residual
                moe_output = _combine_expert_outputs(
                    expert_outputs, routing['selected_indices'], routing['selected_weights']
                )
                hidden_states = _attention_forward(model, weights, layer_idx, hidden_states)
                hidden_states = hidden_states + moe_output
            else:
                # Dense FFN layer: just forward
                hidden_states = _dense_layer_forward(model, weights, layer_idx, hidden_states)
            
            mx.eval(hidden_states)  # Release per-layer graph
    
    return observer_data
```

### 4.4 Accumulator (`src/reap/mlx/accumulator.py`)

```python
def _init_layer_accumulators(num_experts: int) -> dict:
    return {
        'total_tokens': 0,
        'expert_frequency': np.zeros(num_experts, dtype=np.int64),
        'ean_sum': np.zeros(num_experts, dtype=np.float64),
        'ean_mean': np.zeros(num_experts, dtype=np.float32),
        'ean_mean_count': np.zeros(num_experts, dtype=np.int64),  # for online mean
        'weighted_ean_sum': np.zeros(num_experts, dtype=np.float64),
        'weighted_expert_frequency_sum': np.zeros(num_experts, dtype=np.float64),
        'reap': np.zeros(num_experts, dtype=np.float32),
        'reap_count': np.zeros(num_experts, dtype=np.int64),  # for online mean
        'max_activations': np.zeros(num_experts, dtype=np.float32),
    }

def _accumulate_batch_stats(state, routing, expert_outputs, attention_mask, E):
    """Accumulate one batch's stats into the per-layer state."""
    selected = np.array(routing['selected_indices'])       # [T, K]
    weights = np.array(routing['selected_weights'])        # [T, K]
    outputs = np.array(expert_outputs)                      # [T, K, H]
    
    # Mask out padding tokens
    if attention_mask is not None:
        mask = np.array(attention_mask).reshape(-1).astype(bool)
        selected = selected[mask]
        weights = weights[mask]
        outputs = outputs[mask]
    
    T = selected.shape[0]
    state['total_tokens'] += T
    
    # Expert frequency
    freq = np.bincount(selected.reshape(-1), minlength=E)
    state['expert_frequency'] += freq
    
    # Per-expert stats
    ean_norms = np.linalg.norm(outputs, axis=-1)  # [T, K]
    
    for e in range(E):
        mask = (selected == e).any(axis=-1)  # [T]
        if not mask.any():
            continue
        
        ean_sum_e = ean_norms[mask][selected[mask] == e].sum()  # norms for expert e
        weighted_sum_e = (ean_norms[mask] * weights[mask])[selected[mask] == e].sum()
        
        state['ean_sum'][e] += ean_sum_e
        state['weighted_ean_sum'][e] += weighted_sum_e
        state['weighted_expert_frequency_sum'][e] += weights[mask][selected[mask] == e].sum()
        
        # Online mean: (old * old_count + new * new_count) / (old_count + new_count)
        cnt = freq[e]
        if cnt > 0:
            new_mean = ean_sum_e / cnt
            old_cnt = state['ean_mean_count'][e]
            old_mean = state['ean_mean'][e]
            state['ean_mean'][e] = (old_mean * old_cnt + new_mean * cnt) / (old_cnt + cnt)
            state['ean_mean_count'][e] += cnt
            
            new_reap = weighted_sum_e / cnt
            old_rc = state['reap_count'][e]
            old_reap = state['reap'][e]
            state['reap'][e] = (old_reap * old_rc + new_reap * cnt) / (old_rc + cnt)
            state['reap_count'][e] += cnt
        
        # Max activation
        expert_max = outputs[mask][selected[mask] == e].max()
        if expert_max > state['max_activations'][e]:
            state['max_activations'][e] = expert_max
```

### 4.5 Pruner (`src/reap/mlx/prune.py`)

```python
def prune_experts(weights: dict, config: dict, observer_data: dict, 
                  prune_method: str, compression_ratio: float,
                  preserve_super_experts: bool = False,
                  preserve_outlier_experts: bool = False) -> dict:
    """Prune experts from an MLX-LM model weight dictionary.
    
    Returns modified weights dict suitable for saving.
    """
    num_experts = config['num_experts']
    n_to_prune = int(num_experts * compression_ratio)
    moe_layers = identify_moe_layers(weights)
    expert_format = None  # auto-detect per layer
    
    for layer_idx in moe_layers:
        # Select experts to keep
        saliency = observer_data[layer_idx][prune_method]
        if preserve_super_experts or preserve_outlier_experts:
            super_indices = _identify_super_experts(observer_data, preserve_outlier_experts)
            saliency[super_indices] = float('inf')
        
        keep_indices = np.argsort(saliency)[::-1][:num_experts - n_to_prune]
        keep_indices = sorted(keep_indices)  # keep sorted for predictable slicing
        
        # Slice weights
        expert_format = detect_expert_format(weights, layer_idx)
        if expert_format == 'stacked':
            _prune_stacked_layer(weights, layer_idx, keep_indices)
        else:
            _prune_individual_layer(weights, layer_idx, keep_indices, num_experts)
    
    # Update config
    config['num_experts'] = num_experts - n_to_prune
    if config['num_experts_per_tok'] > config['num_experts']:
        config['num_experts_per_tok'] = config['num_experts']
    
    return weights, config
```

### 4.6 Save (`src/reap/mlx/save.py`)

```python
def save_pruned_model(weights: dict, config: dict, tokenizer, output_dir: str):
    """Save pruned model in MLX-LM format."""
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save weights (sharded safetensors)
    mlx_lm.utils.save_weights(output_dir, weights)
    
    # Or use mx.save_safetensors with sharding:
    # _save_sharded(weights, output_dir, max_size_gb=5)
```

---

## 5. File Structure

```
src/reap/mlx/
├── __init__.py              # Empty, or exports run_pipeline()
├── model_loader.py          # ~80 lines — load(), identify_moe_layers(), detect_expert_format()
├── router.py                # ~80 lines — Qwen3MoeRouterAdapter (extensible per-arch)
├── observer.py              # ~250 lines — observe_model(), layerwise replay
├── accumulator.py           # ~120 lines — init/accumulate batch stats (NumPy)
├── prune.py                 # ~100 lines — saliency selection, weight slicing
├── save.py                  # ~50 lines — save_pruned_model()
├── smoke_test.py            # ~40 lines — load pruned model, run test generation
└── pipeline.py              # ~60 lines — run_full_pipeline() orchestration

experiments/
└── mlx-pruning.sh           # Shell script entry point
```

Total: ~780 lines of new code (excluding shell scripts and tests).

---

## 6. Existing Code Changes

| File | Change | Lines |
|---|---|---|
| `src/reap/args.py` | Add `--backend` flag (choices: `torch`, `mlx`; default: `torch`) | +5 |
| `src/reap/data.py` | No changes (framework-agnostic) | 0 |
| `src/reap/cluster.py` | No changes | 0 |
| All `src/reap/*.py` in root | No changes (torch backend stays intact) | 0 |
| `pyproject.toml` | Add optional dependency: `mlx-lm>=0.24.0` | +2 |

---

## 7. Implementation Order (v0.1 → v1.0)

### Milestone M1: Observation Pipeline (week 1)
1. Create `src/reap/mlx/` package structure
2. Implement `model_loader.py` (load, detect MoE layers, detect format)
3. Implement `router.py` for Qwen3-MoE
4. Implement `accumulator.py` (NumPy accumulators)
5. Implement `observer.py` (layerwise replay with selected-expert stats)
6. Test: run observation on Qwen3-30B-A3B with evol-codealpaca-v1
7. Validation: compare REAP scores against torch backend for same model+dataset+seed

### Milestone M2: Pruning (week 1-2)
8. Implement `prune.py` for stacked and individual formats
9. Implement `save.py`
10. Implement `smoke_test.py`
11. End-to-end test: prune Qwen3-30B-A3B at 0.25 ratio, verify smoke test passes

### Milestone M3: CLI + Polish (week 2)
12. Implement `pipeline.py` orchestration
13. Add `--backend mlx` to `args.py`
14. Create `experiments/mlx-pruning.sh`
15. Test full pipeline end-to-end

### Milestone M4: Additional Architectures (post-v0.1)
16. Mixtral router adapter
17. Llama-4 router adapter (fused MoE)
18. DeepSeek V2 router adapter (group-limited routing)
19. Quantized expert support (dequantize → prune → requantize path)

---

## 8. Risk Register

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| MLX-LM Qwen3MoE uses individual (not stacked) expert format | Medium | Low (implementation differs, equally tractable) | Auto-detect format at load time |
| Router reimplementation diverges from native routing | Low | High (wrong REAP scores) | Validate by comparing against native model forward outputs; fall back to instrumented native forward |
| Selected-expert-only misses some pruning information | None | — | Traced through all metrics; confirmed 100% equivalent for pruning |
| 61GB bf16 model doesn't fit in unified memory | Medium | Medium | Use 4-bit quantized model for observation (17GB fits easily); prune full-precision after observation |
| Quantized model observation produces different REAP scores | Medium | Medium | Acceptable for v0.1 (scores are relative, ranking is what matters); document the precision tradeoff |
| MLX lazy evaluation causes OOM across batches | Low | Medium | `mx.eval()` after each layer; NumPy accumulators prevent cross-batch graph retention |
| `mlx-lm` API changes between versions | Low | Low | Pin `mlx-lm>=0.24,<1.0`; wrap in adapter functions |
