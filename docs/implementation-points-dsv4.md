# REAP MLX Backend Implementation Plan

## Overview

Document detailing the key aspects and concrete steps required to port REAP (Router-weighted Expert Activation Pruning) from PyTorch/CUDA to Apple Silicon MLX, while maintaining the original codebase architecture.

---

## Alignment Notice - 2026-05-31

The production and official experimentation workflow remains the original
PyTorch/CUDA REAP implementation. The MLX backend is a parallel Apple Silicon
experimentation path.

The MLX goal is adapter-driven support for compatible MoE weights, not a
one-model port. Qwen3-MoE is the bootstrap/reference adapter only; model-specific
routing, expert layout, shared-expert behavior, and config updates must live
behind explicit MLX adapter contracts.

---

## 1. Architectural Strategy: Abstract Backend Layer

Create a thin abstraction layer for framework-specific operations, allowing the core REAP algorithms to remain framework-agnostic while PyTorch and MLX backends coexist.

### Target Structure

```
src/reap/
├── backends/
│   ├── __init__.py          # get_backend() factory
│   ├── interface.py         # AbstractBackend ABC
│   ├── torch_backend.py     # TorchBackend — current code, refactored
│   └── mlx_backend.py       # MlxBackend — new implementation
├── observer.py              # → uses backend for model ops
├── metrics.py               # → uses backend for tensor ops
├── merge.py                 # → uses backend for weight ops
├── prune.py                 # → uses backend for weight ops
├── cluster.py               # → largely unchanged (scipy-based)
├── data.py                  # → minimal changes (tokenizer is HF, not ML-dependent)
├── args.py                  # → add backend selection argument
├── main.py                  # → route to correct backend
└── ...
```

### Backend Interface

```python
class AbstractBackend(ABC):
    """Framework-agnostic interface for model loading, inference, and weight surgery."""

    # -- Model lifecycle --
    @abstractmethod
    def load_model(model_name: str) -> Any: ...
    @abstractmethod
    def load_tokenizer(model_name: str) -> Any: ...
    @abstractmethod
    def save_model(model: Any, path: str) -> None: ...
    @abstractmethod
    def get_model_class_name(model: Any) -> str: ...

    # -- Forward pass --
    @abstractmethod
    def forward_pass(model, input_ids, attention_mask) -> MoEIntermediateValues: ...

    # -- Architecture introspection --
    @abstractmethod
    def get_num_layers(model) -> int: ...
    @abstractmethod
    def get_num_experts(model, layer_idx: int) -> int: ...
    @abstractmethod
    def get_top_k(model, layer_idx: int) -> int: ...
    @abstractmethod
    def get_hidden_dim(model, layer_idx: int) -> int: ...
    @abstractmethod
    def is_fused_experts(model) -> bool: ...

    # -- Weight access --
    @abstractmethod
    def get_expert_weights(model, layer_idx: int, expert_idx: int) -> Dict[str, Array]: ...
    @abstractmethod
    def set_expert_weights(model, layer_idx: int, expert_idx: int, weights: Dict[str, Array]) -> None: ...
    @abstractmethod
    def get_router_weights(model, layer_idx: int) -> Array: ...
    @abstractmethod
    def set_router_weights(model, layer_idx: int, weights: Array) -> None: ...
    @abstractmethod
    def remove_experts(model, layer_idx: int, indices_to_remove: List[int]) -> None: ...

    # -- Tensor primitives (framework-agnostic) --
    @abstractmethod
    def tensor_zeros(shape, dtype) -> Array: ...
    @abstractmethod
    def tensor_ones(shape, dtype) -> Array: ...
    @abstractmethod
    def tensor_arange(start, stop) -> Array: ...
    @abstractmethod
    def concatenate(arrays, axis) -> Array: ...
    @abstractmethod
    def stack(arrays, axis) -> Array: ...

    # -- Linalg ops --
    @abstractmethod
    def norm(x, axis) -> Array: ...
    @abstractmethod
    def topk(x, k, axis) -> Tuple[Array, Array]: ...
    @abstractmethod
    def softmax(x, axis) -> Array: ...
    @abstractmethod
    def cosine_similarity(x, y, axis) -> Array: ...
    @abstractmethod
    def bincount(x, minlength) -> Array: ...
    @abstractmethod
    def scatter_add(target, indices, source, axis) -> Array: ...
    @abstractmethod
    def gather(x, indices, axis) -> Array: ...

    # -- I/O --
    @abstractmethod
    def save_tensor(path, data) -> None: ...
    @abstractmethod
    def load_tensor(path) -> Any: ...
```

---

## 2. Model Architecture Registry Redesign

The current registry maps PyTorch class names to attribute paths. For MLX, models are identified differently.

### Current (Torch)

```python
MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    ...
}
```

### Target (Unified)

```python
MODEL_REGISTRY = {
    "torch": {
        "Qwen3MoeForCausalLM": TorchModelConfig(
            moe_block_attr="mlp",
            gate_proj_attr="gate_proj",
            up_proj_attr="up_proj",
            down_proj_attr="down_proj",
            experts_attr="experts",
            fused=False,
            router_attr="gate",
            num_experts_config_key="num_experts",
            num_experts_per_tok_config_key="num_experts_per_tok",
        ),
        ...
    },
    "mlx": {
        "Qwen3-30B-A3B": MlxModelConfig(
            # MLX weight key patterns
            expert_weight_pattern="model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
            router_weight_pattern="model.layers.{layer}.mlp.gate.weight",
            num_layers=...,
            num_experts=...,
            top_k=...,
            hidden_dim=...,
            fused=False,
        ),
        ...
    },
}
```

### Key Insight

The MLX backend identifies models by HF model ID (e.g., `"mlx-community/Qwen3-30B-A3B-4bit"`) rather than Python class name. The registry tells the backend:

- Where to find expert weights in the weight dictionary
- How many layers/experts exist
- How to map logical operations (get gate_proj) to physical weight keys

---

## 3. Observation Pipeline — The Core Challenge

### Current Flow (PyTorch)

```python
# Step 1: Register hooks on MoE blocks
observer = MoETransformerObserver(model, hook_config)
# hooks call _hook_fn for each MoE layer forward pass

# Step 2: Run data through model
for sample in dataloader:
    with observer.set_attention_mask(attention_mask):
        model(**sample)
    # hook_fn captures:
    #   - flat_input (hidden states)
    #   - router_logits
    #   - selected_experts (from topk)
    #   - per-expert activations (via explicit expert(i, flat_input) loop)

# Step 3: Hook computes metrics online
#   pruning_metrics.update_pruning_state(
#       activations, selected_experts, router_logits, ...
#   )
#   metrics.ttm_online(activations, ...)
#   metrics.ca_dist_online(activations, ...)
#   metrics.get_routed_characteristic_activation(...)
```

### MLX Approach

MLX has **no hooks** — models are typically functions that take `(weights, input) → output`. Solution: write an **explicit observation forward function**.

```python
def moe_forward_with_observations(
    hidden_states: mx.array,      # [batch, seq, hidden_dim]
    expert_weights: List[Dict],   # each dict: {"gate_proj", "up_proj", "down_proj"}
    router_weight: mx.array,      # [num_experts, hidden_dim]
    top_k: int,
) -> dict:
    """Forward pass through MoE layer returning all intermediate values."""
    flat = hidden_states.reshape(-1, hidden_dim)
    
    # Router
    router_logits = flat @ router_weight.T  # [tokens, num_experts]
    _, selected = mx.topk(router_logits, top_k, axis=-1)  # [tokens, top_k]
    
    # Per-expert activations
    activations = []
    for expert_w in expert_weights:
        x = flat @ expert_w["gate_proj"].T  # or fused gate_up_proj
        x = mx.silu(x) * (flat @ expert_w["up_proj"].T)  # SwiGLU
        x = x @ expert_w["down_proj"].T
        activations.append(x)
    activations = mx.stack(activations)  # [num_experts, tokens, hidden_dim]
    
    # Weighted output
    router_probs = mx.softmax(router_logits, axis=-1)
    topk_probs = mx.take_along_axis(router_probs, selected, axis=-1)
    # ... compute weighted sum of selected expert outputs
    
    return {
        "activations": activations,
        "router_logits": router_logits,
        "selected_experts": selected,
        "output": output,
    }
```

### Metrics Update (MLX)

```python
def update_pruning_state_mlx(layer_state, activations, selected_experts, router_logits, num_experts, valid_token_mask):
    """MLX-native version of pruning_metrics.update_pruning_state"""
    
    # Filter padding tokens
    if valid_token_mask is not None:
        activations = activations[:, valid_token_mask, :]
        selected_experts = selected_experts[valid_token_mask]
        router_logits = router_logits[valid_token_mask]
    
    num_tokens = selected_experts.shape[0]
    
    # Expert frequency via bincount
    if num_tokens > 0:
        expert_frequency = mx.bincount(selected_experts.reshape(-1), minlength=num_experts)
    else:
        expert_frequency = mx.zeros(num_experts, dtype=mx.int32)
    
    pairwise_expert_frequency = expert_frequency[:, None] + expert_frequency[None, :]  # broadcasting
    
    layer_state["total_tokens"] += num_tokens
    layer_state["expert_frequency"] += expert_frequency
    layer_state["pairwise_expert_frequency"] += pairwise_expert_frequency
    
    routing_weights = mx.softmax(router_logits, axis=-1)
    # Renormalize if needed
    if renormalize_router_weights and num_tokens > 0:
        topk_weights = mx.take_along_axis(routing_weights, selected_experts, axis=-1)
        routing_weights = routing_weights / topk_weights.sum(axis=-1, keepdims=True)
        routing_weights = mx.clip(routing_weights, a_min=1e-8, a_max=None)
    
    # Per-expert accumulation
    ean_sum = mx.zeros(num_experts, dtype=mx.float32)
    ean_mean = mx.zeros(num_experts, dtype=mx.float32)
    weighted_ean_sum = mx.zeros(num_experts, dtype=mx.float32)
    reap = mx.zeros(num_experts, dtype=mx.float32)
    
    for i in range(num_experts):
        active_mask = (selected_experts == i).any(axis=-1)
        if not active_mask.any():
            continue
        
        selected_acts = activations[i, active_mask, :]
        active_weights = routing_weights[active_mask, i]
        ean_norm = mx.linalg.norm(selected_acts, axis=-1)
        ean_sum = ean_sum.at[i].add(ean_norm.sum())
        ean_mean = ean_mean.at[i].add(ean_norm.mean())
        weighted_ean_sum = weighted_ean_sum.at[i].add((ean_norm * active_weights).sum())
        reap = reap.at[i].add((ean_norm * active_weights).mean())
    
    # Update running statistics
    layer_state["ean_sum"] += ean_sum
    layer_state["ean_mean"].update(ean_mean, expert_frequency)
    layer_state["weighted_ean_sum"] += weighted_ean_sum
    layer_state["reap"].update(reap, expert_frequency)
    layer_state["weighted_expert_frequency_sum"] += routing_weights.sum(axis=0)
    
    # Max activations
    for i in range(num_experts):
        if active_mask.any():
            expert_max = activations[i, active_mask].max()
            if expert_max > layer_state["max_activations"][i]:
                layer_state["max_activations"] = layer_state["max_activations"].at[i].set(expert_max)
    
    return activations, selected_experts, router_logits
```

---

## 4. Weight Surgery — Merging & Pruning

### Pruning (MLX)

MLX doesn't have mutable `nn.Module` objects. Instead, we work with weight dictionaries.

```python
def prune_experts_mlx(weights: dict, model_config, layer_idx: int, indices_to_keep: List[int]):
    """Prune experts from an MLX model weight dictionary."""
    num_kept = len(indices_to_keep)
    
    # Update expert weights — create new arrays with pruned dimensions
    for expert_idx in range(num_kept):
        old_idx = indices_to_keep[expert_idx]
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            old_key = model_config.format_expert_key(layer_idx, old_idx, proj)
            new_key = model_config.format_expert_key(layer_idx, expert_idx, proj)
            weights[new_key] = weights.pop(old_key)
    
    # Prune router
    router_key = model_config.format_router_key(layer_idx)
    weights[router_key] = weights[router_key][indices_to_keep, :]
    
    # Update config
    weights["config"]["num_experts"] = num_kept
    
    return weights
```

### Merging (MLX)

```python
def merge_experts_mlx(weights: dict, model_config, layer_idx: int, cluster_labels: Array, expert_proba: Array):
    """Merge experts within clusters for MLX model weights."""
    for cluster_id in mx.unique(cluster_labels):
        expert_indices = mx.where(cluster_labels == cluster_id)[0]
        if len(expert_indices) <= 1:
            continue
        
        dom_idx = expert_indices[expert_proba[expert_indices].argmax()]
        non_dom = [i for i in expert_indices if i != dom_idx]
        
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            dom_key = model_config.format_expert_key(layer_idx, dom_idx, proj)
            dom_weight = weights[dom_key]
            other_weights = [weights[model_config.format_expert_key(layer_idx, i, proj)] for i in non_dom]
            
            # Weighted average merge
            dom_freq = expert_proba[dom_idx]
            probs = mx.concatenate([mx.array([dom_freq]), expert_proba[non_dom]])
            merged = (mx.stack([dom_weight] + other_weights) * probs[:, None, None]).sum(axis=0) / probs.sum()
            
            weights[dom_key] = merged
        
        # Remove non-dominant experts or tie weights
        for i in non_dom:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                del weights[model_config.format_expert_key(layer_idx, i, proj)]
    
    return weights
```

---

## 5. OnlineStatsTracker — MLX Port

The `OnlineStatsTracker` uses Welford's algorithm with Kahan summation. Key changes:

```python
class MlxOnlineStatsTracker:
    """MLX-native online statistics tracker using Welford's algorithm."""
    
    def __init__(self, shape, count_shape=1, dtype=mx.float32):
        self.shape = shape
        self.count_shape = count_shape
        self.dtype = dtype
        self.count = mx.zeros(count_shape, dtype=mx.int32)
        self.mean = mx.zeros(shape, dtype=dtype)
        self.mean_compensation = mx.zeros(shape, dtype=dtype)
    
    def update(self, new_mean, new_count):
        """Update with new batch of data."""
        new_count = mx.array(new_count, dtype=mx.int32)
        
        updated_count = self.count + new_count
        delta = new_mean - self.mean
        y = delta * new_count.astype(mx.float32) / updated_count.astype(mx.float32)
        y = mx.where(mx.isnan(y), mx.zeros_like(y), y)  # Replace NaN with 0
        y = y - self.mean_compensation
        t = self.mean + y
        self.mean_compensation = (t - self.mean) - y
        self.mean = t
        self.count = updated_count
```

---

## 6. Distance Functions & Metrics

### Torch → MLX Equivalents

| Operation | Torch | MLX | Notes |
|---|---|---|---|
| Norm | `torch.linalg.norm(x, dim=-1)` | `mx.linalg.norm(x, axis=-1)` | |
| Top-K | `torch.topk(x, k, dim=-1)` | `mx.topk(x, k, axis=-1)` | |
| Softmax | `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | |
| Gather | `torch.gather(x, dim, idx)` | `mx.take_along_axis(x, idx, axis)` | |
| Take | `x[idx]` | `x[idx]` or `mx.take(x, idx, axis)` | |
| Bincount | `torch.bincount(x, minlength=n)` | `mx.bincount(x, minlength=n)` | ✅ MLX has this |
| Scatter Add | `x.scatter_add_(dim, idx, src)` | `x.at[idx].add(src)` or manual loop | ⚠️ No direct equivalent |
| Clip | `torch.clamp(x, min, max)` | `mx.clip(x, min, max)` | |
| Cdist | `torch.cdist(a, b, p=2)` | `mx.linalg.norm(a[:,None]-b, axis=-1)` | ⚠️ Memory heavy |
| Allclose | `torch.allclose(x, y)` | `mx.allclose(x, y)` | ✅ MLX has this |
| Save/Load | `torch.save/load` | `mx.save/load` or `mx.save_safetensors/load_safetensors` | |

### Cosine Similarity (Custom)

MLX doesn't have a built-in batched cosine similarity. Need custom implementation:

```python
def cosine_similarity_mlx(x, y, axis=-1):
    """Compute cosine similarity between two tensors along given axis."""
    x_norm = mx.linalg.norm(x, axis=axis, keepdims=True)
    y_norm = mx.linalg.norm(y, axis=axis, keepdims=True)
    return (x * y).sum(axis=axis) / (x_norm * y_norm + 1e-8)
```

### TTM Online (Token-to-Token Matching)

The key challenge is replacing `scatter_add_`. For MLX, we can use a loop-based approach:

```python
def ttm_online_mlx(activations, selected, distance_fn, num_experts):
    """Vectorized TTM distance computation for MLX."""
    E, S, H = activations.shape  # experts, seq, hidden
    K = selected.shape[1]  # top-k
    
    act_t = activations.transpose(1, 0, 2)  # S, E, H
    selected_acts = mx.take_along_axis(act_t, selected[..., None], axis=1)  # S, K, H
    
    # Compute distances, chunk to avoid memory blowup
    CHUNK = 16
    pairwise = mx.zeros((E, E), dtype=mx.float32)
    for s_start in range(0, S, CHUNK):
        s_end = min(s_start + CHUNK, S)
        chunk_acts = act_t[s_start:s_end]  # chunk_size, E, H
        chunk_sel = selected_acts[s_start:s_end]  # chunk_size, K, H
        
        # distances: chunk, K, E (selected vs all experts)
        dists = distance_fn(chunk_sel[:, :, None, :], chunk_acts[:, None, :, :])
        flat_dists = dists.reshape(-1, E)  # chunk*K, E
        idx0 = selected[s_start:s_end].reshape(-1)  # chunk*K
        
        # Scatter add by looping over experts
        for e in range(E):
            mask = (idx0 == e)
            if mask.any():
                pairwise = pairwise.at[e].add(flat_dists[mask].sum(axis=0))
    
    pairwise = pairwise + pairwise.T
    # Normalize by pairwise frequency
    # (pairwise_expert_frequency is computed externally and passed in)
    return pairwise
```

### Characteristic Activation Distance (Online)

```python
def ca_dist_online_mlx(activations, distance_fn):
    """Compute pairwise distance between expert characteristic activations."""
    activations_t = activations.transpose(1, 0, 2)  # E, S, H
    # Chunk along token dimension
    CHUNK = 16
    S = activations.shape[1]
    distances = []
    for start in range(0, S, CHUNK):
        end = min(start + CHUNK, S)
        chunk = activations_t[:, start:end, :]  # E, chunk, H
        dists = distance_fn(chunk[:, :, None, :], chunk[:, None, :, :])
        distances.append(dists)
    return mx.stack(distances).mean(axis=0)  # E, E
```

---

## 7. Device/Memory Management — Simplification

### Current Torch Code to Remove/Replace

```python
# ALL of these become NO-OPs or are removed in MLX backend:
device_map="auto"                           # HF model loading param
torch_dtype="auto"
model.to(device)                             # explicit transfers
model.to("cpu")
value.cpu()                                  # tensor transfer
value.to(device)
torch.cuda.empty_cache()
torch.cuda.synchronize()
torch.cuda.device_count()
torch.cuda.is_available()                   # → mx.default_device()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.amp.autocast(device_type="cuda")      # → MLX handles internally
gc.collect() + cuda cleanup                 # → mx.eval() for forced evaluation

# Layerwise observer complexity:
# _move_block() / _offload_current_block() / _load_block_for_replay()
# → All go away! MLX uses unified memory, no block loading/unloading needed
```

### MLX Memory Management

```python
# MLX memory is managed automatically with unified memory
# The only thing needed:
mx.eval()           # Force evaluation of lazy computation graph
mx.metal.clear_cache()  # Optional: clear Metal shader cache
```

### Layerwise Mode Simplification

The layerwise observer was designed for GPU-poor environments. On Apple Silicon:
- Unified memory (e.g., 64GB M-series Max) can handle large models directly
- If layerwise is still needed for very large models, it becomes much simpler:
  - No CPU↔GPU transfers — everything stays in unified memory
  - No block loading/unloading — all blocks accessible simultaneously
  - Just process blocks sequentially to manage peak memory

---

## 8. Data Pipeline — Minimal Changes

The data pipeline uses HuggingFace `datasets` + `transformers`, which are framework-agnostic.

### Changes Needed

```python
# Current: tokenizer returns torch tensors
tokens = tokenizer(text, return_tensors="pt")  # torch.Tensor

# MLX: convert to MLX arrays at boundary
tokens = tokenizer(text, return_tensors="pt")
mlx_tokens = {k: mx.array(v.numpy()) for k, v in tokens.items()}

# OR use return_tensors="np":
tokens = tokenizer(text, return_tensors="np")
mlx_tokens = {k: mx.array(v) for k, v in tokens.items()}
```

That's essentially it. The tokenizer and dataset loading code in `data.py` stays the same.

---

## 9. Clustering — Framework Agnostic

The clustering algorithms in `cluster.py` operate on:
- Distance matrices (numpy/torch arrays from observation data)
- Uses `scipy.cluster.hierarchy.linkage`, `scipy.spatial.distance.squareform`, `scipy.cluster.vq.kmeans2`

### Changes Needed

```python
# At clustering time, observation data is already extracted and saved to disk
# Just ensure it's in numpy format for scipy:
if isinstance(data, mx.array):
    data = np.array(data)  # convert at boundary
```

Zero algorithmic changes needed. This is the easiest part.

---

## 10. Evaluation — Separate Concern

The evaluation pipeline (`eval.py`) is heavily CUDA-dependent via vLLM:

```python
# vLLM requires CUDA:
subprocess.Popen(["vllm", "serve", model_name, "--gpu-memory-utilization", ...])
```

### Options for MLX

1. **Option A: MLX LM Server** — Use `mlx-lm` for serving:
   ```bash
   mlx_lm.server --model mlx-community/Qwen3-30B-A3B-4bit
   ```

2. **Option B: Direct Generation** — Use `mlx-lm` `generate()` function:
   ```python
   from mlx_lm import load, generate
   model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")
   response = generate(model, tokenizer, prompt="...")
   ```

3. **Option C: Deferred** — Keep `--do-eval false` as default; evaluation runs separately with appropriate tools.

The `lm-eval-harness` supports multiple backends, including a local-completions API that works with any OpenAI-compatible server. MLX LM server provides this.

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Backend Layer + Registry)
- [ ] Create `src/reap/backends/interface.py` with `AbstractBackend` ABC
- [ ] Create `src/reap/backends/torch_backend.py` — extract existing torch code into backend
- [ ] Create `src/reap/backends/mlx_backend.py` — MLX stubs
- [ ] Create unified `MODEL_REGISTRY` with torch and mlx entries
- [ ] Add `--backend` argument to `args.py`

### Phase 2: Core Pipeline (Observation + Metrics)
- [ ] Port `metrics.py` distance functions to MLX
- [ ] Port `OnlineStatsTracker` to MLX
- [ ] Port `pruning_metrics.py` (`update_pruning_state`, `_prepare_pruning_batch`)
- [ ] Implement `MlxBackend.forward_pass()` for MoE observation
- [ ] Update `observer.py` to use backend abstraction

### Phase 3: Weight Surgery (Merge + Prune)
- [ ] Implement `MlxBackend` weight access/manipulation methods
- [ ] Port `merge.py` to work with backend interface
- [ ] Port `prune.py` weight surgery to backend interface
- [ ] Update `permute.py` (if permutation is used with merging)

### Phase 4: Integration & Cleanup
- [ ] Adapt `main.py` and `layerwise_prune.py` entry points
- [ ] Add MLX experiment scripts
- [ ] Remove CUDA-specific code paths
- [ ] Test with actual MLX models

### Phase 5: Evaluation (Deferred/Optional)
- [ ] Implement MLX evaluation using `mlx-lm` server or direct generation
- [ ] Add MLX-compatible eval backend

---

## 12. Key Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| No available MLX versions of supported models | High | Start with well-supported models (Qwen3-MoE, Mixtral); for others, implement conversion scripts |
| MLX `scatter_add` performance | Medium | Use vectorized alternatives where possible; manual loops for correctness |
| MLX lazy evaluation memory blowup | Medium | Insert `mx.eval()` checkpoints; monitor memory during development |
| Model format incompatibility (HF PyTorch → MLX) | Medium | Use `mlx-lm convert` for supported architectures |
| vLLM replacement for evaluation | Low | Evaluation is optional; can defer to post-processing |
