# REAP MLX Backend — Synthesized Implementation Plan

> Cross-analysis of dsv4 (deep architecture analysis) and gpt (MLX-specific optimization points).
> Key insight: the MLX path should be **architecturally different** from the PyTorch path,
> not a 1:1 port. MLX's functional paradigm, unified memory, and stacked-tensor MoE layout
> enable fundamentally better approaches.

---

## 0. Critical Architectural Differences: Torch vs MLX

| Concern | Torch (current) | MLX (target) |
|---|---|---|
| MoE expert storage | `nn.ModuleList` of individual `nn.Module`s | Stacked weight tensors in `switch_mlp` (dim-0 slice) |
| Forward pass interception | `register_forward_hook` | Explicit function returning intermediates |
| Expert activations | Computes ALL experts `(num_experts, tokens, hidden)` | Only selected `top_k` experts per token |
| Accumulators | `OnlineStatsTracker` in Torch with scatter_add | NumPy accumulators on CPU (compact, no graph retention) |
| Device memory | Explicit CPU↔GPU transfers, device_map | Unified memory, no transfers needed |
| Weight modification | In-place `nn.Parameter.data.copy_()` | Array slicing & reconstruct weight dict |
| Evaluation | vLLM CUDA server | `mlx-lm` server or deferred |
| Lazy evaluation | Eager | Lazy — requires explicit `mx.eval()` boundaries |

---

## 1. Primary Optimization: Selected-Only Expert Activations

**This is the single most impactful change for MLX.**

### Current (Torch) — wasteful path:

```python
# observer.py line 362-365: computes ALL experts' outputs
activations = torch.zeros((num_experts, *flat_input.shape), device=device)
for idx, expert in enumerate(module.experts):
    activations[idx] = expert(flat_input).to(device)
# Shape: (128, 4096, 2048) for Qwen3-30B-A3B — ~1GB per batch per layer
```

### Target (MLX) — selected-only path:

```python
# Only compute outputs for top_k selected experts per token
# MLX-LM's SwitchGLU already does this internally
selected_outputs = switch_glu(hidden_states, router_logits, top_k)
# Shape: (tokens, hidden) — only the final weighted sum

# For REAP, aggregate per-expert stats from selected outputs:
for token_idx, (token_out, selected_ids, selected_weights) in enumerate(per_token_data):
    for expert_idx, weight in zip(selected_ids, selected_weights):
        norm = mx.linalg.norm(token_out)
        reap_accum[expert_idx] += norm * weight
```

### What about merging metrics (TTM, CA)?

The merging metrics (`ttm_similarity_matrix`, `characteristic_activation`, etc.) DO require all-expert activations. However:

- **Pruning-only REAP** uses `record_pruning_metrics_only=True` — these are skipped entirely
- For pruning, the selected-only path covers all needed metrics:
  - `expert_frequency` ✓ (from router decisions)
  - `ean_sum/ean_mean` ✓ (from selected expert norms)
  - `weighted_ean_sum` ✓
  - `reap` ✓
  - `max_activations` ✓
- For merging (if ever needed on MLX): materialize all-expert activations only when explicitly requested

**Decision: Default to selected-only; gate all-expert behind a flag.**

---

## 2. CPU-Side NumPy Accumulators (No OnlineStatsTracker Port Needed)

GPT-5.5's point #7 eliminates the need to port `OnlineStatsTracker` and work around `scatter_add`.

### Strategy:

```
MLX computation → mx.eval() → compact numpy arrays → NumPy accumulators
```

```python
# Per-batch accumulation (in MLX, per layer):
def collect_batch_stats_mlx(selected_experts, router_weights, expert_outputs):
    """Returns compact arrays suitable for transfer to CPU."""
    num_experts = E
    freq = mx.bincount(selected_experts.reshape(-1), minlength=E)  # [E]
    
    # Per-expert EAN sum (only for selected experts)
    ean_norms = mx.linalg.norm(expert_outputs, axis=-1)  # [tokens, K]
    ean_sum = mx.zeros(E)
    weighted_ean_sum = mx.zeros(E)
    reap = mx.zeros(E)
    
    for i in range(E):
        mask = (selected_experts == i).any(axis=-1)
        if not mask.any():
            continue
        ean_sum = ean_sum.at[i].add(ean_norms[mask].sum())
        weighted_ean_sum = weighted_ean_sum.at[i].add(
            (ean_norms[mask] * router_weights[mask, i]).sum()
        )
        reap = reap.at[i].add(
            (ean_norms[mask] * router_weights[mask, i]).mean()
        )
    
    mx.eval(freq, ean_sum, weighted_ean_sum, reap)
    return {
        "expert_frequency": np.array(freq),
        "ean_sum": np.array(ean_sum),
        "weighted_ean_sum": np.array(weighted_ean_sum),
        "reap": np.array(reap),
    }

# Accumulate across batches (NumPy, CPU):
layer_state["expert_frequency"] += batch["expert_frequency"]
layer_state["ean_sum"] += batch["ean_sum"]
layer_state["reap"] = running_mean(layer_state["reap"], batch["reap"], batch["freq"])
```

**Benefit:** Zero lazy graph retention across batches. No Welford/Kahan complexity. Trivially debuggable.

---

## 3. Native Router Reuse (Don't Reimplement Routing)

Each architecture has **different routing semantics**. Don't abstract them into a common `softmax+topk` approximation.

| Architecture | Routing Algorithm | Key Detail |
|---|---|---|
| Qwen3-MoE | Full softmax → top-k | Optional top-k renormalization (`norm_topk_prob`) |
| Mixtral | Top-k over raw logits → softmax over selected | Different from Qwen! |
| DeepSeek V2 | Group-limited greedy / scoring function | `scoring_func` + `topk_method` config |
| Llama 4 | Fused MoE block | `gate_up_proj` stacked tensor |
| GLM-4.5 | Sigmoid + correction bias + group selection | Non-standard gate |
| ERNIE 4.5 | Softmax/sigmoid gate, shared experts | Output signature differs |

### Approach:

Wrap the model's **actual forward pass** to return router decisions as side-channel data:

```python
def forward_with_routing_data(model, weights, tokens, layer_idx):
    """Run one layer's forward pass, capturing routing decisions."""
    # Use the model's native forward — don't reimplement routing
    output, router_logits, selected_indices, selected_weights = (
        model.layers[layer_idx].forward_with_routing(tokens)
    )
    return {
        "hidden_states": output,
        "selected_experts": selected_indices,      # [tokens, top_k]
        "router_weights": selected_weights,          # [tokens, top_k]
        "router_logits": router_logits,              # [tokens, num_experts] (may be None)
    }
```

For MLX-LM models that don't expose router intermediates, monkey-patch or subclass the MoE layer to return the needed data alongside output.

---

## 4. Pruning = Tensor Slicing (Not Module Surgery)

MLX-LM stores experts as **stacked tensors** in `switch_mlp`:

```
switch_mlp.gate_proj.weight  → [num_experts, hidden, intermediate]
switch_mlp.up_proj.weight     → [num_experts, hidden, intermediate]
switch_mlp.down_proj.weight   → [num_experts, intermediate, hidden]
router weight                  → [num_experts, hidden]
```

Pruning is a simple slice on dim 0:

```python
def prune_experts_mlx(weights, indices_to_keep):
    """Slice expert and router tensors to keep only selected experts."""
    keep = mx.array(indices_to_keep)
    
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}.mlp"
        
        # Slice expert stacked tensors on dim 0
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{prefix}.switch_mlp.{proj}.weight"
            weights[key] = weights[key][keep, :, :]
        
        # Slice router weight on dim 0
        router_key = f"{prefix}.gate.weight"
        weights[router_key] = weights[router_key][keep, :]
        
        # Update config
        weights["config"]["num_experts"] = len(keep)
        
        # Clamp top_k if it exceeds num_experts after pruning
        if weights["config"]["num_experts_per_tok"] > len(keep):
            weights["config"]["num_experts_per_tok"] = len(keep)
    
    return weights
```

**For quantized models** (GPT-5.5 #10): additionally slice `scales` and `biases` arrays. If the quantization format makes slicing unsafe, dequantize → prune → requantize as a two-step process.

---

## 5. Memory Management: Explicit `mx.eval()` Boundaries

MLX is **lazy** — operations build a computation graph that isn't evaluated until needed. Without explicit `mx.eval()` calls, the graph grows unbounded across calibration batches.

### Eval checkpoints:

```python
# After each batch through each layer:
mx.eval(output, router_data)
# After accumulating batch stats:
mx.eval(batch_stats)       # force computation before CPU transfer
# Periodically:
mx.metal.clear_cache()     # release Metal shader cache

# Memory monitoring (during development):
allocated = mx.metal.get_active_memory()
peak = mx.metal.get_peak_memory()
logger.debug(f"MLX memory: {allocated/1e9:.2f}GB active, {peak/1e9:.2f}GB peak")
```

### Layerwise replay on MLX:

The `LayerwiseMoEObserver`'s CPU↔GPU block loading/unloading is unnecessary on Apple Silicon (unified memory). However, layerwise **replay** is still useful to:
- Control peak memory by processing one layer at a time
- Avoid constructing the full model computation graph at once

The MLX version should:
1. Load full model weights into memory (unified, no device transfer)
2. Replay each calibration batch through layers sequentially
3. Call `mx.eval()` after each layer to release the per-layer graph
4. Cache hidden states for the next layer as NumPy arrays (not in MLX graph)

---

## 6. Backend Structure: Separate Paths, Shared Schema

After analysis, a heavy `AbstractBackend` ABC with 30+ methods is over-engineering. The correct approach is:

### Shared (framework-agnostic):

```
src/reap/
├── data.py              # Tokenization & dataset loading (uses HF, not torch/mlx)
├── cluster.py           # Scipy clustering — zero changes needed
├── args.py              # Add --backend {torch,mlx} flag
├── schema.py            # Observation data format spec (dict[int, dict[str, array]])
├── pruning_decisions.py # Given saliency data → which experts to prune (pure logic)
└── entry.py             # dispatch: if backend=="mlx" → mlx_pipeline else torch_pipeline
```

### MLX Backend (new):

```
src/reap/mlx/
├── __init__.py
├── model_loader.py      # Load MLX-LM models, identify MoE layers
├── router.py            # Architecture-specific router adapters (capture decisions)
├── observer.py          # Layerwise replay with selected-expert stats
├── accumulator.py       # NumPy-based REAP metric accumulation
├── prune.py             # Stacked-tensor slicing + config update
├── save.py              # MLX-LM save with weight format
└── smoke_test.py        # Local generation test
```

### Torch Backend (existing, refactored):

```
src/reap/torch/
├── __init__.py
├── observer.py          # Current hook-based observer (unchanged)
├── metrics.py           # Current OnlineStatsTracker + distance fns (unchanged)
├── prune.py             # Current Module surgery pruning (unchanged)
├── merge.py             # Current expert merging (unchanged)
├── model_util.py        # Current MODEL_ATTRS + helpers (unchanged)
└── eval.py              # Current vLLM eval (unchanged)
```

**Key principle:** The MLX backend is NOT a drop-in replacement for the torch backend. It's a parallel implementation that shares only the data schema and pruning-decision logic.

---

## 7. Implementation Order (Revised)

### Step 1: Make imports backend-safe (GPT-5.5 #1)
- Delay `import torch` / `import vllm` behind conditional guards
- Ensure `import reap.mlx` works without CUDA/PyTorch installed
- Add `--backend mlx|torch` flag to args

### Step 2: MLX model loading and registry
- Identify MLX-LM MoE models available (Qwen3-MoE, Mixtral are well-supported; Llama-4, DeepSeek, GLM, ERNIE may need conversion)
- Map MLX weight key patterns to logical names (gate_proj, up_proj, down_proj)
- Implement `load_model()` returning (weights_dict, config, tokenizer)

### Step 3: MLX router adapters
- Per-architecture wrappers that return routing decisions from native forward
- Qwen3-MoE first (best MLX support), then Mixtral, then others
- Output: `{selected_experts, router_weights, router_logits}` per layer per batch

### Step 4: Selected-expert REAP accumulation
- Implement per-batch stats collection (selected-only, not all-expert)
- Implement NumPy-based running accumulators
- Validate against known PyTorch REAP results for the same model

### Step 5: MLX layerwise observer
- Replay calibration batches through model layers
- Collect per-layer routing data and compute REAP stats
- `mx.eval()` boundaries at each layer to control graph growth

### Step 6: MLX pruning (tensor slicing)
- Slice stacked expert tensors on dim 0
- Update config fields (num_experts, num_experts_per_tok)
- Handle quantized layers if present

### Step 7: Save + smoke test
- Save pruned model via `mlx_lm` utilities or `mx.save_safetensors`
- Reload and run generation smoke test
- Verify output not garbage

### Step 8: Architecture coverage (incremental)
- Add Mixtral, DeepSeek, Llama-4, GLM, ERNIE support as MLX models become available
- Each new architecture mainly needs: router adapter + weight key mapping

---

## 8. What Stays Unchanged

| Component | Status | Notes |
|---|---|---|
| `data.py` — dataset loading & tokenization | ✅ Unchanged | Uses HF `datasets` + `transformers`, framework-agnostic |
| `cluster.py` — all clustering algorithms | ✅ Unchanged | Uses scipy, operates on numpy arrays from observation data |
| `pruning_decisions.py` — which experts to prune | ✅ Unchanged | Pure logic: given saliency scores → select top-k lowest |
| `args.py` — most argument definitions | Minor addition | Add `--backend` flag |
| `cluster_plots.py` | ✅ Unchanged | Matplotlib-based visualization |
| All experiment shell scripts | New MLX variants | Keep torch scripts; add `experiments/mlx-pruning.sh` |

---

## 9. Key Risks & Decisions

| Decision Point | Recommendation | Rationale |
|---|---|---|
| Selected-only vs all-expert activations | **Selected-only for pruning** | 10-20× memory reduction; all-expert only needed for merging metrics (TTM, CA) |
| NumPy accumulators vs OnlineStatsTracker | **NumPy on CPU** | No scatter_add, no lazy graph retention, trivially correct |
| Abstract backend ABC vs separate paths | **Separate paths, shared schema** | MLX execution model is too different from torch for a thin abstraction |
| Full vLLM eval vs deferred | **Deferred** | `--do-eval false` by default; add `mlx-lm` server eval later |
| Quantized expert support | **Dequantize first** | Prune in float16, optionally requantize |

---

## 10. First Concrete Deliverable

**Target: Qwen3-30B-A3B pruning via REAP on MLX**

This model has the best MLX community support (`mlx-community/Qwen3-30B-A3B-*` variants available). It's also the primary benchmark model in the paper.

### What "done" looks like:

```bash
python -m reap.mlx.prune \
    --model-name "mlx-community/Qwen3-30B-A3B-4bit" \
    --dataset-name "theblackcat102/evol-codealpaca-v1" \
    --compression-ratio 0.25 \
    --prune-method reap \
    --seed 42

# → produces pruned safetensors
# → smoke test passes ("What is your name?" returns coherent text)
# → REAP saliency scores match torch backend within numerical tolerance
```

### Files needed (new):

```
src/reap/mlx/__init__.py          # Empty or exports
src/reap/mlx/model_loader.py      # ~100 lines — load, inspect, identify MoE layers
src/reap/mlx/router.py            # ~80 lines — Qwen3 router adapter
src/reap/mlx/observer.py          # ~200 lines — layerwise replay + stats collection
src/reap/mlx/accumulator.py       # ~100 lines — NumPy accumulators
src/reap/mlx/prune.py             # ~80 lines — stacked tensor slicing
src/reap/mlx/save.py              # ~50 lines — save pruned model
src/reap/mlx/smoke_test.py        # ~40 lines — generation sanity check
experiments/mlx-pruning.sh        # Shell script

# Modified:
src/reap/args.py                  # + backend flag
src/reap/entry.py                 # dispatch logic (new file)
```

---

## Alignment Notice — 2026-05-31

This is a historical brainstorming document. If any statement here conflicts
with `implementation-plan-draft.md`, the active draft wins.

The current objective is not to replace the original CUDA REAP workflow. The
CUDA/PyTorch implementation remains the production and official experimentation
path on CUDA-compatible systems. The MLX work is a parallel Apple Silicon
experimentation backend.

The MLX backend must be adapter-driven: Qwen3 is a bootstrap/reference adapter,
not the model-support boundary. The target is to support any MoE weights whose
routing and expert layout can be represented by an MLX model/weight adapter,
while preserving REAP pruning semantics and schema compatibility.
