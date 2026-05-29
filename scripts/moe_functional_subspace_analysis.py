#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-efficient layerwise MoE analysis for ERNIE / Qwen3-MoE / Mixtral / Llama4-MoE.

Key properties:
- No deepcopy of models/experts (in-place prune/merge).
- Only one full model in VRAM at a time (Original → Pruned → Merged).
- Calibration samples live on CPU; moved to model device per forward.
- Expert activations aggregated online on CPU (sum/count).
- ERNIE fix: resize/average moe_statics.e_score_correction_bias when pruning/merging.
- Clamp layer.top_k and config.moe_k to new #experts after structure changes.
"""

import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.stats import entropy

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Global / Imports of MoE types
# ---------------------------

MOE_CLASSES: Dict[str, type] = {}
Llama4TextMoe = None  # ensure symbol exists even if import fails

# Qwen3 MoE
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    MOE_CLASSES['Qwen3MoeSparseMoeBlock'] = Qwen3MoeSparseMoeBlock
    print("✅ Successfully imported 'Qwen3MoeSparseMoeBlock'")
except Exception:
    print("ℹ️ Info: Could not import 'Qwen3MoeSparseMoeBlock'")

# Mixtral MoE
try:
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    MOE_CLASSES['MixtralSparseMoeBlock'] = MixtralSparseMoeBlock
    print("✅ Successfully imported 'MixtralSparseMoeBlock'")
except Exception:
    print("ℹ️ Info: Could not import 'MixtralSparseMoeBlock'")

# ERNIE 4.5 MoE
try:
    from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeSparseMoeBlock
    MOE_CLASSES['Ernie4_5_MoeSparseMoeBlock'] = Ernie4_5_MoeSparseMoeBlock
    print("✅ Successfully imported 'Ernie4_5_MoeSparseMoeBlock'")
except Exception:
    print("ℹ️ Info: Could not import 'Ernie4_5_MoeSparseMoeBlock'")

# Llama4 MoE (router-returning)
try:
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe as _Llama4TextMoe
    Llama4TextMoe = _Llama4TextMoe
    MOE_CLASSES['Llama4TextMoe'] = _Llama4TextMoe
    print("✅ Successfully imported 'Llama4TextMoe'")
except Exception:
    print("ℹ️ Info: Could not import 'Llama4TextMoe'")

Llama4TextExperts = None
try:
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts as _Llama4TextExperts
    Llama4TextExperts = _Llama4TextExperts
    print("✅ Successfully imported 'Llama4TextExperts'")
except Exception:
    print("ℹ️ Info: Could not import 'Llama4TextExperts'")

if not MOE_CLASSES:
    print("⚠️ Warning: No specific MoE classes could be imported. Will use generic MoE detection.")

# ---------------------------
# Output Directory
# ---------------------------

def create_output_directory(model_name, base_dir="moe_analysis_outputs") -> Path:
    model_short_name = model_name.replace("/", "_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / model_short_name / timestamp
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "layer_analysis").mkdir(exist_ok=True)
    (output_dir / "fim_analysis").mkdir(exist_ok=True)
    with open(output_dir / "analysis_metadata.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"\n📁 Output directory created: {output_dir}")
    return output_dir

# ---------------------------
# MoE detection and hooks
# ---------------------------

def get_moe_layers(model):
    """Identify MoE layers across several architectures (Qwen3, Mixtral, ERNIE, Llama4, generic)."""
    moe_layers = []
    print("Scanning model for MoE layers...")

    for name, module in model.named_modules():
        is_specific = any(isinstance(module, cls) for cls in MOE_CLASSES.values())

        is_generic = (
            hasattr(module, "gate")
            and hasattr(module, "experts")
            and isinstance(getattr(module, "experts"), torch.nn.ModuleList)
        )

        is_llama4 = (
            hasattr(module, "router")
            and hasattr(module, "experts")
            and hasattr(module, "num_experts")
            and hasattr(module, "top_k")
        )
        if is_specific or is_generic or is_llama4:
            if is_llama4 and not hasattr(module, "gate"):
                module.gate = module.router  # normalize interface
            moe_layers.append((name, module))

    if not moe_layers:
        warnings.warn("🚨 WARNING: No MoE layers were found. Analysis cannot proceed.")
    else:
        print(f"✅ Found {len(moe_layers)} MoE layers.")
    return moe_layers

@contextmanager
def _capture_router_logits_via_hooks(model):
    """
    Capture router-like logits per MoE layer (robust).
    Cache each capture on CPU to avoid GPU growth.
    """
    moe_layers = get_moe_layers(model)
    cache = {i: [] for i, _ in enumerate(moe_layers)}
    hooks = []

    def _mk_hook(layer_idx):
        def _hook(_module, _inp, out):
            # out may be (scores, logits) or logits
            t = out[1] if (isinstance(out, tuple) and len(out) >= 2 and out[1] is not None) else out
            if t.dim() == 2:  # (B*S, E)
                t = t.unsqueeze(0)
            cache[layer_idx].append(t.detach().to("cpu"))
        return _hook

    for i, (_name, layer) in enumerate(moe_layers):
        router = getattr(layer, "router", None)
        if router is not None and hasattr(router, "register_forward_hook"):
            hooks.append(router.register_forward_hook(_mk_hook(i)))
            continue
        if Llama4TextMoe is not None and isinstance(layer, Llama4TextMoe):
            hooks.append(layer.register_forward_hook(_mk_hook(i)))
            continue
        gate = getattr(layer, "gate", None)
        if gate is not None and hasattr(gate, "register_forward_hook"):
            hooks.append(gate.register_forward_hook(_mk_hook(i)))

    try:
        yield cache
    finally:
        for h in hooks:
            h.remove()

# ---------------------------
# Calibration data (CPU only)
# ---------------------------

def get_calibration_data(tokenizer, dataset_name, num_sequences, seq_length):
    """
    Stream C4 and produce CPU tensors only.
    Inputs are moved to the model device right before forward().
    """
    print(f"\nLoading and processing calibration data from '{dataset_name}'...")
    print(f"Target: {num_sequences} sequences of length {seq_length}.")
    dataset = load_dataset(dataset_name, "en", streaming=True)
    train_data = dataset["train"]

    token_buffer: List[int] = []
    all_sequences: List[Dict[str, torch.Tensor]] = []

    for sample in train_data:
        text = sample['text']
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        token_buffer.extend(tokens)
        while len(token_buffer) >= seq_length:
            seq = token_buffer[:seq_length]
            token_buffer = token_buffer[seq_length:]
            all_sequences.append({
                'input_ids': torch.tensor(seq, dtype=torch.long).unsqueeze(0),       # CPU
                'attention_mask': torch.ones(1, seq_length, dtype=torch.long)        # CPU
            })
            if len(all_sequences) >= num_sequences:
                print(f"\n✅ Generated {len(all_sequences)} calibration sequences.")
                return all_sequences
    warnings.warn(f"⚠️ End of dataset; only generated {len(all_sequences)} sequences.")
    return all_sequences

# ---------------------------
# Helpers for ERNIE corrections / shape sync
# ---------------------------

def _clamp_top_k(layer, new_E: int):
    if hasattr(layer, "top_k"):
        layer.top_k = int(min(int(layer.top_k), int(new_E)))


def _update_config_num_experts(model, new_E: int):
    """
    Update both config AND model attributes after structural changes.
    This is crucial for models like ERNIE that cache num_experts on the model itself.
    """
    # Update config
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    
    # ERNIE-specific config fields
    if hasattr(cfg, "moe_num_experts"):
        cfg.moe_num_experts = int(new_E)
    if hasattr(cfg, "moe_k"):
        cfg.moe_k = int(min(int(cfg.moe_k), int(new_E)))
    
    # Generic fields some models use
    for attr in ("num_experts", "num_local_experts"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, int(new_E))
    
    # CRITICAL: Also update model-level attributes (not just config)
    # This is what was missing and causing the error
    if hasattr(model, "num_experts"):
        model.num_experts = int(new_E)
    if hasattr(model, "num_experts_per_tok"):
        model.num_experts_per_tok = int(min(int(model.num_experts_per_tok), int(new_E)))
    if hasattr(model, "moe_k"):
        model.moe_k = int(min(int(model.moe_k), int(new_E)))

def _shrink_moe_statics_for_keep(layer, keep_idx: List[int]):
    """For pruning: index the correction bias to the kept experts."""
    stats = getattr(layer, "moe_statics", None)
    if stats is None or not hasattr(stats, "e_score_correction_bias"):
        return
    bias = stats.e_score_correction_bias  # shape: [1, E], float32 param (frozen)
    device = bias.device
    with torch.no_grad():
        new_bias = bias[:, keep_idx].to(device)
    stats.e_score_correction_bias = torch.nn.Parameter(new_bias, requires_grad=False)

def _shrink_moe_statics_for_merge(layer, keep_idx: List[int], pairs: List[Tuple[int, int]]):
    """
    For merging: build new bias as:
      - for survivor i that merged with j: 0.5*(bias[i] + bias[j])
      - else: bias[i]
    """
    stats = getattr(layer, "moe_statics", None)
    if stats is None or not hasattr(stats, "e_score_correction_bias"):
        return
    bias = stats.e_score_correction_bias  # [1, E]
    device = bias.device
    with torch.no_grad():
        b = bias.squeeze(0)  # [E]
        partner = {i: j for (i, j) in pairs}  # only i survives
        new_vals = []
        for k in keep_idx:
            if k in partner:
                v = 0.5 * (b[k] + b[partner[k]])
            else:
                v = b[k]
            new_vals.append(v)
        new_b = torch.stack(new_vals, dim=0).unsqueeze(0).to(device)  # [1, new_E]
    stats.e_score_correction_bias = torch.nn.Parameter(new_b, requires_grad=False)

# ---------------------------
# Usage and Activations
# ---------------------------

@torch.no_grad()
def get_expert_usage(model, calibration_data):
    """Counts expert selections per MoE layer via router logits (robust across archs)."""
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return {}

    expert_usage: Dict[str, torch.Tensor] = {}
    for name, layer in moe_layers:
        if hasattr(layer, "num_experts"):
            nE = int(layer.num_experts)
        elif hasattr(layer, "experts") and hasattr(layer.experts, "__len__"):
            nE = len(layer.experts)
        else:
            print(f"⚠️ Cannot determine number of experts for layer {name}")
            continue
        expert_usage[name] = torch.zeros(nE, device="cpu")

    input_device = next(model.parameters()).device

    for model_inputs in calibration_data:
        local_inputs = {k: v.to(input_device, non_blocking=True) for k, v in model_inputs.items()}
        with _capture_router_logits_via_hooks(model) as cache:
            _ = model(**local_inputs)

        for i, (name, layer) in enumerate(moe_layers):
            if not cache[i]:
                continue
            x = torch.cat(cache[i], dim=0)  # (B?, S?, E)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if hasattr(layer, "top_k"):
                top_k = int(layer.top_k)
            elif hasattr(layer, "num_experts_per_tok"):
                top_k = int(layer.num_experts_per_tok)
            else:
                top_k = min(2, x.shape[-1])

            probs = torch.softmax(x, dim=-1)
            _, selected = torch.topk(probs, k=min(top_k, x.shape[-1]), dim=-1)
            idx = selected.reshape(-1)
            counts = torch.bincount(idx.cpu(), minlength=expert_usage[name].numel())
            expert_usage[name] += counts

    return expert_usage

@torch.no_grad()
def get_expert_activations_online(model, calibration_data):
    """
    Capture mean activation per expert using online accumulation to avoid large CPU lists.
    For each expert: running SUM over tokens and COUNT; final mean = SUM/COUNT.
    Special handling for Llama4's parameter-based experts.
    """
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return {}

    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.text_config.hidden_size
    activ_sum: Dict[str, torch.Tensor] = {}
    activ_cnt: Dict[str, torch.Tensor] = {}

    for name, layer in moe_layers:
        if hasattr(layer, "num_experts"):
            E = int(layer.num_experts)
        elif hasattr(layer, "experts") and hasattr(layer.experts, "__len__"):
            E = len(layer.experts)
        else:
            continue
        activ_sum[name] = torch.zeros(E, hidden_dim, dtype=torch.float32)
        activ_cnt[name] = torch.zeros(E, dtype=torch.long)

    # Register forward hooks per expert (ModuleList) or on the experts module (Llama4)
    input_device = next(model.parameters()).device

    def make_hook(layer_name, expert_idx):
        def _hook(_module, _inp, output):
            if isinstance(output, tuple):
                output = output[0]
            if output is None:
                return
            out = output.detach()
            if out.ndim == 1:
                out = out.unsqueeze(0)
            n = out.shape[0]
            activ_sum[layer_name][expert_idx].add_(out.sum(dim=0).to("cpu"))
            activ_cnt[layer_name][expert_idx] += n
        return _hook

    def make_llama4_hook(layer_name):
        """Special hook for Llama4TextExperts that processes batched expert outputs"""
        def _hook(_module, _inp, output):
            # For Llama4TextExperts, output is already the combined result
            # We need to capture the intermediate states per expert
            # Since we can't easily separate them after BMM, we'll use router scores
            # to approximate which experts were most active
            pass  # We'll handle this differently
        return _hook

    hooks = []
    llama4_layers = []  # Track Llama4 layers for special handling
    
    for name, layer in moe_layers:
        if isinstance(layer.experts, torch.nn.ModuleList):
            # Standard ModuleList approach (Qwen3, Mixtral, ERNIE)
            for i, expert in enumerate(layer.experts):
                hooks.append(expert.register_forward_hook(make_hook(name, i)))
        elif Llama4TextExperts is not None and isinstance(layer.experts, Llama4TextExperts):
            # Mark for special handling
            llama4_layers.append((name, layer))
        else:
            # Generic fallback
            def _fallback(layer_name=name):
                def __h(_m, _inp, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    if output is None:
                        return
                    out = output.detach()
                    if out.ndim == 1:
                        out = out.unsqueeze(0)
                    n = out.shape[0]
                    E = activ_sum[layer_name].shape[0]
                    share = (out.sum(dim=0).to("cpu") / float(E))
                    activ_sum[layer_name].add_(share.unsqueeze(0).expand(E, -1))
                    activ_cnt[layer_name] += n // max(E, 1)
                return __h
            hooks.append(layer.register_forward_hook(_fallback()))

    # For Llama4, we need to capture activations differently
    # We'll use the router scores to weight the shared expert output
    if llama4_layers:
        def capture_llama4_activations(layer_name, layer, hidden_states):
            """Approximate expert activations using router scores and shared output"""
            with torch.no_grad():
                # Get router scores for this input
                if hasattr(layer, 'router'):
                    router_scores, _ = layer.router(hidden_states.view(-1, hidden_dim))
                    # router_scores shape: [batch*seq, num_experts]
                    
                    # Use shared expert output as a proxy for all experts
                    # (This is an approximation since we can't easily separate individual expert outputs)
                    if hasattr(layer, 'shared_expert'):
                        shared_out = layer.shared_expert(hidden_states.view(-1, hidden_dim))
                    else:
                        # Use the MoE output
                        moe_out, _ = layer(hidden_states)
                        shared_out = moe_out.view(-1, hidden_dim)
                    
                    # Weight the output by router scores to approximate per-expert activation
                    for expert_idx in range(layer.num_experts):
                        expert_weight = router_scores[:, expert_idx].unsqueeze(1)  # [batch*seq, 1]
                        weighted_out = shared_out * expert_weight
                        
                        # Accumulate
                        n_active = (expert_weight > 0.01).sum().item()  # Count significantly active tokens
                        if n_active > 0:
                            activ_sum[layer_name][expert_idx].add_(weighted_out.sum(dim=0).to("cpu"))
                            activ_cnt[layer_name][expert_idx] += n_active

    # Run calibration
    for model_inputs in calibration_data:
        local_inputs = {k: v.to(input_device, non_blocking=True) for k, v in model_inputs.items()}
        
        # For Llama4, we need to capture hidden states before MoE layers
        if llama4_layers:
            # Use hooks to capture inputs to MoE layers
            llama4_input_hooks = []
            llama4_inputs = {}
            
            def make_input_capture_hook(layer_name):
                def _hook(_module, inp, _output):
                    llama4_inputs[layer_name] = inp[0].detach() if isinstance(inp, tuple) else inp.detach()
                return _hook
            
            for name, layer in llama4_layers:
                llama4_input_hooks.append(layer.register_forward_hook(make_input_capture_hook(name)))
            
            _ = model(**local_inputs)
            
            # Process captured inputs
            for name, layer in llama4_layers:
                if name in llama4_inputs:
                    capture_llama4_activations(name, layer, llama4_inputs[name])
            
            # Remove temporary hooks
            for h in llama4_input_hooks:
                h.remove()
        else:
            _ = model(**local_inputs)

    for h in hooks:
        h.remove()

    # Compute means
    expert_means: Dict[str, List[torch.Tensor]] = {}
    for name in activ_sum.keys():
        E, H = activ_sum[name].shape
        means = []
        for i in range(E):
            cnt = int(activ_cnt[name][i].item())
            if cnt > 0:
                means.append(activ_sum[name][i] / float(cnt))
            else:
                means.append(torch.zeros(H, dtype=torch.float32))
        expert_means[name] = means
    return expert_means

# ---------------------------
# In-place PRUNE and MERGE
# ---------------------------

@torch.no_grad()
def prune_model_by_usage_inplace(model, expert_usage, compression_factor: float):
    """
    In-place architectural prune with ultra memory-efficient handling for Llama4.
    Deletes old parameters before creating new ones to avoid OOM.
    """
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return model

    print("\n--- ✂️ Architecturally Pruning Model (in-place) ---")
    new_E_global = None

    for name, layer in moe_layers:
        # how many experts exist?
        if hasattr(layer, 'num_experts'):
            E = int(layer.num_experts)
        elif hasattr(layer, 'experts') and hasattr(layer.experts, '__len__'):
            E = len(layer.experts)
        else:
            continue

        k = int(E * compression_factor)
        if k == 0 or name not in expert_usage:
            _clamp_top_k(layer, E)
            if new_E_global is None:
                new_E_global = E
            continue

        usage = expert_usage[name].to('cpu')
        idx_prune = torch.topk(usage, k=k, largest=False).indices.tolist()
        keep = [i for i in range(E) if i not in set(idx_prune)]

        # Handle different expert architectures
        if isinstance(layer.experts, torch.nn.ModuleList):
            # Standard ModuleList approach (Qwen3, Mixtral, ERNIE)
            layer.experts = torch.nn.ModuleList([layer.experts[i] for i in keep])
        elif Llama4TextExperts is not None and isinstance(layer.experts, Llama4TextExperts):
            # Llama4 parameter-based experts - ULTRA MEMORY EFFICIENT VERSION
            print(f"  Pruning Llama4 experts in layer '{name}'...")
            
            # Process gate_up_proj first
            old_gate_up = layer.experts.gate_up_proj
            device = old_gate_up.device
            dtype = old_gate_up.dtype
            
            # Move to CPU and slice
            with torch.cuda.device(device):
                old_gate_up_cpu = old_gate_up.detach().to('cpu', non_blocking=False)
                # Delete the old GPU tensor immediately
                del layer.experts.gate_up_proj
                del old_gate_up
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Slice on CPU
                new_gate_up_cpu = old_gate_up_cpu[keep].contiguous()
                del old_gate_up_cpu  # Free CPU memory
                
                # Move back to GPU and create parameter
                new_gate_up = new_gate_up_cpu.to(device, dtype=dtype, non_blocking=False)
                del new_gate_up_cpu  # Free CPU memory
                layer.experts.gate_up_proj = torch.nn.Parameter(new_gate_up)
                del new_gate_up
                torch.cuda.empty_cache()
            
            # Process down_proj
            old_down = layer.experts.down_proj
            
            with torch.cuda.device(device):
                old_down_cpu = old_down.detach().to('cpu', non_blocking=False)
                # Delete the old GPU tensor immediately
                del layer.experts.down_proj
                del old_down
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Slice on CPU
                new_down_cpu = old_down_cpu[keep].contiguous()
                del old_down_cpu  # Free CPU memory
                
                # Move back to GPU and create parameter
                new_down = new_down_cpu.to(device, dtype=dtype, non_blocking=False)
                del new_down_cpu  # Free CPU memory
                layer.experts.down_proj = torch.nn.Parameter(new_down)
                del new_down
                torch.cuda.empty_cache()
            
            layer.experts.num_experts = len(keep)
        else:
            print(f"⚠️ Warning: Cannot prune custom expert structure in layer '{name}'")
            continue

        # new expert count
        new_E = len(keep)
        layer.num_experts = new_E
        _clamp_top_k(layer, new_E)

        # shrink gate/router - FIXED FOR LLAMA4
        gate = layer.gate if hasattr(layer, 'gate') else getattr(layer, 'router', None)
        if gate is not None:
            device = gate.weight.data.device
            dtype = gate.weight.data.dtype
            
            # Check if this is a Llama4Router that needs special handling
            try:
                from transformers.models.llama4.modeling_llama4 import Llama4Router
                is_llama4_router = isinstance(gate, Llama4Router)
            except ImportError:
                is_llama4_router = False
            
            with torch.cuda.device(device):
                old_weight_cpu = gate.weight.data.detach().to('cpu', non_blocking=False)
                new_weight_cpu = old_weight_cpu[keep].contiguous()
                del old_weight_cpu
                
                if is_llama4_router:
                    # Create a new Llama4Router instance
                    class MinimalConfig:
                        def __init__(self, hidden_size, num_local_experts, num_experts_per_tok):
                            self.hidden_size = hidden_size
                            self.num_local_experts = num_local_experts
                            self.num_experts_per_tok = num_experts_per_tok
                    
                    config = MinimalConfig(
                        hidden_size=gate.in_features,
                        num_local_experts=new_E,
                        num_experts_per_tok=min(gate.top_k, new_E)
                    )
                    
                    new_gate = Llama4Router(config)
                else:
                    # For other router types, create a plain Linear layer
                    new_gate = torch.nn.Linear(gate.in_features, new_E, bias=False)
                    
                    # Preserve any attributes from the old router
                    if hasattr(gate, 'num_experts'):
                        new_gate.num_experts = new_E
                    if hasattr(gate, 'top_k'):
                        new_gate.top_k = min(gate.top_k, new_E)
                
                # Set the weights
                new_gate.weight.data = new_weight_cpu.to(device, dtype=dtype, non_blocking=False)
                del new_weight_cpu
                
                # Replace the gate/router
                if hasattr(layer, 'gate'):
                    del layer.gate
                    torch.cuda.empty_cache()
                    layer.gate = new_gate
                if hasattr(layer, 'router'):
                    del layer.router
                    torch.cuda.empty_cache()
                    layer.router = new_gate

        # shrink correction bias in moe_statics (ERNIE-specific)
        _shrink_moe_statics_for_keep(layer, keep)

        print(f"Layer '{name}': pruned {k}. New expert count: {new_E}")
        new_E_global = new_E
        
        # Force memory cleanup after each layer
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory._dump_snapshot'):
            torch.cuda.memory._dump_snapshot()

    # sync config (use last layer's new_E if changed)
    if new_E_global is not None:
        _update_config_num_experts(model, new_E_global)

    return model


@torch.no_grad()
def merge_model_by_similarity_inplace(model, expert_activations, compression_factor: float, merge_on_cpu: bool = False):
    """
    In-place architectural merge with ultra memory-efficient handling for Llama4.
    Processes parameters one at a time and deletes old ones before creating new.
    """
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return model

    print("\n--- 🤝 Architecturally Merging Model (in-place, zero-copy) ---")
    new_E_global = None

    for name, layer in moe_layers:
        # Determine expert count
        if hasattr(layer, 'num_experts'):
            E = int(layer.num_experts)
        elif hasattr(layer, 'experts') and hasattr(layer.experts, '__len__'):
            E = len(layer.experts)
        else:
            continue

        merges = int(E * compression_factor)
        if merges == 0:
            _clamp_top_k(layer, E)
            if new_E_global is None:
                new_E_global = E
            continue

        if name not in expert_activations or len(expert_activations[name]) != E:
            print(f"  ⚠ Layer '{name}': missing/size-mismatch activations; skipping.")
            _clamp_top_k(layer, E)
            if new_E_global is None:
                new_E_global = E
            continue

        # similarity on means
        acts = torch.stack([
            a if isinstance(a, torch.Tensor) else torch.tensor(a)
            for a in expert_activations[name]
        ])
        norms = acts.norm(dim=1)
        active = norms > 1e-10
        sim = torch.full((E, E), -1.0)
        idx_active = active.nonzero(as_tuple=True)[0].tolist()
        for i in idx_active:
            ai = acts[i]
            ni = ai.norm()
            if ni == 0:
                continue
            for j in idx_active:
                if i == j:
                    continue
                aj = acts[j]
                nj = aj.norm()
                if nj > 0:
                    sim[i, j] = F.cosine_similarity(ai.unsqueeze(0), aj.unsqueeze(0)).item()

        # pick pairs (greedy)
        pairs, used = [], set()
        for _ in range(merges):
            max_val = sim.max().item()
            if max_val <= -1:
                pool = [k for k in range(E) if k not in used]
                if len(pool) < 2:
                    break
                i, j = pool[0], pool[1]
            else:
                ii, jj = (sim == max_val).nonzero(as_tuple=False)[0].tolist()
                i, j = ii, jj
            if j < i:
                i, j = j, i
            if i in used or j in used or i == j:
                sim[i, :] = -1; sim[:, i] = -1
                sim[j, :] = -1; sim[:, j] = -1
                continue
            pairs.append((i, j))
            used.update([i, j])
            sim[i, :] = -1; sim[:, i] = -1
            sim[j, :] = -1; sim[:, j] = -1

        if not pairs:
            print(f"  Layer '{name}': nothing to merge.")
            _clamp_top_k(layer, E)
            if new_E_global is None:
                new_E_global = E
            continue

        gate = layer.gate if hasattr(layer, "gate") else getattr(layer, "router", None)
        assert gate is not None, f"Expected gate/router on MoE layer '{name}'"

        # Determine keep indices
        removed = set([j for (_, j) in pairs])
        keep = [k for k in range(E) if k not in removed]

        # Handle different expert architectures
        if isinstance(layer.experts, torch.nn.ModuleList):
            # Standard ModuleList approach (Qwen3, Mixtral, ERNIE)
            def _avg_(dst: torch.nn.Module, src: torch.nn.Module):
                for (p_dst, p_src) in zip(dst.parameters(), src.parameters()):
                    if merge_on_cpu:
                        cpu_sum = (p_dst.detach().to("cpu", copy=True) + p_src.detach().to("cpu", copy=True)) * 0.5
                        p_dst.data.copy_(cpu_sum.to(p_dst.device))
                    else:
                        p_dst.data.add_(p_src.data).mul_(0.5)

            # average experts
            for (i, j) in pairs:
                _avg_(layer.experts[i], layer.experts[j])
            
            layer.experts = torch.nn.ModuleList([layer.experts[k] for k in keep])
                
        elif Llama4TextExperts is not None and isinstance(layer.experts, Llama4TextExperts):
            # Llama4 parameter-based experts - ULTRA MEMORY EFFICIENT VERSION
            print(f"  Merging Llama4 experts in layer '{name}'...")
            
            # Process gate_up_proj
            gate_up = layer.experts.gate_up_proj
            device = gate_up.device
            dtype = gate_up.dtype
            
            with torch.cuda.device(device):
                # Move to CPU for processing
                gate_up_cpu = gate_up.detach().to('cpu', non_blocking=False)
                del layer.experts.gate_up_proj
                del gate_up
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Average merged pairs on CPU
                for (i, j) in pairs:
                    gate_up_cpu[i] = (gate_up_cpu[i] + gate_up_cpu[j]) * 0.5
                
                # Slice to keep only survivors
                new_gate_up_cpu = gate_up_cpu[keep].contiguous()
                del gate_up_cpu
                
                # Move back to GPU
                new_gate_up = new_gate_up_cpu.to(device, dtype=dtype, non_blocking=False)
                del new_gate_up_cpu
                layer.experts.gate_up_proj = torch.nn.Parameter(new_gate_up)
                del new_gate_up
                torch.cuda.empty_cache()
            
            # Process down_proj
            down = layer.experts.down_proj
            
            with torch.cuda.device(device):
                # Move to CPU for processing
                down_cpu = down.detach().to('cpu', non_blocking=False)
                del layer.experts.down_proj
                del down
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Average merged pairs on CPU
                for (i, j) in pairs:
                    down_cpu[i] = (down_cpu[i] + down_cpu[j]) * 0.5
                
                # Slice to keep only survivors
                new_down_cpu = down_cpu[keep].contiguous()
                del down_cpu
                
                # Move back to GPU
                new_down = new_down_cpu.to(device, dtype=dtype, non_blocking=False)
                del new_down_cpu
                layer.experts.down_proj = torch.nn.Parameter(new_down)
                del new_down
                torch.cuda.empty_cache()
            
            layer.experts.num_experts = len(keep)
        else:
            print(f"⚠️ Warning: Cannot merge custom expert structure in layer '{name}'")
            continue

        new_E = len(keep)
        layer.num_experts = new_E
        _clamp_top_k(layer, new_E)

        # Process gate/router weights - FIXED FOR LLAMA4
        gate_device = gate.weight.data.device
        gate_dtype = gate.weight.data.dtype
        
        # Check if this is a Llama4Router that needs special handling
        try:
            from transformers.models.llama4.modeling_llama4 import Llama4Router
            is_llama4_router = isinstance(gate, Llama4Router)
        except ImportError:
            is_llama4_router = False
        
        with torch.cuda.device(gate_device):
            # Move to CPU for operations
            gate_weight_cpu = gate.weight.data.detach().to('cpu', non_blocking=False)
            
            # Average merged pairs
            for (i, j) in pairs:
                gate_weight_cpu[i] = (gate_weight_cpu[i] + gate_weight_cpu[j]) * 0.5
            
            # Slice to keep only survivors
            new_weight_cpu = gate_weight_cpu[keep].contiguous()
            del gate_weight_cpu
            
            if is_llama4_router:
                # Create a new Llama4Router instance
                class MinimalConfig:
                    def __init__(self, hidden_size, num_local_experts, num_experts_per_tok):
                        self.hidden_size = hidden_size
                        self.num_local_experts = num_local_experts
                        self.num_experts_per_tok = num_experts_per_tok
                
                config = MinimalConfig(
                    hidden_size=gate.in_features,
                    num_local_experts=new_E,
                    num_experts_per_tok=min(gate.top_k, new_E)
                )
                
                new_gate = Llama4Router(config)
            else:
                # Create new standard Linear gate
                new_gate = torch.nn.Linear(gate.in_features, new_E, bias=False)
                
                if hasattr(gate, 'num_experts'):
                    new_gate.num_experts = new_E
                if hasattr(gate, 'top_k'):
                    new_gate.top_k = min(gate.top_k, new_E)
            
            # Set the weights
            new_gate.weight.data = new_weight_cpu.to(gate_device, dtype=gate_dtype, non_blocking=False)
            del new_weight_cpu
            
            # Replace gate/router
            if hasattr(layer, "gate"):
                del layer.gate
                torch.cuda.empty_cache()
                layer.gate = new_gate
            if hasattr(layer, "router"):
                del layer.router
                torch.cuda.empty_cache()
                layer.router = new_gate

        # shrink + average correction bias (ERNIE-specific)
        _shrink_moe_statics_for_merge(layer, keep, pairs)

        print(f"Layer '{name}': ✓ merged {len(pairs)} pairs. New count: {new_E} experts")
        new_E_global = new_E
        
        # Aggressive memory cleanup after each layer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if new_E_global is not None:
        _update_config_num_experts(model, new_E_global)

    return model


def create_pruned_router(old_router, new_E, keep_indices, device, dtype):
    """
    Create a new router with pruned weights, preserving the router type.
    For Llama4Router, we need to maintain the custom forward method.
    """
    # Check if this is a Llama4Router
    is_llama4_router = old_router.__class__.__name__ == 'Llama4Router'
    
    if is_llama4_router:
        # Create a new Llama4Router instance
        # We need to create a minimal config-like object for initialization
        class MinimalConfig:
            def __init__(self, hidden_size, num_local_experts, num_experts_per_tok):
                self.hidden_size = hidden_size
                self.num_local_experts = num_local_experts
                self.num_experts_per_tok = num_experts_per_tok
        
        config = MinimalConfig(
            hidden_size=old_router.in_features,
            num_local_experts=new_E,
            num_experts_per_tok=min(old_router.top_k, new_E)
        )
        
        # Import the Llama4Router class
        from transformers.models.llama4.modeling_llama4 import Llama4Router
        new_router = Llama4Router(config)
    else:
        # For other router types, create a plain Linear layer
        new_router = torch.nn.Linear(old_router.in_features, new_E, bias=False)
        
        # Preserve any attributes from the old router
        if hasattr(old_router, 'num_experts'):
            new_router.num_experts = new_E
        if hasattr(old_router, 'top_k'):
            new_router.top_k = min(old_router.top_k, new_E)
    
    # Copy the pruned weights
    old_weight_cpu = old_router.weight.data.detach().to('cpu', non_blocking=False)
    new_weight_cpu = old_weight_cpu[keep_indices].contiguous()
    new_router.weight.data = new_weight_cpu.to(device, dtype=dtype, non_blocking=False)
    
    return new_router

def create_merged_router(old_router, new_E, keep_indices, pairs, device, dtype):
    """
    Create a new router with merged weights, preserving the router type.
    """
    # Try to import Llama4Router
    try:
        from transformers.models.llama4.modeling_llama4 import Llama4Router
        is_llama4_router = isinstance(old_router, Llama4Router)
    except ImportError:
        is_llama4_router = False
    
    if is_llama4_router:
        # Create a new Llama4Router instance
        class MinimalConfig:
            def __init__(self, hidden_size, num_local_experts, num_experts_per_tok):
                self.hidden_size = hidden_size
                self.num_local_experts = num_local_experts
                self.num_experts_per_tok = num_experts_per_tok
        
        config = MinimalConfig(
            hidden_size=old_router.in_features,
            num_local_experts=new_E,
            num_experts_per_tok=min(old_router.top_k, new_E)
        )
        
        new_router = Llama4Router(config)
    else:
        new_router = torch.nn.Linear(old_router.in_features, new_E, bias=False)
        
        if hasattr(old_router, 'num_experts'):
            new_router.num_experts = new_E
        if hasattr(old_router, 'top_k'):
            new_router.top_k = min(old_router.top_k, new_E)
    
    # Process weights with merging
    with torch.cuda.device(device):
        gate_weight_cpu = old_router.weight.data.detach().to('cpu', non_blocking=False)
        
        # Average merged pairs
        for (i, j) in pairs:
            gate_weight_cpu[i] = (gate_weight_cpu[i] + gate_weight_cpu[j]) * 0.5
        
        # Slice to keep only survivors
        new_weight_cpu = gate_weight_cpu[keep_indices].contiguous()
        del gate_weight_cpu
        new_router.weight.data = new_weight_cpu.to(device, dtype=dtype, non_blocking=False)
        del new_weight_cpu
        torch.cuda.empty_cache()
    
    return new_router

# ---------------------------
# Metrics & Visualizations
# ---------------------------

def compute_collapse_metrics(activations_dict):
    """Compute PCA-based metrics for each layer."""
    metrics = {}
    for layer_name, layer_acts in activations_dict.items():
        try:
            data = torch.stack(layer_acts).to(torch.float32).numpy()
        except Exception:
            try:
                data = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in layer_acts]).to(torch.float32).numpy()
            except Exception:
                continue

        if data.ndim != 2 or np.var(data) < 1e-10:
            metrics[layer_name] = {'variance_ratio':0,'spread':0,'entropy':0,'convex_hull_area':0}
            continue

        pca = PCA(n_components=min(2, data.shape[1], data.shape[0]))
        data_2d = pca.fit_transform(data)
        variance_ratio = sum(pca.explained_variance_ratio_[:2]) if len(pca.explained_variance_ratio_) >= 2 else pca.explained_variance_ratio_[0]
        spread = np.sqrt(np.var(data_2d[:,0]) + (np.var(data_2d[:,1]) if data_2d.shape[1] > 1 else 0))

        if data_2d.shape[0] > 1:
            hist, _, _ = np.histogram2d(data_2d[:,0], data_2d[:,1] if data_2d.shape[1] > 1 else np.zeros_like(data_2d[:,0]), bins=10)
            hist = hist / (hist.sum() + 1e-12)
            hflat = hist.flatten(); hflat = hflat[hflat > 0]
            ent = entropy(hflat)
        else:
            ent = 0

        if data_2d.shape[1] >= 2 and data_2d.shape[0] >= 3:
            try:
                hull = ConvexHull(data_2d)
                hull_area = hull.volume
            except Exception:
                hull_area = 0
        else:
            hull_area = 0

        metrics[layer_name] = {
            'variance_ratio': float(variance_ratio),
            'spread': float(spread),
            'entropy': float(ent),
            'convex_hull_area': float(hull_area),
        }
    return metrics

def plot_collapse_progression(original_metrics, pruned_metrics, merged_metrics, model_name_str, output_dir: Path):
    print("\n--- 📈 Generating Collapse Progression Analysis ---")
    def _layer_idx(name):
        parts = name.split('.')
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    layer_names = sorted(list(original_metrics.keys()), key=_layer_idx)
    layer_indices = [_layer_idx(n) for n in layer_names]
    metrics_to_plot = ['spread', 'entropy', 'convex_hull_area']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Functional Collapse Progression Across Layers - {model_name_str.split('/')[-1]}", fontsize=16, y=1.02)

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        o = np.array([original_metrics[n].get(metric_name, 0) for n in layer_names])
        p = np.array([pruned_metrics.get(n, {}).get(metric_name, 0) for n in layer_names])
        m = np.array([merged_metrics.get(n, {}).get(metric_name, 0) for n in layer_names])

        o_safe = np.where(o > 1e-10, o, 1.0)
        pr = p / o_safe
        mr = m / o_safe

        ax.plot(layer_indices, np.ones_like(layer_indices), label='Original', linewidth=2, alpha=0.7)
        ax.plot(layer_indices, pr, label='Pruned', linewidth=2, marker='o', markersize=4)
        ax.plot(layer_indices, mr, label='Merged', linewidth=2, marker='s', markersize=4)

        if len(layer_indices) > 1:
            ax.plot(layer_indices, np.poly1d(np.polyfit(layer_indices, pr, 1))(layer_indices), 'r--', alpha=0.3, linewidth=1)
            ax.plot(layer_indices, np.poly1d(np.polyfit(layer_indices, mr, 1))(layer_indices), 'g--', alpha=0.3, linewidth=1)

        ax.set_ylabel(f'Relative {metric_name.replace("_"," ").title()}')
        if idx == 2: ax.set_xlabel('Layer Index')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        for k, val in enumerate(mr):
            if val < 0.5:
                ax.axvspan(layer_indices[k]-0.5, layer_indices[k]+0.5, alpha=0.08, color='red')

    plt.tight_layout()
    fname = output_dir / "metrics" / "collapse_progression_analysis.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Collapse progression analysis saved to '{fname}'")

def visualize_single_layer_subspace(original_acts, pruned_acts, merged_acts, layer_name, compression_factor, model_name_str, output_dir: Path):
    if layer_name not in original_acts:
        return
    try:
        pca = PCA(n_components=2)
        O = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in original_acts[layer_name]]).float().numpy()
        if O.shape[0] < 2:
            return
        P = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in pruned_acts.get(layer_name, [])]).float().numpy() if pruned_acts.get(layer_name) else np.array([[]])
        M = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in merged_acts.get(layer_name, [])]).float().numpy() if merged_acts.get(layer_name) else np.array([[]])

        O2 = pca.fit_transform(O)
        P2 = pca.transform(P) if P.size > 0 else np.array([[]])
        M2 = pca.transform(M) if M.size > 0 else np.array([[]])

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        clean_name = layer_name.replace('model.', '').replace('.mlp', '')

        fig.suptitle(f"{model_name_str.split('/')[-1]} | Layer: {clean_name} ({int((1-compression_factor)*100)}% of original)", fontsize=18, y=0.99)

        # Panel 1: Original (blue)
        axes[0].scatter(O2[:,0], O2[:,1], alpha=0.8, s=50, c='blue', label='Original Experts')
        axes[0].set_title("1. Original")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].legend()

        # Panel 2: Pruned (red for surviving experts)
        axes[1].scatter(O2[:,0], O2[:,1], c='grey', alpha=0.15, s=50)  # ghost of original
        if P2.size > 0:
            axes[1].scatter(P2[:,0], P2[:,1], c='red', s=60, label='Surviving')
        axes[1].set_title("2. Pruned")
        axes[1].set_xlabel("PC1")
        axes[1].legend()

        # Panel 3: Merged (green for merged experts)
        axes[2].scatter(O2[:,0], O2[:,1], c='grey', alpha=0.15, s=50)  # ghost of original
        if M2.size > 0:
            axes[2].scatter(M2[:,0], M2[:,1], c='green', marker='X', s=90, linewidth=2, label='Merged')
        axes[2].set_title("3. Merged")
        axes[2].set_xlabel("PC1")
        axes[2].legend()

        try:
            idx = int(layer_name.split('.')[-2])
            fname = output_dir / "layer_analysis" / f"layer_{idx:03d}_{clean_name.replace('.','_')}.png"
        except Exception:
            fname = output_dir / "layer_analysis" / f"{clean_name.replace('.','_')}.png"

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved visualization for layer '{clean_name}' to {fname}")
    except Exception as e:
        print(f"  ⚠ Error visualizing layer '{layer_name}': {e}")


def visualize_all_functional_subspaces(original_acts, pruned_acts, merged_acts, compression_factor, model_name_str, output_dir: Path, plot_all_layers=False):
    if plot_all_layers:
        print(f"\n--- 📊 Generating Functional Subspace Analysis for All Layers ---")
        layer_names = list(original_acts.keys())
        def _idx(n):
            parts = n.split('.')
            for p in reversed(parts):
                if p.isdigit(): return int(p)
            return 0
        layer_names.sort(key=_idx)
        for ln in layer_names:
            visualize_single_layer_subspace(original_acts, pruned_acts, merged_acts, ln, compression_factor, model_name_str, output_dir)

    if original_acts:
        first_layer_name = next(iter(original_acts))
        summary_dir = Path(str(output_dir).replace("/layer_analysis", "/visualizations"))
        visualize_single_layer_subspace(original_acts, pruned_acts, merged_acts, first_layer_name, compression_factor, model_name_str, summary_dir)

# ---------------------------
# Router policy (sequential)
# ---------------------------

def _router_metrics_for_model(model, calibration_data, tag):
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return None
    num_experts = moe_layers[0][1].num_experts
    all_logits = []

    input_device = next(model.parameters()).device
    with torch.no_grad():
        for model_inputs in calibration_data:
            local_inputs = {k: v.to(input_device, non_blocking=True) for k, v in model_inputs.items()}
            out = model(**local_inputs, output_router_logits=True)
            if hasattr(out, "router_logits") and out.router_logits:
                all_logits.append(out.router_logits[0].view(-1, num_experts).cpu())

    if not all_logits:
        return None

    x = torch.cat(all_logits, dim=0).float().numpy()
    var = np.var(x, axis=0)
    active = var > 1e-6
    if not np.any(active):
        return None
    fim_proxy = np.corrcoef(x[:, active], rowvar=False)
    off = fim_proxy[~np.eye(fim_proxy.shape[0], dtype=bool)]
    coupling = np.mean(np.abs(off))
    eig = np.linalg.eigvalsh(fim_proxy)
    eig = np.maximum(eig, 1e-9)
    eff_rank = (eig.sum() ** 2) / (eig ** 2).sum()
    return {"offdiag": off, "coupling": coupling, "eff_rank": eff_rank, "E": num_experts}

def analyze_and_visualize_router_policy_sequential(model_name, calibration_data, model_name_str, output_dir: Path, compression_factor: float, merge_on_cpu: bool):
    print("\n--- 📈 Router Policy Analysis (sequential) ---")
    names, metrics = [], []

    def _load(name):
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        mdl.eval()
        return tok, mdl

    # Original
    tok, mdl = _load(model_name)
    m = _router_metrics_for_model(mdl, calibration_data, "Original")
    del tok, mdl; torch.cuda.empty_cache()
    if m is not None:
        names.append("Original"); metrics.append(m)

    # Pruned
    tok, mdl = _load(model_name)
    usage = get_expert_usage(mdl, calibration_data)
    prune_model_by_usage_inplace(mdl, usage, compression_factor)
    m = _router_metrics_for_model(mdl, calibration_data, "Pruned")
    del tok, mdl; torch.cuda.empty_cache()
    if m is not None:
        names.append("Pruned"); metrics.append(m)

    # Merged
    tok, mdl = _load(model_name)
    acts = get_expert_activations_online(mdl, calibration_data)
    merge_model_by_similarity_inplace(mdl, acts, compression_factor, merge_on_cpu=merge_on_cpu)
    m = _router_metrics_for_model(mdl, calibration_data, "Merged")
    del tok, mdl; torch.cuda.empty_cache()
    if m is not None:
        names.append("Merged"); metrics.append(m)

    if not metrics:
        warnings.warn("No router metrics collected.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Router Policy Analysis for {model_name_str.split('/')[-1]}", fontsize=20, y=0.98)

    ax1 = axes[0]
    for tag, mm in zip(names, metrics):
        sns.kdeplot(mm["offdiag"], ax=ax1, label=tag, fill=True, alpha=0.5, linewidth=2.5)
    ax1.set_title("Distribution of Off-Diagonal Coupling"); ax1.set_xlabel("Correlation"); ax1.legend(); ax1.grid(True); ax1.axvline(0, ls="--")

    ax2 = axes[1]
    bars = ax2.bar(names, [mm["coupling"] for mm in metrics])
    ax2.set_title("Mean |Off-diagonal|"); ax2.set_ylabel("Lower is better"); ax2.bar_label(bars, fmt="%.4f"); ax2.grid(axis="y")

    ax3 = axes[2]
    norm_rank = [mm["eff_rank"]/mm["E"] for mm in metrics]
    bars3 = ax3.bar(names, norm_rank)
    ax3.set_title("Effective Rank / #Experts"); ax3.set_ylim(0, 1.05); ax3.bar_label(bars3, fmt="%.3f"); ax3.grid(axis="y")

    out_png = output_dir / "visualizations" / "router_policy_analysis.png"
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✅ Router Policy Analysis saved to '{out_png}'")

# ---------------------------
# Fisher Information (sequential)
# ---------------------------

def parse_layer_idxs(spec: str, m: int):
    if spec.strip().lower() == "auto":
        return sorted({0, m // 2, m - 1})
    idxs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok: continue
        i = int(tok)
        if i < 0: i = m + i
        if 0 <= i < m: idxs.append(i)
    return sorted(set(idxs))

def compute_fisher_information_matrix(model, calibration_data, num_samples=10, layer_indices=None):
    print("\n--- 🔬 Computing Fisher Information Matrix ---")
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        return {}

    m = len(moe_layers)
    if layer_indices is None:
        layer_indices = sorted({0, m // 2, m - 1})

    fim_data = {}
    fim_samples = calibration_data[:min(num_samples, len(calibration_data))]
    input_device = next(model.parameters()).device

    for li in layer_indices:
        if li < 0: li = m + li
        if not (0 <= li < m): continue

        layer_name, layer = moe_layers[li]
        print(f"  Computing FIM for layer {layer_name} (idx {li})...")

        if not hasattr(layer, "gate"):
            print(f"    ⚠ Layer {layer_name} has no 'gate'; skipping.")
            continue

        num_experts = int(layer.num_experts)
        per_sample_grad_mats = []

        layer.gate.weight.requires_grad_(True)
        for sample in fim_samples:
            local_inputs = {k: v.to(input_device, non_blocking=True) for k, v in sample.items()}
            outputs = model(**local_inputs, output_router_logits=True)
            if not hasattr(outputs, "router_logits") or outputs.router_logits is None:
                continue
            if li >= len(outputs.router_logits):
                continue

            router_logits = outputs.router_logits[li]
            gate_probs = F.softmax(router_logits, dim=-1)
            if gate_probs.shape[-1] != num_experts:
                continue

            sample_grad_rows = []
            for expert_idx in range(num_experts):
                if layer.gate.weight.grad is not None:
                    layer.gate.weight.grad.zero_()
                model.zero_grad(set_to_none=True)

                probs_e = gate_probs[..., expert_idx]
                if probs_e.ndim > 1:
                    probs_e = probs_e.reshape(-1)
                log_prob = torch.log(probs_e + 1e-10).mean()

                grad_w = torch.autograd.grad(
                    log_prob, layer.gate.weight,
                    retain_graph=True, allow_unused=True, create_graph=False
                )[0]

                if grad_w is None:
                    sample_grad_rows.append(torch.zeros(layer.gate.weight.shape[1], device="cpu"))
                else:
                    sample_grad_rows.append(grad_w[expert_idx].detach().cpu())

            G_s = torch.stack(sample_grad_rows, dim=0)  # [E, H]
            per_sample_grad_mats.append(G_s)

        layer.gate.weight.requires_grad_(False)

        if not per_sample_grad_mats:
            print(f"    ⚠ No gradients collected for {layer_name}; skipping.")
            continue

        E = len(per_sample_grad_mats)
        fim = torch.zeros(num_experts, num_experts)
        for G_s in per_sample_grad_mats:
            fim += G_s @ G_s.T
        fim /= E
        fim_data[layer_name] = fim.numpy()

    return fim_data

def compute_fim_for_variants_sequential(model_name, calibration_data, layer_idxs, fim_samples, compression_factor, merge_on_cpu, output_dir: Path, model_tag: str):
    """
    Compute FIMs for Original, Pruned, Merged by reloading base each time.
    Saves a visualization and metrics file.
    """
    def _load():
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        mdl.eval()
        return tok, mdl

    # Original
    tok, mdl = _load()
    original_fim = compute_fisher_information_matrix(mdl, calibration_data, num_samples=fim_samples, layer_indices=layer_idxs)
    del tok, mdl; torch.cuda.empty_cache()

    # Pruned
    tok, mdl = _load()
    usage = get_expert_usage(mdl, calibration_data)
    prune_model_by_usage_inplace(mdl, usage, compression_factor)
    pruned_fim = compute_fisher_information_matrix(mdl, calibration_data, num_samples=fim_samples, layer_indices=layer_idxs)
    del tok, mdl; torch.cuda.empty_cache()

    # Merged
    tok, mdl = _load()
    acts = get_expert_activations_online(mdl, calibration_data)
    merge_model_by_similarity_inplace(mdl, acts, compression_factor, merge_on_cpu=merge_on_cpu)
    merged_fim = compute_fisher_information_matrix(mdl, calibration_data, num_samples=fim_samples, layer_indices=layer_idxs)
    del tok, mdl; torch.cuda.empty_cache()

    visualize_fisher_information_matrices(original_fim, pruned_fim, merged_fim, model_tag, output_dir)

def visualize_fisher_information_matrices(original_fim, pruned_fim, merged_fim, model_name_str, output_dir: Path):
    print("\n--- 📊 Visualizing Fisher Information Matrices ---")
    if not original_fim:
        print("  No original FIM data available."); return

    # Choose first available layer for visualization
    layer_name = list(original_fim.keys())[0]
    fim_orig = original_fim[layer_name]
    fim_prun_raw = pruned_fim.get(layer_name, np.zeros((1,1)))
    fim_merg_raw = merged_fim.get(layer_name, np.zeros((1,1)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Fisher Information Matrix Analysis - {model_name_str.split('/')[-1]}\nLayer: {layer_name}", fontsize=14, y=1.02)

    def _show(ax, title, fim):
        if fim.size == 1 and fim.shape == (1,1) and title != "Original":
            ax.set_title(title); ax.axis("off"); return
        denom = np.max(np.abs(fim)) + 1e-10
        im = ax.imshow(fim / denom, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(title); ax.set_xlabel('Expert Index'); ax.set_ylabel('Expert Index')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.plot([0, fim.shape[0]-1], [0, fim.shape[1]-1], 'k--', alpha=0.3, linewidth=1)

    _show(axes[0,0], "Original", fim_orig)
    _show(axes[0,1], "Pruned", fim_prun_raw)
    _show(axes[0,2], "Merged", fim_merg_raw)

    axes[1,0].axis("off"); axes[1,1].axis("off"); axes[1,2].axis("off")
    plt.tight_layout()
    filename = output_dir / "fim_analysis" / "fisher_information_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()

    # Save metrics
    metrics_file = output_dir / "fim_analysis" / "fim_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("Fisher Information Matrix Metrics\n")
        f.write("="*60 + "\n")
        f.write(f"{'Variant':<15} | {'Frobenius Norm':<20} | {'Off-Diag Mean':<20} | {'Condition Number':<20}\n")
        f.write("-"*75 + "\n")

        def _metrics(fim):
            if fim.size == 0:
                return (0.0, 0.0, float('inf'))
            frob = np.linalg.norm(fim, 'fro')
            off_mask = ~np.eye(fim.shape[0], dtype=bool)
            off_mean = np.mean(np.abs(fim[off_mask])) if np.any(off_mask) else 0.0
            try:
                cond = np.linalg.cond(fim)
            except Exception:
                cond = np.inf
            return (frob, off_mean, cond)

        for name, fim in [("Original", fim_orig), ("Pruned", fim_prun_raw), ("Merged", fim_merg_raw)]:
            frob, off_mean, cond = _metrics(fim)
            f.write(f"{name:<15} | {frob:<20.4f} | {off_mean:<20.6f} | {cond:<20.2e}\n")
        f.write("="*60 + "\n")

    print(f"✅ Fisher Information analysis saved to '{filename}'")
    print(f"✅ FIM metrics saved to '{metrics_file}'")

# ---------------------------
# Utilities
# ---------------------------

def _load_model_and_tokenizer(name, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map=device_map, trust_remote_code=True)
    mdl.eval()
    return tok, mdl

# ---------------------------
# Main
# ---------------------------

def main(args):
    # Better CUDA allocator behavior for long runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    output_dir = create_output_directory(args.model_name, args.base_dir)

     # ---- ORIGINAL (collect stats) ----
    print(f"Loading model '{args.model_name}'... (This may take several minutes and require significant VRAM)")
    tokenizer, model = _load_model_and_tokenizer(args.model_name, device_map="auto")
    print(f"Model loaded. Main input device: {next(model.parameters()).device}")

    calibration_data = get_calibration_data(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        num_sequences=args.num_calibration_sequences,
        seq_length=args.sequence_length
    )

    if not args.plots_only:

        # Save run config
        with open(output_dir / "analysis_config.txt", "w") as f:
            f.write(f"Model Name: {args.model_name}\n")
            f.write(f"Compression Factor: {args.compression_factor}\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Num Calibration Sequences: {args.num_calibration_sequences}\n")
            f.write(f"Sequence Length: {args.sequence_length}\n")
            f.write(f"Plot All Layers: {args.plot_all_layers}\n")
            f.write(f"Compute FIM: {args.compute_fim}\n")
            f.write(f"FIM Layer Idxs: {args.fim_layer_idxs}\n")
            f.write(f"FIM Samples: {args.fim_samples}\n")
            f.write(f"Merge on CPU: {args.merge_on_cpu}\n")


        print("\n--- Step 1: Analyzing the Original Model ---")
        original_expert_usage = get_expert_usage(model, calibration_data)
        if not original_expert_usage:
            print("Analysis halted as no MoE layers were found."); return
        original_expert_activations = get_expert_activations_online(model, calibration_data)

        # free original model before proceeding
        del tokenizer, model
        torch.cuda.empty_cache()

        # ---- PRUNED ----
        print("\n--- Step 2A: Building PRUNED model (in-place) ---")
        tokenizer, pruned_model = _load_model_and_tokenizer(args.model_name, device_map="auto")
        prune_model_by_usage_inplace(pruned_model, original_expert_usage, args.compression_factor)
        pruned_expert_activations = get_expert_activations_online(pruned_model, calibration_data)
        del tokenizer, pruned_model
        torch.cuda.empty_cache()

        # ---- MERGED ----
        print("\n--- Step 2B: Building MERGED model (in-place) ---")
        tokenizer, merged_model = _load_model_and_tokenizer(args.model_name, device_map="auto")
        merge_model_by_similarity_inplace(merged_model, original_expert_activations, args.compression_factor, merge_on_cpu=args.merge_on_cpu)
        merged_expert_activations = get_expert_activations_online(merged_model, calibration_data)
        del tokenizer, merged_model
        torch.cuda.empty_cache()

        # save outputs
        data = {
            "original_expert_usage": original_expert_usage,
            "original_expert_activations": original_expert_activations,
            "pruned_expert_activations": pruned_expert_activations,
            "merged_expert_activations": merged_expert_activations,
        }
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        torch.save(data, output_dir / "data" / "collected_data.pt")
    
    if args.plots_only:
        # free original model before proceeding
        del tokenizer, model
        torch.cuda.empty_cache()

        print("Skipping to visualizations as --plots_only was set.")
        data = torch.load(output_dir / "data" / "collected_data.pt")
        original_expert_usage = data["original_expert_usage"]
        original_expert_activations = data["original_expert_activations"]
        pruned_expert_activations = data["pruned_expert_activations"]
        merged_expert_activations = data["merged_expert_activations"]



    # ---- Visualizations that need only activations ----
    print("\n--- Step 3: Generating Visualizations ---")
    visualize_all_functional_subspaces(
        original_expert_activations,
        pruned_expert_activations,
        merged_expert_activations,
        args.compression_factor,
        args.model_name,
        output_dir,
        plot_all_layers=args.plot_all_layers
    )

    # Router policy (sequential reloads)
    analyze_and_visualize_router_policy_sequential(
        args.model_name, calibration_data, args.model_name, output_dir,
        compression_factor=args.compression_factor, merge_on_cpu=args.merge_on_cpu
    )

    # Collapse progression
    if args.plot_collapse_progression:
        orig_m = compute_collapse_metrics(original_expert_activations)
        prun_m = compute_collapse_metrics(pruned_expert_activations)
        merg_m = compute_collapse_metrics(merged_expert_activations)
        plot_collapse_progression(orig_m, prun_m, merg_m, args.model_name, output_dir)

    # FIM (sequential reloads)
    if args.compute_fim:
        # Determine layer idxs against a small, temporary load to get m
        tok, mdl = _load_model_and_tokenizer(args.model_name, device_map="auto")
        m = len(get_moe_layers(mdl))
        del tok, mdl; torch.cuda.empty_cache()
        layer_idxs = parse_layer_idxs(args.fim_layer_idxs, m)
        compute_fim_for_variants_sequential(
            args.model_name, calibration_data, layer_idxs, args.fim_samples,
            args.compression_factor, args.merge_on_cpu, output_dir, args.model_name
        )

    # Final summary
    with open(output_dir / "analysis_complete.txt", "w") as f:
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total MoE layers analyzed: {len(original_expert_usage)}\n")

    print("\n" + "="*60)
    print(f"✅ Analysis complete!")
    print(f"📁 All outputs saved to: {output_dir}")
    print("="*60)
    if args.plot_all_layers:
        print(f"📊 Layer visualizations: {output_dir / 'layer_analysis'}")
    print(f"📈 Main visualizations: {output_dir / 'visualizations'}")
    print(f"📉 Metrics and summaries: {output_dir / 'metrics'}")
    if args.compute_fim:
        print(f"🔬 FIM analysis: {output_dir / 'fim_analysis'}")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare pruning vs. merging for MoE models using C4 calibration data.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="MoE model from Hugging Face Hub. Supports Qwen3, Mixtral, Llama4, and ERNIE models.")
    parser.add_argument("--base_dir", type=str, default="artifacts/moe_functional_subspace_analysis_outputs")
    parser.add_argument("--compression_factor", type=float, default=0.5,
                        help="Target compression ratio (0.5 = reduce to 50% of original expert count).")
    parser.add_argument("--dataset_name", type=str, default="allenai/c4",
                        help="Calibration dataset from Hugging Face Hub.")
    parser.add_argument("--num_calibration_sequences", type=int, default=32,
                        help="Number of sequences to generate for calibration.")
    parser.add_argument("--sequence_length", type=int, default=2048,
                        help="Token length of each calibration sequence.")
    parser.add_argument("--plot_all_layers", action="store_true",
                        help="Generate functional subspace plots for all MoE layers.")
    parser.add_argument("--plot_collapse_progression", action="store_true",
                        help="Plot the progression of functional collapse across all layers.")
    parser.add_argument("--compute_fim", action="store_true",
                        help="Compute and visualize Fisher Information Matrices to verify coupling predictions.")
    parser.add_argument("--fim_layer_idxs", type=str, default="auto",
                        help="Comma-separated layer indices (e.g., '0,5,-1') or 'auto' for {first, middle, last}.")
    parser.add_argument("--fim_samples", type=int, default=10,
                        help="Number of samples to use for FIM computation (default: 10).")
    parser.add_argument("--merge_on_cpu", action="store_true",
                        help="Average expert and gate params on CPU during merge to minimize VRAM spikes.")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip analysis and load previously saved data to regenerate plots only.")

    args = parser.parse_args()
    main(args)


    """
# 8 h100s required
CUDA_VISIBLE_DEVIES=01,2,3,4,5,6,7 python scripts/moe_functional_subspace_analysis.py --model_name "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
                --compression_factor 0.5 \
                --plot_all_layers 

# 2 h100s required
CUDA_VISIBLE_DEVICES=2,3 python scripts/moe_functional_subspace_analysis.py --model_name "baidu/ERNIE-4.5-21B-A3B-PT" \
                 --compression_factor 0.5 \
                 --plot_all_layers

# 4 h100s required
CUDA_VISIBLE_DEVIES=2,3,4,5 python scripts/moe_functional_subspace_analysis.py --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
                 --compression_factor 0.5 \
                 --plot_all_layers

# 8 h100s required
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/moe_functional_subspace_analysis.py --model_name "Qwen/Qwen3-30B-A3B" \
                 --compression_factor 0.5 \
                 --plot_all_layers
    """