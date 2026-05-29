import argparse
from matplotlib.patches import bbox_artist
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, set_start_method
from collections import Counter

# --- UTILITY AND PREPARATION FUNCTIONS ---

FIGSIZE=(4.5, 6)

COLOR_MAP = {
    "Baseline": "#999999",
    "REAP": "#4477aa",
    "M-SMoE": "#66ccee",
    "HC-SMoE": "#ccbb44",
}

def map_fn(sample, tokenizer):
    """Prepares a sample for logit collection."""
    prompt_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["instruction"]}],
        add_generation_prompt=True, tokenize=False
    )
    teacher_forced_input = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ],
        add_generation_prompt=True, tokenize=False
    )
    return {
        "prompt_input": prompt_input,
        "teacher_forced_input": teacher_forced_input,
        "output": sample["output"],
    }

def get_model_outputs(model, tokenizer, mapped_dataset, num_samples=100):
    """Collects logits, generated tokens, and hidden states for a given model."""
    model_outputs = {"generated_logits": [], "teacher_forced_logits": [], "generated_ids": []}
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc=f"Processing samples for {model.config._name_or_path.split('/')[-1]}"):
            sample = mapped_dataset[idx]
            
            prompt_tokens = tokenizer(sample["prompt_input"], return_tensors="pt").to(model.device)
            prompt_length = prompt_tokens.input_ids.shape[1]

            # --- Generated outputs (logits and tokens) ---
            generated_output = model.generate(
                **prompt_tokens,
                max_new_tokens=256,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # Use greedy decoding for reproducibility
            )
            # Stack logits: (num_generated_tokens, vocab_size)
            stacked_generated_logits = torch.stack(generated_output.logits, dim=1).squeeze(0)
            model_outputs["generated_logits"].append(stacked_generated_logits.cpu())
            model_outputs["generated_ids"].append(generated_output.sequences[0, prompt_length:].cpu())

            # --- Teacher-forced logits ---
            teacher_forced_tokens = tokenizer(sample["teacher_forced_input"], return_tensors="pt").to(model.device)
            teacher_forced_output = model(**teacher_forced_tokens, output_logits=True)
            # Get logits only for the response part
            teacher_logits = teacher_forced_output.logits[0, prompt_length - 1 : -1, :].cpu()
            model_outputs["teacher_forced_logits"].append(teacher_logits)
            
    return model_outputs

# --- CORE ANALYSIS FUNCTIONS ---

def calculate_entropy(logits_list, model_type, output_type):
    """Calculates entropy for a set of logits (vectorized)."""
    records = []
    for sample_idx, logits in enumerate(logits_list):
        if logits.numel() == 0: continue
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        for pos, ent_val in enumerate(entropy):
            records.append({
                "entropy": ent_val.item(), "model_type": model_type,
                "output_type": output_type, "seq": sample_idx, "pos": pos
            })
    return pd.DataFrame(records)

def calculate_divergences(base_logits_list, other_logits_list, model_type):
    """Calculates Forward KL, Reverse KL, JS, and Wasserstein divergences."""
    records = []
    for sample_idx, (base_logits, other_logits) in enumerate(zip(base_logits_list, other_logits_list)):
        trunc_len = min(base_logits.shape[0], other_logits.shape[0])
        if trunc_len == 0: continue
        
        base_logits, other_logits = base_logits[:trunc_len], other_logits[:trunc_len]
        
        p_dist = torch.softmax(base_logits, dim=-1)  # Base model distribution (P)
        q_dist = torch.softmax(other_logits, dim=-1) # Other model distribution (Q)
        
        kld_forward = torch.nn.functional.kl_div(q_dist.log(), p_dist, reduction='none').sum(dim=-1)
        kld_reverse = torch.nn.functional.kl_div(p_dist.log(), q_dist, reduction='none').sum(dim=-1)
        
        m_dist = 0.5 * (p_dist + q_dist)
        jsd = 0.5 * (torch.nn.functional.kl_div(m_dist.log(), p_dist, reduction='none').sum(dim=-1) + \
                     torch.nn.functional.kl_div(m_dist.log(), q_dist, reduction='none').sum(dim=-1))

        p_cdf = torch.cumsum(p_dist, dim=-1)
        q_cdf = torch.cumsum(q_dist, dim=-1)
        wasserstein = torch.sum(torch.abs(p_cdf - q_cdf), dim=-1)

        for pos in range(trunc_len):
            records.append({
                "kld_forward": kld_forward[pos].item(),
                "kld_reverse": kld_reverse[pos].item(),
                "jsd": jsd[pos].item(), 
                "wasserstein": wasserstein[pos].item(),
                "model_type": model_type, "seq": sample_idx, "pos": pos
            })
    return pd.DataFrame(records)

def calculate_ngram_diversity(generated_ids_list, model_type):
    """Calculates n-gram diversity for generated token sequences."""
    records = []
    for n in [2, 3, 4]:
        for sample_idx, token_ids in enumerate(generated_ids_list):
            if len(token_ids) < n: continue
            ngrams = [tuple(token_ids[i:i+n].tolist()) for i in range(len(token_ids) - n + 1)]
            if not ngrams: continue
            diversity = len(set(ngrams)) / len(ngrams)
            records.append({
                "diversity": diversity, "n_gram": n, "model_type": model_type, "sample_idx": sample_idx
            })
    return pd.DataFrame(records)

def calculate_cross_perplexity(base_model, tokenizer, generated_ids_dict, device):
    """Calculates cross-perplexity of generated text under the base model."""
    base_model.to(device)
    records = []
    with torch.no_grad():
        for model_type, id_list in generated_ids_dict.items():
            if model_type == 'base': continue
            for sample_idx, token_ids in tqdm(enumerate(id_list), desc=f"Cross-PPL for {model_type}"):
                if len(token_ids) == 0: continue
                loss = base_model(input_ids=token_ids.unsqueeze(0).to(device), labels=token_ids.unsqueeze(0).to(device)).loss
                perplexity = torch.exp(loss).item()
                records.append({
                    "cross_perplexity": perplexity, "model_type": model_type, "sample_idx": sample_idx
                })
    return pd.DataFrame(records)
    
# --- NEW ANALYSIS FUNCTION ---
def calculate_running_cross_perplexity(base_model, generated_ids_dict, device):
    """Calculates running cross-perplexity of generated text under the base model."""
    base_model.to(device)
    records = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for model_type, id_list in generated_ids_dict.items():
            if model_type == 'base': continue
            for sample_idx, token_ids in tqdm(enumerate(id_list), desc=f"Running Cross-PPL for {model_type}"):
                if len(token_ids) < 2: continue
                
                input_ids = token_ids.unsqueeze(0).to(device)
                outputs = base_model(input_ids)
                
                # Shift logits and labels for Causal LM loss calculation
                logits = outputs.logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                
                # Get loss for each token
                per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Calculate cumulative average loss and then perplexity
                cumulative_loss = torch.cumsum(per_token_loss, dim=0)
                token_counts = torch.arange(1, len(per_token_loss) + 1, device=device)
                running_avg_loss = cumulative_loss / token_counts
                running_ppl = torch.exp(running_avg_loss)
                
                for pos, ppl in enumerate(running_ppl):
                    records.append({
                        "running_perplexity": ppl.item(),
                        "model_type": model_type,
                        "sample_idx": sample_idx,
                        "pos": pos
                    })
    return pd.DataFrame(records)


# --- PLOTTING FUNCTIONS ---

def plot_results(df_dict, output_dir):
    """Creates and saves all plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Entropy Plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_dict["entropy"], x="pos", y="entropy", hue="model_type", style="output_type")
    plt.title("Entropy of Logits vs. Token Position"); plt.xlabel("Token Position"); plt.ylabel("Entropy")
    plt.xlim(0, 256); plt.legend(title="Model & Output"); plt.savefig(os.path.join(output_dir, "entropy.png")); plt.close()

    # Divergence Plots
    for metric in ["kld_forward", "kld_reverse", "jsd", "wasserstein"]:
        skip_tokens = 3
        plt.figure(figsize=(18, 6))
        div_df = df_dict["divergence"].copy()
        div_df = div_df.loc[div_df["pos"] >= skip_tokens]
        div_df['pos'] = div_df['pos'] - skip_tokens
        color_palette = [COLOR_MAP.get(model) for model in df_dict["divergence"]["model_type"].unique()]
        sns.lineplot(data=div_df, x="pos", y=metric, hue="model_type", palette=color_palette)
        if metric == "kld_forward": title = "Forward KL Divergence D_KL(Base || Other) vs. Position"
        elif metric == "kld_reverse": title = "Reverse KL Divergence D_KL(Other || Base) vs. Position"
        else: title = f"{metric.upper()} vs. Token Position (vs. Base Model)"
        # plt.title(title)
        plt.xlabel("Token Position")
        if metric == "jsd":
            plt.ylabel("JSD vs. baseline logits")
        else:
            plt.ylabel(metric.replace("_", " ").upper())
        plt.xlim(0, 25)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, f"{metric}.pdf"), bbox_inches="tight")
        plt.close()

    # small divergence plots
    for metric in ["kld_forward", "kld_reverse", "jsd", "wasserstein"]:
        skip_tokens = 3
        plt.figure(figsize=(4.5, 6))
        div_df = df_dict["divergence"].copy()
        div_df = div_df.loc[div_df["pos"] >= skip_tokens]
        div_df['pos'] = div_df['pos'] - skip_tokens
        color_palette = [COLOR_MAP.get(model) for model in df_dict["divergence"]["model_type"].unique()]
        sns.lineplot(data=div_df, x="pos", y=metric, hue="model_type", palette=color_palette)
        if metric == "kld_forward": title = "Forward KL Divergence D_KL(Base || Other) vs. Position"
        elif metric == "kld_reverse": title = "Reverse KL Divergence D_KL(Other || Base) vs. Position"
        else: title = f"{metric.upper()} vs. Token Position (vs. Base Model)"
        # plt.title(title);
        plt.xlabel("Token Position"); plt.ylabel(metric.replace("_", " ").upper())
        plt.xlim(0, 25)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric}_small.png"), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, f"{metric}_small.pdf"), bbox_inches="tight")
        plt.close()

    # N-gram Diversity Plot
    plt.figure(figsize=FIGSIZE)
    df_dict['diversity']['model_type'] = df_dict['diversity']['model_type'].astype(pd.CategoricalDtype(categories=COLOR_MAP.keys(), ordered=True))
    df_dict['diversity'] = df_dict['diversity'].sort_values(by=['model_type', 'n_gram'])

    color_palette = [COLOR_MAP.get(model) for model in df_dict["diversity"]["model_type"].unique()]
    try:
        sns.boxplot(data=df_dict["diversity"], x="n_gram", y="diversity", hue="model_type", palette=color_palette)
    except Exception as e:
        import pdb; breakpoint()
    # plt.title("N-gram Diversity of Generated Text");
    plt.xlabel("N-gram size")
    plt.ylabel("N-gram diversity")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[1:], labels=labels[1:])
    ax.legend(handles=handles[:], labels=labels[:], frameon=True)
    plt.savefig(os.path.join(output_dir, "ngram_diversity.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "ngram_diversity.pdf"), bbox_inches="tight")
    plt.close()
    
    # Cross-Perplexity (Violin) Plot
    plt.figure(figsize=FIGSIZE)
    model_order = ["REAP", "M-SMoE", "HC-SMoE"]
    filtered_df = df_dict["cross_perplexity"][df_dict["cross_perplexity"]["model_type"].isin(model_order)]
    palette = {model: COLOR_MAP.get(model) for model in model_order}
    sns.violinplot(data=filtered_df, x="model_type", y="cross_perplexity", order=model_order, palette=palette, zorder=2)

    # plt.title("Cross-Perplexity of Generated Text (under Base Model)");
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=18) 
    plt.xlabel("Generator model"); plt.ylabel("Cross perplexity")
    plt.yscale("log")
    ax = plt.gca()
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5,zorder=1)
    plt.savefig(os.path.join(output_dir, "cross_perplexity.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "cross_perplexity.pdf"), bbox_inches="tight")
    plt.close()
    
    # Running Cross-Perplexity (Line) Plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_dict["running_cross_perplexity"], x="pos", y="running_perplexity", hue="model_type")
    plt.xlabel("Token Position"); plt.ylabel("Running Perplexity")
    plt.yscale("log"); plt.legend(title="Generator Model"); plt.grid(True, which="both", ls="--")
    plt.xlim(0, 256)
    plt.savefig(os.path.join(output_dir, "running_cross_perplexity.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "running_cross_perplexity.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"All plots saved to {output_dir}")

def process_model(args):
    """Loads a model and collects outputs in a separate process."""
    model_type, model_path, device, tokenizer_path, dataset_name, num_samples = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(dataset_name, split="train")
    mapped_dataset = dataset.map(lambda sample: map_fn(sample, tokenizer))
    
    print(f"Loading {model_type} from {model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    outputs = get_model_outputs(model, tokenizer, mapped_dataset, num_samples)
    del model; torch.cuda.empty_cache()
    return model_type, outputs

def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of auto-regressive model collapse.")
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--reap_model_dir", type=str, required=False)
    parser.add_argument("--m_smoe_model_dir", type=str, required=False)
    parser.add_argument("--hc_smoe_model_dir", type=str, required=False)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="artifacts/fig/generation_quality_analysis/qwen-generation_quality_analysis-all-0.50")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plots without running analyses.")
    args = parser.parse_args()

    if torch.cuda.device_count() < 4 and not args.plot_only:
        raise ValueError("This script is optimized for at least 4 GPUs.")
    set_start_method("spawn", force=True)
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    dataset_name = "theblackcat102/evol-codealpaca-v1"
    tasks = [
        ("Baseline", args.base_model_name, devices[0], args.base_model_name, dataset_name, args.num_samples),
        ("REAP", args.reap_model_dir, devices[1], args.base_model_name, dataset_name, args.num_samples),
        ("M-SMoE", args.m_smoe_model_dir, devices[2], args.base_model_name, dataset_name, args.num_samples),
        ("HC-SMoE", args.hc_smoe_model_dir, devices[3], args.base_model_name, dataset_name, args.num_samples),
    ]

    if not args.plot_only:
        with Pool(processes=len(tasks)) as pool:
            results = pool.map(process_model, tasks)
        
        all_outputs = {model_type: output for model_type, output in results}
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(all_outputs, os.path.join(args.output_dir, "all_model_outputs.pt"))

        # --- Run All Analyses ---
        print("Running analyses...")
        entropy_dfs, divergence_dfs, diversity_dfs = [], [], []
        
        model_types = ["Baseline", "REAP", "M-SMoE", "HC-SMoE"]
        for model_type in model_types:
            entropy_dfs.append(calculate_entropy(all_outputs[model_type]["generated_logits"], model_type, "generated"))
            entropy_dfs.append(calculate_entropy(all_outputs[model_type]["teacher_forced_logits"], model_type, "teacher_forced"))
            diversity_dfs.append(calculate_ngram_diversity(all_outputs[model_type]["generated_ids"], model_type))

            if model_type != "Baseline":
                divergence_dfs.append(calculate_divergences(
                    all_outputs["Baseline"]["generated_logits"], 
                    all_outputs[model_type]["generated_logits"], 
                    f"{model_type} vs Baseline"
                ))

        # --- Load base model once for perplexity calculations ---
        print("Loading base model for perplexity calculations...")
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=False, trust_remote_code=True)
        
        print("Calculating cross-perplexity...")
        cross_ppl_df = calculate_cross_perplexity(
            base_model, tokenizer, {m_type: out["generated_ids"] for m_type, out in all_outputs.items() if m_type != "Baseline"}, devices[0]
        )
        
        print("Calculating running cross-perplexity...")
        running_cross_ppl_df = calculate_running_cross_perplexity(
            base_model, {m_type: out["generated_ids"] for m_type, out in all_outputs.items() if m_type != "Baseline"}, devices[0]
        )
        
        del base_model; torch.cuda.empty_cache()

        # --- Consolidate and Save DataFrames ---
        df_dict = {
            "entropy": pd.concat(entropy_dfs, ignore_index=True),
            "divergence": pd.concat(divergence_dfs, ignore_index=True),
            "diversity": pd.concat(diversity_dfs, ignore_index=True),
            "cross_perplexity": cross_ppl_df,
            "running_cross_perplexity": running_cross_ppl_df,
        }
        for name, df in df_dict.items():
            df.to_csv(os.path.join(args.output_dir, f"{name}_results.csv"), index=False)
        print(f"Analysis dataframes saved to {args.output_dir}")
    else:
        print("Plotting only mode enabled. Skipping analyses.")
        df_dict = {
            "entropy": pd.read_csv(os.path.join(args.output_dir, "entropy_results.csv")),
            "divergence": pd.read_csv(os.path.join(args.output_dir, "divergence_results.csv")),
            "diversity": pd.read_csv(os.path.join(args.output_dir, "diversity_results.csv")),
            "cross_perplexity": pd.read_csv(os.path.join(args.output_dir, "cross_perplexity_results.csv")),
            "running_cross_perplexity": pd.read_csv(os.path.join(args.output_dir, "running_cross_perplexity_results.csv")),
        }
        map_names = True # Names are already correct from generation
        if map_names:
            name_map = {
                "WMEAN (ours)": "REAP",
                "WMEAN (ours) vs Baseline": "REAP",
                "M-SMoE vs Baseline": "M-SMoE",
                "HC-SMoE vs Baseline": "HC-SMoE",
            }
            df_dict["entropy"]["model_type"] = df_dict["entropy"]["model_type"].apply(lambda x: name_map.get(x, x))
            df_dict["divergence"]["model_type"] = df_dict["divergence"]["model_type"].apply(lambda x: name_map.get(x, x))
            df_dict["diversity"]["model_type"] = df_dict["diversity"]["model_type"].apply(lambda x: name_map.get(x, x))
            df_dict["cross_perplexity"]["model_type"] = df_dict["cross_perplexity"]["model_type"].apply(lambda x: name_map.get(x, x))
            df_dict["running_cross_perplexity"]["model_type"] = df_dict["running_cross_perplexity"]["model_type"].apply(lambda x: name_map.get(x, x))

    # --- Plot Results ---

    plot_results(df_dict, os.path.join(args.output_dir, "plots"))

if __name__ == "__main__":
    plt.style.use("config/plt_plot_style.mplstyle")
    main()