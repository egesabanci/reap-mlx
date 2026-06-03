# Telemetry

Every CLI run writes structured validation telemetry through
`reap.validation_metrics.RunMetrics`.

Default path:

```txt
<output-dir>/validation-metrics.json
```

If `--metrics-file` is an absolute path, telemetry is written there instead.

## Top-Level Shape

```json
{
  "status": "success",
  "started_at": "...",
  "finished_at": "...",
  "duration_seconds": 0.0,
  "model": {},
  "runtime": {},
  "run_config": {},
  "memory": {"samples": {}},
  "timings": {"phases": {}, "phase_percentages": {}},
  "throughput": {},
  "observer": {},
  "pruning": {},
  "save_reload": {},
  "smoke": {},
  "failure": null
}
```

Status is `running` while in memory, then `success` or `failed` when written.

## Runtime

Runtime metadata includes:

- full Python version;
- Python executable;
- platform, machine, processor;
- package versions for `mlx`, `mlx-lm`, `datasets`, `huggingface-hub`, and
  `safetensors`;
- process memory sample.

Package versions can be `null` when packages are not installed in the active
environment.

## Run Config

Run config records user-facing CLI inputs:

- `model_name`
- `dataset_name`
- `dataset_config_name`
- `split`
- `seed`
- `max_samples`
- `max_seq_length`
- `eval_frequency`
- `prune_method`
- `resolved_prune_method`
- `compression_ratio`
- `output_dir`
- `metrics_file`
- `smoke_enabled`

After calibration, it also records actual sample and token counts:

- `actual_sample_count`
- `actual_total_tokens`
- `actual_token_counts`
- `actual_min_tokens`
- `actual_max_tokens`
- `actual_mean_tokens`

## Model

Model telemetry includes adapter-visible metadata:

- model name, revision, and source path when available;
- adapter name;
- model type and architectures;
- quantization metadata;
- hidden size;
- configured and adapter-visible layer counts;
- MoE and dense layer indices;
- expert count and top-k before pruning;
- expert count and top-k after pruning;
- `norm_topk_prob`;
- `use_expert_bias`;
- first MoE layer shape summary before pruning.

The shape summary captures gate fields, expert bias shape, and switch projection
field shapes when available.

## Memory

Memory samples are taken at named pipeline points:

- `start`
- `after_model_load`
- `after_calibration`
- `after_observe`
- `after_prune`
- `after_save_reload_smoke`
- `failure` when an exception is handled

Each sample includes:

- timestamp;
- MLX memory when `mlx.core` is available;
- process max RSS from `resource.getrusage`.

MLX memory keys are:

- `active_bytes`
- `peak_bytes`
- `cache_bytes`

When MLX is unavailable or sampling fails, the sample records availability or
error information instead of raising.

## Timings

Phase timings are recorded for:

- `model_load`
- `calibration`
- `observe`
- `prune`
- `save_reload_smoke`

`save_pruned_model` also reports nested timings:

- `save_seconds`
- `reload_seconds`
- `reload_validation_seconds`
- `smoke_seconds` when smoke is enabled

`phase_percentages` divides each phase time by total run duration.

## Observer

Observer telemetry is a summary, not a full dump of every metric array.

Top-level observer fields:

- `observed_moe_layer_count`
- `observed_moe_layer_indices`
- `total_input_tokens`
- `per_layer`

Per layer:

- `total_tokens`
- `expert_frequency_sum`
- `saliency_key`
- `saliency_count`
- `saliency_finite_count`
- `saliency_non_finite_count`
- `saliency_min`
- `saliency_max`
- `saliency_mean`

Non-finite saliency values are counted and JSON-safe output replaces non-finite
floating values with `null` where needed.

## Pruning

Pruning telemetry includes:

- pruned layer count;
- total experts removed;
- expert count before and after;
- top-k before and after;
- whether top-k was clamped;
- per-layer original, retained, removed counts;
- retained expert indices.

Use this section to compare pruning decisions across methods and calibration
sets.

## Save Reload

Save/reload telemetry includes:

- output directory;
- expected expert count;
- reloaded config expert count;
- reloaded adapter-visible MoE layer indices;
- artifact summary;
- save/reload timings;
- reloaded shape summary.

Artifact summary includes file count, total bytes, and per-file byte counts.

## Smoke

Smoke telemetry records:

- whether smoke was enabled;
- whether it completed;
- prompt;
- max tokens;
- elapsed seconds;
- generated token count when tokenizer encoding is available;
- result preview.

The result preview is truncated to 500 characters.

## Throughput

Derived throughput fields include:

- calibration samples per second;
- calibration tokens per second;
- observer input tokens per second;
- observer layer-tokens per second;
- observer layers per second;
- observer seconds per layer;
- pruning layers per second;
- pruning experts per second;
- save MB per second;
- reload MB per second;
- generation tokens per second.

Rates are `null` when the numerator or denominator is missing or the elapsed
time is zero.

## Failure Payload

On failure, telemetry status is `failed` and `failure` contains:

- `phase`
- `type`
- `message`
- `elapsed_seconds_before_failure`
- `memory_at_failure`

The original exception is re-raised after telemetry is written.
