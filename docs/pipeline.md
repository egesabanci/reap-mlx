# Pipeline

The CLI pipeline lives in `reap.entrypoint.main`. It is written as a sequence of
small phases with injectable functions so tests can run the full orchestration
without loading real models, downloading datasets, or importing MLX.

## Phase Overview

```txt
model_load
  -> calibration
  -> observe
  -> prune
  -> save_reload_smoke
  -> metrics write
```

| Phase | Main function | Responsibility | Output |
| --- | --- | --- | --- |
| `model_load` | `_default_load_model` | Load model, tokenizer, and config through MLX-LM. | `(model, tokenizer, config)` |
| `calibration` | `load_calibration_sequences` | Load text records, tokenize, truncate, and keep non-empty samples. | `list[{"input_ids": np.ndarray}]` |
| `observe` | `observe_model` | Replay layers and collect selected-route expert statistics. | `dict[layer_idx, observer_report]` |
| `prune` | `prune_experts` | Rank experts, slice expert-stacked tensors, update runtime attrs and config. | `dict[layer_idx, keep_indices]` |
| `save_reload_smoke` | `save_pruned_model` | Save artifact, reload it, validate shapes, optionally generate. | `SaveReloadResult` |
| metrics write | `RunMetrics.write` | Finalize success or failure telemetry. | `validation-metrics.json` |

## Model Loading

The default loader imports MLX-LM only inside `_default_load_model`.

Current MLX-LM loaders are called with:

```python
mlx_lm.load(model_name, return_config=True)
```

When the installed loader does not support `return_config`, the fallback calls:

```python
mlx_lm.load(model_name)
mlx_lm.utils.get_model_path(model_name)
mlx_lm.utils.load_config(model_path)
```

The pipeline requires the final config to be a mapping. Adapter inference then
uses config metadata and model layout to select `lfm2_moe` or `qwen3_moe`.

## Calibration

Calibration uses the loaded tokenizer and a Hugging Face dataset. It loads the
requested split, optionally passes a dataset config name, shuffles when the
dataset object exposes `.shuffle(seed=...)`, extracts text, tokenizes, truncates,
and drops empty sequences.

The observer expects unpadded batch-size-1 sequences. The loader returns
one-dimensional `np.int32` arrays, which the observer converts to `[1, seq]`.

See [Calibration](calibration.md) for supported record shapes.

## Observation

Observation performs explicit layer replay rather than registering hooks. For
each calibration sequence:

1. Convert `input_ids` to an MLX array.
2. Add a batch dimension when needed.
3. Run embeddings.
4. Replay every layer in order.
5. For dense layers, run the dense MLP path.
6. For MoE layers, route tokens, call `switch_mlp` for selected experts, and
   accumulate selected-output statistics.
7. Call `mx.eval(h)` after every layer by default.

Qwen3-style models use standard attention replay. LFM2-style models choose an
attention mask or SSM/conv mask based on each layer's operator type.

Observation returns reports for MoE layers only.

## Pruning

Pruning is in-place. It does not create a new model object.

For each adapter-visible MoE layer:

1. Resolve the requested pruning method.
2. Read the per-expert saliency vector from observer data.
3. Compute retained count as:

   ```txt
   retained_count = max(num_experts - int(num_experts * compression_ratio), 1)
   ```

4. Keep the highest-saliency experts using deterministic tie-breaking by lower
   expert id.
5. Return keep indices in ascending order.
6. Slice switch projections, router gate fields, and LFM2 expert bias when
   applicable.
7. Update live `num_experts` and top-k attributes.
8. Update global config `num_experts` and `num_experts_per_tok`.

The implementation requires every pruned MoE layer to retain the same expert
count because the saved config stores a single global `num_experts`.

## Save, Reload, And Smoke

Saving delegates to `mlx_lm.utils.save` with the mutated model and mutated
config. After saving, validation requires:

- `config.json` exists;
- at least one `*.safetensors` or `*.npz` weight artifact exists;
- the saved model reloads successfully;
- reloaded config `num_experts` matches the expected expert count;
- every adapter-visible MoE layer exposes switch projections and router gate
  weights with first dimension equal to the expected expert count;
- LFM2 `expert_bias` first dimension is valid when present or enabled.

The optional smoke test runs on the reloaded model, not the original model.
`--no-smoke` disables it.

## Error Handling

Argument validation happens before pipeline functions run. Invalid compression
ratios, pruning methods, sample counts, and sequence lengths fail with parser
exit code 2.

`KeyboardInterrupt` returns exit code 130 and does not write success telemetry.

For other runtime exceptions, the pipeline samples memory, writes failed
telemetry with the current phase and exception summary, then re-raises.

