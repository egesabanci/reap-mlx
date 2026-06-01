# `prune_experts()` Mutation Behavior

## Function Overview

`prune_experts()` in `src/reap/prune.py` **mutates the config dict in-place** via `update_qwen3_moe_config()` or `update_lfm2_moe_config()`.

## Caller Safety

The caller in `entrypoint.py` snapshots the config before invocation:

```python
config_before_prune = dict(config)
```

This mutation is currently undocumented and should be noted for developers working on the pruning module to avoid unintended side effects.