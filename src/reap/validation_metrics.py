"""Structured validation telemetry for MLX pruning runs.

The helpers in this module are intentionally import-light. Optional runtime
packages are imported only inside collection methods so the package keeps
its no-heavy-import package boundary.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import resource
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from reap.model_adapters import infer_model_adapter
from reap.prune import resolve_prune_method
from reap.save import artifact_summary

logger = logging.getLogger(__name__)


_PACKAGE_VERSION_NAMES = (
    "mlx",
    "mlx-lm",
    "datasets",
    "huggingface-hub",
    "safetensors",
)


@dataclass
class RunMetrics:
    """Mutable structured metrics for one MLX pruning CLI run."""

    output_dir: str | Path
    metrics_file: str | Path = "validation-metrics.json"
    clock: Any = time.perf_counter
    data: dict[str, Any] = field(init=False)
    _started_seconds: float = field(init=False)
    _phase_starts: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.metrics_file = Path(self.metrics_file)
        self._started_seconds = float(self.clock())
        self.data = {
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "duration_seconds": None,
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
            "failure": None,
        }

    @property
    def path(self) -> Path:
        """Return the destination path for the metrics JSON artifact."""
        if self.metrics_file.is_absolute():
            return self.metrics_file
        return self.output_dir / self.metrics_file

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a named phase."""
        self.start_phase(name)
        try:
            yield
        finally:
            self.finish_phase(name)

    def start_phase(self, name: str) -> None:
        """Mark ``name`` as currently running."""
        self._phase_starts[name] = float(self.clock())

    def finish_phase(self, name: str) -> float:
        """Finish a phase and return elapsed seconds."""
        started = self._phase_starts.pop(name, None)
        if started is None:
            return 0.0

        elapsed = max(float(self.clock()) - started, 0.0)
        self.data["timings"]["phases"][name] = elapsed
        return elapsed

    def sample_memory(self, label: str) -> dict[str, Any]:
        """Record MLX and process memory for a named point in the run."""
        sample = {
            "timestamp": _utc_now(),
            "mlx": _sample_mlx_memory(),
            "process": _sample_process_memory(),
        }
        self.data["memory"]["samples"][label] = sample
        return sample

    def record_runtime(self) -> None:
        """Record Python, platform, and package version metadata."""
        self.data["runtime"] = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "package_versions": {
                name: _package_version(name) for name in _PACKAGE_VERSION_NAMES
            },
            "process": _sample_process_memory(),
        }

    def record_run_config(self, args: Any) -> None:
        """Record user-facing CLI configuration."""
        model_name = getattr(args, "model_name", None)
        self.data["run_config"].update(
            {
                "model_name": model_name,
                "dataset_name": getattr(args, "dataset_name", None),
                "dataset_config_name": getattr(args, "dataset_config_name", None),
                "split": getattr(args, "split", None),
                "seed": getattr(args, "seed", None),
                "max_samples": getattr(args, "max_samples", None),
                "max_seq_length": getattr(args, "max_seq_length", None),
                "eval_frequency": getattr(args, "eval_frequency", None),
                "prune_method": getattr(args, "prune_method", None),
                "resolved_prune_method": _resolve_method_or_none(
                    getattr(args, "prune_method", None)
                ),
                "compression_ratio": getattr(args, "compression_ratio", None),
                "output_dir": str(getattr(args, "output_dir", self.output_dir)),
                "metrics_file": str(getattr(args, "metrics_file", self.metrics_file)),
                "smoke_enabled": not bool(getattr(args, "no_smoke", False)),
            }
        )
        self.data["model"]["model_name"] = model_name

    def record_model_metadata(
        self,
        model: Any,
        config: Mapping[str, Any],
        *,
        adapter: Any | None = None,
    ) -> None:
        """Record model config and adapter-visible architecture facts."""
        adapter = _safe_adapter(model, config, adapter)
        layers = _safe_layers(adapter, model)
        moe_layer_indices = _safe_moe_layers(adapter, model)
        dense_layer_indices = [
            idx for idx in range(len(layers)) if idx not in set(moe_layer_indices)
        ]
        before_layer_config = _first_layer_config(adapter, layers, config)

        self.data["model"].update(
            {
                "model_name": self.data["model"].get("model_name")
                or config.get("_name_or_path"),
                "model_revision": config.get("_commit_hash") or config.get("revision"),
                "model_source_path": _first_attr(
                    model,
                    "model_path",
                    "path",
                    "name_or_path",
                ),
                "adapter_name": getattr(adapter, "adapter_name", None),
                "model_type": config.get("model_type"),
                "architectures": _json_safe(config.get("architectures")),
                "quantization": _json_safe(config.get("quantization")),
                "quantization_config": _json_safe(config.get("quantization_config")),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers", len(layers)),
                "layer_count": len(layers) if layers else config.get("num_hidden_layers"),
                "layer_types": _json_safe(config.get("layer_types")),
                "moe_layer_indices": moe_layer_indices,
                "dense_layer_indices": dense_layer_indices,
                "moe_layer_count": len(moe_layer_indices),
                "expert_count_before": _layer_config_value(
                    before_layer_config,
                    "num_experts",
                    config.get("num_experts"),
                ),
                "top_k_before": _layer_config_value(
                    before_layer_config,
                    "top_k",
                    config.get("num_experts_per_tok", config.get("top_k")),
                ),
                "norm_topk_prob": _layer_config_value(
                    before_layer_config,
                    "norm_topk_prob",
                    config.get("norm_topk_prob"),
                ),
                "use_expert_bias": _layer_config_value(
                    before_layer_config,
                    "use_expert_bias",
                    config.get("use_expert_bias"),
                ),
                "first_moe_shapes_before": _first_moe_shape_summary(
                    adapter,
                    layers,
                ),
            }
        )

    def record_calibration(self, calibration_sequences: list[Any]) -> None:
        """Record sample and token counts for loaded calibration sequences."""
        token_counts = [_sequence_length(sequence) for sequence in calibration_sequences]
        total_tokens = sum(token_counts)
        self.data["run_config"].update(
            {
                "actual_sample_count": len(calibration_sequences),
                "actual_total_tokens": total_tokens,
                "actual_token_counts": token_counts,
                "actual_min_tokens": min(token_counts) if token_counts else 0,
                "actual_max_tokens": max(token_counts) if token_counts else 0,
                "actual_mean_tokens": (
                    total_tokens / len(token_counts) if token_counts else 0.0
                ),
            }
        )

    def record_observer(
        self,
        observer_data: Mapping[int, Mapping[str, Any]],
        prune_method: str,
    ) -> None:
        """Record per-layer observer summary and saliency finite counts."""
        resolved_method = _resolve_method_or_none(prune_method) or prune_method
        per_layer: dict[str, Any] = {}
        total_tokens = 0

        for layer_idx, layer_data in sorted(observer_data.items()):
            layer_key = str(layer_idx)
            layer_tokens = int(layer_data.get("total_tokens", 0))
            total_tokens += layer_tokens
            frequency = _numeric_list(layer_data.get("expert_frequency", []))
            saliency = _numeric_list(layer_data.get(resolved_method, []))
            finite_saliency = [
                value for value in saliency if isinstance(value, float) and math.isfinite(value)
            ]
            non_finite_count = len(saliency) - len(finite_saliency)

            per_layer[layer_key] = {
                "total_tokens": layer_tokens,
                "expert_frequency_sum": int(sum(frequency)),
                "saliency_key": resolved_method,
                "saliency_count": len(saliency),
                "saliency_finite_count": len(finite_saliency),
                "saliency_non_finite_count": non_finite_count,
                "saliency_min": min(finite_saliency) if finite_saliency else None,
                "saliency_max": max(finite_saliency) if finite_saliency else None,
                "saliency_mean": (
                    sum(finite_saliency) / len(finite_saliency)
                    if finite_saliency
                    else None
                ),
            }

        self.data["observer"] = {
            "observed_moe_layer_count": len(observer_data),
            "observed_moe_layer_indices": [int(idx) for idx in sorted(observer_data)],
            "total_input_tokens": total_tokens,
            "per_layer": per_layer,
        }

    def record_pruning(
        self,
        keep_by_layer: Mapping[int, Any] | None,
        *,
        config_before: Mapping[str, Any],
        config_after: Mapping[str, Any],
        observer_data: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Record retained experts and before/after expert counts."""
        keep_by_layer = {} if keep_by_layer is None else keep_by_layer
        per_layer: dict[str, Any] = {}
        total_removed = 0

        for layer_idx, keep_indices in sorted(keep_by_layer.items()):
            keep_list = [int(idx) for idx in _list_like(keep_indices)]
            original_count = _original_expert_count(
                observer_data.get(int(layer_idx), {}),
                config_before,
            )
            removed_count = max(original_count - len(keep_list), 0)
            total_removed += removed_count
            per_layer[str(layer_idx)] = {
                "original_expert_count": original_count,
                "retained_expert_count": len(keep_list),
                "removed_expert_count": removed_count,
                "retained_indices": keep_list,
            }

        top_k_before = config_before.get(
            "num_experts_per_tok",
            config_before.get("top_k"),
        )
        top_k_after = config_after.get("num_experts_per_tok", config_after.get("top_k"))
        self.data["pruning"] = {
            "layer_count": len(keep_by_layer),
            "total_experts_removed": total_removed,
            "expert_count_before": config_before.get("num_experts"),
            "expert_count_after": config_after.get("num_experts"),
            "top_k_before": top_k_before,
            "top_k_after": top_k_after,
            "top_k_was_clamped": (
                top_k_before is not None
                and top_k_after is not None
                and int(top_k_after) < int(top_k_before)
            ),
            "per_layer": per_layer,
        }
        self.data["model"]["expert_count_after"] = config_after.get("num_experts")
        self.data["model"]["top_k_after"] = top_k_after

    def record_save_reload(
        self,
        result: Any,
        *,
        adapter: Any | None = None,
    ) -> None:
        """Record save/reload artifact, shape validation, and smoke metrics."""
        output_dir = Path(getattr(result, "output_dir", self.output_dir))
        result_metrics = getattr(result, "metrics", None) or {}
        artifact_result = result_metrics.get("artifacts") or artifact_summary(
            output_dir
        )
        reloaded_config = getattr(result, "reloaded_config", {}) or {}
        reloaded_model = getattr(result, "reloaded_model", None)
        reloaded_moe_layers = _safe_moe_layers(adapter, reloaded_model)

        self.data["save_reload"] = {
            "output_dir": str(output_dir),
            "expected_expert_count": getattr(result, "expected_expert_count", None),
            "reloaded_config_expert_count": reloaded_config.get("num_experts"),
            "reloaded_adapter_moe_layer_count": len(reloaded_moe_layers),
            "reloaded_adapter_moe_layer_indices": reloaded_moe_layers,
            "artifacts": artifact_result,
            "timings": result_metrics.get("timings", {}),
            "shape_summary": _reloaded_shape_summary(adapter, reloaded_model),
        }

        self.data["smoke"] = result_metrics.get(
            "smoke",
            {
                "enabled": getattr(result, "smoke_result", None) is not None,
                "completed": getattr(result, "smoke_result", None) is not None,
                "result_preview": _preview(getattr(result, "smoke_result", None)),
            },
        )

    def write(
        self,
        *,
        status: str,
        failure: Mapping[str, Any] | None = None,
    ) -> Path:
        """Finalize and write the metrics JSON artifact."""
        self.data["status"] = status
        self.data["finished_at"] = _utc_now()
        self.data["duration_seconds"] = max(
            float(self.clock()) - self._started_seconds,
            0.0,
        )
        self.data["failure"] = _json_safe(failure)
        self._derive_phase_percentages()
        self._derive_throughput()

        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        # _json_safe should already replace non-finite floats with None, but
        # guard against any residual NaN/Inf that slipped through (e.g. via
        # custom objects) so a metrics write can never crash the failure
        # telemetry path itself.
        payload = _strip_nonfinite(_json_safe(self.data))
        try:
            text = json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False
            )
        except ValueError:
            logger.warning(
                "metrics payload still contained non-finite values after "
                "sanitization; writing with allow_nan=True to preserve telemetry",
            )
            text = json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=True
            )
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def failure_payload(self, phase: str, exc: BaseException) -> dict[str, Any]:
        """Build a JSON-safe failure record for ``exc``."""
        return {
            "phase": phase,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "elapsed_seconds_before_failure": max(
                float(self.clock()) - self._started_seconds,
                0.0,
            ),
            "memory_at_failure": self.data["memory"]["samples"].get("failure"),
        }

    def _derive_phase_percentages(self) -> None:
        total = float(self.data.get("duration_seconds") or 0.0)
        phases = self.data["timings"]["phases"]
        percentages = {}
        for name, seconds in phases.items():
            percentages[name] = (seconds / total * 100.0) if total > 0 else None
        self.data["timings"]["phase_percentages"] = percentages

    def _derive_throughput(self) -> None:
        phases = self.data["timings"]["phases"]
        run_config = self.data["run_config"]
        observer = self.data["observer"]
        pruning = self.data["pruning"]
        save_reload = self.data["save_reload"]
        smoke = self.data["smoke"]

        sample_count = int(run_config.get("actual_sample_count") or 0)
        total_tokens = int(run_config.get("actual_total_tokens") or 0)
        observed_layers = int(observer.get("observed_moe_layer_count") or 0)
        pruning_layers = int(pruning.get("layer_count") or 0)
        experts_removed = int(pruning.get("total_experts_removed") or 0)
        artifact_bytes = int(
            (save_reload.get("artifacts") or {}).get("total_bytes") or 0
        )

        calibration_seconds = phases.get("calibration", 0.0)
        observe_seconds = phases.get("observe", 0.0)
        prune_seconds = phases.get("prune", 0.0)
        save_reload_timings = save_reload.get("timings") or {}
        smoke_seconds = smoke.get("elapsed_seconds")
        generated_tokens = smoke.get("generated_token_count")

        self.data["throughput"] = {
            "calibration_samples_per_second": _rate(sample_count, calibration_seconds),
            "calibration_tokens_per_second": _rate(total_tokens, calibration_seconds),
            "observer_input_tokens_per_second": _rate(total_tokens, observe_seconds),
            "observer_layer_tokens_per_second": _rate(
                total_tokens * observed_layers,
                observe_seconds,
            ),
            "observer_layers_per_second": _rate(observed_layers, observe_seconds),
            "observer_seconds_per_layer": _rate(observe_seconds, observed_layers),
            "pruning_layers_per_second": _rate(pruning_layers, prune_seconds),
            "pruning_experts_per_second": _rate(experts_removed, prune_seconds),
            "save_mb_per_second": _rate(
                artifact_bytes / 1_000_000,
                save_reload_timings.get("save_seconds"),
            ),
            "reload_mb_per_second": _rate(
                artifact_bytes / 1_000_000,
                save_reload_timings.get("reload_seconds"),
            ),
            "generation_tokens_per_second": _rate(generated_tokens, smoke_seconds),
        }


def _safe_adapter(
    model: Any,
    config: Mapping[str, Any],
    adapter: Any | None,
) -> Any | None:
    if adapter is not None:
        return adapter
    try:
        return infer_model_adapter(model, config)
    except Exception:
        return None


def _safe_layers(adapter: Any | None, model: Any) -> list[Any]:
    if adapter is None or model is None:
        return []
    try:
        return list(adapter.layers(model))
    except Exception:
        return []


def _safe_moe_layers(adapter: Any | None, model: Any) -> list[int]:
    if adapter is None or model is None:
        return []
    try:
        return [int(idx) for idx in adapter.identify_moe_layers(model)]
    except Exception:
        return []


def _first_layer_config(
    adapter: Any | None,
    layers: list[Any],
    config: Mapping[str, Any],
) -> Any | None:
    if adapter is None:
        return None
    for layer in layers:
        try:
            if adapter.is_moe_layer(layer):
                return adapter.get_layer_config(layer, config)
        except Exception:
            continue
    return None


def _first_moe_shape_summary(
    adapter: Any | None,
    layers: list[Any],
) -> dict[str, Any]:
    if adapter is None:
        return {}
    for layer_idx, layer in enumerate(layers):
        try:
            if adapter.is_moe_layer(layer):
                return {
                    "layer_idx": layer_idx,
                    "shapes": _moe_shape_summary(adapter.get_moe(layer)),
                }
        except Exception:
            continue
    return {}


def _reloaded_shape_summary(adapter: Any | None, model: Any) -> dict[str, Any]:
    if adapter is None or model is None:
        return {}
    layers = _safe_layers(adapter, model)
    shape_summary = {}
    for layer_idx in _safe_moe_layers(adapter, model):
        try:
            moe = adapter.get_moe(layers[layer_idx])
        except Exception:
            continue
        shape_summary[str(layer_idx)] = _moe_shape_summary(moe)
    return shape_summary


def _moe_shape_summary(moe: Any) -> dict[str, Any]:
    switch_mlp = getattr(moe, "switch_mlp", None)
    summary: dict[str, Any] = {
        "gate": _field_shapes(
            getattr(moe, "gate", None),
            ("weight", "scales", "biases", "bias"),
        ),
        "expert_bias": _shape(getattr(moe, "expert_bias", None)),
        "switch_mlp": {},
    }

    if switch_mlp is not None:
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            summary["switch_mlp"][projection_name] = _field_shapes(
                getattr(switch_mlp, projection_name, None),
                ("weight", "scales", "biases", "bias"),
            )
    return summary


def _field_shapes(module: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    if module is None:
        return {}
    return {
        field_name: _shape(_get_module_value(module, field_name))
        for field_name in field_names
        if _get_module_value(module, field_name) is not None
    }


def _get_module_value(module: Any, field_name: str) -> Any | None:
    getter = getattr(module, "get", None)
    if callable(getter):
        try:
            value = getter(field_name)
        except (KeyError, TypeError, AttributeError):
            value = None
        if value is not None:
            return value
    return getattr(module, field_name, None)


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _first_attr(root: Any, *attrs: str) -> Any:
    for attr in attrs:
        value = getattr(root, attr, None)
        if value is not None and not callable(value):
            return _json_safe(value)
    return None


def _layer_config_value(layer_config: Any | None, attr: str, default: Any) -> Any:
    if layer_config is None:
        return default
    return getattr(layer_config, attr, default)


def _sequence_length(sequence: Any) -> int:
    input_ids = sequence.get("input_ids") if isinstance(sequence, Mapping) else sequence
    shape = getattr(input_ids, "shape", None)
    if shape:
        return int(shape[-1])

    values = _list_like(input_ids)
    if values and isinstance(values[0], (list, tuple)):
        return len(values[0])
    return len(values)


def _original_expert_count(
    layer_data: Mapping[str, Any],
    config_before: Mapping[str, Any],
) -> int:
    frequency = layer_data.get("expert_frequency")
    if frequency is not None:
        shape = getattr(frequency, "shape", None)
        if shape:
            return int(shape[0])
        try:
            return len(frequency)
        except TypeError:
            pass
    return int(config_before.get("num_experts") or 0)


def _numeric_list(value: Any) -> list[float]:
    values = _list_like(value)
    result = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _list_like(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _resolve_method_or_none(prune_method: str | None) -> str | None:
    if prune_method is None:
        return None
    try:
        return resolve_prune_method(prune_method)
    except ValueError:
        return None



def _sample_mlx_memory() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception as exc:
        return {
            "available": False,
            "error": exc.__class__.__name__,
        }

    sample: dict[str, Any] = {"available": True}
    for key, attr in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        getter = getattr(mx, attr, None)
        if callable(getter):
            try:
                sample[key] = int(getter())
            except Exception as exc:
                sample[key] = None
                sample[f"{key}_error"] = exc.__class__.__name__
        else:
            sample[key] = None
    return sample


def _sample_process_memory() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = int(usage.ru_maxrss)
    # ru_maxrss is in bytes on macOS, kilobytes on Linux/BSD
    divisor = 1_000_000 if sys.platform == "darwin" else 1_000
    return {
        "max_rss_mb": round(raw / divisor, 1),
        "max_rss_bytes": raw if sys.platform == "darwin" else raw * 1000,
        "max_rss_units": "megabytes",
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _rate(numerator: Any, seconds: Any) -> float | None:
    if numerator is None or seconds is None:
        return None
    try:
        numerator = float(numerator)
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return numerator / seconds


def _preview(value: Any, *, max_chars: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {
            "type": value.__class__.__name__,
            "shape": [int(dim) for dim in shape],
        }
    return str(value)


def _strip_nonfinite(value: Any) -> Any:
    """Recursively replace any remaining non-finite floats with None.

    _json_safe already handles scalars/arrays, but a custom object may fall
    through to str() while still yielding a nested float elsewhere. This is a
    final safety net so json.dumps(..., allow_nan=False) cannot raise and lose
    failure telemetry.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _strip_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_strip_nonfinite(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["RunMetrics"]
