"""Helpers for organized per-model config, metrics, and matmul outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import write_markdown_report
from .sources import NANO_MOE_JAX_DEFAULT_CONFIG, collectStatsFromConfig, fetch_hf_config
from .verification import (
    ModelContext,
    OperatingPoint,
    build_computed_metric_rows,
    parse_sheet,
    render_computed_metrics_markdown,
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    slug: str
    kind: str
    hf_model: str | None = None
    revision: str = "main"
    config: Mapping[str, Any] | None = None
    source: str | None = None
    metrics_config_path: tuple[str, ...] = ()
    operating_points: Mapping[str, OperatingPoint] | None = None


DEFAULT_OLMOE_MODEL = "allenai/OLMoE-1B-7B-0125"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_DEEPSEEK_V4_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_KIMI_K25_MODEL = "moonshotai/Kimi-K2.5"

NANO_OPERATING_POINTS = {
    "Prefill": OperatingPoint(batch=1, sequence=128, kv_context=512),
    "Decode": OperatingPoint(batch=1, sequence=512, kv_context=512),
}

OLMOE_OPERATING_POINTS = {
    "Prefill": OperatingPoint(batch=32, sequence=512, kv_context=2048),
    "Decode": OperatingPoint(batch=32, sequence=2048, kv_context=2048),
}

LARGE_MODEL_OPERATING_POINTS = {
    "Prefill": OperatingPoint(batch=32, sequence=1024, kv_context=4096),
    "Decode": OperatingPoint(batch=32, sequence=4096, kv_context=4096),
}


def default_model_specs(
    *,
    olmoe_model: str = DEFAULT_OLMOE_MODEL,
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,
    deepseek_v4_model: str = DEFAULT_DEEPSEEK_V4_MODEL,
    kimi_model: str = DEFAULT_KIMI_K25_MODEL,
    revision: str = "main",
) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            label="Nano-MoE-JAX",
            slug="nano-moe-jax",
            kind="nano",
            config=dict(NANO_MOE_JAX_DEFAULT_CONFIG),
            source="github:carrycooldude/Nano-MoE-JAX defaults",
            operating_points=NANO_OPERATING_POINTS,
        ),
        ModelSpec(
            label="OLMoE-1B-7B",
            slug="olmoe-1b-7b",
            kind="olmoe",
            hf_model=olmoe_model,
            revision=revision,
            operating_points=OLMOE_OPERATING_POINTS,
        ),
        ModelSpec(
            label="DeepSeek-V3",
            slug="deepseek-v3",
            kind="deepseek",
            hf_model=deepseek_model,
            revision=revision,
            operating_points=LARGE_MODEL_OPERATING_POINTS,
        ),
        ModelSpec(
            label="DeepSeek-V4-Pro",
            slug="deepseek-v4-pro",
            kind="deepseek_v4",
            hf_model=deepseek_v4_model,
            revision=revision,
            operating_points=LARGE_MODEL_OPERATING_POINTS,
        ),
        ModelSpec(
            label="Kimi-K2.5",
            slug="kimi-k2.5",
            kind="deepseek",
            hf_model=kimi_model,
            revision=revision,
            metrics_config_path=("text_config",),
            operating_points=LARGE_MODEL_OPERATING_POINTS,
        ),
    )


def write_model_results(
    *,
    specs: tuple[ModelSpec, ...],
    models_root: str | Path = "models",
    results_root: str | Path = "results",
    metric_sheet_path: str | Path = "verify/ai_filled_metrics_sheet.tsv",
) -> list[Path]:
    """Write per-model config.json, metrics.md, and matmul.md files."""

    written: list[Path] = []
    metric_units = _metric_units(metric_sheet_path)

    for spec in specs:
        config, source = _load_spec_config(spec)

        model_dir = Path(models_root) / spec.slug
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(config_path)

        stats = collectStatsFromConfig(model=spec.label, source=source, config=config)
        result_dir = Path(results_root) / spec.slug
        matmul_path = write_markdown_report([stats], result_dir / "matmul.md")
        written.append(matmul_path)

        metrics_config = _nested_config(config, spec.metrics_config_path)
        context = ModelContext(
            label=spec.label,
            kind=spec.kind,
            source=source,
            config=metrics_config,
            stats=stats,
        )
        operating_points = {
            (spec.label, phase): point
            for phase, point in _spec_operating_points(spec).items()
        }
        rows = build_computed_metric_rows(
            contexts={spec.label: context},
            operating_points=operating_points,
            metric_units=metric_units,
        )
        metrics_markdown = render_computed_metrics_markdown(
            rows,
            title=f"{spec.label} Metrics",
            model_labels=(spec.label,),
            operating_points=operating_points,
        )
        metrics_path = result_dir / "metrics.md"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(metrics_markdown, encoding="utf-8")
        written.append(metrics_path)

    return written


def write_new_model_results(
    *,
    specs: tuple[ModelSpec, ...] | None = None,
    models_root: str | Path = "models",
    results_root: str | Path = "results",
    metric_sheet_path: str | Path = "verify/ai_filled_metrics_sheet.tsv",
) -> list[Path]:
    """Backward-compatible wrapper for writing organized per-model results."""

    return write_model_results(
        specs=specs or default_model_specs()[3:],
        models_root=models_root,
        results_root=results_root,
        metric_sheet_path=metric_sheet_path,
    )


def _load_spec_config(spec: ModelSpec) -> tuple[dict[str, Any], str]:
    if spec.config is not None:
        source = spec.source or f"local:{spec.slug}"
        return dict(spec.config), source
    if not spec.hf_model:
        raise ValueError(f"ModelSpec {spec.slug!r} needs either config or hf_model.")
    source = spec.source or f"huggingface:{spec.hf_model}@{spec.revision}"
    return fetch_hf_config(spec.hf_model, revision=spec.revision), source


def _spec_operating_points(spec: ModelSpec) -> Mapping[str, OperatingPoint]:
    return spec.operating_points or LARGE_MODEL_OPERATING_POINTS


def _nested_config(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(path)
            raise KeyError(f"Config does not contain nested path {joined!r}.")
        value = value[key]
    if not isinstance(value, dict):
        joined = ".".join(path)
        raise TypeError(f"Config path {joined!r} must resolve to a JSON object.")
    return dict(value)


def _metric_units(sheet_path: str | Path) -> dict[str, str]:
    try:
        cells = parse_sheet(sheet_path)
    except FileNotFoundError:
        return {}

    units: dict[str, str] = {}
    for metric, _, _ in cells:
        if metric not in units:
            units[metric] = cells[(metric, "Nano-MoE-Jax", "Prefill")].unit
    return units
