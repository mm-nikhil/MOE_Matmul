"""Helpers for organized per-model config, metrics, and matmul outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import write_markdown_report
from .sources import collectStatsFromConfig, fetch_hf_config
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
    hf_model: str
    kind: str
    revision: str = "main"
    metrics_config_path: tuple[str, ...] = ()


DEFAULT_DEEPSEEK_V4_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_KIMI_K25_MODEL = "moonshotai/Kimi-K2.5"

DEFAULT_NEW_MODEL_SPECS = (
    ModelSpec(
        label="DeepSeek-V4-Pro",
        slug="deepseek-v4-pro",
        hf_model=DEFAULT_DEEPSEEK_V4_MODEL,
        kind="deepseek_v4",
    ),
    ModelSpec(
        label="Kimi-K2.5",
        slug="kimi-k2.5",
        hf_model=DEFAULT_KIMI_K25_MODEL,
        kind="deepseek",
        metrics_config_path=("text_config",),
    ),
)

DEFAULT_LARGE_MODEL_OPERATING_POINTS = {
    "Prefill": OperatingPoint(batch=32, sequence=1024, kv_context=4096),
    "Decode": OperatingPoint(batch=32, sequence=4096, kv_context=4096),
}


def write_new_model_results(
    *,
    specs: tuple[ModelSpec, ...] = DEFAULT_NEW_MODEL_SPECS,
    models_root: str | Path = "models",
    results_root: str | Path = "results",
    metric_sheet_path: str | Path = "verify/ai_filled_metrics_sheet.tsv",
) -> list[Path]:
    """Fetch configs and write per-model metrics.md and matmul.md files."""

    written: list[Path] = []
    metric_units = _metric_units(metric_sheet_path)

    for spec in specs:
        config = fetch_hf_config(spec.hf_model, revision=spec.revision)
        source = f"huggingface:{spec.hf_model}@{spec.revision}"

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
            for phase, point in DEFAULT_LARGE_MODEL_OPERATING_POINTS.items()
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
