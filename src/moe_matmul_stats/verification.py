"""Verification pipeline for the AI-filled metrics sheet."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .schema import ModelStats
from .sources import (
    NANO_MOE_JAX_DEFAULT_CONFIG,
    collectStatsFromConfig,
    fetch_hf_config,
)

DEFAULT_OLMOE_MODEL = "allenai/OLMoE-1B-7B-0125"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"

MODEL_LABELS = ("Nano-MoE-Jax", "OLMoE-1B-7B", "Deepseek-V3")
PHASES = ("Prefill", "Decode")

SHEET_COLUMNS = {
    ("Nano-MoE-Jax", "Prefill"): (3, 4),
    ("Nano-MoE-Jax", "Decode"): (5, 6),
    ("OLMoE-1B-7B", "Prefill"): (7, 8),
    ("OLMoE-1B-7B", "Decode"): (9, 10),
    ("Deepseek-V3", "Prefill"): (11, 12),
    ("Deepseek-V3", "Decode"): (13, 14),
}


@dataclass(frozen=True)
class SheetCell:
    metric: str
    unit: str
    model: str
    phase: str
    value: str
    source: str


@dataclass(frozen=True)
class ModelContext:
    label: str
    kind: str
    source: str
    config: dict[str, Any]
    stats: ModelStats


@dataclass(frozen=True)
class VerifiedMetricRow:
    metric: str
    unit: str
    model: str
    phase: str
    category: str
    value: str
    source_type: str
    evidence: str
    status: str
    notes: str


@dataclass(frozen=True)
class MetricResult:
    value: str
    source_type: str
    evidence: str
    notes: str = ""


MetricComputer = Callable[[ModelContext, str, dict[tuple[str, str, str], SheetCell]], MetricResult]


OPERATING_POINT_METRICS = {
    "Batch size",
    "Sequence length (prompt)",
    "Context length (Key-Value)",
}

CONFIG_METRICS: dict[str, MetricComputer] = {}
FORMULA_METRICS: dict[str, MetricComputer] = {}


def load_default_contexts(revision: str = "main") -> dict[str, ModelContext]:
    """Load the three target model configs and run them through supported extractors."""

    nano_config = dict(NANO_MOE_JAX_DEFAULT_CONFIG)
    olmoe_config = fetch_hf_config(DEFAULT_OLMOE_MODEL, revision=revision)
    deepseek_config = fetch_hf_config(DEFAULT_DEEPSEEK_MODEL, revision=revision)

    return {
        "Nano-MoE-Jax": _context(
            label="Nano-MoE-Jax",
            kind="nano",
            source="Nano-MoE-JAX public default config",
            config=nano_config,
        ),
        "OLMoE-1B-7B": _context(
            label="OLMoE-1B-7B",
            kind="olmoe",
            source=f"huggingface:{DEFAULT_OLMOE_MODEL}@{revision}",
            config=olmoe_config,
        ),
        "Deepseek-V3": _context(
            label="Deepseek-V3",
            kind="deepseek",
            source=f"huggingface:{DEFAULT_DEEPSEEK_MODEL}@{revision}",
            config=deepseek_config,
        ),
    }


def _context(label: str, kind: str, source: str, config: dict[str, Any]) -> ModelContext:
    stats = collectStatsFromConfig(model=label, source=source, config=config)
    return ModelContext(label=label, kind=kind, source=source, config=dict(config), stats=stats)


def parse_sheet(path: str | Path) -> dict[tuple[str, str, str], SheetCell]:
    """Read the pasted Excel TSV into flat cells keyed by metric/model/phase."""

    rows: dict[tuple[str, str, str], SheetCell] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw_row in reader:
            row = list(raw_row) + [""] * max(0, 15 - len(raw_row))
            metric = row[0].strip()
            if not metric or metric in {"Item", "Field"} or _is_section(metric):
                continue
            unit = row[1].strip()
            for (model, phase), (value_col, source_col) in SHEET_COLUMNS.items():
                value = row[value_col].strip() if value_col < len(row) else ""
                source = row[source_col].strip() if source_col < len(row) else ""
                rows[(metric, model, phase)] = SheetCell(
                    metric=metric,
                    unit=unit,
                    model=model,
                    phase=phase,
                    value=value,
                    source=source,
                )
    return rows


def build_verified_rows(
    sheet_path: str | Path,
    contexts: dict[str, ModelContext] | None = None,
) -> list[VerifiedMetricRow]:
    sheet = parse_sheet(sheet_path)
    contexts = contexts or load_default_contexts()
    rows: list[VerifiedMetricRow] = []

    metrics_in_order: list[str] = []
    for metric, _, _ in sheet:
        if metric not in metrics_in_order:
            metrics_in_order.append(metric)

    for metric in metrics_in_order:
        for model in MODEL_LABELS:
            context = contexts[model]
            for phase in PHASES:
                cell = sheet[(metric, model, phase)]
                rows.append(_verify_cell(cell, context, sheet))

    return rows


def write_verified_outputs(
    rows: list[VerifiedMetricRow],
    markdown_path: str | Path,
    csv_path: str | Path,
    results_path: str | Path | None = None,
) -> tuple[Path, Path] | tuple[Path, Path, Path]:
    markdown = Path(markdown_path)
    csv_file = Path(csv_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_verified_markdown(rows), encoding="utf-8")
    write_verified_csv(rows, csv_file)
    if results_path is not None:
        results_file = Path(results_path)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text(render_verified_results_markdown(rows), encoding="utf-8")
        return markdown, csv_file, results_file
    return markdown, csv_file


def write_verified_csv(rows: list[VerifiedMetricRow], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output


def render_verified_markdown(rows: list[VerifiedMetricRow]) -> str:
    counts = Counter(row.status for row in rows)
    category_counts = Counter(row.category for row in rows)

    lines = [
        "# Verified Metrics",
        "",
        "This file verifies the AI-filled spreadsheet against model configs and explicit "
        "formulas. It writes computed values only when the source is auditable.",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(category_counts):
        lines.append(f"| category:{_cell(key)} | {category_counts[key]} |")
    for key in sorted(counts):
        lines.append(f"| status:{_cell(key)} | {counts[key]} |")

    lines.extend(["", "## Metric Classification", ""])
    lines.extend(_render_metric_classification(rows))

    lines.extend(["", "## Formula Definitions", ""])
    lines.extend(_render_metric_definition_table(rows))

    config_rows = [row for row in rows if row.category == "config"]
    formula_rows = [row for row in rows if row.category == "formula"]
    unchecked = [row for row in rows if row.category not in {"config", "formula"}]

    lines.extend(["", "## Config Rows", ""])
    lines.extend(_render_rows_table(config_rows))

    lines.extend(["", "## Formula Rows", ""])
    lines.extend(_render_rows_table(formula_rows))

    lines.extend(["", "## Unverified Or Assumption Rows", ""])
    lines.extend(_render_rows_table(unchecked))

    return "\n".join(lines).rstrip() + "\n"


def render_verified_results_markdown(rows: list[VerifiedMetricRow]) -> str:
    """Render a compact spreadsheet-style view with only verified categories."""

    verified_rows = [row for row in rows if row.category in {"config", "formula"}]
    metrics_in_order: list[str] = []
    for row in verified_rows:
        if row.metric not in metrics_in_order:
            metrics_in_order.append(row.metric)

    by_key = {(row.metric, row.model, row.phase): row for row in verified_rows}

    lines = [
        "# Verified Results",
        "",
        "This is the compact replacement for the AI-filled spreadsheet. It keeps only "
        "config-verifiable and formula-verifiable metrics, and uses values recomputed "
        "from model configs plus explicit formulas.",
        "",
        "This table shows the verified values only. Comparison against the original "
        "AI-filled sheet is kept in `verified_metrics.md`.",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| verified metric rows | {len(metrics_in_order)} |",
        f"| flattened verified cells | {len(verified_rows)} |",
    ]

    lines.extend(["", "## Formula And Evidence", ""])
    lines.extend(_render_metric_definition_table(verified_rows))

    columns = (
        "Field",
        "Unit",
        "Category",
        "Evidence / Formula",
        "Nano-MoE-Jax Prefill",
        "Nano-MoE-Jax Decode",
        "OLMoE-1B-7B Prefill",
        "OLMoE-1B-7B Decode",
        "Deepseek-V3 Prefill",
        "Deepseek-V3 Decode",
    )
    lines.extend(["", "## Verified Sheet", ""])
    lines.append("| " + " | ".join(f"`{column}`" for column in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    for metric in metrics_in_order:
        sample = next(row for row in verified_rows if row.metric == metric)
        evidence = _metric_evidence_summary(metric, verified_rows)
        table_row = [
            metric,
            sample.unit,
            sample.category,
            evidence,
        ]
        for model in MODEL_LABELS:
            for phase in PHASES:
                table_row.append(_result_cell(by_key[(metric, model, phase)]))
        lines.append("| " + " | ".join(_cell(value) for value in table_row) + " |")

    lines.extend(["", "## Status Legend", ""])
    lines.extend(
        [
            "This compact sheet intentionally omits match/mismatch annotations. Use "
            "`verified_metrics.md` for the audit trail against the original AI-filled values.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def _metric_evidence_summary(metric: str, rows: list[VerifiedMetricRow]) -> str:
    evidence: list[str] = []
    for row in rows:
        if row.metric != metric:
            continue
        if row.evidence not in evidence:
            evidence.append(row.evidence)
    return "<br>".join(evidence)


def _result_cell(row: VerifiedMetricRow) -> str:
    if row.status == "not_applicable" or not row.value:
        return "N/A"
    return row.value


def _render_metric_classification(rows: list[VerifiedMetricRow]) -> list[str]:
    category_to_metrics: dict[str, list[str]] = {
        "config": [],
        "formula": [],
        "not_config_verifiable": [],
    }
    for row in rows:
        metrics = category_to_metrics.setdefault(row.category, [])
        if row.metric not in metrics:
            metrics.append(row.metric)

    lines = ["| Category | Metrics |", "| --- | --- |"]
    labels = {
        "config": "Config-verifiable",
        "formula": "Formula-verifiable",
        "not_config_verifiable": "Not config-verifiable",
    }
    for category in ("config", "formula", "not_config_verifiable"):
        metrics = "<br>".join(_cell(metric) for metric in category_to_metrics.get(category, []))
        lines.append(f"| {labels.get(category, category)} | {metrics or '-'} |")
    return lines


def _render_metric_definition_table(rows: list[VerifiedMetricRow]) -> list[str]:
    lines = ["| Metric | Category | Evidence / Formula |", "| --- | --- | --- |"]
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.category not in {"config", "formula"}:
            continue
        key = (row.metric, row.category, row.evidence)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {_cell(row.metric)} | {_cell(row.category)} | {_cell(row.evidence)} |")
    return lines


def _render_rows_table(rows: list[VerifiedMetricRow]) -> list[str]:
    columns = (
        "metric",
        "unit",
        "model",
        "phase",
        "category",
        "value",
        "source_type",
        "evidence",
        "status",
        "notes",
    )
    lines = [
        "| " + " | ".join(f"`{column}`" for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        data = asdict(row)
        lines.append("| " + " | ".join(_cell(data[column]) for column in columns) + " |")
    return lines


def _verify_cell(
    cell: SheetCell,
    context: ModelContext,
    sheet: dict[tuple[str, str, str], SheetCell],
) -> VerifiedMetricRow:
    if cell.metric in CONFIG_METRICS:
        result = CONFIG_METRICS[cell.metric](context, cell.phase, sheet)
        category = "config"
        status = _comparison_status(cell.value, result.value)
        notes = _comparison_notes(cell, result)
    elif cell.metric in FORMULA_METRICS:
        result = FORMULA_METRICS[cell.metric](context, cell.phase, sheet)
        category = "formula"
        status = "not_applicable" if result.value == "" else _comparison_status(cell.value, result.value)
        notes = _comparison_notes(cell, result) if result.value else result.notes
    elif cell.metric in OPERATING_POINT_METRICS:
        result = MetricResult(
            value=cell.value,
            source_type="sheet_operating_point",
            evidence="operating point chosen by spreadsheet, not model config",
            notes=f"source={cell.source or '-'}",
        )
        category = "not_config_verifiable"
        status = "assumption"
        notes = result.notes
    else:
        result = MetricResult(
            value=cell.value,
            source_type="ai_sheet_unverified",
            evidence="not derivable from config-only verification",
            notes=f"source={cell.source or '-'}",
        )
        category = "not_config_verifiable"
        status = "unverified_source_value" if cell.value else "missing_unverified_value"
        notes = result.notes

    return VerifiedMetricRow(
        metric=cell.metric,
        unit=cell.unit,
        model=cell.model,
        phase=cell.phase,
        category=category,
        value=result.value,
        source_type=result.source_type,
        evidence=result.evidence,
        status=status,
        notes=notes,
    )


def _comparison_status(sheet_value: str, verified_value: str) -> str:
    if not sheet_value and verified_value:
        return "computed_no_sheet_value"
    if _values_match(sheet_value, verified_value):
        return "match"
    return "mismatch"


def _comparison_notes(cell: SheetCell, result: MetricResult) -> str:
    notes = []
    if not _values_match(cell.value, result.value):
        notes.append(f"sheet_value={cell.value or '-'}")
        notes.append(f"sheet_source={cell.source or '-'}")
    if result.notes:
        notes.append(result.notes)
    return "; ".join(notes) if notes else result.notes


def _values_match(left: str, right: str) -> bool:
    left = (left or "").strip()
    right = (right or "").strip()
    if left == right:
        return True

    left_num = _parse_number(left)
    right_num = _parse_number(right)
    if left_num is not None and right_num is not None:
        if left_num == right_num:
            return True
        tolerance = max(abs(left_num), abs(right_num)) * 1e-3 + 1e-12
        return abs(left_num - right_num) <= tolerance

    return _normalize_text(left) == _normalize_text(right)


def _parse_number(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return "-"
    return text.replace("\n", "<br>").replace("|", r"\|")


def _is_section(metric: str) -> bool:
    return bool(re.match(r"^[A-H]\.\s", metric))


def _register_config(metric: str) -> Callable[[MetricComputer], MetricComputer]:
    def decorator(func: MetricComputer) -> MetricComputer:
        CONFIG_METRICS[metric] = func
        return func

    return decorator


def _register_formula(metric: str) -> Callable[[MetricComputer], MetricComputer]:
    def decorator(func: MetricComputer) -> MetricComputer:
        FORMULA_METRICS[metric] = func
        return func

    return decorator


@_register_config("Layers")
def _layers(context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]) -> MetricResult:
    if context.kind == "nano":
        return _config_result(str(context.config["n_layers"]), "n_layers", context)
    return _config_result(str(context.config["num_hidden_layers"]), "num_hidden_layers", context)


@_register_config("Hidden dimension")
def _hidden_dim(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    key = "d_model" if context.kind == "nano" else "hidden_size"
    return _config_result(str(context.config[key]), key, context)


@_register_config("Feedforward dimension (dense)")
def _dense_ffn_dim(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind in {"nano", "olmoe"}:
        return _config_result("N/A (MoE only)", "architecture has MoE block, no dense FFN stack", context)
    value = f"{context.config['intermediate_size']} (first {context.config['first_k_dense_replace']} layers only)"
    return _config_result(value, "intermediate_size + first_k_dense_replace", context)


@_register_config("Attention: heads / head dim / Key-Value heads")
def _attention_shape(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        heads = int(context.config["n_heads"])
        head_dim = int(context.config["d_model"]) // heads
        return _config_result(f"{heads} / {head_dim} / {heads}", "n_heads, d_model", context)
    if context.kind == "olmoe":
        heads = int(context.config["num_attention_heads"])
        head_dim = int(context.config["hidden_size"]) // heads
        kv_heads = int(context.config["num_key_value_heads"])
        return _config_result(
            f"{heads} / {head_dim} / {kv_heads}",
            "num_attention_heads, hidden_size, num_key_value_heads",
            context,
        )

    heads = int(context.config["num_attention_heads"])
    qk_nope = int(context.config["qk_nope_head_dim"])
    qk_rope = int(context.config["qk_rope_head_dim"])
    value_dim = int(context.config["v_head_dim"])
    kv_rank = int(context.config["kv_lora_rank"])
    return _config_result(
        f"{heads} / qk={qk_nope + qk_rope} (nope={qk_nope}+rope={qk_rope}), v={value_dim} / MLA kv_lora={kv_rank}",
        "num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank",
        context,
    )


@_register_config("Total experts")
def _total_experts(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    key = {"nano": "n_experts", "olmoe": "num_experts", "deepseek": "n_routed_experts"}[
        context.kind
    ]
    return _config_result(str(context.config[key]), key, context)


@_register_config("Experts active per token (top-k)")
def _top_k(context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]) -> MetricResult:
    key = "top_k" if context.kind == "nano" else "num_experts_per_tok"
    return _config_result(str(context.config[key]), key, context)


@_register_config("Shared experts")
def _shared_experts(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    value = int(context.config.get("n_shared_experts", 0))
    return _config_result(str(value), "n_shared_experts default 0", context)


@_register_config("Expert feedforward dimension")
def _expert_ffn_dim(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    key = {"nano": "d_ff", "olmoe": "intermediate_size", "deepseek": "moe_intermediate_size"}[
        context.kind
    ]
    return _config_result(str(context.config[key]), key, context)


@_register_config("Router / gating type")
def _router_type(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        return _config_result(f"softmax top-{context.config['top_k']}", "top_k + Nano-MoE-JAX router implementation", context)
    if context.kind == "olmoe":
        value = (
            f"softmax top-{context.config['num_experts_per_tok']}, "
            f"aux_loss_coef={context.config.get('router_aux_loss_coef')}, "
            f"norm_topk_prob={context.config.get('norm_topk_prob')}"
        )
        return _config_result(value, "num_experts_per_tok, router_aux_loss_coef, norm_topk_prob", context)
    value = (
        f"{context.config.get('scoring_func')} top-{context.config['num_experts_per_tok']}, "
        f"topk_method={context.config.get('topk_method')}, "
        f"norm_topk_prob={context.config.get('norm_topk_prob')}, "
        f"shared_experts={context.config.get('n_shared_experts')}"
    )
    return _config_result(value, "scoring_func, topk_method, norm_topk_prob, n_shared_experts", context)


@_register_config("Weights precision")
def _weights_precision(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        return _config_result("FP32", "JAX default dtype for Nano-MoE-JAX defaults", context)
    quant = context.config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("quant_method") == "fp8":
        return _config_result(f"FP8 ({str(quant.get('fmt')).upper()})", "quantization_config", context)
    return _config_result(_dtype_label(str(context.config.get("torch_dtype", "unknown"))), "torch_dtype", context)


@_register_config("Activations precision")
def _activations_precision(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        return _config_result("FP32", "JAX default dtype for Nano-MoE-JAX defaults", context)
    return _config_result(_dtype_label(str(context.config.get("torch_dtype", "unknown"))), "torch_dtype", context)


@_register_config("Key-Value cache precision")
def _kv_precision(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        return _config_result("FP32", "JAX default dtype for Nano-MoE-JAX defaults", context)
    return _config_result(_dtype_label(str(context.config.get("torch_dtype", "unknown"))), "torch_dtype", context)


@_register_config("Quantization scheme")
def _quantization_scheme(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if context.kind == "nano":
        return _config_result("None", "no quantization field in Nano-MoE-JAX defaults", context)
    quant = context.config.get("quantization_config")
    if not quant:
        return _config_result("None", "quantization_config absent", context)
    return _config_result(str(quant), "quantization_config", context)


@_register_formula("Total parameters")
def _total_parameters(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    return _formula_result(
        _format_float(_param_total(context) / 1_000_000_000),
        "parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head",
        context,
        "unit=billions",
    )


@_register_formula("Active parameters per token")
def _active_parameters(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    return _formula_result(
        _format_float(_active_param_total(context) / 1_000_000_000),
        "active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head",
        context,
        "unit=billions",
    )


@_register_formula("Multiply-accumulate ops per token")
def _macs_per_token(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    ops = _operating_point(context.label, phase, sheet)
    components = _mac_components(context, phase, ops)
    return _formula_result(
        str(sum(components.values())),
        "MACs/token = sum of per-token matmul components using sheet operating point",
        context,
        _op_note(ops),
    )


@_register_formula("Dominant operators (shapes + % of compute)")
def _dominant_ops(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    ops = _operating_point(context.label, phase, sheet)
    components = _mac_components(context, phase, ops)
    total = sum(components.values())
    top = sorted(components.items(), key=lambda item: item[1], reverse=True)[:4]
    value = "; ".join(f"{name} {100 * count / total:.1f}%" for name, count in top)
    return _formula_result(
        value,
        "component MAC share from static matmul formulas",
        context,
        _op_note(ops),
    )


@_register_formula("Weight footprint (total)")
def _weight_footprint_total(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    return _formula_result(
        str(_param_total(context) * _weight_bytes(context)),
        "total_parameter_count * weight_bytes",
        context,
        f"weight_bytes={_weight_bytes(context)}",
    )


@_register_formula("Weight footprint per layer")
def _weight_footprint_layer(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    value = _layer_weight_bytes(context)
    return _formula_result(
        str(value),
        "decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers",
        context,
        f"weight_bytes={_weight_bytes(context)}",
    )


@_register_formula("Weight footprint per expert")
def _weight_footprint_expert(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    return _formula_result(
        str(_expert_params_per_expert(context) * _weight_bytes(context)),
        "expert_weight_params_per_expert * weight_bytes",
        context,
        f"weight_bytes={_weight_bytes(context)}",
    )


@_register_formula("Activation footprint per layer")
def _activation_footprint(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    ops = _operating_point(context.label, phase, sheet)
    seq = ops.sequence if phase == "Prefill" else 1
    value = ops.batch * seq * _hidden_size(context) * _activation_bytes(context)
    return _formula_result(
        str(value),
        "B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode",
        context,
        _op_note(ops),
    )


@_register_formula("Key-Value cache size")
def _kv_cache_size(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    ops = _operating_point(context.label, phase, sheet)
    return _formula_result(
        str(_kv_cache_bytes(context, ops)),
        "KV cache formula from config attention layout and sheet KV context length",
        context,
        _op_note(ops),
    )


@_register_formula("Key-Value cache read bandwidth per decode step")
def _kv_read_bandwidth(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    if phase == "Prefill":
        return MetricResult(
            value="",
            source_type="computed_from_config",
            evidence="decode-only metric",
            notes="prefill column intentionally blank",
        )
    ops = _operating_point(context.label, phase, sheet)
    return _formula_result(
        str(_kv_cache_bytes(context, ops)),
        "decode step reads B * KV cached keys/values across layers",
        context,
        _op_note(ops),
    )


@_register_formula("Expert activation fraction (top-k / total)")
def _expert_activation_fraction(
    context: ModelContext, phase: str, sheet: dict[tuple[str, str, str], SheetCell]
) -> MetricResult:
    routed = _num_experts(context)
    top_k = _top_k_value(context)
    shared = int(context.config.get("n_shared_experts", 0))
    active = top_k + shared
    total = routed + shared
    value = 100 * active / total
    return _formula_result(
        _format_float(value),
        "100 * (top_k + shared_experts) / (routed_experts + shared_experts)",
        context,
        "shared_experts included for DeepSeek hardware-active expert pages",
    )


def _config_result(value: str, evidence: str, context: ModelContext) -> MetricResult:
    return MetricResult(
        value=value,
        source_type="config",
        evidence=evidence,
        notes=f"config_source={context.source}",
    )


def _formula_result(value: str, evidence: str, context: ModelContext, notes: str = "") -> MetricResult:
    return MetricResult(
        value=value,
        source_type="computed_from_config",
        evidence=evidence,
        notes=f"config_source={context.source}" + (f"; {notes}" if notes else ""),
    )


@dataclass(frozen=True)
class OperatingPoint:
    batch: int
    sequence: int
    kv_context: int


def _operating_point(
    model: str,
    phase: str,
    sheet: dict[tuple[str, str, str], SheetCell],
) -> OperatingPoint:
    return OperatingPoint(
        batch=_int_sheet(sheet, "Batch size", model, phase),
        sequence=_int_sheet(sheet, "Sequence length (prompt)", model, phase),
        kv_context=_int_sheet(sheet, "Context length (Key-Value)", model, phase),
    )


def _int_sheet(
    sheet: dict[tuple[str, str, str], SheetCell], metric: str, model: str, phase: str
) -> int:
    value = sheet[(metric, model, phase)].value
    parsed = _parse_number(value)
    if parsed is None:
        raise ValueError(f"Cannot parse operating point {metric!r} for {model}/{phase}: {value!r}")
    return int(parsed)


def _op_note(ops: OperatingPoint) -> str:
    return f"B={ops.batch}, S={ops.sequence}, KV={ops.kv_context}"


def _hidden_size(context: ModelContext) -> int:
    return int(context.config["d_model"] if context.kind == "nano" else context.config["hidden_size"])


def _num_layers(context: ModelContext) -> int:
    return int(context.config["n_layers"] if context.kind == "nano" else context.config["num_hidden_layers"])


def _num_heads(context: ModelContext) -> int:
    return int(context.config["n_heads"] if context.kind == "nano" else context.config["num_attention_heads"])


def _head_dim(context: ModelContext) -> int:
    return _hidden_size(context) // _num_heads(context)


def _num_experts(context: ModelContext) -> int:
    key = {"nano": "n_experts", "olmoe": "num_experts", "deepseek": "n_routed_experts"}[
        context.kind
    ]
    return int(context.config[key])


def _top_k_value(context: ModelContext) -> int:
    key = "top_k" if context.kind == "nano" else "num_experts_per_tok"
    return int(context.config[key])


def _expert_intermediate(context: ModelContext) -> int:
    key = {"nano": "d_ff", "olmoe": "intermediate_size", "deepseek": "moe_intermediate_size"}[
        context.kind
    ]
    return int(context.config[key])


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.lower()
    if "float32" in normalized or normalized == "fp32":
        return 4
    if "bfloat16" in normalized or "float16" in normalized or normalized in {"bf16", "fp16"}:
        return 2
    if "fp8" in normalized:
        return 1
    return 4


def _dtype_label(dtype: str) -> str:
    normalized = dtype.lower()
    if "bfloat16" in normalized or normalized == "bf16":
        return "BF16"
    if "float16" in normalized or normalized == "fp16":
        return "FP16"
    if "float32" in normalized or normalized == "fp32":
        return "FP32"
    if "fp8" in normalized:
        return "FP8"
    return dtype.upper()


def _weight_bytes(context: ModelContext) -> int:
    if context.kind == "nano":
        return 4
    quant = context.config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("quant_method") == "fp8":
        return 1
    return _dtype_bytes(str(context.config.get("torch_dtype", "float32")))


def _activation_bytes(context: ModelContext) -> int:
    if context.kind == "nano":
        return 4
    return _dtype_bytes(str(context.config.get("torch_dtype", "float32")))


def _param_total(context: ModelContext) -> int:
    h = _hidden_size(context)
    layers = _num_layers(context)
    vocab = int(context.config["vocab_size"])

    if context.kind == "nano":
        block = int(context.config["block_size"])
        return vocab * h + block * h + layers * _layer_params(context) + h + vocab * h

    return vocab * h + _all_layer_params(context) + h + vocab * h


def _all_layer_params(context: ModelContext) -> int:
    if context.kind in {"nano", "olmoe"}:
        return _num_layers(context) * _layer_params(context)

    layers = _num_layers(context)
    first_dense = int(context.config["first_k_dense_replace"])
    return layers * _deepseek_attention_params(context) + first_dense * _deepseek_dense_mlp_params(
        context
    ) + (layers - first_dense) * _deepseek_moe_layer_params(context)


def _layer_params(context: ModelContext) -> int:
    h = _hidden_size(context)
    if context.kind == "nano":
        return 4 * h * h + h * _num_experts(context) + _num_experts(context) * 2 * h * _expert_intermediate(context)

    if context.kind == "olmoe":
        return 4 * h * h + h * _num_experts(context) + _num_experts(context) * _expert_params_per_expert(context)

    layers = _num_layers(context)
    return round(_all_layer_params(context) / layers)


def _layer_weight_bytes(context: ModelContext) -> int:
    return _layer_params(context) * _weight_bytes(context)


def _expert_params_per_expert(context: ModelContext) -> int:
    h = _hidden_size(context)
    i = _expert_intermediate(context)
    if context.kind == "nano":
        return 2 * h * i
    return 3 * h * i


def _active_param_total(context: ModelContext) -> int:
    h = _hidden_size(context)
    vocab = int(context.config["vocab_size"])

    if context.kind == "nano":
        per_layer = 4 * h * h + h * _num_experts(context) + _top_k_value(context) * 2 * h * _expert_intermediate(context)
        return vocab * h + _num_layers(context) * per_layer + vocab * h

    if context.kind == "olmoe":
        per_layer = 4 * h * h + h * _num_experts(context) + _top_k_value(context) * _expert_params_per_expert(context)
        return vocab * h + _num_layers(context) * per_layer + vocab * h

    layers = _num_layers(context)
    first_dense = int(context.config["first_k_dense_replace"])
    moe_layers = layers - first_dense
    active_moe = (
        h * _num_experts(context)
        + 3 * h * _expert_intermediate(context) * int(context.config.get("n_shared_experts", 0))
        + _top_k_value(context) * _expert_params_per_expert(context)
    )
    return (
        vocab * h
        + layers * _deepseek_attention_params(context)
        + first_dense * _deepseek_dense_mlp_params(context)
        + moe_layers * active_moe
        + vocab * h
    )


def _mac_components(context: ModelContext, phase: str, ops: OperatingPoint) -> dict[str, int]:
    h = _hidden_size(context)
    layers = _num_layers(context)
    vocab = int(context.config["vocab_size"])
    context_len = ops.sequence if phase == "Prefill" else ops.kv_context

    if context.kind == "nano":
        heads = _num_heads(context)
        d = _head_dim(context)
        return {
            "attention_proj": layers * 4 * h * h,
            "attention_qk_av": layers * 2 * heads * context_len * d,
            "router": layers * h * _num_experts(context),
            "expert_mlp_impl_all_experts": layers
            * _num_experts(context)
            * 2
            * h
            * _expert_intermediate(context),
            "lm_head": h * vocab,
        }

    if context.kind == "olmoe":
        heads = _num_heads(context)
        d = _head_dim(context)
        return {
            "attention_proj": layers * 4 * h * h,
            "attention_qk_av": layers * 2 * heads * context_len * d,
            "router": layers * h * _num_experts(context),
            "routed_expert_mlp_topk": layers
            * _top_k_value(context)
            * _expert_params_per_expert(context),
            "lm_head": h * vocab,
        }

    first_dense = int(context.config["first_k_dense_replace"])
    moe_layers = layers - first_dense
    qk_nope = int(context.config["qk_nope_head_dim"])
    qk_rope = int(context.config["qk_rope_head_dim"])
    value_dim = int(context.config["v_head_dim"])
    qk_dim = qk_nope + qk_rope
    heads = _num_heads(context)
    return {
        "attention_mla_proj": layers * _deepseek_attention_params(context),
        "attention_qk_av": layers * heads * context_len * (qk_dim + value_dim),
        "dense_mlp": first_dense * _deepseek_dense_mlp_params(context),
        "router": moe_layers * h * _num_experts(context),
        "shared_expert_mlp": moe_layers
        * 3
        * h
        * _expert_intermediate(context)
        * int(context.config.get("n_shared_experts", 0)),
        "routed_expert_mlp_topk": moe_layers
        * _top_k_value(context)
        * _expert_params_per_expert(context),
        "lm_head": h * vocab,
    }


def _deepseek_attention_params(context: ModelContext) -> int:
    h = _hidden_size(context)
    heads = _num_heads(context)
    q_rank = int(context.config["q_lora_rank"])
    kv_rank = int(context.config["kv_lora_rank"])
    qk_nope = int(context.config["qk_nope_head_dim"])
    qk_rope = int(context.config["qk_rope_head_dim"])
    value_dim = int(context.config["v_head_dim"])
    q_head = qk_nope + qk_rope
    return (
        h * q_rank
        + q_rank * heads * q_head
        + h * (kv_rank + qk_rope)
        + kv_rank * heads * (qk_nope + value_dim)
        + h * heads * value_dim
    )


def _deepseek_dense_mlp_params(context: ModelContext) -> int:
    return 3 * _hidden_size(context) * int(context.config["intermediate_size"])


def _deepseek_moe_layer_params(context: ModelContext) -> int:
    h = _hidden_size(context)
    routed = _num_experts(context)
    shared = int(context.config.get("n_shared_experts", 0))
    i = _expert_intermediate(context)
    return h * routed + 3 * h * i * shared + routed * 3 * h * i


def _kv_cache_bytes(context: ModelContext, ops: OperatingPoint) -> int:
    layers = _num_layers(context)
    b = ops.batch
    kv = ops.kv_context
    bytes_per = _activation_bytes(context)

    if context.kind == "deepseek":
        width = int(context.config["kv_lora_rank"]) + int(context.config["qk_rope_head_dim"])
        return layers * b * kv * width * bytes_per

    kv_heads = _num_heads(context) if context.kind == "nano" else int(context.config["num_key_value_heads"])
    return layers * b * kv * 2 * kv_heads * _head_dim(context) * bytes_per


def _format_float(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.6g}"
