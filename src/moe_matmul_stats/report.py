"""Markdown rendering for matmul statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import MatmulRecord, ModelStats


RECORD_COLUMNS = (
    "layer_range",
    "block",
    "op_name",
    "op_kind",
    "lhs_shape",
    "rhs_shape",
    "output_shape",
    "batching",
    "repeat_count",
    "active_condition",
    "logical_vs_implementation",
    "activation_after",
    "numeric_format",
    "notes",
)


def render_markdown(stats: Iterable[ModelStats]) -> str:
    """Render one combined Markdown report."""

    model_stats = list(stats)
    lines: list[str] = [
        "# MOE Matmul Stats Report",
        "",
        "This report contains static, config-derived matmul shape families. It does not "
        "require model weights.",
        "",
    ]

    for index, model in enumerate(model_stats):
        if index:
            lines.append("")
        lines.extend(_render_model(model))

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(stats: Iterable[ModelStats], output_path: str | Path) -> Path:
    """Render and write a Markdown report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(stats), encoding="utf-8")
    return path


def _render_model(model: ModelStats) -> list[str]:
    lines = [
        f"## {model.model}",
        "",
        f"- Source: `{model.source}`",
        f"- Matmul families: `{len(model.records)}`",
    ]

    if model.config_summary:
        lines.extend(["", "### Config Summary", ""])
        lines.extend(_render_key_value_table(model.config_summary))

    if model.records:
        lines.extend(["", "### Matmul Families", ""])
        lines.extend(_render_record_table(model.records))
    else:
        lines.extend(["", "No matmul records collected."])

    if model.notes:
        lines.extend(["", "### Notes", ""])
        lines.extend(f"- {note}" for note in model.notes)

    return lines


def _render_key_value_table(values: dict[str, object]) -> list[str]:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key in sorted(values):
        lines.append(f"| `{_cell(key)}` | `{_cell(values[key])}` |")
    return lines


def _render_record_table(records: tuple[MatmulRecord, ...]) -> list[str]:
    header = "| " + " | ".join(f"`{column}`" for column in RECORD_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in RECORD_COLUMNS) + " |"
    lines = [header, separator]

    for record in records:
        lines.append(
            "| "
            + " | ".join(_cell(getattr(record, column)) for column in RECORD_COLUMNS)
            + " |"
        )

    return lines


def _cell(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", r"\|")
