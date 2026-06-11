"""Shared data structures for static matmul statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MatmulRecord:
    """One symbolic matmul family in a model architecture."""

    model: str
    layer_range: str
    block: str
    op_name: str
    op_kind: str
    lhs_shape: str
    rhs_shape: str
    output_shape: str
    batching: str
    repeat_count: str
    active_condition: str
    logical_vs_implementation: str
    activation_after: str | None = None
    numeric_format: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ModelStats:
    """Static matmul records plus short model-level context."""

    model: str
    source: str
    records: tuple[MatmulRecord, ...]
    config_summary: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @classmethod
    def from_records(
        cls,
        *,
        model: str,
        source: str,
        records: list[MatmulRecord] | tuple[MatmulRecord, ...],
        config_summary: dict[str, Any] | None = None,
        notes: list[str] | tuple[str, ...] | None = None,
    ) -> "ModelStats":
        return cls(
            model=model,
            source=source,
            records=tuple(records),
            config_summary=dict(config_summary or {}),
            notes=tuple(notes or ()),
        )
