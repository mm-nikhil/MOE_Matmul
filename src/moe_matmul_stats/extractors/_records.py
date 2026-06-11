"""Helpers for building matmul records."""

from __future__ import annotations

from moe_matmul_stats.schema import MatmulRecord


def record(
    *,
    model: str,
    layer_range: str,
    block: str,
    op_name: str,
    op_kind: str,
    lhs_shape: str,
    rhs_shape: str,
    output_shape: str,
    batching: str,
    repeat_count: str,
    active_condition: str,
    logical_vs_implementation: str,
    activation_after: str | None = None,
    numeric_format: str | None = None,
    notes: str | None = None,
) -> MatmulRecord:
    return MatmulRecord(
        model=model,
        layer_range=layer_range,
        block=block,
        op_name=op_name,
        op_kind=op_kind,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        output_shape=output_shape,
        batching=batching,
        repeat_count=repeat_count,
        active_condition=active_condition,
        logical_vs_implementation=logical_vs_implementation,
        activation_after=activation_after,
        numeric_format=numeric_format,
        notes=notes,
    )


def layer_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}..{end}"


def numeric_format_from_config(config: dict | object) -> str:
    if not isinstance(config, dict):
        return "unknown"

    torch_dtype = config.get("torch_dtype")
    quant = config.get("quantization_config")
    parts: list[str] = []
    if torch_dtype:
        parts.append(f"torch_dtype={torch_dtype}")
    if isinstance(quant, dict):
        method = quant.get("quant_method")
        fmt = quant.get("fmt")
        if method and fmt:
            parts.append(f"quant={method}/{fmt}")
        elif method:
            parts.append(f"quant={method}")
    return "; ".join(parts) if parts else "unknown"
