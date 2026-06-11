"""Nano-MoE-JAX static matmul extractor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from moe_matmul_stats.schema import ModelStats

from ._records import layer_range, record
from ._summary import summarize_config


SUMMARY_KEYS = (
    "model_type",
    "vocab_size",
    "n_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "n_experts",
    "top_k",
    "block_size",
    "dropout_rate",
    "batch_size",
)


def extract_nano_jax(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    cfg = dict(config)
    hidden_size = int(cfg["d_model"])
    intermediate_size = int(cfg["d_ff"])
    num_layers = int(cfg["n_layers"])
    num_heads = int(cfg["n_heads"])
    num_experts = int(cfg["n_experts"])
    top_k = int(cfg["top_k"])
    vocab_size = int(cfg["vocab_size"])
    head_dim = hidden_size // num_heads
    layer_span = layer_range(0, num_layers - 1)
    numeric_format = "jax default float32 unless caller casts parameters"

    records = [
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="qkv_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[3*H={3 * hidden_size}, H={hidden_size}]",
            output_shape=f"[T, 3*H={3 * hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per transformer block",
            active_condition="every token",
            logical_vs_implementation="implementation fused QKV projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="qk_scores",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            rhs_shape=f"[B, A={num_heads}, D={head_dim}, S]",
            output_shape=f"[B, A={num_heads}, S, S]",
            batching="attention batch over B*A heads; causal prefill",
            repeat_count="1 per transformer block",
            active_condition="every token after QKV projection",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="attn_values",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, S]",
            rhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            output_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            batching="attention batch over B*A heads; causal prefill",
            repeat_count="1 per transformer block",
            active_condition="after attention softmax",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="o_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[H={hidden_size}, H={hidden_size}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per transformer block",
            active_condition="every token after attention value matmul",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="router",
            op_name="router_logits",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[E={num_experts}, H={hidden_size}]",
            output_shape=f"[T, E={num_experts}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per MoE layer",
            active_condition=f"every token; top K={top_k} experts selected",
            logical_vs_implementation="implementation",
            activation_after="softmax + topk",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="expert_mlp",
            op_name="expert_fc1",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[I={intermediate_size}, H={hidden_size}] per expert",
            output_shape=f"[T, I={intermediate_size}] per expert",
            batching="dense token batch T=B*S for every expert",
            repeat_count=f"E={num_experts} experts per layer",
            active_condition="all experts are computed in this educational implementation",
            logical_vs_implementation="implementation; logical sparse route would use [N_e, H]",
            activation_after="gelu",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="expert_mlp",
            op_name="expert_fc2",
            op_kind="linear",
            lhs_shape=f"[T, I={intermediate_size}] per expert",
            rhs_shape=f"[H={hidden_size}, I={intermediate_size}] per expert",
            output_shape=f"[T, H={hidden_size}] per expert",
            batching="dense token batch T=B*S for every expert",
            repeat_count=f"E={num_experts} experts per layer",
            active_condition="all experts are computed, then selected outputs are gathered",
            logical_vs_implementation="implementation; logical sparse route would use [N_e, I]",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range="final",
            block="lm_head",
            op_name="lm_head",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[VOCAB={vocab_size}, H={hidden_size}]",
            output_shape=f"[T, VOCAB={vocab_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1",
            active_condition="after final LayerNorm",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
    ]

    return ModelStats.from_records(
        model=model,
        source=source,
        records=records,
        config_summary=summarize_config(cfg, SUMMARY_KEYS),
        notes=[
            "Nano-MoE-JAX is a small educational implementation, not a production sparse MoE kernel.",
            f"Router selects K={top_k}, but the implementation computes all E={num_experts} experts first.",
        ],
    )
