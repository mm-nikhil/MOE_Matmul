"""OLMoE static matmul extractor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from moe_matmul_stats.schema import ModelStats

from ._records import layer_range, numeric_format_from_config, record
from ._summary import summarize_config


SUMMARY_KEYS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "num_experts",
    "num_experts_per_tok",
    "hidden_act",
    "torch_dtype",
    "vocab_size",
)


def extract_olmoe(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    cfg = dict(config)
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    vocab_size = int(cfg["vocab_size"])
    hidden_act = str(cfg["hidden_act"])

    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim
    layer_span = layer_range(0, num_layers - 1)
    numeric_format = numeric_format_from_config(cfg)

    records = [
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="q_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[A*D={num_heads * head_dim}, H={hidden_size}]",
            output_shape=f"[T, A*D={num_heads * head_dim}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="k_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[KVH*D={kv_width}, H={hidden_size}]",
            output_shape=f"[T, KVH*D={kv_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="v_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[KVH*D={kv_width}, H={hidden_size}]",
            output_shape=f"[T, KVH*D={kv_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="qk_scores",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            rhs_shape=f"[B, A={num_heads}, D={head_dim}, KV]",
            output_shape=f"[B, A={num_heads}, S, KV]",
            batching="attention batch over B*A heads; prefill KV=S, decode S=1",
            repeat_count="1 per decoder layer",
            active_condition="every token after q/k projection",
            logical_vs_implementation="logical attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="attn_values",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, KV]",
            rhs_shape=f"[B, A={num_heads}, KV, D={head_dim}]",
            output_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            batching="attention batch over B*A heads; prefill KV=S, decode S=1",
            repeat_count="1 per decoder layer",
            active_condition="after attention softmax",
            logical_vs_implementation="logical attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="attention",
            op_name="o_proj",
            op_kind="linear",
            lhs_shape=f"[T, A*D={num_heads * head_dim}]",
            rhs_shape=f"[H={hidden_size}, A*D={num_heads * head_dim}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
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
            repeat_count="1 per decoder layer",
            active_condition=f"every token; top K={top_k} experts selected",
            logical_vs_implementation="implementation",
            activation_after="softmax + topk",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="expert_mlp",
            op_name="gate_up_proj",
            op_kind="grouped_expert_matmul",
            lhs_shape=f"[N_e, H={hidden_size}]",
            rhs_shape=f"[2*I={2 * intermediate_size}, H={hidden_size}] per expert",
            output_shape=f"[N_e, 2*I={2 * intermediate_size}]",
            batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
            repeat_count=f"up to E={num_experts} nonempty experts per layer",
            active_condition=f"selected experts only, K={top_k} per token",
            logical_vs_implementation="implementation fused gate+up projection",
            activation_after=f"{hidden_act} on gate chunk, multiplied by up chunk",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=layer_span,
            block="expert_mlp",
            op_name="down_proj",
            op_kind="grouped_expert_matmul",
            lhs_shape=f"[N_e, I={intermediate_size}]",
            rhs_shape=f"[H={hidden_size}, I={intermediate_size}] per expert",
            output_shape=f"[N_e, H={hidden_size}]",
            batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
            repeat_count=f"up to E={num_experts} nonempty experts per layer",
            active_condition="selected experts only after gated activation",
            logical_vs_implementation="implementation",
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
            batching="dense token batch T=B*S; often last token only during decode",
            repeat_count="1",
            active_condition="after final RMSNorm",
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
            "Routed expert rows use N_e because exact expert token counts are runtime-dependent.",
            "HF OLMoE fuses expert gate and up projections into one gate_up_proj weight.",
        ],
    )
