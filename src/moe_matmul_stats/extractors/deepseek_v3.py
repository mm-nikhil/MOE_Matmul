"""DeepSeek-V3 static matmul extractor."""

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
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_nextn_predict_layers",
    "first_k_dense_replace",
    "moe_layer_freq",
    "num_attention_heads",
    "num_key_value_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "hidden_act",
    "torch_dtype",
    "quantization_config",
    "vocab_size",
)


def extract_deepseek_v3(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    cfg = dict(config)
    hidden_size = int(cfg["hidden_size"])
    dense_intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])
    first_dense = int(cfg["first_k_dense_replace"])
    moe_freq = int(cfg["moe_layer_freq"])
    num_heads = int(cfg["num_attention_heads"])
    q_rank = int(cfg["q_lora_rank"])
    kv_rank = int(cfg["kv_lora_rank"])
    qk_nope = int(cfg["qk_nope_head_dim"])
    qk_rope = int(cfg["qk_rope_head_dim"])
    value_dim = int(cfg["v_head_dim"])
    num_experts = int(cfg["n_routed_experts"])
    shared_experts = int(cfg.get("n_shared_experts") or 0)
    top_k = int(cfg["num_experts_per_tok"])
    vocab_size = int(cfg["vocab_size"])
    hidden_act = str(cfg["hidden_act"])

    q_head_dim = qk_nope + qk_rope
    q_width = num_heads * q_head_dim
    kv_a_width = kv_rank + qk_rope
    kv_b_width = num_heads * (qk_nope + value_dim)
    o_width = num_heads * value_dim
    shared_intermediate = moe_intermediate * shared_experts

    full_span = layer_range(0, num_layers - 1)
    dense_span = layer_range(0, first_dense - 1)
    moe_indices = [
        layer_idx
        for layer_idx in range(first_dense, num_layers)
        if (layer_idx - first_dense) % moe_freq == 0
    ]
    moe_span = _format_layer_indices(moe_indices)
    numeric_format = numeric_format_from_config(cfg)

    records = [
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="q_a_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[Q_RANK={q_rank}, H={hidden_size}]",
            output_shape=f"[T, Q_RANK={q_rank}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation MLA low-rank query A projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="q_b_proj",
            op_kind="linear",
            lhs_shape=f"[T, Q_RANK={q_rank}]",
            rhs_shape=f"[A*QD={q_width}, Q_RANK={q_rank}]",
            output_shape=f"[T, A*QD={q_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="after q_a RMSNorm",
            logical_vs_implementation="implementation MLA low-rank query B projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="kv_a_proj_with_mqa",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[KV_RANK+ROPE={kv_a_width}, H={hidden_size}]",
            output_shape=f"[T, KV_RANK+ROPE={kv_a_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation MLA compressed KV + rope projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="kv_b_proj",
            op_kind="linear",
            lhs_shape=f"[T, KV_RANK={kv_rank}]",
            rhs_shape=f"[A*(NOPE+VD)={kv_b_width}, KV_RANK={kv_rank}]",
            output_shape=f"[T, A*(NOPE+VD)={kv_b_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="after kv_a RMSNorm",
            logical_vs_implementation="implementation MLA low-rank KV B projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="qk_scores",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, QD={q_head_dim}]",
            rhs_shape=f"[B, A={num_heads}, QD={q_head_dim}, KV]",
            output_shape=f"[B, A={num_heads}, S, KV]",
            batching="attention batch over B*A heads; prefill KV=S, decode S=1",
            repeat_count="1 per decoder layer",
            active_condition="after MLA Q/K construction and RoPE",
            logical_vs_implementation="logical attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="attn_values",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, KV]",
            rhs_shape=f"[B, A={num_heads}, KV, VD={value_dim}]",
            output_shape=f"[B, A={num_heads}, S, VD={value_dim}]",
            batching="attention batch over B*A heads; prefill KV=S, decode S=1",
            repeat_count="1 per decoder layer",
            active_condition="after attention softmax",
            logical_vs_implementation="logical attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="o_proj",
            op_kind="linear",
            lhs_shape=f"[T, A*VD={o_width}]",
            rhs_shape=f"[H={hidden_size}, A*VD={o_width}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="after attention value matmul",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=dense_span,
            block="dense_mlp",
            op_name="gate_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[I={dense_intermediate}, H={hidden_size}]",
            output_shape=f"[T, I={dense_intermediate}]",
            batching="dense token batch T=B*S",
            repeat_count=f"first {first_dense} decoder layers",
            active_condition="dense MLP layers before MoE replacement",
            logical_vs_implementation="implementation",
            activation_after=hidden_act,
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=dense_span,
            block="dense_mlp",
            op_name="up_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[I={dense_intermediate}, H={hidden_size}]",
            output_shape=f"[T, I={dense_intermediate}]",
            batching="dense token batch T=B*S",
            repeat_count=f"first {first_dense} decoder layers",
            active_condition="dense MLP layers before MoE replacement",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=dense_span,
            block="dense_mlp",
            op_name="down_proj",
            op_kind="linear",
            lhs_shape=f"[T, I={dense_intermediate}]",
            rhs_shape=f"[H={hidden_size}, I={dense_intermediate}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count=f"first {first_dense} decoder layers",
            active_condition="after gated dense MLP activation",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="router",
            op_name="router_logits",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[E={num_experts}, H={hidden_size}]",
            output_shape=f"[T, E={num_experts}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per MoE decoder layer",
            active_condition=f"MoE layers only; top K={top_k} routed experts selected",
            logical_vs_implementation="implementation",
            activation_after="sigmoid + grouped topk",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="shared_expert_mlp",
            op_name="shared_gate_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[I_shared={shared_intermediate}, H={hidden_size}]",
            output_shape=f"[T, I_shared={shared_intermediate}]",
            batching="dense token batch T=B*S",
            repeat_count=f"1 shared expert block representing {shared_experts} shared expert(s)",
            active_condition="every token in MoE layers",
            logical_vs_implementation="implementation",
            activation_after=hidden_act,
            numeric_format=numeric_format,
            notes="Skipped if n_shared_experts is 0.",
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="shared_expert_mlp",
            op_name="shared_up_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[I_shared={shared_intermediate}, H={hidden_size}]",
            output_shape=f"[T, I_shared={shared_intermediate}]",
            batching="dense token batch T=B*S",
            repeat_count=f"1 shared expert block representing {shared_experts} shared expert(s)",
            active_condition="every token in MoE layers",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
            notes="Skipped if n_shared_experts is 0.",
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="shared_expert_mlp",
            op_name="shared_down_proj",
            op_kind="linear",
            lhs_shape=f"[T, I_shared={shared_intermediate}]",
            rhs_shape=f"[H={hidden_size}, I_shared={shared_intermediate}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count=f"1 shared expert block representing {shared_experts} shared expert(s)",
            active_condition="after shared expert gated activation",
            logical_vs_implementation="implementation",
            numeric_format=numeric_format,
            notes="Skipped if n_shared_experts is 0.",
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="routed_expert_mlp",
            op_name="routed_gate_proj",
            op_kind="grouped_expert_matmul",
            lhs_shape=f"[N_e, H={hidden_size}]",
            rhs_shape=f"[I_moe={moe_intermediate}, H={hidden_size}] per routed expert",
            output_shape=f"[N_e, I_moe={moe_intermediate}]",
            batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
            repeat_count=f"up to E={num_experts} nonempty routed experts per MoE layer",
            active_condition=f"selected routed experts only, K={top_k} per token",
            logical_vs_implementation="logical routed expert matmul",
            activation_after=hidden_act,
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="routed_expert_mlp",
            op_name="routed_up_proj",
            op_kind="grouped_expert_matmul",
            lhs_shape=f"[N_e, H={hidden_size}]",
            rhs_shape=f"[I_moe={moe_intermediate}, H={hidden_size}] per routed expert",
            output_shape=f"[N_e, I_moe={moe_intermediate}]",
            batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
            repeat_count=f"up to E={num_experts} nonempty routed experts per MoE layer",
            active_condition=f"selected routed experts only, K={top_k} per token",
            logical_vs_implementation="logical routed expert matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=moe_span,
            block="routed_expert_mlp",
            op_name="routed_down_proj",
            op_kind="grouped_expert_matmul",
            lhs_shape=f"[N_e, I_moe={moe_intermediate}]",
            rhs_shape=f"[H={hidden_size}, I_moe={moe_intermediate}] per routed expert",
            output_shape=f"[N_e, H={hidden_size}]",
            batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
            repeat_count=f"up to E={num_experts} nonempty routed experts per MoE layer",
            active_condition="selected routed experts only after gated activation",
            logical_vs_implementation="logical routed expert matmul",
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
            "DeepSeek-V3 uses MLA attention, so Q and KV projections are low-rank/compressed rather than standard Q/K/V projections.",
            f"First {first_dense} decoder layers use dense MLP; later MoE layers are {moe_span}.",
            "Routed expert rows use N_e because exact expert token counts are runtime-dependent.",
            "Config exposes num_nextn_predict_layers for MTP; this report covers the main decoder stack and LM head.",
        ],
    )


def _format_layer_indices(indices: list[int]) -> str:
    if not indices:
        return "-"

    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(layer_range(start, previous))
        start = previous = value
    ranges.append(layer_range(start, previous))
    return ",".join(ranges)
