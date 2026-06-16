"""DeepSeek-V4 static matmul extractor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from moe_matmul_stats.schema import ModelStats

from ._records import layer_range, numeric_format_from_config, record
from ._summary import summarize_config


SUMMARY_KEYS = (
    "model_type",
    "hidden_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_hash_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "q_lora_rank",
    "qk_rope_head_dim",
    "sliding_window",
    "compress_ratios",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "o_groups",
    "o_lora_rank",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "scoring_func",
    "topk_method",
    "norm_topk_prob",
    "hidden_act",
    "swiglu_limit",
    "hc_mult",
    "hc_sinkhorn_iters",
    "max_position_embeddings",
    "expert_dtype",
    "torch_dtype",
    "quantization_config",
    "vocab_size",
)

LAYER_TYPE_BY_COMPRESS_RATIO = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


def extract_deepseek_v4(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    cfg = dict(config)
    hidden_size = int(cfg["hidden_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", 1))
    head_dim = int(cfg["head_dim"])
    q_rank = int(cfg["q_lora_rank"])
    rope_dim = int(cfg.get("qk_rope_head_dim") or 64)
    sliding_window = int(cfg["sliding_window"])
    index_heads = int(cfg["index_n_heads"])
    index_head_dim = int(cfg["index_head_dim"])
    index_topk = int(cfg["index_topk"])
    o_groups = int(cfg["o_groups"])
    o_rank = int(cfg["o_lora_rank"])
    num_experts = int(cfg["n_routed_experts"])
    shared_experts = int(cfg.get("n_shared_experts") or 0)
    top_k = int(cfg["num_experts_per_tok"])
    vocab_size = int(cfg["vocab_size"])
    hidden_act = str(cfg["hidden_act"])
    hc_mult = int(cfg["hc_mult"])

    layer_types = _expand_attention_layer_types(cfg)
    mlp_layer_types = _expand_mlp_layer_types(cfg)
    hca_indices = _indices(layer_types, "heavily_compressed_attention")
    csa_indices = _indices(layer_types, "compressed_sparse_attention")
    sliding_indices = _indices(layer_types, "sliding_attention")
    hash_indices = _indices(mlp_layer_types, "hash_moe")
    learned_moe_indices = _indices(mlp_layer_types, "moe")

    full_span = layer_range(0, num_layers - 1)
    hca_span = _format_layer_indices(hca_indices)
    csa_span = _format_layer_indices(csa_indices)
    sliding_span = _format_layer_indices(sliding_indices)
    hash_span = _format_layer_indices(hash_indices)
    learned_moe_span = _format_layer_indices(learned_moe_indices)

    compress_rates = _compress_rates(cfg)
    csa_rate = int(compress_rates["compressed_sparse_attention"])
    hca_rate = int(compress_rates["heavily_compressed_attention"])

    q_width = num_heads * head_dim
    kv_width = kv_heads * head_dim
    group_in = q_width // o_groups
    grouped_o_width = o_groups * o_rank
    index_width = index_heads * index_head_dim
    hc_input = hc_mult * hidden_size
    hc_mix = (2 + hc_mult) * hc_mult
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
            logical_vs_implementation="implementation low-rank query A projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="q_b_proj",
            op_kind="linear",
            lhs_shape=f"[T, Q_RANK={q_rank}]",
            rhs_shape=f"[A*D={q_width}, Q_RANK={q_rank}]",
            output_shape=f"[T, A*D={q_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="after q_a RMSNorm",
            logical_vs_implementation="implementation low-rank query B projection",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="kv_proj",
            op_kind="linear",
            lhs_shape=f"[T, H={hidden_size}]",
            rhs_shape=f"[KVH*D={kv_width}, H={hidden_size}]",
            output_shape=f"[T, KVH*D={kv_width}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="every token",
            logical_vs_implementation="implementation shared-KV MQA projection; K and V use same tensor",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="qk_scores_sliding",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            rhs_shape=f"[B, A={num_heads}, D={head_dim}, KV_local<=W={sliding_window}]",
            output_shape=f"[B, A={num_heads}, S, KV_local<=W={sliding_window}]",
            batching="attention batch over B*A heads; sliding-window local branch",
            repeat_count="1 per decoder layer",
            active_condition="every token; local KV branch in every V4 attention type",
            logical_vs_implementation="logical sliding-window attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="attn_values_sliding",
            op_kind="batched_matmul",
            lhs_shape=f"[B, A={num_heads}, S, KV_local<=W={sliding_window}]",
            rhs_shape=f"[B, A={num_heads}, KV_local<=W={sliding_window}, D={head_dim}]",
            output_shape=f"[B, A={num_heads}, S, D={head_dim}]",
            batching="attention batch over B*A heads; sliding-window local branch",
            repeat_count="1 per decoder layer",
            active_condition="after local attention softmax",
            logical_vs_implementation="logical sliding-window attention matmul",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="o_a_proj_grouped",
            op_kind="grouped_linear",
            lhs_shape=f"[T, G={o_groups}, A*D/G={group_in}]",
            rhs_shape=f"[G={o_groups}, O_RANK={o_rank}, A*D/G={group_in}]",
            output_shape=f"[T, G={o_groups}, O_RANK={o_rank}]",
            batching="dense token batch T=B*S, grouped over output groups",
            repeat_count="1 per decoder layer",
            active_condition="after attention value matmul and inverse RoPE on output",
            logical_vs_implementation="implementation grouped output projection A",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="attention",
            op_name="o_b_proj",
            op_kind="linear",
            lhs_shape=f"[T, G*O_RANK={grouped_o_width}]",
            rhs_shape=f"[H={hidden_size}, G*O_RANK={grouped_o_width}]",
            output_shape=f"[T, H={hidden_size}]",
            batching="dense token batch T=B*S",
            repeat_count="1 per decoder layer",
            active_condition="after grouped output projection A",
            logical_vs_implementation="implementation grouped output projection B",
            numeric_format=numeric_format,
        ),
        record(
            model=model,
            layer_range=full_span,
            block="hyper_connection",
            op_name="attn_hc_mixer_and_ffn_hc_mixer",
            op_kind="linear",
            lhs_shape=f"[T, HC*H={hc_input}]",
            rhs_shape=f"[(2+HC)*HC={hc_mix}, HC*H={hc_input}]",
            output_shape=f"[T, (2+HC)*HC={hc_mix}]",
            batching="dense token batch T=B*S",
            repeat_count="2 per decoder layer",
            active_condition="before attention and before MoE block",
            logical_vs_implementation="implementation mHC stream mixer projection",
            numeric_format="fp32 parameters/accumulation path in Transformers implementation",
        ),
    ]

    if hca_indices:
        records.extend(
            [
                record(
                    model=model,
                    layer_range=hca_span,
                    block="attention_compressor",
                    op_name="hca_compressor_kv_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[D={head_dim}, H={hidden_size}]",
                    output_shape=f"[T, D={head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per HCA layer",
                    active_condition=f"HCA layers only; one compressed entry per {hca_rate} source tokens",
                    logical_vs_implementation="implementation HCA compressor KV projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=hca_span,
                    block="attention_compressor",
                    op_name="hca_compressor_gate_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[D={head_dim}, H={hidden_size}]",
                    output_shape=f"[T, D={head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per HCA layer",
                    active_condition=f"HCA layers only; softmax-gates {hca_rate}-token compression windows",
                    logical_vs_implementation="implementation HCA compressor gate projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=hca_span,
                    block="attention",
                    op_name="qk_scores_hca_compressed",
                    op_kind="batched_matmul",
                    lhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
                    rhs_shape=f"[B, A={num_heads}, D={head_dim}, C_hca=floor(KV/{hca_rate})]",
                    output_shape=f"[B, A={num_heads}, S, C_hca=floor(KV/{hca_rate})]",
                    batching="attention batch over B*A heads; compressed long-range branch",
                    repeat_count="1 per HCA layer",
                    active_condition="HCA layers only, in addition to the sliding-window branch",
                    logical_vs_implementation="logical compressed attention matmul",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=hca_span,
                    block="attention",
                    op_name="attn_values_hca_compressed",
                    op_kind="batched_matmul",
                    lhs_shape=f"[B, A={num_heads}, S, C_hca=floor(KV/{hca_rate})]",
                    rhs_shape=f"[B, A={num_heads}, C_hca=floor(KV/{hca_rate}), D={head_dim}]",
                    output_shape=f"[B, A={num_heads}, S, D={head_dim}]",
                    batching="attention batch over B*A heads; compressed long-range branch",
                    repeat_count="1 per HCA layer",
                    active_condition="HCA layers only, after compressed attention softmax",
                    logical_vs_implementation="logical compressed attention matmul",
                    numeric_format=numeric_format,
                ),
            ]
        )

    if csa_indices:
        records.extend(
            [
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_compressor_kv_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[2*D={2 * head_dim}, H={hidden_size}]",
                    output_shape=f"[T, 2*D={2 * head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition=f"CSA layers only; two compressed series over {csa_rate}-token windows",
                    logical_vs_implementation="implementation CSA compressor KV projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_compressor_gate_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[2*D={2 * head_dim}, H={hidden_size}]",
                    output_shape=f"[T, 2*D={2 * head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition=f"CSA layers only; softmax-gates overlapped {csa_rate}-token compression windows",
                    logical_vs_implementation="implementation CSA compressor gate projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_indexer_kv_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[2*ID={2 * index_head_dim}, H={hidden_size}]",
                    output_shape=f"[T, 2*ID={2 * index_head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition="CSA layers only; builds Lightning Indexer compressed keys",
                    logical_vs_implementation="implementation CSA indexer KV projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_indexer_gate_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[2*ID={2 * index_head_dim}, H={hidden_size}]",
                    output_shape=f"[T, 2*ID={2 * index_head_dim}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition="CSA layers only; gates Lightning Indexer compressed keys",
                    logical_vs_implementation="implementation CSA indexer gate projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_indexer_q_b_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, Q_RANK={q_rank}]",
                    rhs_shape=f"[IH*ID={index_width}, Q_RANK={q_rank}]",
                    output_shape=f"[T, IH*ID={index_width}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition="CSA layers only; after q_a residual",
                    logical_vs_implementation="implementation CSA indexer query projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_indexer_weights_proj",
                    op_kind="linear",
                    lhs_shape=f"[T, H={hidden_size}]",
                    rhs_shape=f"[IH={index_heads}, H={hidden_size}]",
                    output_shape=f"[T, IH={index_heads}]",
                    batching="dense token batch T=B*S",
                    repeat_count="1 per CSA layer",
                    active_condition="CSA layers only; weights indexer head scores",
                    logical_vs_implementation="implementation CSA indexer scorer weights projection",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention_compressor",
                    op_name="csa_indexer_scores",
                    op_kind="batched_matmul",
                    lhs_shape=f"[B, S, IH={index_heads}, ID={index_head_dim}]",
                    rhs_shape=f"[B, C_csa=floor(KV/{csa_rate}), ID={index_head_dim}]",
                    output_shape=f"[B, S, IH={index_heads}, C_csa=floor(KV/{csa_rate})]",
                    batching="indexer batch over B*S queries and index heads",
                    repeat_count="1 per CSA layer",
                    active_condition=f"CSA layers only; top I={index_topk} compressed entries kept per query",
                    logical_vs_implementation="implementation Lightning Indexer score matmul",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention",
                    op_name="qk_scores_csa_compressed",
                    op_kind="batched_matmul",
                    lhs_shape=f"[B, A={num_heads}, S, D={head_dim}]",
                    rhs_shape=f"[B, A={num_heads}, D={head_dim}, C_csa=floor(KV/{csa_rate})]",
                    output_shape=f"[B, A={num_heads}, S, C_csa=floor(KV/{csa_rate})]",
                    batching="attention batch over B*A heads; compressed sparse long-range branch",
                    repeat_count="1 per CSA layer",
                    active_condition=(
                        f"CSA layers only; indexer masks to top I={index_topk} "
                        "compressed entries per query"
                    ),
                    logical_vs_implementation="implementation matmul over compressed axis with indexer block bias",
                    numeric_format=numeric_format,
                ),
                record(
                    model=model,
                    layer_range=csa_span,
                    block="attention",
                    op_name="attn_values_csa_compressed",
                    op_kind="batched_matmul",
                    lhs_shape=f"[B, A={num_heads}, S, C_csa=floor(KV/{csa_rate})]",
                    rhs_shape=f"[B, A={num_heads}, C_csa=floor(KV/{csa_rate}), D={head_dim}]",
                    output_shape=f"[B, A={num_heads}, S, D={head_dim}]",
                    batching="attention batch over B*A heads; compressed sparse long-range branch",
                    repeat_count="1 per CSA layer",
                    active_condition="CSA layers only, after compressed sparse attention softmax",
                    logical_vs_implementation="implementation matmul over compressed axis with indexer block bias",
                    numeric_format=numeric_format,
                ),
            ]
        )

    records.extend(
        [
            record(
                model=model,
                layer_range=hash_span,
                block="router",
                op_name="hash_router_logits",
                op_kind="linear",
                lhs_shape=f"[T, H={hidden_size}]",
                rhs_shape=f"[E={num_experts}, H={hidden_size}]",
                output_shape=f"[T, E={num_experts}]",
                batching="dense token batch T=B*S",
                repeat_count="1 per Hash-MoE layer",
                active_condition=f"Hash-MoE bootstrap layers; token-id table selects K={top_k} experts",
                logical_vs_implementation=(
                    "implementation router score projection; expert indices come from tid2eid lookup"
                ),
                activation_after=f"{cfg.get('scoring_func')} + hash lookup weights",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=learned_moe_span,
                block="router",
                op_name="router_logits",
                op_kind="linear",
                lhs_shape=f"[T, H={hidden_size}]",
                rhs_shape=f"[E={num_experts}, H={hidden_size}]",
                output_shape=f"[T, E={num_experts}]",
                batching="dense token batch T=B*S",
                repeat_count="1 per learned MoE layer",
                active_condition=f"learned MoE layers; top K={top_k} routed experts selected",
                logical_vs_implementation="implementation top-k router score projection",
                activation_after=f"{cfg.get('scoring_func')} + topk",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=full_span,
                block="shared_expert_mlp",
                op_name="shared_gate_proj",
                op_kind="linear",
                lhs_shape=f"[T, H={hidden_size}]",
                rhs_shape=f"[I_shared={moe_intermediate}, H={hidden_size}]",
                output_shape=f"[T, I_shared={moe_intermediate}]",
                batching="dense token batch T=B*S",
                repeat_count=f"1 shared expert block per layer; config n_shared_experts={shared_experts}",
                active_condition="every token in every MoE layer",
                logical_vs_implementation="implementation shared expert MLP",
                activation_after=hidden_act,
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=full_span,
                block="shared_expert_mlp",
                op_name="shared_up_proj",
                op_kind="linear",
                lhs_shape=f"[T, H={hidden_size}]",
                rhs_shape=f"[I_shared={moe_intermediate}, H={hidden_size}]",
                output_shape=f"[T, I_shared={moe_intermediate}]",
                batching="dense token batch T=B*S",
                repeat_count=f"1 shared expert block per layer; config n_shared_experts={shared_experts}",
                active_condition="every token in every MoE layer",
                logical_vs_implementation="implementation shared expert MLP",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=full_span,
                block="shared_expert_mlp",
                op_name="shared_down_proj",
                op_kind="linear",
                lhs_shape=f"[T, I_shared={moe_intermediate}]",
                rhs_shape=f"[H={hidden_size}, I_shared={moe_intermediate}]",
                output_shape=f"[T, H={hidden_size}]",
                batching="dense token batch T=B*S",
                repeat_count=f"1 shared expert block per layer; config n_shared_experts={shared_experts}",
                active_condition="after shared expert gated activation",
                logical_vs_implementation="implementation shared expert MLP",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=full_span,
                block="routed_expert_mlp",
                op_name="routed_gate_up_proj",
                op_kind="grouped_expert_matmul",
                lhs_shape=f"[N_e, H={hidden_size}]",
                rhs_shape=f"[2*I_moe={2 * moe_intermediate}, H={hidden_size}] per routed expert",
                output_shape=f"[N_e, 2*I_moe={2 * moe_intermediate}]",
                batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
                repeat_count=f"up to E={num_experts} nonempty routed experts per layer",
                active_condition=f"selected routed experts only, K={top_k} per token",
                logical_vs_implementation="implementation fused routed expert gate+up projection",
                activation_after=f"{hidden_act} on gate chunk, multiplied by up chunk",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range=full_span,
                block="routed_expert_mlp",
                op_name="routed_down_proj",
                op_kind="grouped_expert_matmul",
                lhs_shape=f"[N_e, I_moe={moe_intermediate}]",
                rhs_shape=f"[H={hidden_size}, I_moe={moe_intermediate}] per routed expert",
                output_shape=f"[N_e, H={hidden_size}]",
                batching=f"per-expert ragged batch N_e; sum_e N_e=T*K={top_k}T",
                repeat_count=f"up to E={num_experts} nonempty routed experts per layer",
                active_condition="selected routed experts only after gated activation",
                logical_vs_implementation="implementation routed expert down projection",
                numeric_format=numeric_format,
            ),
            record(
                model=model,
                layer_range="final",
                block="hyper_connection",
                op_name="final_hc_head",
                op_kind="linear",
                lhs_shape=f"[T, HC*H={hc_input}]",
                rhs_shape=f"[HC={hc_mult}, HC*H={hc_input}]",
                output_shape=f"[T, HC={hc_mult}]",
                batching="dense token batch T=B*S",
                repeat_count="1",
                active_condition="after final decoder layer before final RMSNorm",
                logical_vs_implementation="implementation final mHC stream collapse projection",
                numeric_format="fp32 parameters/accumulation path in Transformers implementation",
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
    )

    type_counts = Counter(layer_types)
    mlp_counts = Counter(mlp_layer_types)
    config_summary = summarize_config(cfg, SUMMARY_KEYS)
    config_summary["derived.layer_type_counts"] = dict(sorted(type_counts.items()))
    config_summary["derived.mlp_layer_type_counts"] = dict(sorted(mlp_counts.items()))

    return ModelStats.from_records(
        model=model,
        source=source,
        records=records,
        config_summary=config_summary,
        notes=[
            "DeepSeek-V4 uses shared-KV MQA: kv_proj creates one KV head and the same "
            "tensor is used for keys and values.",
            f"Attention layer types from config: HCA={len(hca_indices)} ({hca_span}), "
            f"CSA={len(csa_indices)} ({csa_span}), sliding={len(sliding_indices)} "
            f"({sliding_span}).",
            "CSA/HCA compressed sequence counts are runtime dependent: "
            f"C_csa=floor(KV/{csa_rate}), C_hca=floor(KV/{hca_rate}); CSA masks "
            "compressed attention to index_topk entries per query.",
            f"MLP layer types from config: Hash-MoE={len(hash_indices)} ({hash_span}), "
            f"learned MoE={len(learned_moe_indices)} ({learned_moe_span}).",
            "Config exposes num_nextn_predict_layers for MTP; this report covers the main decoder stack and LM head.",
        ],
    )


def _expand_attention_layer_types(config: Mapping[str, Any]) -> list[str]:
    num_layers = int(config["num_hidden_layers"])
    ratios = config.get("compress_ratios")
    if isinstance(ratios, list):
        return [LAYER_TYPE_BY_COMPRESS_RATIO[int(ratio)] for ratio in ratios[:num_layers]]

    interleave = [
        "compressed_sparse_attention" if index % 2 else "heavily_compressed_attention"
        for index in range(max(num_layers - 2, 0))
    ]
    return (["heavily_compressed_attention"] * min(num_layers, 2) + interleave)[:num_layers]


def _expand_mlp_layer_types(config: Mapping[str, Any]) -> list[str]:
    num_layers = int(config["num_hidden_layers"])
    num_hash_layers = int(config.get("num_hash_layers", 3))
    return ["hash_moe"] * min(num_layers, num_hash_layers) + ["moe"] * max(
        0, num_layers - num_hash_layers
    )


def _compress_rates(config: Mapping[str, Any]) -> dict[str, int]:
    rates = config.get("compress_rates")
    if isinstance(rates, Mapping):
        return {
            "compressed_sparse_attention": int(rates.get("compressed_sparse_attention", 4)),
            "heavily_compressed_attention": int(rates.get("heavily_compressed_attention", 128)),
        }
    return {
        "compressed_sparse_attention": int(config.get("compress_rate_csa", 4)),
        "heavily_compressed_attention": int(config.get("compress_rate_hca", 128)),
    }


def _indices(values: list[str], needle: str) -> list[int]:
    return [index for index, value in enumerate(values) if value == needle]


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
