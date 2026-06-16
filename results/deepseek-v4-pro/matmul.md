# MOE Matmul Stats Report

This report contains static, config-derived matmul shape families. It does not require model weights.

## DeepSeek-V4-Pro

- Source: `huggingface:deepseek-ai/DeepSeek-V4-Pro@main`
- Matmul families: `30`

### Config Summary

| Key | Value |
| --- | --- |
| `compress_ratios` | `[128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]` |
| `derived.layer_type_counts` | `{"compressed_sparse_attention": 30, "heavily_compressed_attention": 31}` |
| `derived.mlp_layer_type_counts` | `{"hash_moe": 3, "moe": 58}` |
| `expert_dtype` | `fp4` |
| `hc_mult` | `4` |
| `hc_sinkhorn_iters` | `20` |
| `head_dim` | `512` |
| `hidden_act` | `silu` |
| `hidden_size` | `7168` |
| `index_head_dim` | `128` |
| `index_n_heads` | `64` |
| `index_topk` | `1024` |
| `max_position_embeddings` | `1048576` |
| `model_type` | `deepseek_v4` |
| `moe_intermediate_size` | `3072` |
| `n_routed_experts` | `384` |
| `n_shared_experts` | `1` |
| `norm_topk_prob` | `True` |
| `num_attention_heads` | `128` |
| `num_experts_per_tok` | `6` |
| `num_hash_layers` | `3` |
| `num_hidden_layers` | `61` |
| `num_key_value_heads` | `1` |
| `o_groups` | `16` |
| `o_lora_rank` | `1024` |
| `q_lora_rank` | `1536` |
| `qk_rope_head_dim` | `64` |
| `quantization_config` | `{"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}` |
| `scoring_func` | `sqrtsoftplus` |
| `sliding_window` | `128` |
| `swiglu_limit` | `10.0` |
| `topk_method` | `noaux_tc` |
| `torch_dtype` | `bfloat16` |
| `vocab_size` | `129280` |

### Matmul Families

| `layer_range` | `block` | `op_name` | `op_kind` | `lhs_shape` | `rhs_shape` | `output_shape` | `batching` | `repeat_count` | `active_condition` | `logical_vs_implementation` | `activation_after` | `numeric_format` | `notes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0..60 | attention | q_a_proj | linear | [T, H=7168] | [Q_RANK=1536, H=7168] | [T, Q_RANK=1536] | dense token batch T=B*S | 1 per decoder layer | every token | implementation low-rank query A projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | q_b_proj | linear | [T, Q_RANK=1536] | [A*D=65536, Q_RANK=1536] | [T, A*D=65536] | dense token batch T=B*S | 1 per decoder layer | after q_a RMSNorm | implementation low-rank query B projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | kv_proj | linear | [T, H=7168] | [KVH*D=512, H=7168] | [T, KVH*D=512] | dense token batch T=B*S | 1 per decoder layer | every token | implementation shared-KV MQA projection; K and V use same tensor | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | qk_scores_sliding | batched_matmul | [B, A=128, S, D=512] | [B, A=128, D=512, KV_local<=W=128] | [B, A=128, S, KV_local<=W=128] | attention batch over B*A heads; sliding-window local branch | 1 per decoder layer | every token; local KV branch in every V4 attention type | logical sliding-window attention matmul | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | attn_values_sliding | batched_matmul | [B, A=128, S, KV_local<=W=128] | [B, A=128, KV_local<=W=128, D=512] | [B, A=128, S, D=512] | attention batch over B*A heads; sliding-window local branch | 1 per decoder layer | after local attention softmax | logical sliding-window attention matmul | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | o_a_proj_grouped | grouped_linear | [T, G=16, A*D/G=4096] | [G=16, O_RANK=1024, A*D/G=4096] | [T, G=16, O_RANK=1024] | dense token batch T=B*S, grouped over output groups | 1 per decoder layer | after attention value matmul and inverse RoPE on output | implementation grouped output projection A | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | attention | o_b_proj | linear | [T, G*O_RANK=16384] | [H=7168, G*O_RANK=16384] | [T, H=7168] | dense token batch T=B*S | 1 per decoder layer | after grouped output projection A | implementation grouped output projection B | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | hyper_connection | attn_hc_mixer_and_ffn_hc_mixer | linear | [T, HC*H=28672] | [(2+HC)*HC=24, HC*H=28672] | [T, (2+HC)*HC=24] | dense token batch T=B*S | 2 per decoder layer | before attention and before MoE block | implementation mHC stream mixer projection | - | fp32 parameters/accumulation path in Transformers implementation | - |
| 0..1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59 | attention_compressor | hca_compressor_kv_proj | linear | [T, H=7168] | [D=512, H=7168] | [T, D=512] | dense token batch T=B*S | 1 per HCA layer | HCA layers only; one compressed entry per 128 source tokens | implementation HCA compressor KV projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59 | attention_compressor | hca_compressor_gate_proj | linear | [T, H=7168] | [D=512, H=7168] | [T, D=512] | dense token batch T=B*S | 1 per HCA layer | HCA layers only; softmax-gates 128-token compression windows | implementation HCA compressor gate projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59 | attention | qk_scores_hca_compressed | batched_matmul | [B, A=128, S, D=512] | [B, A=128, D=512, C_hca=floor(KV/128)] | [B, A=128, S, C_hca=floor(KV/128)] | attention batch over B*A heads; compressed long-range branch | 1 per HCA layer | HCA layers only, in addition to the sliding-window branch | logical compressed attention matmul | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59 | attention | attn_values_hca_compressed | batched_matmul | [B, A=128, S, C_hca=floor(KV/128)] | [B, A=128, C_hca=floor(KV/128), D=512] | [B, A=128, S, D=512] | attention batch over B*A heads; compressed long-range branch | 1 per HCA layer | HCA layers only, after compressed attention softmax | logical compressed attention matmul | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_compressor_kv_proj | linear | [T, H=7168] | [2*D=1024, H=7168] | [T, 2*D=1024] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; two compressed series over 4-token windows | implementation CSA compressor KV projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_compressor_gate_proj | linear | [T, H=7168] | [2*D=1024, H=7168] | [T, 2*D=1024] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; softmax-gates overlapped 4-token compression windows | implementation CSA compressor gate projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_indexer_kv_proj | linear | [T, H=7168] | [2*ID=256, H=7168] | [T, 2*ID=256] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; builds Lightning Indexer compressed keys | implementation CSA indexer KV projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_indexer_gate_proj | linear | [T, H=7168] | [2*ID=256, H=7168] | [T, 2*ID=256] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; gates Lightning Indexer compressed keys | implementation CSA indexer gate projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_indexer_q_b_proj | linear | [T, Q_RANK=1536] | [IH*ID=8192, Q_RANK=1536] | [T, IH*ID=8192] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; after q_a residual | implementation CSA indexer query projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_indexer_weights_proj | linear | [T, H=7168] | [IH=64, H=7168] | [T, IH=64] | dense token batch T=B*S | 1 per CSA layer | CSA layers only; weights indexer head scores | implementation CSA indexer scorer weights projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention_compressor | csa_indexer_scores | batched_matmul | [B, S, IH=64, ID=128] | [B, C_csa=floor(KV/4), ID=128] | [B, S, IH=64, C_csa=floor(KV/4)] | indexer batch over B*S queries and index heads | 1 per CSA layer | CSA layers only; top I=1024 compressed entries kept per query | implementation Lightning Indexer score matmul | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention | qk_scores_csa_compressed | batched_matmul | [B, A=128, S, D=512] | [B, A=128, D=512, C_csa=floor(KV/4)] | [B, A=128, S, C_csa=floor(KV/4)] | attention batch over B*A heads; compressed sparse long-range branch | 1 per CSA layer | CSA layers only; indexer masks to top I=1024 compressed entries per query | implementation matmul over compressed axis with indexer block bias | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60 | attention | attn_values_csa_compressed | batched_matmul | [B, A=128, S, C_csa=floor(KV/4)] | [B, A=128, C_csa=floor(KV/4), D=512] | [B, A=128, S, D=512] | attention batch over B*A heads; compressed sparse long-range branch | 1 per CSA layer | CSA layers only, after compressed sparse attention softmax | implementation matmul over compressed axis with indexer block bias | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..2 | router | hash_router_logits | linear | [T, H=7168] | [E=384, H=7168] | [T, E=384] | dense token batch T=B*S | 1 per Hash-MoE layer | Hash-MoE bootstrap layers; token-id table selects K=6 experts | implementation router score projection; expert indices come from tid2eid lookup | sqrtsoftplus + hash lookup weights | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 3..60 | router | router_logits | linear | [T, H=7168] | [E=384, H=7168] | [T, E=384] | dense token batch T=B*S | 1 per learned MoE layer | learned MoE layers; top K=6 routed experts selected | implementation top-k router score projection | sqrtsoftplus + topk | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | shared_expert_mlp | shared_gate_proj | linear | [T, H=7168] | [I_shared=3072, H=7168] | [T, I_shared=3072] | dense token batch T=B*S | 1 shared expert block per layer; config n_shared_experts=1 | every token in every MoE layer | implementation shared expert MLP | silu | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | shared_expert_mlp | shared_up_proj | linear | [T, H=7168] | [I_shared=3072, H=7168] | [T, I_shared=3072] | dense token batch T=B*S | 1 shared expert block per layer; config n_shared_experts=1 | every token in every MoE layer | implementation shared expert MLP | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | shared_expert_mlp | shared_down_proj | linear | [T, I_shared=3072] | [H=7168, I_shared=3072] | [T, H=7168] | dense token batch T=B*S | 1 shared expert block per layer; config n_shared_experts=1 | after shared expert gated activation | implementation shared expert MLP | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | routed_expert_mlp | routed_gate_up_proj | grouped_expert_matmul | [N_e, H=7168] | [2*I_moe=6144, H=7168] per routed expert | [N_e, 2*I_moe=6144] | per-expert ragged batch N_e; sum_e N_e=T*K=6T | up to E=384 nonempty routed experts per layer | selected routed experts only, K=6 per token | implementation fused routed expert gate+up projection | silu on gate chunk, multiplied by up chunk | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| 0..60 | routed_expert_mlp | routed_down_proj | grouped_expert_matmul | [N_e, I_moe=3072] | [H=7168, I_moe=3072] per routed expert | [N_e, H=7168] | per-expert ragged batch N_e; sum_e N_e=T*K=6T | up to E=384 nonempty routed experts per layer | selected routed experts only after gated activation | implementation routed expert down projection | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |
| final | hyper_connection | final_hc_head | linear | [T, HC*H=28672] | [HC=4, HC*H=28672] | [T, HC=4] | dense token batch T=B*S | 1 | after final decoder layer before final RMSNorm | implementation final mHC stream collapse projection | - | fp32 parameters/accumulation path in Transformers implementation | - |
| final | lm_head | lm_head | linear | [T, H=7168] | [VOCAB=129280, H=7168] | [T, VOCAB=129280] | dense token batch T=B*S; often last token only during decode | 1 | after final RMSNorm | implementation | - | torch_dtype=bfloat16; expert_dtype=fp4; quant=fp8/e4m3 | - |

### Notes

- DeepSeek-V4 uses shared-KV MQA: kv_proj creates one KV head and the same tensor is used for keys and values.
- Attention layer types from config: HCA=31 (0..1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59), CSA=30 (2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60), sliding=0 (-).
- CSA/HCA compressed sequence counts are runtime dependent: C_csa=floor(KV/4), C_hca=floor(KV/128); CSA masks compressed attention to index_topk entries per query.
- MLP layer types from config: Hash-MoE=3 (0..2), learned MoE=58 (3..60).
- Config exposes num_nextn_predict_layers for MTP; this report covers the main decoder stack and LM head.
