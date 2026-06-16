# MOE Matmul Stats Report

This report contains static, config-derived matmul shape families. It does not require model weights.

## DeepSeek-V3

- Source: `huggingface:deepseek-ai/DeepSeek-V3@main`
- Matmul families: `18`

### Config Summary

| Key | Value |
| --- | --- |
| `first_k_dense_replace` | `3` |
| `hidden_act` | `silu` |
| `hidden_size` | `7168` |
| `intermediate_size` | `18432` |
| `kv_lora_rank` | `512` |
| `model_type` | `deepseek_v3` |
| `moe_intermediate_size` | `2048` |
| `moe_layer_freq` | `1` |
| `n_routed_experts` | `256` |
| `n_shared_experts` | `1` |
| `num_attention_heads` | `128` |
| `num_experts_per_tok` | `8` |
| `num_hidden_layers` | `61` |
| `num_key_value_heads` | `128` |
| `num_nextn_predict_layers` | `1` |
| `q_lora_rank` | `1536` |
| `qk_nope_head_dim` | `128` |
| `qk_rope_head_dim` | `64` |
| `quantization_config` | `{"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128]}` |
| `torch_dtype` | `bfloat16` |
| `v_head_dim` | `128` |
| `vocab_size` | `129280` |

### Matmul Families

| `layer_range` | `block` | `op_name` | `op_kind` | `lhs_shape` | `rhs_shape` | `output_shape` | `batching` | `repeat_count` | `active_condition` | `logical_vs_implementation` | `activation_after` | `numeric_format` | `notes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0..60 | attention | q_a_proj | linear | [T, H=7168] | [Q_RANK=1536, H=7168] | [T, Q_RANK=1536] | dense token batch T=B*S | 1 per decoder layer | every token | implementation MLA low-rank query A projection | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | q_b_proj | linear | [T, Q_RANK=1536] | [A*QD=24576, Q_RANK=1536] | [T, A*QD=24576] | dense token batch T=B*S | 1 per decoder layer | after q_a RMSNorm | implementation MLA low-rank query B projection | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | kv_a_proj_with_mqa | linear | [T, H=7168] | [KV_RANK+ROPE=576, H=7168] | [T, KV_RANK+ROPE=576] | dense token batch T=B*S | 1 per decoder layer | every token | implementation MLA compressed KV + rope projection | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | kv_b_proj | linear | [T, KV_RANK=512] | [A*(NOPE+VD)=32768, KV_RANK=512] | [T, A*(NOPE+VD)=32768] | dense token batch T=B*S | 1 per decoder layer | after kv_a RMSNorm | implementation MLA low-rank KV B projection | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | qk_scores | batched_matmul | [B, A=128, S, QD=192] | [B, A=128, QD=192, KV] | [B, A=128, S, KV] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | after MLA Q/K construction and RoPE | logical attention matmul | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | attn_values | batched_matmul | [B, A=128, S, KV] | [B, A=128, KV, VD=128] | [B, A=128, S, VD=128] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | after attention softmax | logical attention matmul | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..60 | attention | o_proj | linear | [T, A*VD=16384] | [H=7168, A*VD=16384] | [T, H=7168] | dense token batch T=B*S | 1 per decoder layer | after attention value matmul | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..2 | dense_mlp | gate_proj | linear | [T, H=7168] | [I=18432, H=7168] | [T, I=18432] | dense token batch T=B*S | first 3 decoder layers | dense MLP layers before MoE replacement | implementation | silu | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..2 | dense_mlp | up_proj | linear | [T, H=7168] | [I=18432, H=7168] | [T, I=18432] | dense token batch T=B*S | first 3 decoder layers | dense MLP layers before MoE replacement | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 0..2 | dense_mlp | down_proj | linear | [T, I=18432] | [H=7168, I=18432] | [T, H=7168] | dense token batch T=B*S | first 3 decoder layers | after gated dense MLP activation | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 3..60 | router | router_logits | linear | [T, H=7168] | [E=256, H=7168] | [T, E=256] | dense token batch T=B*S | 1 per MoE decoder layer | MoE layers only; top K=8 routed experts selected | implementation | sigmoid + grouped topk | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 3..60 | shared_expert_mlp | shared_gate_proj | linear | [T, H=7168] | [I_shared=2048, H=7168] | [T, I_shared=2048] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | every token in MoE layers | implementation | silu | torch_dtype=bfloat16; quant=fp8/e4m3 | Skipped if n_shared_experts is 0. |
| 3..60 | shared_expert_mlp | shared_up_proj | linear | [T, H=7168] | [I_shared=2048, H=7168] | [T, I_shared=2048] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | every token in MoE layers | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | Skipped if n_shared_experts is 0. |
| 3..60 | shared_expert_mlp | shared_down_proj | linear | [T, I_shared=2048] | [H=7168, I_shared=2048] | [T, H=7168] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | after shared expert gated activation | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | Skipped if n_shared_experts is 0. |
| 3..60 | routed_expert_mlp | routed_gate_proj | grouped_expert_matmul | [N_e, H=7168] | [I_moe=2048, H=7168] per routed expert | [N_e, I_moe=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=256 nonempty routed experts per MoE layer | selected routed experts only, K=8 per token | logical routed expert matmul | silu | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 3..60 | routed_expert_mlp | routed_up_proj | grouped_expert_matmul | [N_e, H=7168] | [I_moe=2048, H=7168] per routed expert | [N_e, I_moe=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=256 nonempty routed experts per MoE layer | selected routed experts only, K=8 per token | logical routed expert matmul | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| 3..60 | routed_expert_mlp | routed_down_proj | grouped_expert_matmul | [N_e, I_moe=2048] | [H=7168, I_moe=2048] per routed expert | [N_e, H=7168] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=256 nonempty routed experts per MoE layer | selected routed experts only after gated activation | logical routed expert matmul | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |
| final | lm_head | lm_head | linear | [T, H=7168] | [VOCAB=129280, H=7168] | [T, VOCAB=129280] | dense token batch T=B*S; often last token only during decode | 1 | after final RMSNorm | implementation | - | torch_dtype=bfloat16; quant=fp8/e4m3 | - |

### Notes

- DeepSeek-V3 uses MLA attention, so Q and KV projections are low-rank/compressed rather than standard Q/K/V projections.
- First 3 decoder layers use dense MLP; later MoE layers are 3..60.
- Routed expert rows use N_e because exact expert token counts are runtime-dependent.
- Config exposes num_nextn_predict_layers for MTP; this report covers the main decoder stack and LM head.
