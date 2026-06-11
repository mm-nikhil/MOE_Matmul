# MOE Matmul Stats Report

This report contains static, config-derived matmul shape families. It does not require model weights.

## Nano-MoE-JAX

- Source: `github:carrycooldude/Nano-MoE-JAX defaults`
- Matmul families: `8`

### Config Summary

| Key | Value |
| --- | --- |
| `batch_size` | `32` |
| `block_size` | `128` |
| `d_ff` | `512` |
| `d_model` | `128` |
| `dropout_rate` | `0.1` |
| `model_type` | `nano_moe_jax` |
| `n_experts` | `4` |
| `n_heads` | `4` |
| `n_layers` | `4` |
| `top_k` | `2` |
| `vocab_size` | `256` |

### Matmul Families

| `layer_range` | `block` | `op_name` | `op_kind` | `lhs_shape` | `rhs_shape` | `output_shape` | `batching` | `repeat_count` | `active_condition` | `logical_vs_implementation` | `activation_after` | `numeric_format` | `notes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0..3 | attention | qkv_proj | linear | [T, H=128] | [3*H=384, H=128] | [T, 3*H=384] | dense token batch T=B*S | 1 per transformer block | every token | implementation fused QKV projection | - | jax default float32 unless caller casts parameters | - |
| 0..3 | attention | qk_scores | batched_matmul | [B, A=4, S, D=32] | [B, A=4, D=32, S] | [B, A=4, S, S] | attention batch over B*A heads; causal prefill | 1 per transformer block | every token after QKV projection | implementation | - | jax default float32 unless caller casts parameters | - |
| 0..3 | attention | attn_values | batched_matmul | [B, A=4, S, S] | [B, A=4, S, D=32] | [B, A=4, S, D=32] | attention batch over B*A heads; causal prefill | 1 per transformer block | after attention softmax | implementation | - | jax default float32 unless caller casts parameters | - |
| 0..3 | attention | o_proj | linear | [T, H=128] | [H=128, H=128] | [T, H=128] | dense token batch T=B*S | 1 per transformer block | every token after attention value matmul | implementation | - | jax default float32 unless caller casts parameters | - |
| 0..3 | router | router_logits | linear | [T, H=128] | [E=4, H=128] | [T, E=4] | dense token batch T=B*S | 1 per MoE layer | every token; top K=2 experts selected | implementation | softmax + topk | jax default float32 unless caller casts parameters | - |
| 0..3 | expert_mlp | expert_fc1 | linear | [T, H=128] | [I=512, H=128] per expert | [T, I=512] per expert | dense token batch T=B*S for every expert | E=4 experts per layer | all experts are computed in this educational implementation | implementation; logical sparse route would use [N_e, H] | gelu | jax default float32 unless caller casts parameters | - |
| 0..3 | expert_mlp | expert_fc2 | linear | [T, I=512] per expert | [H=128, I=512] per expert | [T, H=128] per expert | dense token batch T=B*S for every expert | E=4 experts per layer | all experts are computed, then selected outputs are gathered | implementation; logical sparse route would use [N_e, I] | - | jax default float32 unless caller casts parameters | - |
| final | lm_head | lm_head | linear | [T, H=128] | [VOCAB=256, H=128] | [T, VOCAB=256] | dense token batch T=B*S | 1 | after final LayerNorm | implementation | - | jax default float32 unless caller casts parameters | - |

### Notes

- Nano-MoE-JAX is a small educational implementation, not a production sparse MoE kernel.
- Router selects K=2, but the implementation computes all E=4 experts first.

## allenai/OLMoE-1B-7B-0125

- Source: `huggingface:allenai/OLMoE-1B-7B-0125@main`
- Matmul families: `10`

### Config Summary

| Key | Value |
| --- | --- |
| `hidden_act` | `silu` |
| `hidden_size` | `2048` |
| `intermediate_size` | `1024` |
| `model_type` | `olmoe` |
| `num_attention_heads` | `16` |
| `num_experts` | `64` |
| `num_experts_per_tok` | `8` |
| `num_hidden_layers` | `16` |
| `num_key_value_heads` | `16` |
| `torch_dtype` | `float32` |
| `vocab_size` | `50304` |

### Matmul Families

| `layer_range` | `block` | `op_name` | `op_kind` | `lhs_shape` | `rhs_shape` | `output_shape` | `batching` | `repeat_count` | `active_condition` | `logical_vs_implementation` | `activation_after` | `numeric_format` | `notes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0..15 | attention | q_proj | linear | [T, H=2048] | [A*D=2048, H=2048] | [T, A*D=2048] | dense token batch T=B*S | 1 per decoder layer | every token | implementation | - | torch_dtype=float32 | - |
| 0..15 | attention | k_proj | linear | [T, H=2048] | [KVH*D=2048, H=2048] | [T, KVH*D=2048] | dense token batch T=B*S | 1 per decoder layer | every token | implementation | - | torch_dtype=float32 | - |
| 0..15 | attention | v_proj | linear | [T, H=2048] | [KVH*D=2048, H=2048] | [T, KVH*D=2048] | dense token batch T=B*S | 1 per decoder layer | every token | implementation | - | torch_dtype=float32 | - |
| 0..15 | attention | qk_scores | batched_matmul | [B, A=16, S, D=128] | [B, A=16, D=128, KV] | [B, A=16, S, KV] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | every token after q/k projection | logical attention matmul | - | torch_dtype=float32 | - |
| 0..15 | attention | attn_values | batched_matmul | [B, A=16, S, KV] | [B, A=16, KV, D=128] | [B, A=16, S, D=128] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | after attention softmax | logical attention matmul | - | torch_dtype=float32 | - |
| 0..15 | attention | o_proj | linear | [T, A*D=2048] | [H=2048, A*D=2048] | [T, H=2048] | dense token batch T=B*S | 1 per decoder layer | every token after attention value matmul | implementation | - | torch_dtype=float32 | - |
| 0..15 | router | router_logits | linear | [T, H=2048] | [E=64, H=2048] | [T, E=64] | dense token batch T=B*S | 1 per decoder layer | every token; top K=8 experts selected | implementation | softmax + topk | torch_dtype=float32 | - |
| 0..15 | expert_mlp | gate_up_proj | grouped_expert_matmul | [N_e, H=2048] | [2*I=2048, H=2048] per expert | [N_e, 2*I=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=64 nonempty experts per layer | selected experts only, K=8 per token | implementation fused gate+up projection | silu on gate chunk, multiplied by up chunk | torch_dtype=float32 | - |
| 0..15 | expert_mlp | down_proj | grouped_expert_matmul | [N_e, I=1024] | [H=2048, I=1024] per expert | [N_e, H=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=64 nonempty experts per layer | selected experts only after gated activation | implementation | - | torch_dtype=float32 | - |
| final | lm_head | lm_head | linear | [T, H=2048] | [VOCAB=50304, H=2048] | [T, VOCAB=50304] | dense token batch T=B*S; often last token only during decode | 1 | after final RMSNorm | implementation | - | torch_dtype=float32 | - |

### Notes

- Routed expert rows use N_e because exact expert token counts are runtime-dependent.
- HF OLMoE fuses expert gate and up projections into one gate_up_proj weight.

## deepseek-ai/DeepSeek-V3

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
