# MOE Matmul Stats Report

This report contains static, config-derived matmul shape families. It does not require model weights.

## OLMoE-1B-7B

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
