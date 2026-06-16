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
