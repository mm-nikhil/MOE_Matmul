# MOE Matmul Stats Report

This report contains static, config-derived matmul shape families. It does not require model weights.

## Kimi-K2.5

- Source: `huggingface:moonshotai/Kimi-K2.5@main`
- Matmul families: `18`

### Config Summary

| Key | Value |
| --- | --- |
| `dtype` | `bfloat16` |
| `model_type` | `kimi_k25` |
| `text.first_k_dense_replace` | `1` |
| `text.hidden_act` | `silu` |
| `text.hidden_size` | `7168` |
| `text.intermediate_size` | `18432` |
| `text.kv_lora_rank` | `512` |
| `text.model_type` | `kimi_k2` |
| `text.moe_intermediate_size` | `2048` |
| `text.moe_layer_freq` | `1` |
| `text.n_routed_experts` | `384` |
| `text.n_shared_experts` | `1` |
| `text.num_attention_heads` | `64` |
| `text.num_experts_per_tok` | `8` |
| `text.num_hidden_layers` | `61` |
| `text.num_key_value_heads` | `64` |
| `text.num_nextn_predict_layers` | `0` |
| `text.q_lora_rank` | `1536` |
| `text.qk_nope_head_dim` | `128` |
| `text.qk_rope_head_dim` | `64` |
| `text.quantization_config` | `{"config_groups": {"group_0": {"input_activations": null, "output_activations": null, "targets": ["Linear"], "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 32, "num_bits": 4, "observer": "minmax", "observer_kwargs": {}, "strategy": "group", "symmetric": true, "type": "int"}}}, "format": "pack-quantized", "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate\|up\|gate_up\|down)_proj.*", "re:.*lm_head.*", "re:vision_tower.*", "re:mm_projector.*"], "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed"}` |
| `text.v_head_dim` | `128` |
| `text.vocab_size` | `163840` |
| `use_unified_vision_chunk` | `True` |
| `vision.text_hidden_size` | `7168` |
| `vision.vt_hidden_size` | `1152` |
| `vision.vt_intermediate_size` | `4304` |
| `vision.vt_num_attention_heads` | `16` |
| `vision.vt_num_hidden_layers` | `27` |

### Matmul Families

| `layer_range` | `block` | `op_name` | `op_kind` | `lhs_shape` | `rhs_shape` | `output_shape` | `batching` | `repeat_count` | `active_condition` | `logical_vs_implementation` | `activation_after` | `numeric_format` | `notes` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0..60 | attention | q_a_proj | linear | [T, H=7168] | [Q_RANK=1536, H=7168] | [T, Q_RANK=1536] | dense token batch T=B*S | 1 per decoder layer | every token | implementation MLA low-rank query A projection | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | q_b_proj | linear | [T, Q_RANK=1536] | [A*QD=12288, Q_RANK=1536] | [T, A*QD=12288] | dense token batch T=B*S | 1 per decoder layer | after q_a RMSNorm | implementation MLA low-rank query B projection | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | kv_a_proj_with_mqa | linear | [T, H=7168] | [KV_RANK+ROPE=576, H=7168] | [T, KV_RANK+ROPE=576] | dense token batch T=B*S | 1 per decoder layer | every token | implementation MLA compressed KV + rope projection | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | kv_b_proj | linear | [T, KV_RANK=512] | [A*(NOPE+VD)=16384, KV_RANK=512] | [T, A*(NOPE+VD)=16384] | dense token batch T=B*S | 1 per decoder layer | after kv_a RMSNorm | implementation MLA low-rank KV B projection | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | qk_scores | batched_matmul | [B, A=64, S, QD=192] | [B, A=64, QD=192, KV] | [B, A=64, S, KV] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | after MLA Q/K construction and RoPE | logical attention matmul | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | attn_values | batched_matmul | [B, A=64, S, KV] | [B, A=64, KV, VD=128] | [B, A=64, S, VD=128] | attention batch over B*A heads; prefill KV=S, decode S=1 | 1 per decoder layer | after attention softmax | logical attention matmul | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0..60 | attention | o_proj | linear | [T, A*VD=8192] | [H=7168, A*VD=8192] | [T, H=7168] | dense token batch T=B*S | 1 per decoder layer | after attention value matmul | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0 | dense_mlp | gate_proj | linear | [T, H=7168] | [I=18432, H=7168] | [T, I=18432] | dense token batch T=B*S | first 1 decoder layers | dense MLP layers before MoE replacement | implementation | silu | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0 | dense_mlp | up_proj | linear | [T, H=7168] | [I=18432, H=7168] | [T, I=18432] | dense token batch T=B*S | first 1 decoder layers | dense MLP layers before MoE replacement | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 0 | dense_mlp | down_proj | linear | [T, I=18432] | [H=7168, I=18432] | [T, H=7168] | dense token batch T=B*S | first 1 decoder layers | after gated dense MLP activation | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 1..60 | router | router_logits | linear | [T, H=7168] | [E=384, H=7168] | [T, E=384] | dense token batch T=B*S | 1 per MoE decoder layer | MoE layers only; top K=8 routed experts selected | implementation | sigmoid + grouped topk | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 1..60 | shared_expert_mlp | shared_gate_proj | linear | [T, H=7168] | [I_shared=2048, H=7168] | [T, I_shared=2048] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | every token in MoE layers | implementation | silu | dtype=bfloat16; quant=compressed-tensors/pack-quantized | Skipped if n_shared_experts is 0. |
| 1..60 | shared_expert_mlp | shared_up_proj | linear | [T, H=7168] | [I_shared=2048, H=7168] | [T, I_shared=2048] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | every token in MoE layers | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | Skipped if n_shared_experts is 0. |
| 1..60 | shared_expert_mlp | shared_down_proj | linear | [T, I_shared=2048] | [H=7168, I_shared=2048] | [T, H=7168] | dense token batch T=B*S | 1 shared expert block representing 1 shared expert(s) | after shared expert gated activation | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | Skipped if n_shared_experts is 0. |
| 1..60 | routed_expert_mlp | routed_gate_proj | grouped_expert_matmul | [N_e, H=7168] | [I_moe=2048, H=7168] per routed expert | [N_e, I_moe=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=384 nonempty routed experts per MoE layer | selected routed experts only, K=8 per token | logical routed expert matmul | silu | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 1..60 | routed_expert_mlp | routed_up_proj | grouped_expert_matmul | [N_e, H=7168] | [I_moe=2048, H=7168] per routed expert | [N_e, I_moe=2048] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=384 nonempty routed experts per MoE layer | selected routed experts only, K=8 per token | logical routed expert matmul | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| 1..60 | routed_expert_mlp | routed_down_proj | grouped_expert_matmul | [N_e, I_moe=2048] | [H=7168, I_moe=2048] per routed expert | [N_e, H=7168] | per-expert ragged batch N_e; sum_e N_e=T*K=8T | up to E=384 nonempty routed experts per MoE layer | selected routed experts only after gated activation | logical routed expert matmul | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |
| final | lm_head | lm_head | linear | [T, H=7168] | [VOCAB=163840, H=7168] | [T, VOCAB=163840] | dense token batch T=B*S; often last token only during decode | 1 | after final RMSNorm | implementation | - | dtype=bfloat16; quant=compressed-tensors/pack-quantized | - |

### Notes

- Kimi K2.5 text decoder uses MLA attention, so Q and KV projections are low-rank/compressed rather than standard Q/K/V projections.
- First 1 decoder layers use dense MLP; later MoE layers are 1..60.
- Routed expert rows use N_e because exact expert token counts are runtime-dependent.
- Kimi K2.5 wraps this text decoder in a multimodal KimiK25 model.
- Config exposes a MoonViT vision tower and multimodal projector; this report covers the text MoE decoder and LM head only.
