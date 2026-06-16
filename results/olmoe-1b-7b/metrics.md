# OLMoE-1B-7B Metrics

This table keeps the same config-derived and formula-derived metric set used by `verify/verified_results.md`. Formula values are deterministic estimates from model configs plus the pinned operating points; they are not runtime measurements.

## Operating Point

| `Model` | `Phase` | `Batch` | `Sequence` | `KV Context` |
| --- | --- | ---: | ---: | ---: |
| OLMoE-1B-7B | Prefill | 32 | 512 | 2048 |
| OLMoE-1B-7B | Decode | 32 | 2048 | 2048 |

## Metric Classification

| Category | Metrics |
| --- | --- |
| Config-verifiable | Layers<br>Hidden dimension<br>Feedforward dimension (dense)<br>Attention: heads / head dim / Key-Value heads<br>Total experts<br>Experts active per token (top-k)<br>Shared experts<br>Expert feedforward dimension<br>Router / gating type<br>Weights precision<br>Activations precision<br>Key-Value cache precision<br>Quantization scheme |
| Formula-verifiable | Total parameters<br>Active parameters per token<br>Multiply-accumulate ops per token<br>Dominant operators (shapes + % of compute)<br>Weight footprint (total)<br>Weight footprint per layer<br>Weight footprint per expert<br>Activation footprint per layer<br>Key-Value cache size<br>Key-Value cache read bandwidth per decode step<br>Expert activation fraction (top-k / total) |
| Not config-verifiable | - |

## Formula And Evidence

| Metric | Category | Evidence / Formula |
| --- | --- | --- |
| Layers | config | num_hidden_layers |
| Hidden dimension | config | hidden_size |
| Feedforward dimension (dense) | config | architecture has MoE block, no dense FFN stack |
| Attention: heads / head dim / Key-Value heads | config | num_attention_heads, hidden_size, num_key_value_heads |
| Total experts | config | num_experts |
| Experts active per token (top-k) | config | num_experts_per_tok |
| Shared experts | config | n_shared_experts default 0 |
| Expert feedforward dimension | config | intermediate_size |
| Router / gating type | config | num_experts_per_tok, router_aux_loss_coef, norm_topk_prob |
| Weights precision | config | torch_dtype or dtype |
| Activations precision | config | torch_dtype or dtype |
| Key-Value cache precision | config | torch_dtype or dtype |
| Quantization scheme | config | quantization_config absent |
| Total parameters | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head |
| Active parameters per token | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head |
| Multiply-accumulate ops per token | formula | MACs/token = sum of per-token matmul components using sheet operating point |
| Dominant operators (shapes + % of compute) | formula | component MAC share from static matmul formulas |
| Weight footprint (total) | formula | total_parameter_count * weight_bytes |
| Weight footprint per layer | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers |
| Weight footprint per expert | formula | expert_weight_params_per_expert * weight_bytes |
| Activation footprint per layer | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode |
| Key-Value cache size | formula | KV cache formula from config attention layout and sheet KV context length |
| Key-Value cache read bandwidth per decode step | formula | decode-only metric |
| Key-Value cache read bandwidth per decode step | formula | decode step reads B * KV cached keys/values across layers |
| Expert activation fraction (top-k / total) | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) |

## Metrics

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `OLMoE-1B-7B Prefill` | `OLMoE-1B-7B Decode` |
| --- | --- | --- | --- | --- | --- |
| Layers | count | config | num_hidden_layers | 16 | 16 |
| Hidden dimension | dim | config | hidden_size | 2048 | 2048 |
| Feedforward dimension (dense) | dim | config | architecture has MoE block, no dense FFN stack | N/A (MoE only) | N/A (MoE only) |
| Attention: heads / head dim / Key-Value heads | counts | config | num_attention_heads, hidden_size, num_key_value_heads | 16 / 128 / 16 | 16 / 128 / 16 |
| Total experts | count | config | num_experts | 64 | 64 |
| Experts active per token (top-k) | count | config | num_experts_per_tok | 8 | 8 |
| Shared experts | count | config | n_shared_experts default 0 | 0 | 0 |
| Expert feedforward dimension | dim | config | intermediate_size | 1024 | 1024 |
| Router / gating type | text | config | num_experts_per_tok, router_aux_loss_coef, norm_topk_prob | softmax top-8, aux_loss_coef=0.01, norm_topk_prob=False | softmax top-8, aux_loss_coef=0.01, norm_topk_prob=False |
| Weights precision | - | config | torch_dtype or dtype | FP32 | FP32 |
| Activations precision | - | config | torch_dtype or dtype | FP32 | FP32 |
| Key-Value cache precision | - | config | torch_dtype or dtype | FP32 | FP32 |
| Quantization scheme | - | config | quantization_config absent | None | None |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 6.91903 | 6.91903 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 1.28188 | 1.28188 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 1212416000 | 1313079296 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | routed_expert_mlp_topk 66.4%; attention_proj 22.1%; lm_head 8.5%; attention_qk_av 2.8% | routed_expert_mlp_topk 61.3%; attention_proj 20.4%; attention_qk_av 10.2%; lm_head 7.8% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 27676123136 | 27676123136 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 1678245888 | 1678245888 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 25165824 | 25165824 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 134217728 | 262144 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 17179869184 | 17179869184 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 17179869184 |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 12.5 | 12.5 |
