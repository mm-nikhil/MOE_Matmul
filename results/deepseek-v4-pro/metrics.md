# DeepSeek-V4-Pro Metrics

This table keeps the same config-derived and formula-derived metric set used by `verify/verified_results.md`. Formula values are deterministic estimates from model configs plus the pinned operating points; they are not runtime measurements.

## Operating Point

| `Model` | `Phase` | `Batch` | `Sequence` | `KV Context` |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V4-Pro | Prefill | 32 | 1024 | 4096 |
| DeepSeek-V4-Pro | Decode | 32 | 4096 | 4096 |

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
| Attention: heads / head dim / Key-Value heads | config | num_attention_heads, head_dim, qk_rope_head_dim, num_key_value_heads, sliding_window |
| Total experts | config | n_routed_experts |
| Experts active per token (top-k) | config | num_experts_per_tok |
| Shared experts | config | n_shared_experts default 0 |
| Expert feedforward dimension | config | moe_intermediate_size |
| Router / gating type | config | scoring_func, num_experts_per_tok, topk_method, norm_topk_prob, num_hash_layers, n_shared_experts |
| Weights precision | config | expert_dtype + quantization_config |
| Activations precision | config | torch_dtype or dtype |
| Key-Value cache precision | config | torch_dtype or dtype |
| Quantization scheme | config | quantization_config |
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

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `DeepSeek-V4-Pro Prefill` | `DeepSeek-V4-Pro Decode` |
| --- | --- | --- | --- | --- | --- |
| Layers | count | config | num_hidden_layers | 61 | 61 |
| Hidden dimension | dim | config | hidden_size | 7168 | 7168 |
| Feedforward dimension (dense) | dim | config | architecture has MoE block, no dense FFN stack | N/A (MoE only) | N/A (MoE only) |
| Attention: heads / head dim / Key-Value heads | counts | config | num_attention_heads, head_dim, qk_rope_head_dim, num_key_value_heads, sliding_window | 128 / head_dim=512 (rope=64) / shared-KV MQA kv_heads=1, sliding_window=128 | 128 / head_dim=512 (rope=64) / shared-KV MQA kv_heads=1, sliding_window=128 |
| Total experts | count | config | n_routed_experts | 384 | 384 |
| Experts active per token (top-k) | count | config | num_experts_per_tok | 6 | 6 |
| Shared experts | count | config | n_shared_experts default 0 | 1 | 1 |
| Expert feedforward dimension | dim | config | moe_intermediate_size | 3072 | 3072 |
| Router / gating type | text | config | scoring_func, num_experts_per_tok, topk_method, norm_topk_prob, num_hash_layers, n_shared_experts | sqrtsoftplus top-6, topk_method=noaux_tc, norm_topk_prob=True, hash_layers=3, shared_experts=1 | sqrtsoftplus top-6, topk_method=noaux_tc, norm_topk_prob=True, hash_layers=3, shared_experts=1 |
| Weights precision | - | config | expert_dtype + quantization_config | FP4 experts + FP8 (E4M3) mixed | FP4 experts + FP8 (E4M3) mixed |
| Activations precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Key-Value cache precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Quantization scheme | - | config | quantization_config | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'scale_fmt': 'ue8m0', 'weight_block_size': [128, 128]} | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'scale_fmt': 'ue8m0', 'weight_block_size': [128, 128]} |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 1572.99 | 1572.99 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 49.7758 | 49.7758 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 50974605312 | 54280765440 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | routed_expert_mlp_topk 47.4%; attention_main_proj 35.9%; shared_expert_mlp 7.9%; attention_compressor_proj 2.3% | routed_expert_mlp_topk 44.5%; attention_main_proj 33.7%; attention_qk_av_compressed 7.7%; shared_expert_mlp 7.4% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 1572993948672 | 1572993948672 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 25756401127 | 25756401127 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 66060288 | 66060288 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 469762048 | 458752 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 1546649600 | 1546649600 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 1546649600 |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 1.81818 | 1.81818 |
