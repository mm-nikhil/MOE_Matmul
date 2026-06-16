# DeepSeek-V3 Metrics

This table keeps the same config-derived and formula-derived metric set used by `verify/verified_results.md`. Formula values are deterministic estimates from model configs plus the pinned operating points; they are not runtime measurements.

## Operating Point

| `Model` | `Phase` | `Batch` | `Sequence` | `KV Context` |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V3 | Prefill | 32 | 1024 | 4096 |
| DeepSeek-V3 | Decode | 32 | 4096 | 4096 |

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
| Feedforward dimension (dense) | config | intermediate_size + first_k_dense_replace |
| Attention: heads / head dim / Key-Value heads | config | num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank |
| Total experts | config | n_routed_experts |
| Experts active per token (top-k) | config | num_experts_per_tok |
| Shared experts | config | n_shared_experts default 0 |
| Expert feedforward dimension | config | moe_intermediate_size |
| Router / gating type | config | scoring_func, topk_method, norm_topk_prob, n_shared_experts |
| Weights precision | config | quantization_config |
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

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `DeepSeek-V3 Prefill` | `DeepSeek-V3 Decode` |
| --- | --- | --- | --- | --- | --- |
| Layers | count | config | num_hidden_layers | 61 | 61 |
| Hidden dimension | dim | config | hidden_size | 7168 | 7168 |
| Feedforward dimension (dense) | dim | config | intermediate_size + first_k_dense_replace | 18432 (first 3 layers only) | 18432 (first 3 layers only) |
| Attention: heads / head dim / Key-Value heads | counts | config | num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank | 128 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 | 128 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 |
| Total experts | count | config | n_routed_experts | 256 | 256 |
| Experts active per token (top-k) | count | config | num_experts_per_tok | 8 | 8 |
| Shared experts | count | config | n_shared_experts default 0 | 1 | 1 |
| Expert feedforward dimension | dim | config | moe_intermediate_size | 2048 | 2048 |
| Router / gating type | text | config | scoring_func, topk_method, norm_topk_prob, n_shared_experts | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 |
| Weights precision | - | config | quantization_config | FP8 (E4M3) | FP8 (E4M3) |
| Activations precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Key-Value cache precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Quantization scheme | - | config | quantization_config | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]} | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]} |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 671.025 | 671.025 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 37.5513 | 37.5513 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 39183122432 | 46858698752 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | routed_expert_mlp_topk 52.2%; attention_mla_proj 29.1%; attention_qk_av 6.5%; shared_expert_mlp 6.5% | routed_expert_mlp_topk 43.6%; attention_mla_proj 24.4%; attention_qk_av 21.8%; shared_expert_mlp 5.5% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 671025404928 | 671025404928 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 10970033437 | 10970033437 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 44040192 | 44040192 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 469762048 | 458752 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 9210691584 | 9210691584 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 9210691584 |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 3.50195 | 3.50195 |
