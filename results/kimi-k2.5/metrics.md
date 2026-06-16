# Kimi-K2.5 Metrics

This table keeps the same config-derived and formula-derived metric set used by `verify/verified_results.md`. Formula values are deterministic estimates from model configs plus the pinned operating points; they are not runtime measurements.

## Operating Point

| `Model` | `Phase` | `Batch` | `Sequence` | `KV Context` |
| --- | --- | ---: | ---: | ---: |
| Kimi-K2.5 | Prefill | 32 | 1024 | 4096 |
| Kimi-K2.5 | Decode | 32 | 4096 | 4096 |

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
| Weights precision | config | quantization_config + dtype |
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

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `Kimi-K2.5 Prefill` | `Kimi-K2.5 Decode` |
| --- | --- | --- | --- | --- | --- |
| Layers | count | config | num_hidden_layers | 61 | 61 |
| Hidden dimension | dim | config | hidden_size | 7168 | 7168 |
| Feedforward dimension (dense) | dim | config | intermediate_size + first_k_dense_replace | 18432 (first 1 layers only) | 18432 (first 1 layers only) |
| Attention: heads / head dim / Key-Value heads | counts | config | num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank | 64 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 | 64 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 |
| Total experts | count | config | n_routed_experts | 384 | 384 |
| Experts active per token (top-k) | count | config | num_experts_per_tok | 8 | 8 |
| Shared experts | count | config | n_shared_experts default 0 | 1 | 1 |
| Expert feedforward dimension | dim | config | moe_intermediate_size | 2048 | 2048 |
| Router / gating type | text | config | scoring_func, topk_method, norm_topk_prob, n_shared_experts | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 |
| Weights precision | - | config | quantization_config + dtype | INT4 compressed-tensors where applied; BF16 for ignored modules | INT4 compressed-tensors where applied; BF16 for ignored modules |
| Activations precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Key-Value cache precision | - | config | torch_dtype or dtype | BF16 | BF16 |
| Quantization scheme | - | config | quantization_config | {'config_groups': {'group_0': {'input_activations': None, 'output_activations': None, 'targets': ['Linear'], 'weights': {'actorder': None, 'block_structure': None, 'dynamic': False, 'group_size': 32, 'num_bits': 4, 'observer': 'minmax', 'observer_kwargs': {}, 'strategy': 'group', 'symmetric': True, 'type': 'int'}}}, 'format': 'pack-quantized', 'ignore': ['re:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate\|up\|gate_up\|down)_proj.*', 're:.*lm_head.*', 're:vision_tower.*', 're:mm_projector.*'], 'kv_cache_scheme': None, 'quant_method': 'compressed-tensors', 'quantization_status': 'compressed'} | {'config_groups': {'group_0': {'input_activations': None, 'output_activations': None, 'targets': ['Linear'], 'weights': {'actorder': None, 'block_structure': None, 'dynamic': False, 'group_size': 32, 'num_bits': 4, 'observer': 'minmax', 'observer_kwargs': {}, 'strategy': 'group', 'symmetric': True, 'type': 'int'}}}, 'format': 'pack-quantized', 'ignore': ['re:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate\|up\|gate_up\|down)_proj.*', 're:.*lm_head.*', 're:vision_tower.*', 're:mm_projector.*'], 'kv_cache_scheme': None, 'quant_method': 'compressed-tensors', 'quantization_status': 'compressed'} |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 1026.41 | 1026.41 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 32.8605 | 32.8605 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 32965328896 | 36803117056 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | routed_expert_mlp_topk 64.1%; attention_mla_proj 18.7%; shared_expert_mlp 8.0%; attention_qk_av 3.9% | routed_expert_mlp_topk 57.4%; attention_mla_proj 16.8%; attention_qk_av 13.9%; shared_expert_mlp 7.2% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 2052814419968 | 2052814419968 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 33575685002 | 33575685002 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 88080384 | 88080384 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 469762048 | 458752 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 9210691584 | 9210691584 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 9210691584 |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 2.33766 | 2.33766 |
