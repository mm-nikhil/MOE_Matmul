# Verified Results

This is the compact replacement for the AI-filled spreadsheet. It keeps only config-verifiable and formula-verifiable metrics, and uses values recomputed from model configs plus explicit formulas.

This table shows the verified values only. Comparison against the original AI-filled sheet is kept in `verified_metrics.md`.

## Summary

| Item | Count |
| --- | ---: |
| verified metric rows | 24 |
| flattened verified cells | 144 |

## Formula And Evidence

| Metric | Category | Evidence / Formula |
| --- | --- | --- |
| Total parameters | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head |
| Active parameters per token | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head |
| Layers | config | n_layers |
| Layers | config | num_hidden_layers |
| Hidden dimension | config | d_model |
| Hidden dimension | config | hidden_size |
| Feedforward dimension (dense) | config | architecture has MoE block, no dense FFN stack |
| Feedforward dimension (dense) | config | intermediate_size + first_k_dense_replace |
| Attention: heads / head dim / Key-Value heads | config | n_heads, d_model |
| Attention: heads / head dim / Key-Value heads | config | num_attention_heads, hidden_size, num_key_value_heads |
| Attention: heads / head dim / Key-Value heads | config | num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank |
| Total experts | config | n_experts |
| Total experts | config | num_experts |
| Total experts | config | n_routed_experts |
| Experts active per token (top-k) | config | top_k |
| Experts active per token (top-k) | config | num_experts_per_tok |
| Shared experts | config | n_shared_experts default 0 |
| Expert feedforward dimension | config | d_ff |
| Expert feedforward dimension | config | intermediate_size |
| Expert feedforward dimension | config | moe_intermediate_size |
| Router / gating type | config | top_k + Nano-MoE-JAX router implementation |
| Router / gating type | config | num_experts_per_tok, router_aux_loss_coef, norm_topk_prob |
| Router / gating type | config | scoring_func, topk_method, norm_topk_prob, n_shared_experts |
| Multiply-accumulate ops per token | formula | MACs/token = sum of per-token matmul components using sheet operating point |
| Dominant operators (shapes + % of compute) | formula | component MAC share from static matmul formulas |
| Weight footprint (total) | formula | total_parameter_count * weight_bytes |
| Weight footprint per layer | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers |
| Weight footprint per expert | formula | expert_weight_params_per_expert * weight_bytes |
| Activation footprint per layer | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode |
| Key-Value cache size | formula | KV cache formula from config attention layout and sheet KV context length |
| Key-Value cache read bandwidth per decode step | formula | decode-only metric |
| Key-Value cache read bandwidth per decode step | formula | decode step reads B * KV cached keys/values across layers |
| Weights precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Weights precision | config | torch_dtype |
| Weights precision | config | quantization_config |
| Activations precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Activations precision | config | torch_dtype |
| Key-Value cache precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Key-Value cache precision | config | torch_dtype |
| Quantization scheme | config | no quantization field in Nano-MoE-JAX defaults |
| Quantization scheme | config | quantization_config absent |
| Quantization scheme | config | quantization_config |
| Expert activation fraction (top-k / total) | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) |

## Verified Sheet

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `Nano-MoE-Jax Prefill` | `Nano-MoE-Jax Decode` | `OLMoE-1B-7B Prefill` | `OLMoE-1B-7B Decode` | `Deepseek-V3 Prefill` | `Deepseek-V3 Decode` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 0.00244339 | 0.00244339 | 6.91903 | 6.91903 | 671.025 | 671.025 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 0.0013783 | 0.0013783 | 1.28188 | 1.28188 | 37.5513 | 37.5513 |
| Layers | count | config | n_layers<br>num_hidden_layers | 4 | 4 | 16 | 16 | 61 | 61 |
| Hidden dimension | dim | config | d_model<br>hidden_size | 128 | 128 | 2048 | 2048 | 7168 | 7168 |
| Feedforward dimension (dense) | dim | config | architecture has MoE block, no dense FFN stack<br>intermediate_size + first_k_dense_replace | N/A (MoE only) | N/A (MoE only) | N/A (MoE only) | N/A (MoE only) | 18432 (first 3 layers only) | 18432 (first 3 layers only) |
| Attention: heads / head dim / Key-Value heads | counts | config | n_heads, d_model<br>num_attention_heads, hidden_size, num_key_value_heads<br>num_attention_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank | 4 / 32 / 4 | 4 / 32 / 4 | 16 / 128 / 16 | 16 / 128 / 16 | 128 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 | 128 / qk=192 (nope=128+rope=64), v=128 / MLA kv_lora=512 |
| Total experts | count | config | n_experts<br>num_experts<br>n_routed_experts | 4 | 4 | 64 | 64 | 256 | 256 |
| Experts active per token (top-k) | count | config | top_k<br>num_experts_per_tok | 2 | 2 | 8 | 8 | 8 | 8 |
| Shared experts | count | config | n_shared_experts default 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Expert feedforward dimension | dim | config | d_ff<br>intermediate_size<br>moe_intermediate_size | 512 | 512 | 1024 | 1024 | 2048 | 2048 |
| Router / gating type | text | config | top_k + Nano-MoE-JAX router implementation<br>num_experts_per_tok, router_aux_loss_coef, norm_topk_prob<br>scoring_func, topk_method, norm_topk_prob, n_shared_experts | softmax top-2 | softmax top-2 | softmax top-8, aux_loss_coef=0.01, norm_topk_prob=False | softmax top-8, aux_loss_coef=0.01, norm_topk_prob=False | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 | sigmoid top-8, topk_method=noaux_tc, norm_topk_prob=True, shared_experts=1 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 2525184 | 2918400 | 1212416000 | 1313079296 | 39183122432 | 46858698752 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | expert_mlp_impl_all_experts 83.0%; attention_proj 10.4%; attention_qk_av 5.2%; lm_head 1.3% | expert_mlp_impl_all_experts 71.9%; attention_qk_av 18.0%; attention_proj 9.0%; lm_head 1.1% | routed_expert_mlp_topk 66.4%; attention_proj 22.1%; lm_head 8.5%; attention_qk_av 2.8% | routed_expert_mlp_topk 61.3%; attention_proj 20.4%; attention_qk_av 10.2%; lm_head 7.8% | routed_expert_mlp_topk 52.2%; attention_mla_proj 29.1%; attention_qk_av 6.5%; shared_expert_mlp 6.5% | routed_expert_mlp_topk 43.6%; attention_mla_proj 24.4%; attention_qk_av 21.8%; shared_expert_mlp 5.5% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 9773568 | 9773568 | 27676123136 | 27676123136 | 671025404928 | 671025404928 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 2361344 | 2361344 | 1678245888 | 1678245888 | 10970033437 | 10970033437 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 524288 | 524288 | 25165824 | 25165824 | 44040192 | 44040192 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 65536 | 512 | 134217728 | 262144 | 469762048 | 458752 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 2097152 | 2097152 | 17179869184 | 17179869184 | 9210691584 | 9210691584 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 2097152 | N/A | 17179869184 | N/A | 9210691584 |
| Weights precision | - | config | JAX default dtype for Nano-MoE-JAX defaults<br>torch_dtype<br>quantization_config | FP32 | FP32 | FP32 | FP32 | FP8 (E4M3) | FP8 (E4M3) |
| Activations precision | - | config | JAX default dtype for Nano-MoE-JAX defaults<br>torch_dtype | FP32 | FP32 | FP32 | FP32 | BF16 | BF16 |
| Key-Value cache precision | - | config | JAX default dtype for Nano-MoE-JAX defaults<br>torch_dtype | FP32 | FP32 | FP32 | FP32 | BF16 | BF16 |
| Quantization scheme | - | config | no quantization field in Nano-MoE-JAX defaults<br>quantization_config absent<br>quantization_config | None | None | None | None | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]} | {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]} |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 50 | 50 | 12.5 | 12.5 | 3.50195 | 3.50195 |

## Status Legend

This compact sheet intentionally omits match/mismatch annotations. Use `verified_metrics.md` for the audit trail against the original AI-filled values.
