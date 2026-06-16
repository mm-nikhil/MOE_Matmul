# Nano-MoE-JAX Metrics

This table keeps the same config-derived and formula-derived metric set used by `verify/verified_results.md`. Formula values are deterministic estimates from model configs plus the pinned operating points; they are not runtime measurements.

## Operating Point

| `Model` | `Phase` | `Batch` | `Sequence` | `KV Context` |
| --- | --- | ---: | ---: | ---: |
| Nano-MoE-JAX | Prefill | 1 | 128 | 512 |
| Nano-MoE-JAX | Decode | 1 | 512 | 512 |

## Metric Classification

| Category | Metrics |
| --- | --- |
| Config-verifiable | Layers<br>Hidden dimension<br>Feedforward dimension (dense)<br>Attention: heads / head dim / Key-Value heads<br>Total experts<br>Experts active per token (top-k)<br>Shared experts<br>Expert feedforward dimension<br>Router / gating type<br>Weights precision<br>Activations precision<br>Key-Value cache precision<br>Quantization scheme |
| Formula-verifiable | Total parameters<br>Active parameters per token<br>Multiply-accumulate ops per token<br>Dominant operators (shapes + % of compute)<br>Weight footprint (total)<br>Weight footprint per layer<br>Weight footprint per expert<br>Activation footprint per layer<br>Key-Value cache size<br>Key-Value cache read bandwidth per decode step<br>Expert activation fraction (top-k / total) |
| Not config-verifiable | - |

## Formula And Evidence

| Metric | Category | Evidence / Formula |
| --- | --- | --- |
| Layers | config | n_layers |
| Hidden dimension | config | d_model |
| Feedforward dimension (dense) | config | architecture has MoE block, no dense FFN stack |
| Attention: heads / head dim / Key-Value heads | config | n_heads, d_model |
| Total experts | config | n_experts |
| Experts active per token (top-k) | config | top_k |
| Shared experts | config | n_shared_experts default 0 |
| Expert feedforward dimension | config | d_ff |
| Router / gating type | config | top_k + Nano-MoE-JAX router implementation |
| Weights precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Activations precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Key-Value cache precision | config | JAX default dtype for Nano-MoE-JAX defaults |
| Quantization scheme | config | no quantization field in Nano-MoE-JAX defaults |
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

| `Field` | `Unit` | `Category` | `Evidence / Formula` | `Nano-MoE-JAX Prefill` | `Nano-MoE-JAX Decode` |
| --- | --- | --- | --- | --- | --- |
| Layers | count | config | n_layers | 4 | 4 |
| Hidden dimension | dim | config | d_model | 128 | 128 |
| Feedforward dimension (dense) | dim | config | architecture has MoE block, no dense FFN stack | N/A (MoE only) | N/A (MoE only) |
| Attention: heads / head dim / Key-Value heads | counts | config | n_heads, d_model | 4 / 32 / 4 | 4 / 32 / 4 |
| Total experts | count | config | n_experts | 4 | 4 |
| Experts active per token (top-k) | count | config | top_k | 2 | 2 |
| Shared experts | count | config | n_shared_experts default 0 | 0 | 0 |
| Expert feedforward dimension | dim | config | d_ff | 512 | 512 |
| Router / gating type | text | config | top_k + Nano-MoE-JAX router implementation | softmax top-2 | softmax top-2 |
| Weights precision | - | config | JAX default dtype for Nano-MoE-JAX defaults | FP32 | FP32 |
| Activations precision | - | config | JAX default dtype for Nano-MoE-JAX defaults | FP32 | FP32 |
| Key-Value cache precision | - | config | JAX default dtype for Nano-MoE-JAX defaults | FP32 | FP32 |
| Quantization scheme | - | config | no quantization field in Nano-MoE-JAX defaults | None | None |
| Total parameters | billions | formula | parameter_count / 1e9; parameter_count = embeddings + decoder projections + experts + lm_head | 0.00244339 | 0.00244339 |
| Active parameters per token | billions | formula | active_parameter_count / 1e9; active weights = attention/dense/shared weights + top-k routed experts + lm_head | 0.0013783 | 0.0013783 |
| Multiply-accumulate ops per token | MACs/token | formula | MACs/token = sum of per-token matmul components using sheet operating point | 2525184 | 2918400 |
| Dominant operators (shapes + % of compute) | text | formula | component MAC share from static matmul formulas | expert_mlp_impl_all_experts 83.0%; attention_proj 10.4%; attention_qk_av 5.2%; lm_head 1.3% | expert_mlp_impl_all_experts 71.9%; attention_qk_av 18.0%; attention_proj 9.0%; lm_head 1.1% |
| Weight footprint (total) | bytes | formula | total_parameter_count * weight_bytes | 9773568 | 9773568 |
| Weight footprint per layer | bytes | formula | decoder_layer_weight_params * weight_bytes; DeepSeek value is average over dense and MoE layers | 2361344 | 2361344 |
| Weight footprint per expert | bytes | formula | expert_weight_params_per_expert * weight_bytes | 524288 | 524288 |
| Activation footprint per layer | bytes | formula | B * S_effective * H * activation_bytes; S_effective=S for prefill and 1 for decode | 65536 | 512 |
| Key-Value cache size | bytes | formula | KV cache formula from config attention layout and sheet KV context length | 2097152 | 2097152 |
| Key-Value cache read bandwidth per decode step | bytes/step | formula | decode-only metric<br>decode step reads B * KV cached keys/values across layers | N/A | 2097152 |
| Expert activation fraction (top-k / total) | % | formula | 100 * (top_k + shared_experts) / (routed_experts + shared_experts) | 50 | 50 |
