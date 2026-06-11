# MOE Matmul Stats

Purpose: collect static matmul statistics for MoE LLM architectures so we can reason about hardware support without downloading full model weights.

Initial target models:

- Nano-MoE-JAX
- OLMoE-1B-7B
- DeepSeek-V3

Generate the combined report:

```bash
PYTHONPATH=src python3 -m moe_matmul_stats.cli --out results/moe_matmul_stats.md
```

## Matmul Record

One record describes one matmul family, not every runtime invocation.

| Field | Meaning |
| --- | --- |
| `model` | Model name and checkpoint/source. |
| `layer_range` | Decoder layers where this matmul family appears, using zero-based inclusive ranges like `0..15`. |
| `block` | Architectural block that owns the matmul: `attention`, `router`, `expert_mlp`, `dense_mlp`, or `lm_head`. |
| `op_name` | Local operation name, such as `q_proj`, `qk_scores`, `gate_proj`, or `down_proj`. |
| `op_kind` | Computation style: `linear`, `batched_matmul`, `grouped_expert_matmul`, or `embedding_lookup`. |
| `lhs_shape` | Symbolic left input shape. |
| `rhs_shape` | Symbolic weight/right input shape. |
| `output_shape` | Symbolic output shape. |
| `batching` | How work is batched: dense token batch, attention batch, per-expert ragged batch, or generation step. |
| `repeat_count` | Number of logical repeats per layer, for example heads, experts, or top-k routes. |
| `active_condition` | When the op runs, for example every token, selected experts only, or first dense layers only. |
| `logical_vs_implementation` | Whether the record is a logical op or the actual implementation form, such as fused QKV or fused gate+up. |
| `activation_after` | Activation directly after the matmul, if any. |
| `numeric_format` | Expected dtype or quantization when known from config. |
| `notes` | Short caveats that affect hardware interpretation. |

## Key Variables

- `B`: batch size.
- `S`: query sequence length.
- `KV`: key/value sequence length. In prefill, usually `KV = S`; in decode, usually `S = 1` and `KV = past_tokens + 1`.
- `T = B * S`: flattened token count.
- `H`: hidden size.
- `A`: attention head count.
- `D`: per-head query/key dimension.
- `V`: per-head value dimension when different from `D`.
- `I`: dense or expert intermediate size.
- `E`: number of routed experts.
- `K`: experts selected per token.
- `N_e`: tokens routed to expert `e` at runtime.

## Shape Formulas

Dense linear projection:

```text
lhs:    [T, in_features]
rhs:    [out_features, in_features]
output: [T, out_features]
```

Attention score matmul:

```text
lhs:    [B, A, S, D]
rhs:    [B, A, D, KV]
output: [B, A, S, KV]
```

Attention value matmul:

```text
lhs:    [B, A, S, KV]
rhs:    [B, A, KV, V]
output: [B, A, S, V]
```

Routed expert matmul:

```text
lhs:    [N_e, in_features]
rhs:    [out_features, in_features]
output: [N_e, out_features]
```

Routing constraint:

```text
sum_e N_e = T * K
```

`N_e` is runtime-dependent, so config-only extraction reports the shape family and the routing constraint, not exact expert batch sizes.
