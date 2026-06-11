# Verification Workspace

This folder is for checking the AI-filled spreadsheet separately from generated model stats.

Files:

- `ai_filled_metrics_sheet.tsv`: exact tab-separated copy of the pasted Excel sheet.
- `generate_verified_metrics.py`: verifier entry point.
- `verified_metrics.md`: generated verification report.
- `verified_metrics.csv`: generated flat table.
- `verified_results.md`: compact spreadsheet-style table containing only verified config/formula rows.

Generate outputs:

```bash
PYTHONPATH=src python3 verify/generate_verified_metrics.py
```

Initial read:

- Config-verifiable rows: layers, hidden size, attention heads, KV heads, expert counts, top-k, expert intermediate size, activation, dtype/quantization fields.
- Formula-verifiable rows: parameter estimates, MACs/token, memory footprint, KV cache size, activation footprint. These need explicit formulas and pinned operating points.
- Not config-verifiable rows: business case, latency targets, throughput targets, accuracy floor, arithmetic intensity, bytes moved off-chip, locality notes, and actual expert-token distribution.

Known issues from current configs:

- Nano-MoE-JAX sheet values do not match the public default config used by this repo: default experts are 4, heads are 4, head dim is 32, and expert FFN dim is 512.
- OLMoE current config has `num_key_value_heads=16`, so the sheet row `16 / 128 / 8 (GQA)` is not correct for `allenai/OLMoE-1B-7B-0125`.
- OLMoE current config reports `torch_dtype=float32`; treating it as BF16 needs a source outside `config.json`.
- Batch size, prompt length, and KV length are operating points, not model architecture facts.
