import unittest
from pathlib import Path

from moe_matmul_stats.schema import ModelStats
from moe_matmul_stats.sources import NANO_MOE_JAX_DEFAULT_CONFIG, collectStatsFromConfig
from moe_matmul_stats.verification import (
    ModelContext,
    build_verified_rows,
    render_verified_results_markdown,
)


SHEET_PATH = Path("verify/ai_filled_metrics_sheet.tsv")


def _context(label: str, kind: str, config: dict) -> ModelContext:
    source = f"unit-test:{label}"
    stats: ModelStats = collectStatsFromConfig(model=label, source=source, config=config)
    return ModelContext(label=label, kind=kind, source=source, config=config, stats=stats)


class VerificationTests(unittest.TestCase):
    def test_build_verified_rows_categorizes_and_flags_key_cases(self) -> None:
        rows = build_verified_rows(SHEET_PATH, contexts=_test_contexts())
        by_key = {(row.metric, row.model, row.phase): row for row in rows}

        self.assertEqual(len(rows), 246)

        nano_experts = by_key[("Total experts", "Nano-MoE-Jax", "Prefill")]
        self.assertEqual(nano_experts.category, "config")
        self.assertEqual(nano_experts.value, "4")
        self.assertEqual(nano_experts.status, "mismatch")
        self.assertIn("sheet_value=8", nano_experts.notes)

        olmoe_attention = by_key[
            ("Attention: heads / head dim / Key-Value heads", "OLMoE-1B-7B", "Prefill")
        ]
        self.assertEqual(olmoe_attention.value, "16 / 128 / 16")
        self.assertEqual(olmoe_attention.status, "mismatch")

        deepseek_activation = by_key[("Activations precision", "Deepseek-V3", "Prefill")]
        self.assertEqual(deepseek_activation.value, "BF16")
        self.assertEqual(deepseek_activation.status, "match")

        business_case = by_key[("Business case / relevance (1 line)", "Nano-MoE-Jax", "Prefill")]
        self.assertEqual(business_case.category, "not_config_verifiable")
        self.assertEqual(business_case.status, "unverified_source_value")

        batch_size = by_key[("Batch size", "OLMoE-1B-7B", "Prefill")]
        self.assertEqual(batch_size.source_type, "sheet_operating_point")
        self.assertEqual(batch_size.status, "assumption")

        kv_read_prefill = by_key[
            ("Key-Value cache read bandwidth per decode step", "Deepseek-V3", "Prefill")
        ]
        self.assertEqual(kv_read_prefill.status, "not_applicable")

    def test_compact_results_doc_contains_only_verified_categories(self) -> None:
        rows = build_verified_rows(SHEET_PATH, contexts=_test_contexts())
        markdown = render_verified_results_markdown(rows)

        self.assertIn("# Verified Results", markdown)
        self.assertIn("## Verified Sheet", markdown)
        self.assertIn("Total experts", markdown)
        self.assertIn("| Total experts | count | config |", markdown)
        self.assertIn("| 4 | 4 | 64 | 64 | 256 | 256 |", markdown)
        self.assertNotIn("Business case / relevance", markdown)
        self.assertNotIn("Latency target", markdown)


def _test_contexts() -> dict[str, ModelContext]:
    nano_config = dict(NANO_MOE_JAX_DEFAULT_CONFIG)
    olmoe_config = {
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "hidden_act": "silu",
        "router_aux_loss_coef": 0.01,
        "norm_topk_prob": False,
        "torch_dtype": "float32",
        "vocab_size": 50304,
    }
    deepseek_config = {
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "hidden_act": "silu",
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        "vocab_size": 129280,
    }
    return {
        "Nano-MoE-Jax": _context("Nano-MoE-Jax", "nano", nano_config),
        "OLMoE-1B-7B": _context("OLMoE-1B-7B", "olmoe", olmoe_config),
        "Deepseek-V3": _context("Deepseek-V3", "deepseek", deepseek_config),
    }


if __name__ == "__main__":
    unittest.main()
