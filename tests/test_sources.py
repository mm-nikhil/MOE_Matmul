import unittest
from unittest.mock import patch

from moe_matmul_stats.extractors import UnsupportedModelError
from moe_matmul_stats.sources import (
    ConfigLoadError,
    collectStatsFromConfig,
    collectStatsHF,
    collectStatsNanoJax,
    hf_config_url,
)


class SourceTests(unittest.TestCase):
    def test_hf_config_url_points_to_raw_config(self) -> None:
        self.assertEqual(
            hf_config_url("allenai/OLMoE-1B-7B-0125"),
            "https://huggingface.co/allenai/OLMoE-1B-7B-0125/raw/main/config.json",
        )

    def test_collect_stats_hf_routes_by_model_type(self) -> None:
        fake_config = {
            "model_type": "olmoe",
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "hidden_act": "silu",
            "torch_dtype": "float32",
            "vocab_size": 50304,
        }

        with patch("moe_matmul_stats.sources.fetch_hf_config", return_value=fake_config):
            stats = collectStatsHF("allenai/OLMoE-1B-7B-0125")

        self.assertEqual(stats.model, "allenai/OLMoE-1B-7B-0125")
        self.assertEqual(stats.source, "huggingface:allenai/OLMoE-1B-7B-0125@main")
        self.assertEqual(stats.config_summary["hidden_size"], 2048)
        self.assertEqual(stats.config_summary["num_experts"], 64)
        self.assertEqual(len(stats.records), 10)
        self.assertEqual(stats.records[0].op_name, "q_proj")

    def test_collect_stats_from_config_routes_deepseek(self) -> None:
        stats = collectStatsFromConfig(
            model="deepseek-ai/DeepSeek-V3",
            source="unit-test",
            config={
                "model_type": "deepseek_v3",
                "hidden_size": 7168,
                "intermediate_size": 18432,
                "moe_intermediate_size": 2048,
                "num_hidden_layers": 61,
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
                "torch_dtype": "bfloat16",
                "vocab_size": 129280,
            },
        )

        self.assertEqual(stats.config_summary["hidden_size"], 7168)
        self.assertEqual(stats.config_summary["n_routed_experts"], 256)
        self.assertEqual(len(stats.records), 18)
        self.assertIn("MLA attention", stats.notes[0])

    def test_collect_stats_nano_jax_uses_default_config(self) -> None:
        stats = collectStatsNanoJax()

        self.assertEqual(stats.model, "Nano-MoE-JAX")
        self.assertEqual(stats.config_summary["n_layers"], 4)
        self.assertEqual(stats.config_summary["n_experts"], 4)
        self.assertEqual(stats.config_summary["top_k"], 2)
        self.assertEqual(len(stats.records), 8)

    def test_missing_model_type_is_config_error(self) -> None:
        with self.assertRaises(ConfigLoadError):
            collectStatsFromConfig(model="missing", source="unit-test", config={})

    def test_unsupported_model_type_is_registry_error(self) -> None:
        with self.assertRaises(UnsupportedModelError):
            collectStatsFromConfig(
                model="unknown",
                source="unit-test",
                config={"model_type": "unknown_moe"},
            )


if __name__ == "__main__":
    unittest.main()
