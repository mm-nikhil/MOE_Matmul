import unittest

from moe_matmul_stats.sources import collectStatsFromConfig, collectStatsNanoJax


class ExtractorTests(unittest.TestCase):
    def test_olmoe_records_key_shapes(self) -> None:
        stats = collectStatsFromConfig(
            model="olmoe-test",
            source="unit-test",
            config={
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
            },
        )

        by_name = {record.op_name: record for record in stats.records}
        self.assertEqual(len(stats.records), 10)
        self.assertEqual(by_name["q_proj"].layer_range, "0..15")
        self.assertEqual(by_name["q_proj"].rhs_shape, "[A*D=2048, H=2048]")
        self.assertEqual(by_name["gate_up_proj"].rhs_shape, "[2*I=2048, H=2048] per expert")
        self.assertIn("sum_e N_e=T*K=8T", by_name["down_proj"].batching)

    def test_nano_jax_records_implementation_difference(self) -> None:
        stats = collectStatsNanoJax()

        by_name = {record.op_name: record for record in stats.records}
        self.assertEqual(len(stats.records), 8)
        self.assertEqual(by_name["qkv_proj"].rhs_shape, "[3*H=384, H=128]")
        self.assertEqual(by_name["expert_fc1"].repeat_count, "E=4 experts per layer")
        self.assertIn("computes all E=4 experts", stats.notes[1])

    def test_deepseek_records_mla_and_moe_layer_spans(self) -> None:
        stats = collectStatsFromConfig(
            model="deepseek-test",
            source="unit-test",
            config={
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
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
                "vocab_size": 129280,
            },
        )

        by_name = {record.op_name: record for record in stats.records}
        self.assertEqual(len(stats.records), 18)
        self.assertEqual(by_name["q_b_proj"].rhs_shape, "[A*QD=24576, Q_RANK=1536]")
        self.assertEqual(by_name["kv_b_proj"].rhs_shape, "[A*(NOPE+VD)=32768, KV_RANK=512]")
        self.assertEqual(by_name["gate_proj"].layer_range, "0..2")
        self.assertEqual(by_name["router_logits"].layer_range, "3..60")
        self.assertIn("sum_e N_e=T*K=8T", by_name["routed_down_proj"].batching)


if __name__ == "__main__":
    unittest.main()
