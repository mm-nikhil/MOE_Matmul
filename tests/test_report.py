from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from moe_matmul_stats.report import render_markdown, write_markdown_report
from moe_matmul_stats.schema import MatmulRecord, ModelStats


class ReportTests(unittest.TestCase):
    def test_render_markdown_includes_model_summary_and_records(self) -> None:
        stats = ModelStats.from_records(
            model="Fake-MoE",
            source="unit-test",
            config_summary={"hidden_size": 128, "num_experts": 4},
            records=[
                MatmulRecord(
                    model="Fake-MoE",
                    layer_range="0..3",
                    block="attention",
                    op_name="q_proj",
                    op_kind="linear",
                    lhs_shape="[T, H]",
                    rhs_shape="[H, H]",
                    output_shape="[T, H]",
                    batching="dense token batch",
                    repeat_count="1 per layer",
                    active_condition="every token",
                    logical_vs_implementation="implementation",
                    numeric_format="bf16",
                )
            ],
            notes=["Fake data only."],
        )

        markdown = render_markdown([stats])

        self.assertIn("# MOE Matmul Stats Report", markdown)
        self.assertIn("## Fake-MoE", markdown)
        self.assertIn("| `hidden_size` | `128` |", markdown)
        self.assertIn("| 0..3 | attention | q_proj | linear |", markdown)
        self.assertIn("- Fake data only.", markdown)

    def test_write_markdown_report_creates_parent_directory(self) -> None:
        stats = ModelStats.from_records(model="Empty", source="unit-test", records=[])

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "report.md"
            written_path = write_markdown_report([stats], output_path)

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertIn("No matmul records collected.", output_path.read_text())


if __name__ == "__main__":
    unittest.main()
