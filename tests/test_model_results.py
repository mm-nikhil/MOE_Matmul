from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from moe_matmul_stats.model_results import default_model_specs, write_model_results


class ModelResultsTests(unittest.TestCase):
    def test_write_model_results_creates_config_metrics_and_matmul(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            written = write_model_results(
                specs=(default_model_specs()[0],),
                models_root=root / "models",
                results_root=root / "results",
            )

            expected = {
                root / "models" / "nano-moe-jax" / "config.json",
                root / "results" / "nano-moe-jax" / "metrics.md",
                root / "results" / "nano-moe-jax" / "matmul.md",
            }
            self.assertEqual(set(written), expected)
            self.assertIn("Nano-MoE-JAX Metrics", (root / "results" / "nano-moe-jax" / "metrics.md").read_text())
            self.assertIn("Matmul families", (root / "results" / "nano-moe-jax" / "matmul.md").read_text())


if __name__ == "__main__":
    unittest.main()
