"""Command line entry point for generating the combined matmul report."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_results import (
    DEFAULT_DEEPSEEK_V4_MODEL,
    DEFAULT_KIMI_K25_MODEL,
    ModelSpec,
    write_new_model_results,
)
from .report import write_markdown_report
from .schema import ModelStats
from .sources import collectStatsHF, collectStatsNanoJax

DEFAULT_OLMOE_MODEL = "allenai/OLMoE-1B-7B-0125"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_OUTPUT = "results/moe_matmul_stats.md"


def collect_default_stats(
    *,
    olmoe_model: str = DEFAULT_OLMOE_MODEL,
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL,
    deepseek_v4_model: str = DEFAULT_DEEPSEEK_V4_MODEL,
    kimi_model: str = DEFAULT_KIMI_K25_MODEL,
    revision: str = "main",
) -> list[ModelStats]:
    return [
        collectStatsNanoJax(),
        collectStatsHF(olmoe_model, revision=revision),
        collectStatsHF(deepseek_model, revision=revision),
        collectStatsHF(deepseek_v4_model, revision=revision),
        collectStatsHF(kimi_model, revision=revision),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static MoE matmul stats.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Markdown output path.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to fetch.")
    parser.add_argument("--olmoe-model", default=DEFAULT_OLMOE_MODEL)
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek-v4-model", default=DEFAULT_DEEPSEEK_V4_MODEL)
    parser.add_argument("--kimi-model", default=DEFAULT_KIMI_K25_MODEL)
    parser.add_argument("--models-root", default="models", help="Directory for downloaded configs.")
    parser.add_argument("--results-root", default="results", help="Directory for per-model outputs.")
    parser.add_argument(
        "--skip-organized-results",
        action="store_true",
        help="Only write the combined report, not per-model config/metrics/matmul files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = collect_default_stats(
        olmoe_model=args.olmoe_model,
        deepseek_model=args.deepseek_model,
        deepseek_v4_model=args.deepseek_v4_model,
        kimi_model=args.kimi_model,
        revision=args.revision,
    )
    output_path = write_markdown_report(stats, Path(args.out))
    print(f"Wrote {output_path}")
    if not args.skip_organized_results:
        written = write_new_model_results(
            specs=(
                ModelSpec(
                    label="DeepSeek-V4-Pro",
                    slug="deepseek-v4-pro",
                    hf_model=args.deepseek_v4_model,
                    kind="deepseek_v4",
                    revision=args.revision,
                ),
                ModelSpec(
                    label="Kimi-K2.5",
                    slug="kimi-k2.5",
                    hf_model=args.kimi_model,
                    kind="deepseek",
                    revision=args.revision,
                    metrics_config_path=("text_config",),
                ),
            ),
            models_root=args.models_root,
            results_root=args.results_root,
        )
        for path in written:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
