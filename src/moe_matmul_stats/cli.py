"""Command line entry point for generating the combined matmul report."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    revision: str = "main",
) -> list[ModelStats]:
    return [
        collectStatsNanoJax(),
        collectStatsHF(olmoe_model, revision=revision),
        collectStatsHF(deepseek_model, revision=revision),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static MoE matmul stats.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Markdown output path.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision to fetch.")
    parser.add_argument("--olmoe-model", default=DEFAULT_OLMOE_MODEL)
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = collect_default_stats(
        olmoe_model=args.olmoe_model,
        deepseek_model=args.deepseek_model,
        revision=args.revision,
    )
    output_path = write_markdown_report(stats, Path(args.out))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
