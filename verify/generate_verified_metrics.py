#!/usr/bin/env python3
"""Generate verified metric tables under verify/."""

from __future__ import annotations

from pathlib import Path

from moe_matmul_stats.verification import (
    build_verified_rows,
    write_verified_outputs,
)


def main() -> int:
    root = Path(__file__).resolve().parent
    rows = build_verified_rows(root / "ai_filled_metrics_sheet.tsv")
    markdown, csv_path, results = write_verified_outputs(
        rows,
        markdown_path=root / "verified_metrics.md",
        csv_path=root / "verified_metrics.csv",
        results_path=root / "verified_results.md",
    )
    print(f"Wrote {markdown}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
