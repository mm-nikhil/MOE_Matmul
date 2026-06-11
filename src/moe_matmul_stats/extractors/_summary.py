"""Small helpers shared by extractor stubs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def summarize_config(config: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config[key] for key in keys if key in config}
