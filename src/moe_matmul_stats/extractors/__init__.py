"""Extractor registry for model-specific architecture handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from moe_matmul_stats.schema import ModelStats

from .deepseek_v3 import extract_deepseek_v3
from .deepseek_v4 import extract_deepseek_v4
from .kimi_k25 import extract_kimi_k2_text, extract_kimi_k25
from .nano_jax import extract_nano_jax
from .olmoe import extract_olmoe

Extractor = Callable[[str, str, Mapping[str, Any]], ModelStats]


class UnsupportedModelError(ValueError):
    """Raised when no extractor is registered for a config model_type."""


EXTRACTORS: dict[str, Extractor] = {
    "deepseek_v3": extract_deepseek_v3,
    "deepseek_v4": extract_deepseek_v4,
    "kimi_k2": extract_kimi_k2_text,
    "kimi_k25": extract_kimi_k25,
    "nano_moe_jax": extract_nano_jax,
    "olmoe": extract_olmoe,
}


def get_extractor(model_type: str) -> Extractor:
    try:
        return EXTRACTORS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(EXTRACTORS))
        raise UnsupportedModelError(
            f"Unsupported model_type {model_type!r}. Supported: {supported}"
        ) from exc
