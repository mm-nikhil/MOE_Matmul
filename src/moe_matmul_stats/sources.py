"""Config sources and top-level collection entry points."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .extractors import get_extractor
from .schema import ModelStats


class ConfigLoadError(RuntimeError):
    """Raised when a model config cannot be loaded."""


NANO_MOE_JAX_DEFAULT_CONFIG: dict[str, Any] = {
    "model_type": "nano_moe_jax",
    "vocab_size": 256,
    "n_layers": 4,
    "n_heads": 4,
    "d_model": 128,
    "d_ff": 512,
    "n_experts": 4,
    "top_k": 2,
    "block_size": 128,
    "dropout_rate": 0.1,
    "aux_loss_coeff": 0.01,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "batch_size": 32,
    "max_iters": 5000,
    "eval_interval": 250,
    "eval_iters": 50,
}


def collectStatsHF(modelName: str, revision: str = "main") -> ModelStats:
    """Collect static stats for a Hugging Face model config.

    This fetches only config.json. It does not instantiate model code or download weights.
    """

    config = fetch_hf_config(modelName, revision=revision)
    return collectStatsFromConfig(
        model=modelName,
        source=f"huggingface:{modelName}@{revision}",
        config=config,
    )


def collectStatsNanoJax(config: Mapping[str, Any] | None = None) -> ModelStats:
    """Collect static stats for Nano-MoE-JAX using its public default config."""

    merged_config = dict(NANO_MOE_JAX_DEFAULT_CONFIG)
    if config:
        merged_config.update(config)

    return collectStatsFromConfig(
        model="Nano-MoE-JAX",
        source="github:carrycooldude/Nano-MoE-JAX defaults",
        config=merged_config,
    )


def collectStatsFromConfig(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    """Route a config dictionary to the registered architecture extractor."""

    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ConfigLoadError(f"Config for {model!r} does not contain a valid model_type.")

    extractor = get_extractor(model_type)
    return extractor(model, source, dict(config))


def fetch_hf_config(modelName: str, revision: str = "main") -> dict[str, Any]:
    """Fetch config.json from Hugging Face as raw JSON."""

    url = hf_config_url(modelName, revision=revision)
    request = Request(url, headers={"User-Agent": "moe-matmul-stats/0.1"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ConfigLoadError(f"Failed to fetch Hugging Face config from {url}: {exc}") from exc

    try:
        config = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConfigLoadError(f"Hugging Face config at {url} is not valid JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigLoadError(f"Hugging Face config at {url} must be a JSON object.")
    return config


def hf_config_url(modelName: str, revision: str = "main") -> str:
    quoted_model = quote(modelName, safe="/")
    quoted_revision = quote(revision, safe="")
    return f"https://huggingface.co/{quoted_model}/raw/{quoted_revision}/config.json"


collect_stats_hf = collectStatsHF
collect_stats_nano_jax = collectStatsNanoJax
collect_stats_from_config = collectStatsFromConfig
