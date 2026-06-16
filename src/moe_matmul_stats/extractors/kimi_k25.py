"""Kimi K2.5 static matmul extractor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from moe_matmul_stats.schema import ModelStats

from .deepseek_v3 import SUMMARY_KEYS as TEXT_SUMMARY_KEYS
from .deepseek_v3 import extract_deepseek_mla_moe


TOP_LEVEL_SUMMARY_KEYS = (
    "model_type",
    "dtype",
    "use_unified_vision_chunk",
)

VISION_SUMMARY_KEYS = (
    "vt_hidden_size",
    "vt_intermediate_size",
    "vt_num_attention_heads",
    "vt_num_hidden_layers",
    "text_hidden_size",
)


def extract_kimi_k25(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    cfg = dict(config)
    text_config = cfg.get("text_config")
    if not isinstance(text_config, Mapping):
        raise ValueError("Kimi K2.5 config must contain a text_config object.")

    text_cfg = dict(text_config)
    text_stats = extract_deepseek_mla_moe(
        model=model,
        source=source,
        config=text_cfg,
        architecture_label="Kimi K2.5 text decoder",
        extra_notes=(
            "Kimi K2.5 wraps this text decoder in a multimodal KimiK25 model.",
            "Config exposes a MoonViT vision tower and multimodal projector; this report covers the text MoE decoder and LM head only.",
        ),
    )

    config_summary: dict[str, Any] = {
        key: cfg[key] for key in TOP_LEVEL_SUMMARY_KEYS if key in cfg
    }
    config_summary.update(
        {f"text.{key}": text_cfg[key] for key in TEXT_SUMMARY_KEYS if key in text_cfg}
    )

    vision_config = cfg.get("vision_config")
    if isinstance(vision_config, Mapping):
        config_summary.update(
            {
                f"vision.{key}": vision_config[key]
                for key in VISION_SUMMARY_KEYS
                if key in vision_config
            }
        )

    return ModelStats.from_records(
        model=model,
        source=source,
        records=text_stats.records,
        config_summary=config_summary,
        notes=text_stats.notes,
    )


def extract_kimi_k2_text(model: str, source: str, config: Mapping[str, Any]) -> ModelStats:
    return extract_deepseek_mla_moe(
        model=model,
        source=source,
        config=config,
        architecture_label="Kimi K2 text decoder",
        extra_notes=(
            "This extractor is for the text_config decoder used by Kimi K2.x checkpoints.",
        ),
    )
