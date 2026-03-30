from __future__ import annotations

from typing import Any, Dict

from transformers import BitsAndBytesConfig

from src.core.device import parse_torch_dtype


def build_quantization_config(quantization_config: Dict[str, Any] | None):
    if not quantization_config or not quantization_config.get("enabled"):
        return None

    kwargs = dict(quantization_config)
    kwargs.pop("enabled", None)

    compute_dtype = kwargs.get("bnb_4bit_compute_dtype")
    if isinstance(compute_dtype, str):
        kwargs["bnb_4bit_compute_dtype"] = parse_torch_dtype(compute_dtype)

    return BitsAndBytesConfig(**kwargs)

