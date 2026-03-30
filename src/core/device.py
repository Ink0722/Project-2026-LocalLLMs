from __future__ import annotations

import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_torch_dtype(dtype_name: str | None):
    if not dtype_name:
        return None
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'")
    return DTYPE_MAP[dtype_name]


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

