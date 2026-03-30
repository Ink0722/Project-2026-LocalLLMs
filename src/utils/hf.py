from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict


def resolve_download_source(
    download_config: Dict[str, Any] | None, source_name: str
) -> tuple[str, str | None]:
    config = download_config or {}

    if source_name == "mirror":
        return "mirror", config.get("mirror_endpoint")
    if source_name == "official":
        return "official", None

    raise ValueError(f"Unsupported download source '{source_name}'")


@contextmanager
def hf_endpoint_override(endpoint: str | None):
    previous = os.environ.get("HF_ENDPOINT")

    try:
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous
