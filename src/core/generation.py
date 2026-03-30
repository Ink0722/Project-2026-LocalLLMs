from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def build_generation_config(
    model_config: Dict[str, Any], overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    generation = deepcopy(model_config.get("generation", {}))
    if not overrides:
        return generation

    for key, value in overrides.items():
        if value is not None:
            generation[key] = value
    return generation

