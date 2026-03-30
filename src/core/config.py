from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import yaml

from src.utils.paths import resolve_model_config_path, resolve_runtime_config_path


def _load_yaml(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_model_config(model_name: str, runtime_name: str = "local_gpu") -> Dict[str, Any]:
    runtime_config = _load_yaml(resolve_runtime_config_path(runtime_name))
    model_config = _load_yaml(resolve_model_config_path(model_name))

    merged = _deep_merge(runtime_config.get("defaults", {}), model_config.get("runtime", {}))
    model_config["runtime"] = merged

    if runtime_config.get("generation"):
        model_config["generation"] = _deep_merge(
            runtime_config["generation"], model_config.get("generation", {})
        )

    if runtime_config.get("download"):
        model_config["download"] = _deep_merge(
            runtime_config["download"], model_config.get("download", {})
        )

    return model_config


def list_available_models() -> list[str]:
    from src.utils.paths import MODELS_DIR

    return sorted(path.stem for path in MODELS_DIR.glob("*.yaml"))

