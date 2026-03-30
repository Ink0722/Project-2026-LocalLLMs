from __future__ import annotations

from typing import Any, Dict, List

from src.core.generation import build_generation_config
from src.core.interfaces import InferenceResult
from src.core.model_loader import load_model_artifacts
from src.core.registry import get_adapter


class InferenceService:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.artifacts = load_model_artifacts(model_config)
        self.adapter = get_adapter(model_config["task_type"])

    def infer(
        self, prompt: str, generation_overrides: Dict[str, Any] | None = None
    ) -> InferenceResult:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        prepared_inputs = self.adapter.prepare_inputs(self.artifacts, messages)
        generation_config = build_generation_config(
            self.model_config, overrides=generation_overrides
        )
        response = self.adapter.generate_text(
            self.artifacts, prepared_inputs, generation_config
        )

        return InferenceResult(
            model_name=self.model_config["name"],
            model_id=self.model_config["model_id"],
            prompt=prompt,
            response=response,
            device=self.artifacts.device,
        )
