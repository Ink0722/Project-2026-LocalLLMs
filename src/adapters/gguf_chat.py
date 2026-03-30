from __future__ import annotations

from typing import Any, Dict, List

from src.adapters.base import BaseAdapter
from src.core.interfaces import LoadedArtifacts


class GGUFChatAdapter(BaseAdapter):
    def prepare_inputs(self, artifacts: LoadedArtifacts, messages: List[Dict[str, Any]]) -> Any:
        return messages

    def generate_text(
        self,
        artifacts: LoadedArtifacts,
        prepared_inputs: Any,
        generation_config: Dict[str, Any],
    ) -> str:
        model = artifacts.model
        completion_kwargs = {
            "messages": prepared_inputs,
            "max_tokens": generation_config.get("max_new_tokens"),
            "temperature": generation_config.get("temperature"),
            "top_p": generation_config.get("top_p"),
            "top_k": generation_config.get("top_k"),
            "repeat_penalty": generation_config.get("repetition_penalty"),
        }

        if generation_config.get("stop") is not None:
            completion_kwargs["stop"] = generation_config["stop"]

        response = model.create_chat_completion(**completion_kwargs)
        return response["choices"][0]["message"]["content"].strip()
