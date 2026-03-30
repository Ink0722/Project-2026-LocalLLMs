from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.adapters.base import BaseAdapter
from src.core.interfaces import LoadedArtifacts


class ChatCausalLMAdapter(BaseAdapter):
    def prepare_inputs(self, artifacts: LoadedArtifacts, messages: List[Dict[str, Any]]) -> Any:
        tokenizer = artifacts.tokenizer
        model = artifacts.model

        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = "\n".join(message["content"] for message in messages)
            inputs = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(inputs, torch.Tensor):
            return inputs.to(model.device)

        return {key: value.to(model.device) for key, value in inputs.items()}

    def generate_text(
        self,
        artifacts: LoadedArtifacts,
        prepared_inputs: Any,
        generation_config: Dict[str, Any],
    ) -> str:
        model = artifacts.model
        tokenizer = artifacts.tokenizer

        if isinstance(prepared_inputs, torch.Tensor):
            outputs = model.generate(prepared_inputs, **generation_config)
            generated_tokens = outputs[0][prepared_inputs.shape[1] :]
        else:
            outputs = model.generate(**prepared_inputs, **generation_config)
            input_length = prepared_inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

