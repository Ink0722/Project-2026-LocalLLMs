from __future__ import annotations

from src.adapters.chat_causal_lm import ChatCausalLMAdapter
from src.adapters.gguf_chat import GGUFChatAdapter
from src.adapters.multimodal import MultimodalAdapter


REGISTRY = {
    "chat_causal_lm": ChatCausalLMAdapter,
    "gguf_chat": GGUFChatAdapter,
    "multimodal": MultimodalAdapter,
}


def get_adapter(task_type: str):
    if task_type not in REGISTRY:
        supported = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unsupported task_type '{task_type}'. Supported types: {supported}")
    return REGISTRY[task_type]()
