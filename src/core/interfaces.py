from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LoadedArtifacts:
    tokenizer: Any | None
    processor: Any | None
    model: Any
    config: Dict[str, Any]
    device: str


@dataclass
class InferenceResult:
    model_name: str
    model_id: str
    prompt: str
    response: str
    device: str


class BaseAdapter(ABC):
    @abstractmethod
    def prepare_inputs(self, artifacts: LoadedArtifacts, messages: List[Dict[str, Any]]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        artifacts: LoadedArtifacts,
        prepared_inputs: Any,
        generation_config: Dict[str, Any],
    ) -> str:
        raise NotImplementedError
