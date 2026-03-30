from __future__ import annotations

from typing import Any, Dict, List

from src.adapters.base import BaseAdapter, AdapterError
from src.core.interfaces import LoadedArtifacts


class MultimodalAdapter(BaseAdapter):
    def prepare_inputs(self, artifacts: LoadedArtifacts, messages: List[Dict[str, Any]]) -> Any:
        raise AdapterError("Multimodal models are not enabled in the first implementation phase.")

    def generate_text(
        self,
        artifacts: LoadedArtifacts,
        prepared_inputs: Any,
        generation_config: Dict[str, Any],
    ) -> str:
        raise AdapterError("Multimodal models are not enabled in the first implementation phase.")

