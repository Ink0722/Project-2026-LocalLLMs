from __future__ import annotations

from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from src.core.device import get_default_device, parse_torch_dtype
from src.core.interfaces import LoadedArtifacts
from src.core.quantization import build_quantization_config


def _load_transformers_artifacts(model_config: Dict[str, Any]) -> LoadedArtifacts:
    tokenizer, processor = _load_tokenizer_or_processor(model_config)

    runtime = model_config.get("runtime", {})
    quantization_config = build_quantization_config(model_config.get("quantization"))
    torch_dtype = parse_torch_dtype(runtime.get("torch_dtype"))

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "low_cpu_mem_usage": True,
    }

    if runtime.get("device_map") is not None:
        load_kwargs["device_map"] = runtime["device_map"]

    if runtime.get("max_memory"):
        load_kwargs["max_memory"] = runtime["max_memory"]

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    elif torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    if runtime.get("device_map") in (None, "single_device"):
        load_kwargs["device_map"] = {"": get_default_device()}

    model = AutoModelForCausalLM.from_pretrained(model_config["model_id"], **load_kwargs)

    if tokenizer is not None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return LoadedArtifacts(
        tokenizer=tokenizer,
        processor=processor,
        model=model,
        config=model_config,
        device=str(model.device),
    )


def _load_llama_cpp_artifacts(model_config: Dict[str, Any]) -> LoadedArtifacts:
    from llama_cpp import Llama

    runtime = model_config.get("runtime", {})
    artifacts = model_config.get("artifacts", {})
    gguf_filename = artifacts.get("gguf_filename")

    if not gguf_filename:
        raise ValueError("GGUF model config must include artifacts.gguf_filename")

    llama_kwargs: Dict[str, Any] = {
        "repo_id": model_config["model_id"],
        "filename": gguf_filename,
        "verbose": False,
    }

    if runtime.get("n_ctx") is not None:
        llama_kwargs["n_ctx"] = runtime["n_ctx"]
    if runtime.get("n_batch") is not None:
        llama_kwargs["n_batch"] = runtime["n_batch"]
    if runtime.get("gpu_layers") is not None:
        llama_kwargs["n_gpu_layers"] = runtime["gpu_layers"]
    if runtime.get("chat_format"):
        llama_kwargs["chat_format"] = runtime["chat_format"]

    model = Llama.from_pretrained(**llama_kwargs)

    return LoadedArtifacts(
        tokenizer=None,
        processor=None,
        model=model,
        config=model_config,
        device="llama.cpp",
    )


def _load_tokenizer_or_processor(model_config: Dict[str, Any]):
    model_id = model_config["model_id"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    processor_type = model_config.get("processor_type", "tokenizer")

    if processor_type == "processor":
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        return None, processor

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    return tokenizer, None


def load_model_artifacts(model_config: Dict[str, Any]) -> LoadedArtifacts:
    loader_type = model_config.get("loader_type", "auto_causal_lm")

    if loader_type == "llama_cpp_gguf":
        return _load_llama_cpp_artifacts(model_config)

    return _load_transformers_artifacts(model_config)
