from __future__ import annotations

from huggingface_hub import snapshot_download

from src.utils.hf import hf_endpoint_override, resolve_download_source


def _build_allow_patterns(model_config) -> list[str] | None:
    artifacts = model_config.get("artifacts", {})
    gguf_filename = artifacts.get("gguf_filename")
    if gguf_filename:
        return [gguf_filename]
    return None


def _download_once(
    model_id: str,
    source_name: str,
    endpoint: str | None,
    allow_patterns: list[str] | None = None,
) -> None:
    endpoint_label = endpoint or "https://huggingface.co"
    if allow_patterns:
        print(
            f"Downloading from {source_name} source: {endpoint_label} "
            f"(files: {', '.join(allow_patterns)})"
        )
    else:
        print(f"Downloading from {source_name} source: {endpoint_label}")

    with hf_endpoint_override(endpoint):
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            allow_patterns=allow_patterns,
        )


def download_model(model_config, source_override: str | None = None):
    download_config = model_config.get("download", {})
    model_id = model_config["model_id"]
    allow_patterns = _build_allow_patterns(model_config)

    if source_override:
        source_name, endpoint = resolve_download_source(download_config, source_override)
        _download_once(model_id, source_name, endpoint, allow_patterns)
        return

    primary_source = download_config.get("primary_source", "official")
    fallback_source = download_config.get("fallback_source")

    source_name, endpoint = resolve_download_source(download_config, primary_source)

    try:
        _download_once(model_id, source_name, endpoint, allow_patterns)
    except Exception as exc:
        if not fallback_source or fallback_source == primary_source:
            raise

        fallback_name, fallback_endpoint = resolve_download_source(
            download_config, fallback_source
        )
        print(f"Primary download failed from {source_name}: {exc}")
        print(f"Retrying with fallback source: {fallback_name}")
        _download_once(model_id, fallback_name, fallback_endpoint, allow_patterns)
