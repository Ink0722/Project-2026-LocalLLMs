from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = CONFIGS_DIR / "models"
RUNTIME_DIR = CONFIGS_DIR / "runtime"


def resolve_model_config_path(model_name: str) -> Path:
    candidate = MODELS_DIR / f"{model_name}.yaml"
    if not candidate.exists():
        available = ", ".join(sorted(path.stem for path in MODELS_DIR.glob("*.yaml")))
        raise FileNotFoundError(
            f"Unknown model config '{model_name}'. Available models: {available}"
        )
    return candidate


def resolve_runtime_config_path(runtime_name: str) -> Path:
    candidate = RUNTIME_DIR / f"{runtime_name}.yaml"
    if not candidate.exists():
        available = ", ".join(sorted(path.stem for path in RUNTIME_DIR.glob("*.yaml")))
        raise FileNotFoundError(
            f"Unknown runtime config '{runtime_name}'. Available runtimes: {available}"
        )
    return candidate

