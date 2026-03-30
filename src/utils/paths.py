from pathlib import Path


def _candidate_roots() -> list[Path]:
    file_path = Path(__file__).resolve()
    cwd = Path.cwd().resolve()

    candidates: list[Path] = []

    for base in [file_path, *file_path.parents, cwd, *cwd.parents]:
        root = base if base.is_dir() else base.parent
        if root not in candidates:
            candidates.append(root)

    return candidates


def _discover_project_root() -> Path:
    for root in _candidate_roots():
        models_dir = root / "configs" / "models"
        runtime_dir = root / "configs" / "runtime"
        if models_dir.exists() and runtime_dir.exists():
            return root

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _discover_project_root()
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
