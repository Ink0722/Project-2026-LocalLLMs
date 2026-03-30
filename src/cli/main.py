from __future__ import annotations

import argparse
from typing import Any, Dict

from src.core.config import list_available_models, load_model_config


def _build_generation_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }


def run_chat(args: argparse.Namespace) -> int:
    from src.services.inference_service import InferenceService

    model_config = load_model_config(args.model, runtime_name=args.runtime)
    service = InferenceService(model_config)

    print(
        f"Model loaded: {model_config['name']} ({model_config['model_id']}). "
        "Type 'exit' to quit."
    )

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Bye!")
            return 0

        result = service.infer(user_input, _build_generation_overrides(args))
        print(f"Assistant: {result.response}")


def run_infer(args: argparse.Namespace) -> int:
    from src.services.inference_service import InferenceService

    model_config = load_model_config(args.model, runtime_name=args.runtime)
    service = InferenceService(model_config)
    result = service.infer(args.prompt, _build_generation_overrides(args))
    print(result.response)
    return 0


def run_download(args: argparse.Namespace) -> int:
    from src.services.download_service import download_model

    model_config = load_model_config(args.model, runtime_name=args.runtime)
    print(f"Downloading {model_config['model_id']} ...")
    download_model(model_config, source_override=args.source)
    print("Download finished.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Config-driven local inference runner for Hugging Face models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "--model",
        required=True,
        choices=list_available_models(),
        help="Model config name under configs/models.",
    )
    common_parent.add_argument(
        "--runtime",
        default="local_gpu",
        help="Runtime config name under configs/runtime.",
    )
    common_parent.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    common_parent.add_argument("--temperature", type=float)
    common_parent.add_argument("--top-p", type=float, dest="top_p")
    common_parent.add_argument("--top-k", type=int, dest="top_k")
    common_parent.add_argument("--repetition-penalty", type=float, dest="repetition_penalty")

    chat_parser = subparsers.add_parser("chat", parents=[common_parent])
    chat_parser.set_defaults(func=run_chat)

    infer_parser = subparsers.add_parser("infer", parents=[common_parent])
    infer_parser.add_argument("--prompt", required=True, help="Single prompt to run.")
    infer_parser.set_defaults(func=run_infer)

    download_parser = subparsers.add_parser("download", parents=[common_parent])
    download_parser.add_argument(
        "--source",
        choices=["mirror", "official"],
        help="Force a specific download source instead of using automatic fallback.",
    )
    download_parser.set_defaults(func=run_download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
