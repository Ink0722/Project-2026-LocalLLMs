from argparse import Namespace

from src.cli.main import run_chat


def run_inference():
    return run_chat(
        Namespace(
            model="deepseek_chat",
            runtime="local_gpu",
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
        )
    )


if __name__ == "__main__":
    raise SystemExit(run_inference())
