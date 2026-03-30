from argparse import Namespace

from src.cli.main import run_download


def download_model():
    return run_download(Namespace(model="qwen2_5_7b_instruct", runtime="local_gpu"))


if __name__ == "__main__":
    raise SystemExit(download_model())
