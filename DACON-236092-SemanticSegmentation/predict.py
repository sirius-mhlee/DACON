import argparse

from src.config import load_config
from src.predictor import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO prediction entrypoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    predict(cfg)


if __name__ == "__main__":
    main()
