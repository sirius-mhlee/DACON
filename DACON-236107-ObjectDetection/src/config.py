from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    device: int = 0
    seed: int = 2026

    train_model: str = "models/yolov8n.pt"
    train_project: str = "runs/train"
    train_name: str = "exp"
    data_yaml: str = "datasets/data.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01

    predict_model: str = "runs/exp/weights/best.pt"
    predict_project: str = "runs/predict"
    predict_name: str = "exp"
    data_folder: str = "datasets/images/test"
    sample_csv: str = "datasets/sample_submission.csv"
    output_csv: str = "submit.csv"
    conf: float = 0.25
    iou: float = 0.7
    save_txt: bool = False
    save_conf: bool = False
    save_result: bool = False


def _filter_keys(data: Dict[str, Any], schema: Type[Config]) -> Dict[str, Any]:
    allowed = {field.name for field in schema.__dataclass_fields__.values()}
    return {key: value for key, value in data.items() if key in allowed}


def load_config(path: str) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return Config(**_filter_keys(data, Config))
