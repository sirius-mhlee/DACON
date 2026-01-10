from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "runs"


@dataclass
class Config:
    device: str = ""
    project: str = "runs"
    name: str = "exp"

    data_yaml: str = "datasets/data.yaml"
    model: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    seed: int = 2023

    source: str = "datasets/images"
    conf: float = 0.25
    iou: float = 0.7
    save_txt: bool = False
    save_conf: bool = False


def load_config(path: str) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return Config(**_filter_keys(data, Config))


def _filter_keys(data: Dict[str, Any], schema: Type[Config]) -> Dict[str, Any]:
    allowed = {field.name for field in schema.__dataclass_fields__.values()}
    return {key: value for key, value in data.items() if key in allowed}
