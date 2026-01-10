from ultralytics import YOLO

from .config import Config


def train(cfg: Config) -> None:
    model = YOLO(cfg.model)
    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        seed=cfg.seed,
    )
