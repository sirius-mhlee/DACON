from ultralytics import YOLO

from .config import Config


def predict(cfg: Config) -> None:
    model = YOLO(cfg.model)
    model.predict(
        source=cfg.source,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        save_txt=cfg.save_txt,
        save_conf=cfg.save_conf,
    )
