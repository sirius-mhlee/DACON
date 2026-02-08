from ultralytics import YOLO

from .config import Config, ROOT_DIR


def train(cfg: Config) -> None:
    model = YOLO(cfg.train_model)
    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        rect=True,
        batch=cfg.batch,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        device=cfg.device,
        project=ROOT_DIR / cfg.train_project,
        name=cfg.train_name,
        seed=cfg.seed,
    )
