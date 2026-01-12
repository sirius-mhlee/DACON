from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

from .config import Config, ROOT_DIR


def _collect_images(data_folder: Path) -> List[Path]:
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    return sorted([path for path in data_folder.rglob("*.png")])


def _xyxy_to_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    x1p = x1 * img_w
    x2p = x2 * img_w
    y1p = y1 * img_h
    y2p = y2 * img_h
    return (x1p, y1p, x2p, y1p, x2p, y2p, x1p, y2p)


def predict(cfg: Config) -> None:
    model = YOLO(cfg.predict_model)
    image_paths = _collect_images(Path(cfg.data_folder))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg.data_folder}")

    submit = pd.read_csv(cfg.sample_csv)
    new_rows = []

    for image_path in tqdm(image_paths, total=len(image_paths), desc="Prediction", unit="file"):
        results = model.predict(
            source=str(image_path),
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            project=ROOT_DIR / cfg.predict_project,
            name=cfg.predict_name,
            save_txt=cfg.save_txt,
            save_conf=cfg.save_conf,
            verbose=False,
            exist_ok=True,
            save=False,
        )

        if not results:
            continue

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            img_h, img_w = result.orig_shape
            xyxy = result.boxes.xyxyn.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
                points = _xyxy_to_points(x1, y1, x2, y2, img_w, img_h)
                new_rows.append(
                    {
                        "file_name": image_path.name,
                        "class_id": int(cls_id),
                        "confidence": float(conf),
                        "point1_x": int(points[0]),
                        "point1_y": int(points[1]),
                        "point2_x": int(points[2]),
                        "point2_y": int(points[3]),
                        "point3_x": int(points[4]),
                        "point3_y": int(points[5]),
                        "point4_x": int(points[6]),
                        "point4_y": int(points[7]),
                    }
                )

    if new_rows:
        submit = pd.DataFrame(data=new_rows, columns=submit.columns)

    output_dir = ROOT_DIR / cfg.predict_project / cfg.predict_name
    submit.to_csv(output_dir / cfg.output_csv, index=False)
