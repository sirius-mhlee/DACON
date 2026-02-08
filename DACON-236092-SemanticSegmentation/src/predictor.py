from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

from .config import Config, ROOT_DIR


def _collect_images(data_folder: Path) -> List[Path]:
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    return sorted([path for path in data_folder.rglob("*.png")])


def _encode_rle(mask: np.ndarray) -> str:
    pixels = mask.flatten()
    if pixels.max() == 0:
        return "-1"

    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _results_to_rle(results) -> str:
    if not results:
        return "-1"

    img_h, img_w = results[0].orig_shape
    merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for result in results:
        if result.masks is None or len(result.masks) == 0:
            continue

        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            binary = (mask > 0.0).astype(np.uint8)
            if binary.shape != (img_h, img_w):
                binary = np.asarray(
                    Image.fromarray(binary).resize((img_w, img_h), resample=Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )

            if binary.max() == 0:
                continue

            merged_mask = np.maximum(merged_mask, binary)

    if merged_mask.max() == 0:
        return "-1"

    return _encode_rle(merged_mask)


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
            imgsz=cfg.imgsz,
            rect=True,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            project=ROOT_DIR / cfg.predict_project,
            name=cfg.predict_name,
            save_txt=cfg.save_txt,
            save_conf=cfg.save_conf,
            verbose=False,
            exist_ok=True,
            save=cfg.save_result,
            show_labels=cfg.show_labels,
        )

        new_rows.append(
            {
                "img_id": image_path.stem,
                "mask_rle": _results_to_rle(results),
            }
        )

    if new_rows:
        submit = pd.DataFrame(data=new_rows, columns=submit.columns)

    output_dir = ROOT_DIR / cfg.predict_project / cfg.predict_name
    submit.to_csv(output_dir / cfg.output_csv, index=False)
