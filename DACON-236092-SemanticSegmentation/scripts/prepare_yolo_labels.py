import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm.auto import tqdm
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RLE mask to YOLO polygon.")
    parser.add_argument("--input-dir", type=str, default="datasets/images/train", help="Input image directory")
    parser.add_argument("--output-dir", type=str, default="datasets/labels/train", help="Output label directory")
    parser.add_argument("--csv-path", type=str, default="datasets/images/train/train.csv", help="CSV path (img_id, img_path, mask_rle)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (1=disabled)")
    parser.add_argument("--class-id", type=int, default=1, help="Class id for all polygons")
    return parser.parse_args()


def _decode_rle(mask_rle: str, img_w: int, img_h: int) -> np.ndarray:
    mask = np.zeros(img_w * img_h, dtype=np.uint8)
    if not mask_rle or mask_rle == "-1":
        return mask.reshape((img_h, img_w), order="F")

    tokens = mask_rle.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"Invalid RLE: {mask_rle[:100]}...")

    starts = np.asarray(tokens[0::2], dtype=np.int64) - 1
    lengths = np.asarray(tokens[1::2], dtype=np.int64)
    ends = starts + lengths

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape((img_h, img_w), order="F")


def _mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[np.ndarray] = []

    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) >= 3:
            polygons.append(points)

    return polygons


def _convert_one(
    image_name: str,
    rles: List[str],
    input_dir_str: str,
    output_dir_str: str,
    class_id: int,
) -> Tuple[bool, int]:
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    image_path = input_dir / image_name
    if not image_path.exists():
        return False, 0

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    lines: List[str] = []
    polygons_count = 0

    for mask_rle in rles:
        if not mask_rle or mask_rle == "-1":
            continue

        mask = _decode_rle(mask_rle, img_w, img_h)
        polygons = _mask_to_polygons(mask)
        for polygon in polygons:
            coords: List[str] = []
            for x, y in polygon:
                coords.append(f"{float(x) / float(img_w):.6f}")
                coords.append(f"{float(y) / float(img_h):.6f}")
            lines.append(f"{class_id} {' '.join(coords)}")
            polygons_count += 1

    out_path = output_dir / f"{Path(image_name).stem}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return True, polygons_count


def _load_rows(csv_path: Path) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = (row.get("img_path") or "").strip()
            if image_name == "":
                continue

            grouped[f"{Path(image_name).name}"].append((row.get("mask_rle") or "").strip())
    return grouped


def convert_labels(input_dir: Path, output_dir: Path, csv_path: Path, workers: int, class_id: int) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    grouped = _load_rows(csv_path)
    if not grouped:
        raise ValueError(f"No rows found in csv: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(grouped.items())

    ok_count = 0
    miss_count = 0
    total_polygons = 0

    if workers <= 1:
        for image_name, rles in tqdm(items, total=len(items), desc="Converting", unit="file"):
            ok, poly_count = _convert_one(image_name, rles, str(input_dir), str(output_dir), class_id)
            if ok:
                ok_count += 1
                total_polygons += poly_count
            else:
                miss_count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_convert_one, image_name, rles, str(input_dir), str(output_dir), class_id)
                for image_name, rles in items
            ]
            for future in tqdm(futures, total=len(futures), desc="Converting", unit="file"):
                ok, poly_count = future.result()
                if ok:
                    ok_count += 1
                    total_polygons += poly_count
                else:
                    miss_count += 1

    print(f"Processed images: {ok_count}")
    print(f"Missing images: {miss_count}")
    print(f"Generated polygons: {total_polygons}")


def main() -> None:
    args = parse_args()
    convert_labels(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        csv_path=Path(args.csv_path),
        workers=args.workers,
        class_id=args.class_id,
    )


if __name__ == "__main__":
    main()
