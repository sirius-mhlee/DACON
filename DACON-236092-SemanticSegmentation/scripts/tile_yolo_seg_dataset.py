import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm


Polygon = Tuple[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile YOLO segmentation dataset")
    parser.add_argument("--images-dir", type=str, default="datasets/images/train", help="Source image directory")
    parser.add_argument("--labels-dir", type=str, default="datasets/labels/train", help="Source label directory")
    parser.add_argument("--output-images-dir", type=str, default="datasets/images/train_tile", help="Output tiled images directory")
    parser.add_argument("--output-labels-dir", type=str, default="datasets/labels/train_tile", help="Output tiled labels directory")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=224, help="Stride for tiling")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (1=disabled)")
    parser.add_argument("--min-area", type=float, default=3.0, help="Minimum contour area to keep")
    parser.add_argument("--clear", action="store_true", help="Clear existing output directories first")
    return parser.parse_args()


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    return sorted(images_dir.rglob("*.png"))


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _parse_yolo_polygons(label_path: Path, img_w: int, img_h: int) -> List[Polygon]:
    polygons: List[Polygon] = []
    if not label_path.exists():
        return polygons

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue

        try:
            cls_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        points = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= float(img_w)
        points[:, 1] *= float(img_h)
        points[:, 0] = np.clip(points[:, 0], 0.0, float(img_w) - 1e-3)
        points[:, 1] = np.clip(points[:, 1], 0.0, float(img_h) - 1e-3)

        if len(points) >= 3:
            polygons.append((cls_id, points))

    return polygons


def _to_yolo_line(cls_id: int, points: np.ndarray, tile_w: int, tile_h: int) -> str:
    norm = []
    for x, y in points:
        norm.append(f"{float(x) / float(tile_w):.6f}")
        norm.append(f"{float(y) / float(tile_h):.6f}")
    return f"{cls_id} {' '.join(norm)}"


def _polygon_bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    return min_x, max_x, min_y, max_y


def _process_one(
    image_path_str: str,
    images_dir_str: str,
    labels_dir_str: str,
    output_images_dir_str: str,
    output_labels_dir_str: str,
    tile_size: int,
    stride: int,
    min_area: float,
) -> Tuple[int, int, int]:
    image_path = Path(image_path_str)
    images_dir = Path(images_dir_str)
    labels_dir = Path(labels_dir_str)
    output_images_dir = Path(output_images_dir_str)
    output_labels_dir = Path(output_labels_dir_str)

    rel = image_path.relative_to(images_dir)
    label_path = labels_dir / rel.with_suffix(".txt")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0, 0, 0

    img_h, img_w = image.shape[:2]
    polygons = _parse_yolo_polygons(label_path, img_w, img_h)

    xs = _tile_starts(img_w, tile_size, stride)
    ys = _tile_starts(img_h, tile_size, stride)

    total_tiles = 0
    label_tiles = 0

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, img_w)
            y1 = min(y0 + tile_size, img_h)
            tile = image[y0:y1, x0:x1]
            tile_h, tile_w = tile.shape[:2]
            if tile_h == 0 or tile_w == 0:
                continue

            total_tiles += 1
            tile_lines: List[str] = []

            for cls_id, points in polygons:
                min_x, max_x, min_y, max_y = _polygon_bbox(points)
                if max_x < x0 or min_x > x1 or max_y < y0 or min_y > y1:
                    continue

                local = points.copy()
                local[:, 0] -= float(x0)
                local[:, 1] -= float(y0)

                mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.round(local).astype(np.int32)], color=1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue

                    contour_points = contour.reshape(-1, 2)
                    if len(contour_points) < 3:
                        continue

                    tile_lines.append(_to_yolo_line(cls_id, contour_points, tile_w, tile_h))

            if tile_lines:
                label_tiles += 1

            tile_name = f"{rel.stem}_x{x0}_y{y0}.png"
            out_img_path = output_images_dir / rel.parent / tile_name
            out_lbl_path = output_labels_dir / rel.parent / f"{Path(tile_name).stem}.txt"
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_img_path), tile)
            out_lbl_path.write_text("\n".join(tile_lines) + "\n", encoding="utf-8")

    return total_tiles, label_tiles


def tile_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    tile_size: int,
    stride: int,
    workers: int,
    min_area: float,
    clear: bool,
) -> None:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive: {tile_size}")

    if stride <= 0:
        raise ValueError(f"stride must be positive: {stride}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if clear:
        _clear_dir(output_images_dir)
        _clear_dir(output_labels_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    label_tiles = 0

    if workers <= 1:
        for image_path in tqdm(image_paths, total=len(image_paths), desc="Tiling", unit="image"):
            t, l = _process_one(
                str(image_path),
                str(images_dir),
                str(labels_dir),
                str(output_images_dir),
                str(output_labels_dir),
                tile_size,
                stride,
                min_area,
            )
            total_tiles += t
            label_tiles += l
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_one,
                    str(image_path),
                    str(images_dir),
                    str(labels_dir),
                    str(output_images_dir),
                    str(output_labels_dir),
                    tile_size,
                    stride,
                    min_area,
                )
                for image_path in image_paths
            ]
            for future in tqdm(futures, total=len(futures), desc="Tiling", unit="image"):
                t, l = future.result()
                total_tiles += t
                label_tiles += l

    print(f"Source images: {len(image_paths)}")
    print(f"Total tiles: {total_tiles}")
    print(f"Tiles with labels: {label_tiles}")


def main() -> None:
    args = parse_args()
    tile_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_images_dir=Path(args.output_images_dir),
        output_labels_dir=Path(args.output_labels_dir),
        tile_size=args.tile_size,
        stride=args.stride,
        workers=args.workers,
        min_area=args.min_area,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
