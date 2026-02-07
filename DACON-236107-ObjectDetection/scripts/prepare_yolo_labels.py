import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe bbox to YOLO bbox.")
    parser.add_argument("--input-dir", type=str, default="datasets/images/train", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="datasets/labels/train", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (1=disabled)")
    parser.add_argument("--clear", action="store_true", help="Clear existing label folders first")
    return parser.parse_args()


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _poly_to_xywh(coords: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    xc = (min_x + max_x) / 2.0 / img_w
    yc = (min_y + max_y) / 2.0 / img_h
    w = (max_x - min_x) / img_w
    h = (max_y - min_y) / img_h
    return xc, yc, w, h


def _convert_one(label_path_str: str, input_dir_str: str, output_dir_str: str) -> None:
    label_path = Path(label_path_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    image_path = input_dir / f"{label_path.stem}.png"
    if not image_path.exists():
        print(f"Skip {label_path.name}: missing image")
        return

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    new_lines: List[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = [float(val) for val in line.split()]
        cls_id = int(parts[0])
        xc, yc, w, h = _poly_to_xywh(parts[1:], img_w, img_h)
        new_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    out_path = output_dir / label_path.name
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def convert_labels(input_dir: Path, output_dir: Path, workers: int, clear: bool) -> None:
    label_paths = sorted(input_dir.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No label files found in {input_dir}")
    
    if clear:
        _clear_dir(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for label_path in tqdm(label_paths, total=len(label_paths), desc="Converting", unit="file"):
            _convert_one(str(label_path), str(input_dir), str(output_dir))
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_convert_one, str(label_path), str(input_dir), str(output_dir))
            for label_path in label_paths
        ]
        for future in tqdm(futures, total=len(futures), desc="Converting", unit="file"):
            future.result()


def main() -> None:
    args = parse_args()
    convert_labels(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        workers=args.workers,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
