import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split train/val with symlinks.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config yaml for defaults")
    parser.add_argument("--images-dir", type=str, default="datasets/images/train", help="Source images directory")
    parser.add_argument("--labels-dir", type=str, default="datasets/labels/train", help="Source labels directory")
    parser.add_argument("--output-root", type=str, default="datasets", help="Dataset root for outputs")
    parser.add_argument("--train-name", type=str, default="train_split", help="Output train split name")
    parser.add_argument("--val-name", type=str, default="val_split", help="Output val split name")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (overrides config)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (1=disabled)")
    parser.add_argument("--clear", action="store_true", help="Clear existing split folders first")
    return parser.parse_args()


def _collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted([path for path in images_dir.rglob("*.png")])


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _make_pair(images_dir: Path, labels_dir: Path, image_path: Path) -> Optional[Tuple[Path, Path, Path]]:
    rel = image_path.relative_to(images_dir)
    label_path = labels_dir / rel.with_suffix(".txt")
    if not label_path.exists():
        return None
    return (image_path, label_path, rel)


def _make_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_paths: List[Path],
    workers: int,
) -> Tuple[List[Tuple[Path, Path, Path]], int]:
    missing_labels = 0
    paired: List[Tuple[Path, Path, Path]] = []
    if workers <= 1:
        for image_path in tqdm(image_paths, total=len(image_paths), desc="Preparing", unit="file"):
            pair = _make_pair(images_dir, labels_dir, image_path)
            if pair is None:
                missing_labels += 1
            else:
                paired.append(pair)
        return paired, missing_labels

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_make_pair, repeat(images_dir), repeat(labels_dir), image_paths)
        for pair in tqdm(results, total=len(image_paths), desc="Preparing", unit="file"):
            if pair is None:
                missing_labels += 1
            else:
                paired.append(pair)
    return paired, missing_labels


def _split_items(
    items: List[Tuple[Path, Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path, Path]], List[Tuple[Path, Path, Path]]]:
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = int(len(items) * val_ratio)
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    val_items = items[:val_count]
    train_items = items[val_count:]
    return train_items, val_items


def _make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            raise IsADirectoryError(f"Refuse to overwrite directory: {dst}")
        dst.unlink()
    os.symlink(src.resolve(), dst)


def _make_symlinks(tasks: List[Tuple[Path, Path]], workers: int, desc: str) -> None:
    if not tasks:
        return

    if workers <= 1:
        for src, dst in tqdm(tasks, total=len(tasks), desc=desc, unit="link"):
            _make_symlink(src, dst)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_make_symlink, [t[0] for t in tasks], [t[1] for t in tasks])
        for _ in tqdm(results, total=len(tasks), desc=desc, unit="link"):
            pass


def split_train_val(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    train_name: str,
    val_name: str,
    val_ratio: float,
    seed: int,
    workers: int,
    clear: bool,
) -> None:
    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    paired, missing_labels = _make_pairs(images_dir, labels_dir, image_paths, workers)

    if not paired:
        raise FileNotFoundError(f"No matching labels found in {labels_dir}")

    train_pairs, val_pairs = _split_items(paired, val_ratio, seed)

    train_images_dir = output_root / "images" / train_name
    val_images_dir = output_root / "images" / val_name
    train_labels_dir = output_root / "labels" / train_name
    val_labels_dir = output_root / "labels" / val_name

    if clear:
        _clear_dir(train_images_dir)
        _clear_dir(val_images_dir)
        _clear_dir(train_labels_dir)
        _clear_dir(val_labels_dir)

    train_tasks = []
    for image_path, label_path, rel in train_pairs:
        train_tasks.append((image_path, train_images_dir / rel))
        train_tasks.append((label_path, train_labels_dir / rel.with_suffix(".txt")))

    val_tasks = []
    for image_path, label_path, rel in val_pairs:
        val_tasks.append((image_path, val_images_dir / rel))
        val_tasks.append((label_path, val_labels_dir / rel.with_suffix(".txt")))

    _make_symlinks(train_tasks, workers, "Symlinking (train)")
    _make_symlinks(val_tasks, workers, "Symlinking (val)")

    print(f"train: {len(train_pairs)} files")
    print(f"val: {len(val_pairs)} files")
    if missing_labels:
        print(f"skipped (missing labels): {missing_labels}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    split_train_val(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_root=Path(args.output_root),
        train_name=args.train_name,
        val_name=args.val_name,
        val_ratio=args.val_ratio,
        seed=seed,
        workers=args.workers,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
