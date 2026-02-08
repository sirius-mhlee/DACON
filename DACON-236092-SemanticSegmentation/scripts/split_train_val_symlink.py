import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config


LabeledPair = Tuple[Path, Path, Path, Tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split train/val with symlinks")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config yaml for defaults")
    parser.add_argument("--images-dir", type=str, default="datasets/images/train", help="Source images directory")
    parser.add_argument("--labels-dir", type=str, default="datasets/labels/train", help="Source labels directory")
    parser.add_argument("--output-root", type=str, default="datasets", help="Dataset root for outputs")
    parser.add_argument("--train-name", type=str, default="train_split", help="Output train split name")
    parser.add_argument("--val-name", type=str, default="val_split", help="Output val split name")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation split")
    parser.add_argument("--split-mode", type=str, choices=["random", "satisfied"], default="satisfied", help="Split strategy")
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


def _read_labels(label_path: Path) -> Tuple[int, ...]:
    labels = set()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)[0]
        try:
            cls_id = int(float(parts))
        except ValueError as exc:
            raise ValueError(f"Invalid label line in {label_path}: {line}") from exc

        labels.add(cls_id)
    return tuple(sorted(labels))


def _make_pair(images_dir: Path, labels_dir: Path, image_path: Path) -> Optional[LabeledPair]:
    rel = image_path.relative_to(images_dir)
    label_path = labels_dir / rel.with_suffix(".txt")
    if not label_path.exists():
        return None

    labels = _read_labels(label_path)
    return (image_path, label_path, rel, labels)


def _make_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_paths: List[Path],
    workers: int,
) -> Tuple[List[LabeledPair], int]:
    missing_labels = 0
    paired: List[LabeledPair] = []
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
    items: List[LabeledPair],
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> Tuple[List[LabeledPair], List[LabeledPair]]:
    if val_ratio <= 0:
        return items, []

    rng = random.Random(seed)
    total = len(items)
    val_count = int(round(total * val_ratio))
    if val_count <= 0:
        val_count = 1
    if val_count > total:
        val_count = total

    if split_mode == "random":
        indices = list(range(total))
        rng.shuffle(indices)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        train_items = [items[i] for i in train_indices]
        val_items = [items[i] for i in val_indices]
        return train_items, val_items

    labels_per_item = [set(pair[3]) for pair in items]
    label_counts: Dict[int, int] = {}
    for labels in labels_per_item:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    if not label_counts:
        indices = list(range(total))
        rng.shuffle(indices)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        train_items = [items[i] for i in train_indices]
        val_items = [items[i] for i in val_indices]
        return train_items, val_items

    desired = {label: count * val_ratio for label, count in label_counts.items()}
    remaining = dict(desired)

    val_indices = set()
    available = set(range(total))

    def score(idx: int) -> float:
        return sum(remaining.get(label, 0.0) for label in labels_per_item[idx])

    def choose(idx: int) -> None:
        val_indices.add(idx)
        available.discard(idx)
        for label in labels_per_item[idx]:
            remaining[label] = max(0.0, remaining[label] - 1.0)

    if val_count >= len(label_counts):
        for label, _ in sorted(label_counts.items(), key=lambda item: item[1]):
            if len(val_indices) >= val_count:
                break

            candidates = [idx for idx in available if label in labels_per_item[idx]]
            if not candidates:
                continue

            best_score = max(score(idx) for idx in candidates)
            best_candidates = [idx for idx in candidates if score(idx) == best_score]
            choose(rng.choice(best_candidates))

    while len(val_indices) < val_count and available:
        scored = [(score(idx), idx) for idx in available]
        max_score = max(scored, key=lambda item: item[0])[0]
        if max_score <= 0:
            remaining_indices = list(available)
            rng.shuffle(remaining_indices)
            for idx in remaining_indices[: val_count - len(val_indices)]:
                choose(idx)
            break

        best_candidates = [idx for sc, idx in scored if sc == max_score]
        choose(rng.choice(best_candidates))

    val_indices_list = sorted(val_indices)
    train_items = [items[idx] for idx in range(total) if idx not in val_indices]
    val_items = [items[idx] for idx in val_indices_list]
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
    split_mode: str,
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

    train_pairs, val_pairs = _split_items(paired, val_ratio, seed, split_mode)
    print("Split completed")

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
    for image_path, label_path, rel, _ in train_pairs:
        train_tasks.append((image_path, train_images_dir / rel))
        train_tasks.append((label_path, train_labels_dir / rel.with_suffix(".txt")))

    val_tasks = []
    for image_path, label_path, rel, _ in val_pairs:
        val_tasks.append((image_path, val_images_dir / rel))
        val_tasks.append((label_path, val_labels_dir / rel.with_suffix(".txt")))

    _make_symlinks(train_tasks, workers, "Symlinking (train)")
    _make_symlinks(val_tasks, workers, "Symlinking (val)")

    print(f"Train: {len(train_pairs)} files")
    print(f"Val: {len(val_pairs)} files")
    if missing_labels:
        print(f"Skipped (missing labels): {missing_labels}")


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
        split_mode=args.split_mode,
        seed=seed,
        workers=args.workers,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
