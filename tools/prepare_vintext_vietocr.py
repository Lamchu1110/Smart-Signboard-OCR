import argparse
import random
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_TEXTS = {"###", "***", "???", ""}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create rectified VinText crops and label files for VietOCR fine-tuning."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="VinText root. Expected to contain labels/ and image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "vintext_recognition",
        help="Output VietOCR recognition dataset directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio when split cannot be inferred from folder names.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio when split cannot be inferred from folder names.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=48,
        help="Target rectified crop height.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=8,
        help="Minimum output crop width.",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=1,
        help="Skip labels shorter than this value.",
    )
    parser.add_argument(
        "--include-ignore",
        action="store_true",
        help="Include ignored/illegible transcripts such as ###.",
    )
    return parser.parse_args()


def find_label_dir(dataset_root: Path) -> Path:
    for name in ("labels", "annotations", "annotation"):
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find label directory under {dataset_root}. "
        "Expected labels/, annotations/, or annotation/."
    )


def collect_image_paths(dataset_root: Path) -> dict[str, Path]:
    image_paths = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths[path.stem.lower()] = path
    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_root}.")
    return image_paths


def image_stem_from_label(label_path: Path) -> str:
    stem = label_path.stem
    if stem.startswith("gt_"):
        number = stem.removeprefix("gt_")
        if number.isdigit():
            return f"im{int(number):04d}"
    return stem


def infer_split(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if {"train", "training", "train_images"} & parts:
        return "train"
    if {"val", "valid", "validation"} & parts:
        return "val"
    if {"test", "testing", "test_image", "test_images"} & parts:
        return "test"
    if {"unseen", "unseen_test", "unseen_test_images"} & parts:
        return "test"
    return None


def parse_vintext_label(label_path: Path, include_ignore: bool, min_text_length: int):
    instances = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(",", 8)
        if len(parts) < 9:
            continue

        try:
            coords = [float(value) for value in parts[:8]]
        except ValueError:
            continue

        text = " ".join(parts[8].strip().split())
        if not include_ignore and text in IGNORE_TEXTS:
            continue
        if len(text) < min_text_length:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        instances.append(
            {
                "polygon": [[xs[index], ys[index]] for index in range(4)],
                "text": text,
            }
        )

    return instances


def order_quad(points: list[list[float]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def rectify_crop(image, polygon, target_height: int, min_width: int):
    quad = order_quad(polygon)

    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])

    source_width = max(int(round(max(width_top, width_bottom))), 1)
    source_height = max(int(round(max(height_left, height_right))), 1)
    scale = target_height / max(source_height, 1)
    target_width = max(int(round(source_width * scale)), min_width)

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))


def assign_missing_splits(records: list[dict], train_ratio: float, val_ratio: float, seed: int):
    missing = [record for record in records if record["split"] is None]
    if not missing:
        return

    random.Random(seed).shuffle(missing)
    train_count = int(len(missing) * train_ratio)
    val_count = int(len(missing) * val_ratio)

    for index, record in enumerate(missing):
        if index < train_count:
            record["split"] = "train"
        elif index < train_count + val_count:
            record["split"] = "val"
        else:
            record["split"] = "test"


def ensure_validation_split(records: list[dict], val_ratio: float, seed: int):
    if any(record["split"] == "val" for record in records):
        return

    train_records = [record for record in records if record["split"] == "train"]
    if len(train_records) < 2:
        return

    val_count = max(1, int(len(train_records) * val_ratio))
    val_count = min(val_count, len(train_records) - 1)
    random.Random(seed).shuffle(train_records)

    for record in train_records[:val_count]:
        record["split"] = "val"


def prepare_vietocr_dataset(args):
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    label_dir = find_label_dir(dataset_root)
    image_paths = collect_image_paths(dataset_root)

    label_paths = sorted(label_dir.rglob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No .txt annotation files found under {label_dir}.")

    records = []
    skipped_missing_images = 0
    skipped_empty_labels = 0

    for label_path in label_paths:
        image_stem = image_stem_from_label(label_path).lower()
        image_path = image_paths.get(image_stem)
        if image_path is None:
            skipped_missing_images += 1
            continue

        instances = parse_vintext_label(
            label_path,
            include_ignore=args.include_ignore,
            min_text_length=args.min_text_length,
        )
        if not instances:
            skipped_empty_labels += 1
            continue

        records.append(
            {
                "label_path": label_path,
                "image_path": image_path,
                "instances": instances,
                "split": infer_split(image_path) or infer_split(label_path),
            }
        )

    if not records:
        raise RuntimeError("No usable VinText recognition records found.")

    assign_missing_splits(records, args.train_ratio, args.val_ratio, args.seed)
    ensure_validation_split(records, args.val_ratio, args.seed)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_rows = {"train": [], "val": [], "test": []}
    crop_counts = {"train": 0, "val": 0, "test": 0}
    failed_crops = 0

    for split in label_rows:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            failed_crops += len(record["instances"])
            continue

        split = record["split"]
        for instance_index, instance in enumerate(record["instances"]):
            try:
                crop = rectify_crop(
                    image,
                    instance["polygon"],
                    target_height=args.height,
                    min_width=args.min_width,
                )
            except cv2.error:
                failed_crops += 1
                continue

            image_name = f"{record['image_path'].stem}_{instance_index:04d}.jpg"
            crop_path = output_dir / "images" / split / image_name
            if not cv2.imwrite(str(crop_path), crop):
                failed_crops += 1
                continue

            relative_crop_path = crop_path.relative_to(output_dir).as_posix()
            label_rows[split].append(f"{relative_crop_path}\t{instance['text']}")
            crop_counts[split] += 1

    for split, rows in label_rows.items():
        (labels_dir / f"{split}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    summary_lines = [
        "VinText VietOCR dataset prepared.",
        f"Dataset root: {dataset_root}",
        f"Output dir: {output_dir}",
        f"Crops by split: {crop_counts}",
        f"Skipped missing images: {skipped_missing_images}",
        f"Skipped empty labels: {skipped_empty_labels}",
        f"Failed crops: {failed_crops}",
        f"Label files: {labels_dir}",
    ]
    print("\n".join(summary_lines))


def main():
    args = parse_args()
    prepare_vietocr_dataset(args)


if __name__ == "__main__":
    main()
