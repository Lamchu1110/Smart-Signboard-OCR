import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_TEXTS = {"###", "***", "???", ""}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VinText-style polygon annotations to YOLO text detection format."
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
        default=REPO_ROOT / "datasets" / "vintext_yolo",
        help="Output YOLO dataset directory.",
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
        "--include-ignore",
        action="store_true",
        help="Include ignored/illegible transcripts such as ### as text boxes.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy images into the YOLO dataset folder.",
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


def parse_vintext_label(label_path: Path, include_ignore: bool) -> list[dict]:
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

        text = parts[8].strip()
        if not include_ignore and text in IGNORE_TEXTS:
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


def polygon_to_yolo_bbox(polygon: list[list[float]], image_width: int, image_height: int):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(image_width), max(xs))
    y2 = min(float(image_height), max(ys))

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 1 or box_height <= 1:
        return None

    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    values = [x_center, y_center, norm_width, norm_height]
    if any(value < 0 or value > 1 for value in values):
        return None
    return values


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


def write_data_yaml(output_dir: Path):
    data_yaml = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: text",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def prepare_yolo_dataset(args):
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

        instances = parse_vintext_label(label_path, include_ignore=args.include_ignore)
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
        raise RuntimeError("No usable VinText records found.")

    assign_missing_splits(records, args.train_ratio, args.val_ratio, args.seed)
    ensure_validation_split(records, args.val_ratio, args.seed)

    split_counts = {"train": 0, "val": 0, "test": 0}
    instance_counts = {"train": 0, "val": 0, "test": 0}

    for split in split_counts:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        image_path = record["image_path"]
        split = record["split"]
        target_image_path = output_dir / "images" / split / image_path.name
        target_label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        yolo_lines = []
        for instance in record["instances"]:
            bbox = polygon_to_yolo_bbox(
                instance["polygon"],
                image_width=image_width,
                image_height=image_height,
            )
            if bbox is None:
                continue
            yolo_lines.append("0 " + " ".join(f"{value:.6f}" for value in bbox))

        if not yolo_lines:
            continue

        if args.copy_images:
            shutil.copy2(image_path, target_image_path)
        target_label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        split_counts[split] += 1
        instance_counts[split] += len(yolo_lines)

    write_data_yaml(output_dir)

    summary_lines = [
        "VinText YOLO dataset prepared.",
        f"Dataset root: {dataset_root}",
        f"Output dir: {output_dir}",
        f"Images by split: {split_counts}",
        f"Text boxes by split: {instance_counts}",
        f"Skipped missing images: {skipped_missing_images}",
        f"Skipped empty labels: {skipped_empty_labels}",
        f"Data YAML: {output_dir / 'data.yaml'}",
    ]
    print("\n".join(summary_lines))


def main():
    args = parse_args()
    prepare_yolo_dataset(args)


if __name__ == "__main__":
    main()
