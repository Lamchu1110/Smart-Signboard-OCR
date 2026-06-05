import argparse
import csv
import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
DEFAULT_ZIP = Path(r"C:\Users\legion\Downloads\vietnamese_original.zip")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "vintext_smoke"


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test YOLO + VietOCR on VinText.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to vietnamese_original.zip.")
    parser.add_argument("--samples", type=int, default=40, help="Number of GT text crops to test.")
    parser.add_argument("--images", type=int, default=3, help="Number of full images to run through YOLO + VietOCR.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def add_deployment_to_path():
    deployment_path = str(DEPLOYMENT_DIR)
    if deployment_path not in sys.path:
        sys.path.insert(0, deployment_path)


def fix_mojibake(text):
    value = str(text or "").strip()
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def normalize_text(text):
    value = fix_mojibake(text)
    value = " ".join(value.strip().split())
    return value.upper()


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def cer(pred, gt):
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not gt_norm:
        return 0.0 if not pred_norm else 1.0
    return levenshtein(pred_norm, gt_norm) / len(gt_norm)


def parse_label_line(line):
    parts = line.rstrip("\n").split(",", 8)
    if len(parts) != 9:
        return None
    try:
        coords = [float(value) for value in parts[:8]]
    except ValueError:
        return None
    text = fix_mojibake(parts[8])
    if not text or text == "###":
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    return {
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "text": text,
    }


def label_to_image_name(label_name):
    number = Path(label_name).stem.replace("gt_", "")
    return f"vietnamese/train_images/im{int(number):04d}.jpg"


def read_label_entries(zip_file, label_name):
    try:
        raw = zip_file.read(label_name).decode("utf-8-sig", errors="replace")
    except KeyError:
        return []
    entries = []
    for line in raw.splitlines():
        item = parse_label_line(line)
        if item:
            entries.append(item)
    return entries


def crop_bbox(image, bbox, pad_ratio=0.08):
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    pad_x = max(2, width * pad_ratio)
    pad_y = max(2, height * pad_ratio)
    return image.crop(
        (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(image.width, int(x2 + pad_x)),
            min(image.height, int(y2 + pad_y)),
        )
    )


def load_vietocr():
    add_deployment_to_path()
    import ocr

    return ocr.OCRCore(
        reg_model=ocr.reg_model,
        det_model=ocr.det_model,
        ocr_engine="vietocr",
        vietocr_config_path=str(ocr.VIETOCR_CONFIG_PATH),
        vietocr_weights_path=str(ocr.VIETOCR_WEIGHTS_PATH),
    )


def choose_label_files(zip_file, samples, seed):
    label_files = [
        entry.filename
        for entry in zip_file.infolist()
        if entry.filename.startswith("vietnamese/labels/gt_") and entry.filename.endswith(".txt")
    ]
    rng = random.Random(seed)
    rng.shuffle(label_files)

    chosen = []
    crop_count = 0
    for label_name in label_files:
        image_name = label_to_image_name(label_name)
        if image_name not in zip_file.namelist():
            continue
        entries = read_label_entries(zip_file, label_name)
        if not entries:
            continue
        chosen.append((label_name, image_name, entries))
        crop_count += len(entries)
        if crop_count >= samples:
            break
    return chosen


def run_gt_crop_recognition(zip_file, service, chosen, samples, output_dir):
    rows = []
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    for label_name, image_name, entries in chosen:
        with zip_file.open(image_name) as handle:
            image = Image.open(handle).convert("RGB")
            for index, entry in enumerate(entries):
                if len(rows) >= samples:
                    return rows
                crop = crop_bbox(image, entry["bbox"])
                pred = service.text_recognition(crop)
                row = {
                    "image_name": image_name,
                    "label_name": label_name,
                    "bbox": entry["bbox"],
                    "gt": entry["text"],
                    "pred": pred,
                    "cer": round(cer(pred, entry["text"]), 4),
                    "exact": normalize_text(pred) == normalize_text(entry["text"]),
                }
                if len(rows) < 12:
                    crop_path = crop_dir / f"crop_{len(rows):03d}.jpg"
                    crop.save(crop_path)
                    row["crop_path"] = str(crop_path)
                rows.append(row)
    return rows


def run_full_pipeline(zip_file, service, chosen, image_count, output_dir):
    add_deployment_to_path()
    from signboard_parser import parse_signboard_fields

    rows = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for _, image_name, _ in chosen[:image_count]:
        image_path = image_dir / Path(image_name).name
        image_path.write_bytes(zip_file.read(image_name))
        predictions = service.process_image(str(image_path))
        parsed = parse_signboard_fields(predictions)
        rows.append(
            {
                "image_name": image_name,
                "prediction_count": len(predictions),
                "raw_text": parsed.get("raw_text", ""),
                "predictions": [
                    {
                        "bbox": [float(value) for value in bbox],
                        "confidence": float(confidence),
                        "text": text,
                    }
                    for bbox, _class_name, confidence, text in predictions
                ],
            }
        )
    return rows


def write_outputs(output_dir, crop_rows, full_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vintext_gt_crop_recognition.csv"
    json_path = output_dir / "vintext_smoke_report.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_name", "label_name", "bbox", "gt", "pred", "cer", "exact", "crop_path"],
        )
        writer.writeheader()
        for row in crop_rows:
            writer.writerow(row)

    exact_count = sum(1 for row in crop_rows if row["exact"])
    avg_cer = sum(row["cer"] for row in crop_rows) / len(crop_rows) if crop_rows else 0
    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gt_crop_recognition": {
            "samples": len(crop_rows),
            "exact_count": exact_count,
            "exact_rate": round(exact_count / len(crop_rows), 4) if crop_rows else 0,
            "avg_cer": round(avg_cer, 4),
            "worst_examples": sorted(crop_rows, key=lambda row: row["cer"], reverse=True)[:10],
        },
        "full_pipeline": full_rows,
        "files": {
            "csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, csv_path, json_path


def main():
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(f"VinText zip not found: {args.zip}")

    service = load_vietocr()
    with ZipFile(args.zip) as zip_file:
        chosen = choose_label_files(zip_file, args.samples, args.seed)
        crop_rows = run_gt_crop_recognition(zip_file, service, chosen, args.samples, args.output_dir)
        full_rows = run_full_pipeline(zip_file, service, chosen, args.images, args.output_dir)

    report, csv_path, json_path = write_outputs(args.output_dir, crop_rows, full_rows)
    metrics = report["gt_crop_recognition"]
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {json_path}")
    print(f"GT crop samples: {metrics['samples']}")
    print(f"Exact rate: {metrics['exact_rate']}")
    print(f"Average CER: {metrics['avg_cer']}")
    print("Worst examples:")
    for row in metrics["worst_examples"][:5]:
        print(f"- GT={row['gt']!r} | PRED={row['pred']!r} | CER={row['cer']}")


if __name__ == "__main__":
    main()
