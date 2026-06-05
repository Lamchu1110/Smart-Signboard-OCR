import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "report_metrics"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_TEXTS = {"###", "***", "???", ""}
CRNN_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CRNN_CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CRNN_CHARS))}
CRNN_IDX_TO_CHAR = {idx: char for char, idx in CRNN_CHAR_TO_IDX.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Smart Signboard OCR comparison metrics for the final report."
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "vietnamese")
    parser.add_argument("--yolo-data", type=Path, default=REPO_ROOT / "datasets" / "vintext_yolo" / "data.yaml")
    parser.add_argument("--yolo-output-dir", type=Path, default=REPO_ROOT / "datasets" / "vintext_yolo")
    parser.add_argument("--old-yolo", type=Path, default=REPO_ROOT / "weights" / "best.pt")
    parser.add_argument("--new-yolo", type=Path, default=REPO_ROOT / "weights" / "yolo_vintext" / "best.pt")
    parser.add_argument("--crnn-weights", type=Path, default=REPO_ROOT / "weights" / "ocr_crnn.pt")
    parser.add_argument("--vietocr-config", type=Path, default=REPO_ROOT / "weights" / "vietocr_vintext" / "config.yml")
    parser.add_argument(
        "--vietocr-weights",
        type=Path,
        default=REPO_ROOT / "weights" / "vietocr_vintext" / "transformerocr.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--det-split", choices=["val", "test"], default="test")
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--rec-samples", type=int, default=200)
    parser.add_argument("--rec-seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--skip-recognition", action="store_true")
    parser.add_argument("--no-prepare-yolo", action="store_true")
    parser.add_argument(
        "--single-cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat all YOLO detections as one text class during validation.",
    )
    return parser.parse_args()


def add_deployment_to_path():
    deployment_path = str(DEPLOYMENT_DIR)
    if deployment_path not in sys.path:
        sys.path.insert(0, deployment_path)


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def file_size_mb(path):
    path = Path(path)
    return round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else 0.0


def parameter_count(model):
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def normalize_text(text):
    return " ".join(str(text or "").strip().split()).upper()


def levenshtein_sequence(a, b):
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


def cer(prediction, target):
    pred = normalize_text(prediction)
    gt = normalize_text(target)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein_sequence(pred, gt) / len(gt)


def wer(prediction, target):
    pred_words = normalize_text(prediction).split()
    gt_words = normalize_text(target).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein_sequence(pred_words, gt_words) / len(gt_words)


def sample_accuracy(prediction, target):
    return 1.0 if normalize_text(prediction) == normalize_text(target) else 0.0


def find_label_dir(dataset_root):
    for name in ("labels", "annotations", "annotation"):
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find labels/ under {dataset_root}")


def image_stem_from_label(label_path):
    stem = label_path.stem
    if stem.startswith("gt_"):
        number = stem.removeprefix("gt_")
        if number.isdigit():
            return f"im{int(number):04d}"
    return stem


def collect_image_paths(dataset_root):
    image_paths = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths[path.stem.lower()] = path
    return image_paths


def parse_label_file(label_path):
    entries = []
    for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.strip().split(",", 8)
        if len(parts) != 9:
            continue
        try:
            coords = [float(value) for value in parts[:8]]
        except ValueError:
            continue
        text = parts[8].strip()
        if text in IGNORE_TEXTS:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        entries.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text})
    return entries


def collect_recognition_samples(dataset_root, sample_count, seed):
    label_dir = find_label_dir(dataset_root)
    image_paths = collect_image_paths(dataset_root)
    samples = []

    for label_path in sorted(label_dir.glob("*.txt")):
        stem = image_stem_from_label(label_path).lower()
        image_path = image_paths.get(stem)
        if image_path is None:
            continue
        for entry in parse_label_file(label_path):
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "bbox": entry["bbox"],
                    "gt": entry["text"],
                }
            )

    if not samples:
        raise RuntimeError(f"No recognition samples found under {dataset_root}")
    random.Random(seed).shuffle(samples)
    return samples[:sample_count]


def crop_bbox(image, bbox, pad_ratio=0.08):
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = max(2.0, width * pad_ratio)
    pad_y = max(2.0, height * pad_ratio)
    return image.crop(
        (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(image.width, int(x2 + pad_x)),
            min(image.height, int(y2 + pad_y)),
        )
    )


class CRNNRecognizer:
    def __init__(self, weights_path, device):
        add_deployment_to_path()
        from crnn import CRNN

        self.device = device
        self.model = CRNN(
            vocab_size=len(CRNN_CHARS),
            hidden_size=256,
            n_layers=3,
            dropout=0.2,
            unfreeze_layers=3,
        )
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def decode(self, encoded_sequences, blank_char="-"):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None
            for token in seq:
                if token != 0:
                    char = CRNN_IDX_TO_CHAR[token.item()]
                    if char != blank_char and (char != prev_char or prev_char == blank_char):
                        decoded_label.append(char)
                    prev_char = char
            decoded_sequences.append("".join(decoded_label))
        return decoded_sequences

    def recognize(self, image):
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(transformed_image).cpu()
        return self.decode(logits.permute(1, 0, 2).argmax(2))[0]


def load_vietocr_recognizer(config_path, weights_path):
    add_deployment_to_path()
    from vietocr_recognizer import VietOCRRecognizer

    return VietOCRRecognizer.from_paths(config_path=Path(config_path), weights_path=Path(weights_path))


def evaluate_recognizer(model_name, recognizer, samples, output_csv):
    rows = []
    image_cache = {}
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        crop = crop_bbox(image_cache[image_path], sample["bbox"])

        item_started = time.perf_counter()
        prediction = recognizer.recognize(crop)
        elapsed_ms = (time.perf_counter() - item_started) * 1000

        rows.append(
            {
                "model": model_name,
                "index": index,
                "image_path": str(sample["image_path"]),
                "label_path": str(sample["label_path"]),
                "bbox": sample["bbox"],
                "gt": sample["gt"],
                "prediction": prediction,
                "cer": round(cer(prediction, sample["gt"]), 6),
                "wer": round(wer(prediction, sample["gt"]), 6),
                "exact": int(sample_accuracy(prediction, sample["gt"])),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    total_elapsed = time.perf_counter() - started
    for image in image_cache.values():
        image.close()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sample_count = len(rows)
    avg_ms = sum(row["elapsed_ms"] for row in rows) / sample_count if rows else 0.0
    return {
        "model": model_name,
        "samples": sample_count,
        "cer": round(sum(row["cer"] for row in rows) / sample_count, 6) if rows else 0.0,
        "wer": round(sum(row["wer"] for row in rows) / sample_count, 6) if rows else 0.0,
        "accuracy": round(sum(row["exact"] for row in rows) / sample_count, 6) if rows else 0.0,
        "avg_inference_ms": round(avg_ms, 3),
        "fps": round(1000 / avg_ms, 3) if avg_ms else 0.0,
        "total_seconds": round(total_elapsed, 3),
        "csv": str(output_csv),
    }


def ensure_yolo_dataset(args):
    if args.yolo_data.exists() or args.no_prepare_yolo:
        return
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from prepare_vintext_yolo import prepare_yolo_dataset

    prepare_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.yolo_output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        include_ignore=False,
        copy_images=True,
    )
    prepare_yolo_dataset(prepare_args)


def metric_value(metrics, path, default=None):
    value = metrics
    for part in path:
        value = getattr(value, part, None)
        if value is None:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_yolo_model(model_name, model_path, data_yaml, split, imgsz, device, single_cls):
    from ultralytics import YOLO

    model_path = Path(model_path)
    started = time.perf_counter()
    model = YOLO(str(model_path))
    params = parameter_count(model.model)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        single_cls=single_cls,
        plots=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    speed = getattr(metrics, "speed", {}) or {}
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms

    return {
        "model": model_name,
        "model_path": str(model_path),
        "dataset": str(data_yaml),
        "split": split,
        "precision": round(metric_value(metrics, ("box", "mp"), 0.0), 6),
        "recall": round(metric_value(metrics, ("box", "mr"), 0.0), 6),
        "map50": round(metric_value(metrics, ("box", "map50"), 0.0), 6),
        "map50_95": round(metric_value(metrics, ("box", "map"), 0.0), 6),
        "preprocess_ms": round(preprocess_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "total_ms_per_image": round(total_ms, 3),
        "fps": round(1000 / total_ms, 3) if total_ms else 0.0,
        "params": params,
        "model_size_mb": file_size_mb(model_path),
        "elapsed_seconds": round(elapsed, 3),
    }


def run_detection_metrics(args, device):
    ensure_yolo_dataset(args)
    if not args.yolo_data.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {args.yolo_data}")

    results = []
    for model_name, model_path in (
        ("old_yolo_icdar2003", args.old_yolo),
        ("new_yolo_vintext", args.new_yolo),
    ):
        if not model_path.exists():
            results.append({"model": model_name, "error": f"Model not found: {model_path}"})
            continue
        try:
            results.append(
                evaluate_yolo_model(
                    model_name=model_name,
                    model_path=model_path,
                    data_yaml=args.yolo_data,
                    split=args.det_split,
                    imgsz=args.det_imgsz,
                    device=device,
                    single_cls=args.single_cls,
                )
            )
        except Exception as exc:
            results.append({"model": model_name, "error": str(exc)})
    return results


def run_recognition_metrics(args, device):
    samples = collect_recognition_samples(
        dataset_root=args.dataset_root,
        sample_count=args.rec_samples,
        seed=args.rec_seed,
    )
    results = []

    if args.crnn_weights.exists():
        try:
            crnn = CRNNRecognizer(args.crnn_weights, device=device)
            crnn_result = evaluate_recognizer(
                "old_crnn",
                crnn,
                samples,
                args.output_dir / "recognition_old_crnn_predictions.csv",
            )
            crnn_result["model_path"] = str(args.crnn_weights)
            crnn_result["params"] = parameter_count(crnn.model)
            crnn_result["model_size_mb"] = file_size_mb(args.crnn_weights)
            results.append(crnn_result)
        except Exception as exc:
            results.append({"model": "old_crnn", "error": str(exc)})
    else:
        results.append({"model": "old_crnn", "error": f"Model not found: {args.crnn_weights}"})

    if args.vietocr_config.exists() and args.vietocr_weights.exists():
        try:
            vietocr = load_vietocr_recognizer(args.vietocr_config, args.vietocr_weights)
            vietocr_result = evaluate_recognizer(
                "new_vietocr_vintext",
                vietocr,
                samples,
                args.output_dir / "recognition_new_vietocr_predictions.csv",
            )
            vietocr_result["model_path"] = str(args.vietocr_weights)
            vietocr_result["config_path"] = str(args.vietocr_config)
            vietocr_result["params"] = parameter_count(vietocr.predictor.model)
            vietocr_result["model_size_mb"] = file_size_mb(args.vietocr_weights)
            results.append(vietocr_result)
        except Exception as exc:
            results.append({"model": "new_vietocr_vintext", "error": str(exc)})
    else:
        results.append(
            {
                "model": "new_vietocr_vintext",
                "error": f"Model/config not found: {args.vietocr_config}, {args.vietocr_weights}",
            }
        )
    return results


def write_csv(path, rows):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    def fmt(value):
        if value is None:
            return ""
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def write_markdown_summary(report, output_path):
    detection_rows = report.get("detection", [])
    recognition_rows = report.get("recognition", [])
    lines = [
        "# Smart Signboard OCR Metrics Comparison",
        "",
        f"Created at: {report['created_at']}",
        f"Device: {report['device']}",
        "",
        "## Detection Models",
        "",
        markdown_table(
            [
                "model",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "fps",
                "total_ms_per_image",
                "params",
                "model_size_mb",
                "error",
            ],
            detection_rows,
        ),
        "",
        "## Recognition Models",
        "",
        markdown_table(
            [
                "model",
                "samples",
                "cer",
                "wer",
                "accuracy",
                "fps",
                "avg_inference_ms",
                "params",
                "model_size_mb",
                "error",
            ],
            recognition_rows,
        ),
        "",
        "## Notes",
        "",
        "- Detection is evaluated on the same VinText YOLO dataset split.",
        "- Recognition is evaluated on the same VinText ground-truth text crops.",
        "- CER/WER lower is better; Accuracy, Precision, Recall, mAP and FPS higher are better.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    report = {
        "project": "Smart Signboard OCR",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "dataset_root": str(args.dataset_root),
        "detection": [],
        "recognition": [],
        "outputs": {},
    }

    if not args.skip_detection:
        report["detection"] = run_detection_metrics(args, device=device)
        detection_csv = args.output_dir / "detection_metrics_comparison.csv"
        write_csv(detection_csv, report["detection"])
        report["outputs"]["detection_csv"] = str(detection_csv)

    if not args.skip_recognition:
        report["recognition"] = run_recognition_metrics(args, device=device)
        recognition_csv = args.output_dir / "recognition_metrics_comparison.csv"
        write_csv(recognition_csv, report["recognition"])
        report["outputs"]["recognition_csv"] = str(recognition_csv)

    json_path = args.output_dir / "metrics_comparison_report.json"
    md_path = args.output_dir / "metrics_comparison_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(report, md_path)
    report["outputs"]["json"] = str(json_path)
    report["outputs"]["markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Metrics JSON: {json_path}")
    print(f"Metrics Markdown: {md_path}")
    if report["detection"]:
        print(f"Detection CSV: {report['outputs'].get('detection_csv')}")
    if report["recognition"]:
        print(f"Recognition CSV: {report['outputs'].get('recognition_csv')}")


if __name__ == "__main__":
    main()
