import argparse
import importlib.util
import os
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"

ARTIFACTS = {
    "yolo_best": REPO_ROOT / "weights" / "yolo_vintext" / "best.pt",
    "vietocr_config": REPO_ROOT / "weights" / "vietocr_vintext" / "config.yml",
    "vietocr_weights": REPO_ROOT / "weights" / "vietocr_vintext" / "transformerocr.pth",
}

REQUIRED_PACKAGES = [
    "torch",
    "ultralytics",
    "vietocr",
    "yaml",
    "numpy",
    "openpyxl",
    "ray",
    "fastapi",
    "streamlit",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify local OCR training artifacts and deployment wiring."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image path for a direct OCRService smoke test.",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Only verify files/imports and skip loading YOLO/VietOCR.",
    )
    return parser.parse_args()


def print_check(ok: bool, message: str):
    prefix = "[OK]" if ok else "[FAIL]"
    print(f"{prefix} {message}")


def require(condition: bool, message: str, failures: list[str]):
    print_check(condition, message)
    if not condition:
        failures.append(message)


def check_artifacts(failures):
    for name, path in ARTIFACTS.items():
        exists = path.exists()
        size_mb = path.stat().st_size / 1024 / 1024 if exists else 0
        require(exists and size_mb > 0, f"{name}: {path} ({size_mb:.2f} MB)", failures)


def check_packages(failures):
    for package in REQUIRED_PACKAGES:
        found = importlib.util.find_spec(package) is not None
        require(found, f"package importable: {package}", failures)

    try:
        import numpy as np

        major = int(np.__version__.split(".", 1)[0])
        require(major < 2, f"numpy version compatible with VietOCR/imgaug: {np.__version__}", failures)
    except Exception as exc:
        require(False, f"numpy version check failed: {exc}", failures)


def add_deployment_to_path():
    deployment_path = str(DEPLOYMENT_DIR)
    if deployment_path not in sys.path:
        sys.path.insert(0, deployment_path)


def check_parser_and_excel(failures):
    add_deployment_to_path()
    try:
        from parser import parse_order_fields

        predictions = [
            ([10, 10, 200, 30], "text", 0.9, "Ma van don: GHNABC123456789"),
            ([10, 40, 200, 60], "text", 0.9, "Nguoi nhan: Nguyen Van A"),
            ([10, 70, 200, 90], "text", 0.9, "SDT: 0912 345 678"),
            ([10, 100, 350, 120], "text", 0.9, "Dia chi: 123 Duong Le Loi, Quan 1, TP HCM"),
            ([10, 130, 200, 150], "text", 0.9, "COD: 250.000d"),
        ]
        parsed = parse_order_fields(predictions)
        require(parsed["tracking_id"] == "GHNABC123456789", "parser extracts tracking_id", failures)
        require(parsed["phone"] == "0912345678", "parser extracts phone", failures)
        require(parsed["cod_amount"] == "250000", "parser extracts cod_amount", failures)
    except Exception as exc:
        require(False, f"parser smoke test failed: {exc}", failures)

    try:
        from tempfile import TemporaryDirectory

        from excel_export import append_order_record, read_order_records

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "orders.xlsx"
            append_order_record({"tracking_id": "GHNABC123456789"}, path)
            rows = read_order_records(path)
            require(len(rows) == 1 and rows[0]["tracking_id"] == "GHNABC123456789", "Excel append/read smoke test", failures)
    except Exception as exc:
        require(False, f"Excel smoke test failed: {exc}", failures)


def check_model_load(failures):
    add_deployment_to_path()
    os.environ["OCR_ENGINE"] = "vietocr"
    try:
        import ocr

        require(ocr.OCR_ENGINE == "vietocr", "deployment selects VietOCR engine", failures)
        require(str(ocr.TEXT_DET_MODEL_PATH).endswith(r"weights\yolo_vintext\best.pt") or str(ocr.TEXT_DET_MODEL_PATH).endswith("weights/yolo_vintext/best.pt"), "deployment selects YOLO VinText model", failures)
    except Exception as exc:
        require(False, f"deployment import/model load failed: {exc}", failures)


def check_image_smoke(image_path: Path, failures):
    if not image_path:
        return

    add_deployment_to_path()
    if not image_path.exists():
        require(False, f"smoke image not found: {image_path}", failures)
        return

    try:
        import ocr

        Image.open(image_path).verify()
        service = ocr.OCRService.options(name="verify_ocr_service_shadow")
        require(service is not None, "Ray Serve deployment object available", failures)
        print_check(True, f"smoke image is readable: {image_path}")
    except Exception as exc:
        require(False, f"smoke image check failed: {exc}", failures)


def main():
    args = parse_args()
    failures = []

    print("Verifying Smart Warehouse OCR artifacts...")
    print(f"Repo root: {REPO_ROOT}")
    check_artifacts(failures)
    check_packages(failures)
    check_parser_and_excel(failures)
    if not args.skip_model_load:
        check_model_load(failures)
    check_image_smoke(args.image, failures)

    if failures:
        print("")
        print("Verification failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("")
    print("Verification passed.")


if __name__ == "__main__":
    main()
