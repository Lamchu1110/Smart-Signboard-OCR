import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
DEFAULT_REPORT_PATH = REPO_ROOT / "outputs" / "signboard_e2e_report.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Smart Signboard OCR E2E report.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional local image path. If provided, run the direct model OCR pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--save-excel",
        action="store_true",
        help="Append the direct OCR result to outputs/signboards.xlsx when --image is provided.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip the lightweight demo pipeline check.",
    )
    return parser.parse_args()


def add_deployment_to_path():
    deployment_path = str(DEPLOYMENT_DIR)
    if deployment_path not in sys.path:
        sys.path.insert(0, deployment_path)


def check_file(path: Path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else 0,
    }


def run_static_checks():
    artifacts = {
        "streamlit_app": DEPLOYMENT_DIR / "app.py",
        "signboard_parser": DEPLOYMENT_DIR / "signboard_parser.py",
        "signboard_excel_export": DEPLOYMENT_DIR / "signboard_excel_export.py",
        "yolo": REPO_ROOT / "weights" / "yolo_vintext" / "best.pt",
        "vietocr_config": REPO_ROOT / "weights" / "vietocr_vintext" / "config.yml",
        "vietocr_weights": REPO_ROOT / "weights" / "vietocr_vintext" / "transformerocr.pth",
    }
    checks = {name: check_file(path) for name, path in artifacts.items()}
    checks["all_required_code_exists"] = all(
        checks[name]["exists"]
        for name in ("streamlit_app", "signboard_parser", "signboard_excel_export")
    )
    checks["model_artifacts_ready"] = all(
        checks[name]["exists"] for name in ("yolo", "vietocr_config", "vietocr_weights")
    )
    return checks


def run_component_checks():
    add_deployment_to_path()
    from signboard_excel_export import append_signboard_record, read_signboard_records
    from signboard_parser import parse_signboard_fields

    sample_predictions = [
        ([20, 20, 360, 64], "text", 0.92, "CA PHE THAO UYEN"),
        ([20, 78, 300, 108], "text", 0.88, "Cafe - Nuoc giai khat"),
        ([20, 122, 240, 152], "text", 0.94, "0909 123 456"),
        ([20, 166, 420, 196], "text", 0.86, "123 Nguyen Hue Quan 1"),
    ]
    parsed = parse_signboard_fields(sample_predictions)
    record = {
        "image_name": "component_sample.jpg",
        "store_name": parsed.get("business_name", ""),
        "category": parsed.get("category_hint", ""),
        "phone": parsed.get("phone", ""),
        "address": parsed.get("address_hint", ""),
        "services": "Cafe, nuoc giai khat",
        "raw_text": parsed.get("raw_text", ""),
        "confidence": 0.9,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "component check",
    }

    with TemporaryDirectory() as tmp:
        excel_path = Path(tmp) / "signboards.xlsx"
        saved = append_signboard_record(record, excel_path)
        rows = read_signboard_records(excel_path)

    return {
        "parser": {
            "ok": bool(parsed.get("business_name") and parsed.get("phone")),
            "parsed": parsed,
        },
        "excel": {
            "ok": len(rows) == 1 and rows[0]["store_name"] == record["store_name"],
            "saved_row": saved["row"],
            "rows": len(rows),
        },
    }


def run_demo_pipeline():
    add_deployment_to_path()
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
        logging.ERROR
    )
    with contextlib.redirect_stderr(io.StringIO()):
        import app

    image = Image.new("RGB", (900, 500), "white")
    started = time.perf_counter()
    result = app.demo_ocr_result(image, "demo_signboard.jpg", show_boxes=True)
    elapsed = time.perf_counter() - started

    parsed = result["parsed"]
    with TemporaryDirectory() as tmp:
        excel_path = Path(tmp) / "signboards.xlsx"
        previous_excel_path = app.EXCEL_PATH
        try:
            app.EXCEL_PATH = str(excel_path)
            saved_id = app.append_record(parsed)
            rows = app.load_records()
        finally:
            app.EXCEL_PATH = previous_excel_path

    return {
        "ok": bool(parsed.get("store_name") and parsed.get("phone")) and len(rows) == 1,
        "elapsed_seconds": round(elapsed, 3),
        "image_name": parsed.get("image_name"),
        "detection_count": len(result["detections"]),
        "parsed": parsed,
        "saved_id": int(saved_id),
        "saved_rows": int(len(rows)),
    }


def predictions_to_dicts(predictions):
    return [
        {
            "bbox": [float(value) for value in bbox],
            "class_name": class_name,
            "confidence": float(confidence),
            "text": text,
        }
        for bbox, class_name, confidence, text in predictions
    ]


def parsed_to_record(parsed, image_name, confidence):
    return {
        "image_name": image_name,
        "store_name": parsed.get("business_name", ""),
        "category": parsed.get("category_hint", ""),
        "phone": parsed.get("phone", ""),
        "address": parsed.get("address_hint", ""),
        "services": "",
        "raw_text": parsed.get("raw_text", ""),
        "confidence": confidence,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "saved by run_e2e_report.py",
    }


def run_direct_ocr(image_path: Path, save_excel: bool):
    add_deployment_to_path()
    os.environ["OCR_ENGINE"] = "vietocr"

    import ocr
    from signboard_excel_export import append_signboard_record
    from signboard_parser import parse_signboard_fields

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    started = time.perf_counter()
    service = ocr.OCRCore(
        reg_model=ocr.reg_model,
        det_model=ocr.det_model,
        ocr_engine=ocr.OCR_ENGINE,
        vietocr_config_path=str(ocr.VIETOCR_CONFIG_PATH),
        vietocr_weights_path=str(ocr.VIETOCR_WEIGHTS_PATH),
    )
    predictions = service.process_image(str(image_path))
    parsed = parse_signboard_fields(predictions)
    confidence = (
        round(sum(float(item[2]) for item in predictions) / len(predictions), 4)
        if predictions
        else 0
    )
    elapsed = time.perf_counter() - started

    excel_result = None
    if save_excel:
        excel_result = append_signboard_record(
            parsed_to_record(parsed, image_path.name, confidence)
        )

    return {
        "image": str(image_path),
        "elapsed_seconds": round(elapsed, 3),
        "prediction_count": len(predictions),
        "average_detection_confidence": confidence,
        "predictions": predictions_to_dicts(predictions),
        "parsed_fields": parsed,
        "excel_result": excel_result,
    }


def derive_status(report):
    checks = [
        report["static_checks"].get("all_required_code_exists", False),
        report["component_checks"].get("parser", {}).get("ok", False),
        report["component_checks"].get("excel", {}).get("ok", False),
    ]
    if report.get("demo_pipeline") is not None:
        checks.append(report["demo_pipeline"].get("ok", False))
    return "passed" if all(checks) else "failed"


def write_report(report, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    report = {
        "project": "Smart Signboard OCR",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "status": "running",
        "static_checks": {},
        "component_checks": {},
        "demo_pipeline": None,
        "direct_ocr": None,
        "errors": [],
    }

    try:
        report["static_checks"] = run_static_checks()
        report["component_checks"] = run_component_checks()
        if not args.skip_demo:
            report["demo_pipeline"] = run_demo_pipeline()
        if args.image:
            report["direct_ocr"] = run_direct_ocr(args.image.resolve(), args.save_excel)
        report["status"] = derive_status(report)
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(str(exc))

    write_report(report, args.output.resolve())
    print(f"E2E report written: {args.output.resolve()}")
    print(f"Status: {report['status']}")
    print(f"Model artifacts ready: {report['static_checks'].get('model_artifacts_ready')}")

    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
