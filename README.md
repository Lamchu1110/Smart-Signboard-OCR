# Smart Signboard OCR

Smart Signboard OCR is a Vietnamese scene-text OCR system for signboard,
banner, and poster images. It detects text regions with YOLO, recognizes
Vietnamese text with VietOCR, extracts business information with a rule-based
parser, and stores reviewed records in Excel for search and lookup.

## Main Features

- Streamlit web interface for OCR, review, save, search, and record editing.
- YOLO text-region detection for natural signboard images.
- Bounding-box post-processing: confidence filtering, merge, and reading-order
  sorting.
- VietOCR recognition for Vietnamese text crops.
- Signboard parser for store name, business category, phone, address, and
  services.
- Excel storage in `outputs/signboards.xlsx`.
- Training and evaluation utilities for VinText-based experiments.

## Pipeline

```text
Input image
-> YOLO text detection
-> Post-processing bbox merge/sort
-> Crop detected text regions
-> VietOCR text recognition
-> Line grouping and raw OCR text
-> Signboard parser
-> User verification
-> Save to Excel
-> Search / review records
```

## Repository Structure

```text
deployment/
  app.py                      Streamlit UI
  run_api_server.py           Direct FastAPI OCR API
  run_ocr_server.py           Optional Ray Serve launcher
  ocr.py                      OCR pipeline core
  crnn.py                     Legacy CRNN fallback/model comparison support
  post_processing.py          Bounding-box merge and sort utilities
  signboard_parser.py         Rule-based signboard parser
  signboard_excel_export.py   Excel read/write helpers
  vietocr_recognizer.py       VietOCR inference wrapper

tools/
  prepare_vintext_yolo.py       Convert VinText annotations to YOLO format
  train_yolo_vintext.py         Train YOLO on VinText
  prepare_vintext_vietocr.py    Prepare VietOCR text crops from VinText
  train_vietocr_vintext.py      Fine-tune VietOCR
  collect_report_metrics.py     Compare detection/recognition metrics
  smoke_test_vintext.py         Lightweight VinText smoke test
  run_e2e_report.py             System-level E2E report
  verify_training_artifacts.py  Check required model files

weights/
  yolo_vintext/
    best.pt                     Not included in Git
  vietocr_vintext/
    config.yml                  Not included in Git
    transformerocr.pth          Not included in Git

outputs/
  signboards.xlsx               Generated at runtime, not included in Git
```

## What Is Not Included

Large and local-only files are intentionally excluded from GitHub:

- VinText dataset
- trained model weights (`.pt`, `.pth`, `.onnx`, etc.)
- runtime outputs and Excel files
- notebooks used during early experiments
- local virtual environments and cache folders

Place trained artifacts manually before running real OCR:

```text
weights/yolo_vintext/best.pt
weights/vietocr_vintext/config.yml
weights/vietocr_vintext/transformerocr.pth
```

## Environment Setup

Use Python 3.10 or 3.11 when possible. Python 3.12 can work, but some OCR
dependencies are more sensitive to package versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Streamlit App

```powershell
cd deployment
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The UI contains:

- Dashboard
- OCR Workspace
- Signboard Records
- Search
- Settings
- About

## Run FastAPI OCR Server

The direct FastAPI server is the recommended local API path on Windows.

```powershell
cd deployment
$env:OCR_ENGINE="vietocr"
python run_api_server.py
```

API docs:

```text
http://localhost:8000/docs
```

Main endpoints:

```text
GET  /ocr?image_url=<url>
POST /ocr/upload
POST /signboards/save
GET  /signboards
```

## Training YOLO on VinText

Prepare YOLO-format data:

```powershell
python tools\prepare_vintext_yolo.py `
  --dataset-root path\to\vietnamese `
  --output-dir datasets\vintext_yolo
```

Train:

```powershell
python tools\train_yolo_vintext.py `
  --data datasets\vintext_yolo\data.yaml `
  --model yolo11s.pt `
  --epochs 50 `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --project weights `
  --name yolo_vintext
```

Expected trained detection model:

```text
weights/yolo_vintext/best.pt
```

## Fine-Tune VietOCR on VinText

Prepare recognition crops:

```powershell
python tools\prepare_vintext_vietocr.py `
  --dataset-root path\to\vietnamese `
  --output-dir datasets\vintext_recognition
```

Fine-tune:

```powershell
python tools\train_vietocr_vintext.py `
  --data-root datasets\vintext_recognition `
  --output-dir weights\vietocr_vintext `
  --device cuda:0 `
  --batch-size 16 `
  --iters 20000
```

Expected VietOCR artifacts:

```text
weights/vietocr_vintext/config.yml
weights/vietocr_vintext/transformerocr.pth
```

## Metrics for Report

After placing the required model files and VinText dataset, run:

```powershell
python tools\collect_report_metrics.py `
  --dataset-root path\to\vietnamese `
  --det-split val `
  --rec-samples 200 `
  --device cuda:0 `
  --output-dir outputs\report_metrics
```

The script generates CSV, JSON, and Markdown metric reports for detection and
recognition comparisons.

## Verify Artifacts

```powershell
python tools\verify_training_artifacts.py
```

For a lightweight check:

```powershell
python tools\verify_training_artifacts.py --skip-model-load
```

## Output Contract

Saved records use this schema:

```text
id, image_name, store_name, category, phone, address, services,
raw_text, confidence, created_at, note
```

See `signboard_ocr_contract.txt` for field-level details.

## Recommended Model Setup

Based on the current VinText comparison, the recommended deployment pipeline is:

```text
YOLO11s + VietOCR fine-tuned on VinText
```

YOLO11s is used for text-region detection, VietOCR is used for Vietnamese text
recognition, and the parser converts OCR text into structured signboard records.
