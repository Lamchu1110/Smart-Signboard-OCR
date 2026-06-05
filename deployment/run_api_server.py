import json
import os
import platform
import tempfile
from io import BytesIO

# Torch can hang on some Windows shells while platform.machine() queries WMI.
# Return the architecture directly before importing torch.
platform.machine = lambda: os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")

# Preload torch before any optional Ray imports on Windows.
import torch  # noqa: F401
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

import ocr
from signboard_excel_export import append_signboard_record, read_signboard_records
from signboard_parser import parse_signboard_fields


os.environ.setdefault("OCR_ENGINE", "vietocr")

app = FastAPI(title="Smart Signboard OCR API")
service = ocr.OCRCore(
    reg_model=ocr.reg_model,
    det_model=ocr.det_model,
    ocr_engine=ocr.OCR_ENGINE,
    vietocr_config_path=str(ocr.VIETOCR_CONFIG_PATH),
    vietocr_weights_path=str(ocr.VIETOCR_WEIGHTS_PATH),
)


def header_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def predictions_to_jsonable(predictions):
    return [
        {
            "bbox": [float(value) for value in bbox],
            "class_name": class_name,
            "confidence": float(confidence),
            "text": text,
        }
        for bbox, class_name, confidence, text in predictions
    ]


def process_image_bytes(image_data: bytes):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        predictions = service.process_image(temp_file_path)
        parsed_fields = parse_signboard_fields(predictions)
        image = Image.open(temp_file_path)
        annotated_image = service.draw_predictions(image, predictions)

        file_stream = BytesIO()
        annotated_image.save(file_stream, format="PNG")
        file_stream.seek(0)

        return Response(
            content=file_stream.getvalue(),
            media_type="image/png",
            headers={
                "X-Predictions": header_json(predictions_to_jsonable(predictions)),
                "X-Parsed-Fields": header_json(parsed_fields),
            },
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/ocr")
def ocr_url(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return process_image_bytes(response.content)
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        return process_image_bytes(await file.read())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {exc}")


@app.post("/signboards/save")
def save_signboard(record: dict):
    try:
        return append_signboard_record(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error saving signboard: {exc}")


@app.get("/signboards")
def list_signboards():
    try:
        return {"signboards": read_signboard_records()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading signboards: {exc}")


@app.post("/orders/save")
def save_order_compat(record: dict):
    return save_signboard(record)


@app.get("/orders")
def list_orders_compat():
    return {"orders": read_signboard_records()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
