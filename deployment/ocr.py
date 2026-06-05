import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from crnn import CRNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from post_processing import post_process_text_detections
from ray import serve
from signboard_excel_export import append_signboard_record, read_signboard_records
from signboard_parser import parse_signboard_fields
from torchvision import transforms
from ultralytics import YOLO
from vietocr_recognizer import VietOCRRecognizer

app = FastAPI(title="Smart Signboard OCR API")

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
NEW_TEXT_DET_MODEL_PATH = BASE_DIR / "weights" / "yolo_vintext" / "best.pt"
LEGACY_TEXT_DET_MODEL_PATH = BASE_DIR / "weights" / "best.pt"
TEXT_DET_MODEL_PATH = NEW_TEXT_DET_MODEL_PATH if NEW_TEXT_DET_MODEL_PATH.exists() else LEGACY_TEXT_DET_MODEL_PATH
OCR_MODEL_PATH = BASE_DIR / "weights" / "ocr_crnn.pt"
VIETOCR_CONFIG_PATH = BASE_DIR / "weights" / "vietocr_vintext" / "config.yml"
VIETOCR_WEIGHTS_PATH = BASE_DIR / "weights" / "vietocr_vintext" / "transformerocr.pth"
DEFAULT_OCR_ENGINE = "vietocr" if VIETOCR_CONFIG_PATH.exists() and VIETOCR_WEIGHTS_PATH.exists() else "crnn"
OCR_ENGINE = os.getenv("OCR_ENGINE", DEFAULT_OCR_ENGINE).strip().lower()


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


def header_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

# Character set configuration
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# Model configuration
HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, ocr_handle):
        self.handle = ocr_handle

    async def process_image(self, image_data: bytes) -> Response:
        """Common image processing logic for both URL and file upload"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name

            # Request OCR results using the temp file path
            predictions = await self.handle.process_image.remote(temp_file_path)
            parsed_fields = parse_signboard_fields(predictions)

            # Load the image and draw predictions
            image = Image.open(temp_file_path)
            annotated_image = await self.handle.draw_predictions.remote(
                image, predictions
            )

            # Convert annotated image to bytes
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="PNG")
            file_stream.seek(0)

            # Clean up the temporary file
            os.unlink(temp_file_path)

            return Response(
                content=file_stream.getvalue(),
                media_type="image/png",
                headers={
                    "X-Predictions": header_json(predictions_to_jsonable(predictions)),
                    "X-Parsed-Fields": header_json(parsed_fields),
                },
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    @app.get("/ocr")
    async def ocr_url(self, image_url: str):
        """Endpoint for processing images from URLs"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return await self.process_image(response.content)
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")

    @app.post("/ocr/upload")
    async def ocr_upload(self, file: UploadFile = File(...)):
        """Endpoint for processing uploaded image files"""
        try:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            content = await file.read()
            return await self.process_image(content)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded file: {e}"
            )

    @app.post("/signboards/save")
    async def save_signboard(self, record: dict):
        try:
            return append_signboard_record(record)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving signboard: {e}")

    @app.get("/signboards")
    async def list_signboards(self):
        try:
            return {"signboards": read_signboard_records()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading signboards: {e}")

    @app.post("/orders/save")
    async def save_order_compat(self, record: dict):
        return await self.save_signboard(record)

    @app.get("/orders")
    async def list_orders_compat(self):
        return {"orders": read_signboard_records()}


class OCRCore:
    def __init__(self, reg_model, det_model, ocr_engine, vietocr_config_path, vietocr_weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_model = det_model.to(self.device)
        self.ocr_engine = ocr_engine
        self.reg_model = None
        self.vietocr_recognizer = None

        if self.ocr_engine == "vietocr":
            self.vietocr_recognizer = VietOCRRecognizer.from_paths(
                config_path=Path(vietocr_config_path),
                weights_path=Path(vietocr_weights_path),
            )
        elif self.ocr_engine == "crnn":
            if reg_model is None:
                raise ValueError("CRNN model is required when OCR_ENGINE=crnn.")
            self.reg_model = reg_model.to(self.device)
            self.reg_model.eval()
        else:
            raise ValueError("OCR_ENGINE must be either 'vietocr' or 'crnn'.")

        # Define transform for inference
        self.transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def text_detection(self, img_path):
        """Detect text regions in the image"""
        results = self.det_model(img_path, verbose=False)[0]
        return (
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.names,
            results.boxes.conf.tolist(),
        )

    def text_recognition(self, img):
        """Recognize text in the cropped image"""
        if self.ocr_engine == "vietocr":
            return self.vietocr_recognizer.recognize(img)

        transformed_image = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.reg_model(transformed_image).cpu()
        text = self.decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
        return text[0]

    def process_image(self, img_path: str):
        """Process the image through the OCR pipeline"""
        try:
            # Detect text regions
            bboxes, classes, names, confs = self.text_detection(img_path)

            detections = [
                {
                    "bbox": bbox,
                    "class_name": names[int(cls_idx)],
                    "confidence": conf,
                }
                for bbox, cls_idx, conf in zip(bboxes, classes, confs)
            ]
            # Load image
            img = Image.open(img_path)
            merged_detections = post_process_text_detections(detections, img.size)
            predictions = []

            # Process each merged text line/region
            for detection in merged_detections:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                name = detection["class_name"]
                conf = detection["confidence"]

                # Crop and recognize text
                cropped_image = img.crop((x1, y1, x2, y2))
                transcribed_text = self.text_recognition(cropped_image)
                predictions.append((bbox, name, conf, transcribed_text))

            return predictions

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    def draw_predictions(self, image, predictions):
        """Draw predictions on the image using PIL for stable local rendering."""
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except OSError:
            font = ImageFont.load_default()

        predictions = sorted(
            predictions, key=lambda x: x[0][1]
        )

        for bbox, class_name, confidence, text in predictions:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = (255, 0, 180)
            label = f"{class_name[:3]} {confidence:.2f}: {text}"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            label_width = text_bbox[2] - text_bbox[0]
            label_height = text_bbox[3] - text_bbox[1]
            label_y1 = max(0, y1 - label_height - 4)
            label_box = [x1, label_y1, x1 + label_width + 6, label_y1 + label_height + 4]
            draw.rectangle(label_box, fill=color)
            draw.text((x1 + 3, label_y1 + 2), label, fill=(255, 255, 255), font=font)

        return annotated

    def decode(self, encoded_sequences, idx_to_char, blank_char="-"):
        decoded_sequences = []

        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None  # To track the previous character

            for token in seq:
                if token != 0:  # Ignore padding (token = 0)
                    char = idx_to_char[token.item()]
                    # Append the character if it's not a blank or the same as the previous character
                    if char != blank_char:
                        if char != prev_char or prev_char == blank_char:
                            decoded_label.append(char)
                    prev_char = char  # Update previous character

            decoded_sequences.append("".join(decoded_label))

        return decoded_sequences


def build_crnn_model():
    model = CRNN(
        vocab_size=len(CHARS),
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT_PROB,
        unfreeze_layers=UNFREEZE_LAYERS,
    )
    model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@serve.deployment(
    ray_actor_options={"num_gpus": 0, "num_cpus": 2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class OCRService(OCRCore):
    pass


# ----------------  Initialize YOLO model
det_model = YOLO(TEXT_DET_MODEL_PATH)

# ----------------  Initialize recognition model
reg_model = build_crnn_model() if OCR_ENGINE == "crnn" else None

# ----------------  Create the service
entrypoint = APIIngress.bind(
    OCRService.bind(
        reg_model=reg_model,
        det_model=det_model,
        ocr_engine=OCR_ENGINE,
        vietocr_config_path=str(VIETOCR_CONFIG_PATH),
        vietocr_weights_path=str(VIETOCR_WEIGHTS_PATH),
    )
)
