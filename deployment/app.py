from datetime import datetime
from io import BytesIO
import html
import json
import os
import re
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


APP_NAME = "Smart Signboard OCR"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXCEL_PATH = os.path.join(OUTPUT_DIR, "signboards.xlsx")
FEEDBACK_PATH = os.path.join(OUTPUT_DIR, "signboard_feedback.xlsx")

RECORD_COLUMNS = [
    "id",
    "image_name",
    "store_name",
    "category",
    "phone",
    "address",
    "services",
    "raw_text",
    "confidence",
    "created_at",
    "note",
]

FEEDBACK_COLUMNS = [
    "created_at",
    "image_name",
    "model_mode",
    "predicted_store_name",
    "corrected_store_name",
    "predicted_category",
    "corrected_category",
    "predicted_phone",
    "corrected_phone",
    "predicted_address",
    "corrected_address",
    "predicted_services",
    "corrected_services",
    "raw_text",
    "detected_lines",
    "store_name_bbox",
    "store_name_score",
    "category_confidence",
    "category_evidence",
    "changed_fields",
    "note",
]

CATEGORIES = [
    "Chưa xác định",
    "Cà phê / Trà sữa",
    "Quán ăn / Nhà hàng",
    "Tiệm tóc / Salon",
    "Spa / Nail",
    "Nhà thuốc",
    "Sửa xe",
    "Khách sạn / Nhà nghỉ",
    "Phòng khám / Nha khoa",
    "Tạp hóa / Bách hóa",
    "Điện thoại / Điện máy",
    "Karaoke",
    "Khác",
]

CATEGORY_LABELS = {
    "cafe": "Cà phê / Trà sữa",
    "quan_an": "Quán ăn / Nhà hàng",
    "sua_xe": "Sửa xe",
    "spa": "Spa / Nail",
    "tiem_toc": "Tiệm tóc / Salon",
    "nha_thuoc": "Nhà thuốc",
    "phong_tro": "Khác",
    "dien_thoai": "Điện thoại / Điện máy",
    "nha_khoa": "Phòng khám / Nha khoa",
    "phong_kham": "Phòng khám / Nha khoa",
    "tap_hoa": "Tạp hóa / Bách hóa",
    "karaoke": "Karaoke",
    "khach_san": "Khách sạn / Nhà nghỉ",
}

TEXT_REPLACEMENTS = {
    "Càphê": "Cà phê",
    "CàPhê": "Cà Phê",
    "càphê": "cà phê",
    "Đặcbiệt": "Đặc biệt",
    "đặcbiệt": "đặc biệt",
    "YOUARE": "YOU ARE",
    "YOU AREE": "YOU ARE",
}

VIETNAMESE_VOWELS = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ"
VIETNAMESE_ONSETS = (
    "b",
    "c",
    "ch",
    "d",
    "đ",
    "g",
    "gh",
    "gi",
    "h",
    "k",
    "kh",
    "l",
    "m",
    "n",
    "ng",
    "ngh",
    "nh",
    "p",
    "ph",
    "q",
    "r",
    "s",
    "t",
    "th",
    "tr",
    "v",
    "x",
)


st.set_page_config(
    page_title=APP_NAME,
    page_icon="OCR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --navy: #12355b;
            --blue: #1f5f99;
            --muted: #667085;
            --border: #d9e2ec;
            --surface: #ffffff;
            --page: #f3f6fa;
        }

        .stApp {
            background: var(--page);
            color: #172033;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        .stDeployButton {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        section[data-testid="stSidebar"] {
            background: #0f2742;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fbff;
        }

        section[data-testid="stSidebar"] .stRadio label {
            color: #f8fbff !important;
        }

        .block-container {
            padding-top: 0.7rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }

        .stApp label,
        .stApp p,
        .stApp span,
        .stApp div[data-testid="stMarkdownContainer"] {
            color: #172033;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
            color: #f8fbff !important;
        }

        h1, h2, h3 {
            color: #102a43;
            letter-spacing: 0;
        }

        .page-title {
            font-size: 34px;
            font-weight: 750;
            color: #102a43;
            margin: 0 0 4px 0;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 15px;
            margin-bottom: 20px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.06);
            padding: 18px 20px;
            margin-bottom: 16px;
        }

        .card:empty {
            display: none;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 20px rgba(16, 42, 67, 0.06);
            min-height: 112px;
        }

        .metric-label {
            color: #667085;
            font-size: 13px;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #12355b;
            font-size: 30px;
            font-weight: 780;
            line-height: 1.1;
        }

        .metric-note {
            color: #697586;
            font-size: 12px;
            margin-top: 7px;
        }

        .control-help {
            color: #52616f;
            font-size: 12px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 12px;
        }

        div[data-testid="stAlert"] p {
            color: #102a43 !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: #12355b;
            color: white;
            border: 1px solid #12355b;
            border-radius: 7px;
            min-height: 40px;
            font-weight: 650;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #1f5f99;
            color: white;
            border-color: #1f5f99;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            background: #ffffff;
            border-radius: 8px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }

        .info-tile {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 86px;
        }

        .info-tile-label {
            color: #52616f;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .info-tile-value {
            color: #102a43;
            font-size: 14px;
            line-height: 1.45;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }

        .json-panel {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 14px;
        }

        .json-panel-title {
            color: #102a43;
            font-size: 13px;
            font-weight: 750;
            margin-bottom: 8px;
        }

        .json-panel pre {
            background: #f8fafc;
            border: 1px solid #e4ebf3;
            border-radius: 7px;
            color: #102a43;
            font-size: 13px;
            line-height: 1.45;
            margin: 0;
            padding: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        .light-table {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            margin: 10px 0 16px 0;
            overflow-x: auto;
        }

        .light-table-title {
            color: #102a43;
            font-size: 13px;
            font-weight: 750;
            margin: 18px 0 8px 0;
        }

        .light-table table {
            border-collapse: collapse;
            min-width: 100%;
            color: #102a43;
            font-size: 13px;
        }

        .light-table th {
            background: #e8f1fb;
            color: #102a43;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #d9e2ec;
            font-weight: 750;
        }

        .light-table td {
            background: #ffffff;
            color: #102a43;
            padding: 10px 12px;
            border-bottom: 1px solid #eef2f6;
            vertical-align: top;
            overflow-wrap: anywhere;
        }

        @media (max-width: 900px) {
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def html_value(value):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value or ""))


def render_info_grid(items):
    tiles = []
    for label, value in items:
        tiles.append(
            f"""
            <div class="info-tile">
                <div class="info-tile-label">{html.escape(str(label))}</div>
                <div class="info-tile-value">{html_value(value)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="info-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_json_panel(title, payload):
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    st.markdown(
        f"""
        <div class="json-panel">
            <div class="json-panel-title">{html.escape(str(title))}</div>
            <pre>{html.escape(pretty)}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(title, rows, columns):
    if not rows:
        return
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "") if isinstance(row, dict) else ""
            cells.append(f"<td>{html_value(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f"""
        <div class="light-table-title">{html.escape(str(title))}</div>
        <div class="light-table">
            <table>
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def empty_records():
    return normalize_records(pd.DataFrame(columns=RECORD_COLUMNS))


def normalize_records(df):
    text_columns = [
        "image_name",
        "store_name",
        "category",
        "phone",
        "address",
        "services",
        "raw_text",
        "created_at",
        "note",
    ]
    for column in RECORD_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[RECORD_COLUMNS].copy()

    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).astype(float)
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)
        df[column] = df[column].replace({"nan": "", "NaN": "", "None": ""})
    return df


def load_records():
    if not os.path.exists(EXCEL_PATH):
        return empty_records()
    try:
        return normalize_records(pd.read_excel(EXCEL_PATH))
    except Exception as exc:
        st.warning(f"Could not read signboards.xlsx: {exc}")
        return empty_records()


def save_records(df):
    ensure_output_dir()
    normalized = normalize_records(df)
    normalized.to_excel(EXCEL_PATH, index=False)
    return normalized


def next_record_id(df):
    if df.empty or "id" not in df.columns:
        return 1
    ids = pd.to_numeric(df["id"], errors="coerce").fillna(0)
    return int(ids.max()) + 1


def append_record(record):
    records = load_records()
    record = {column: record.get(column, "") for column in RECORD_COLUMNS}
    record["id"] = next_record_id(records)
    new_row = pd.DataFrame([record], columns=RECORD_COLUMNS)
    records = new_row if records.empty else pd.concat([records, new_row], ignore_index=True)
    save_records(records)
    return record["id"]


def append_feedback_record(feedback):
    ensure_output_dir()
    row = {column: feedback.get(column, "") for column in FEEDBACK_COLUMNS}
    existing = (
        pd.read_excel(FEEDBACK_PATH)
        if os.path.exists(FEEDBACK_PATH)
        else pd.DataFrame(columns=FEEDBACK_COLUMNS)
    )
    for column in FEEDBACK_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
    existing = existing[FEEDBACK_COLUMNS]
    updated = pd.concat([existing, pd.DataFrame([row], columns=FEEDBACK_COLUMNS)], ignore_index=True)
    updated.to_excel(FEEDBACK_PATH, index=False)
    return FEEDBACK_PATH


def compare_changed_fields(original, corrected):
    fields = ["store_name", "category", "phone", "address", "services"]
    changed = []
    for field in fields:
        before = str(original.get(field, "") or "").strip()
        after = str(corrected.get(field, "") or "").strip()
        if before != after:
            changed.append(field)
    return ", ".join(changed)


def dataframe_to_excel_bytes(df):
    buffer = BytesIO()
    normalize_records(df).to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def draw_demo_boxes(image, detections, show_boxes=True):
    annotated = image.convert("RGB").copy()
    if not show_boxes:
        return annotated

    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        label = f'{item["text"]}  {item["confidence"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline="#1f5f99", width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        label_box = [x1, max(0, y1 - 22), x1 + text_bbox[2] - text_bbox[0] + 10, max(22, y1)]
        draw.rectangle(label_box, fill="#12355b")
        draw.text((x1 + 5, label_box[1] + 3), label, fill="white", font=font)
    return annotated


def clean_ocr_text(text):
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    for source, target in TEXT_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", value)
    value = re.sub(r"(?i)\b(cà)\s*(phê)\b", r"\1 \2", value)
    value = re.sub(r"(?i)\b(đặc)\s*(biệt)\b", r"\1 \2", value)
    value = fix_split_vietnamese_syllables(value)
    return value.strip()


def fix_split_vietnamese_syllables(text):
    tokens = str(text or "").split()
    if len(tokens) < 2:
        return str(text or "")

    merged = []
    index = 0
    while index < len(tokens):
        current = tokens[index]
        if index + 1 < len(tokens):
            next_token = tokens[index + 1]
            current_l = current.lower()
            next_l = next_token.lower()
            if (
                current_l in VIETNAMESE_ONSETS
                and next_l
                and next_l[0] in VIETNAMESE_VOWELS
            ):
                merged.append(current + next_token)
                index += 2
                continue
            if (
                len(current_l) <= 3
                and any(char in VIETNAMESE_VOWELS for char in current_l)
                and next_l
                and next_l[0] in VIETNAMESE_VOWELS
            ):
                merged.append(current + next_token)
                index += 2
                continue
        merged.append(current)
        index += 1
    return " ".join(merged)


def clean_predictions(predictions):
    cleaned = []
    for bbox, class_name, confidence, text in predictions:
        cleaned_text = clean_ocr_text(text)
        if cleaned_text:
            cleaned.append((bbox, class_name, confidence, cleaned_text))
    return cleaned


def filter_noise_predictions(predictions):
    filtered = []
    for bbox, class_name, confidence, text in predictions:
        alpha_count = sum(char.isalpha() for char in str(text))
        if alpha_count < 2 and len(str(text).strip()) <= 3:
            continue
        if float(confidence) < 0.35 and alpha_count <= 2:
            continue
        filtered.append((bbox, class_name, confidence, text))
    return filtered


def bbox_geometry(bbox):
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    return {
        "bbox": [x1, y1, x2, y2],
        "x": x1,
        "y": y1,
        "x2": x2,
        "y2": y2,
        "cx": x1 + width / 2,
        "cy": y1 + height / 2,
        "height": height,
    }


def merge_sort_bbox(box_a, box_b):
    return [
        min(float(box_a[0]), float(box_b[0])),
        min(float(box_a[1]), float(box_b[1])),
        max(float(box_a[2]), float(box_b[2])),
        max(float(box_a[3]), float(box_b[3])),
    ]


def reading_line_match(item_geo, line_geo):
    overlap = min(item_geo["y2"], line_geo["y2"]) - max(item_geo["y"], line_geo["y"])
    overlap_ratio = overlap / max(min(item_geo["height"], line_geo["height"]), 1.0)
    avg_height = (item_geo["height"] + line_geo["height"]) / 2
    center_delta = abs(item_geo["cy"] - line_geo["cy"])
    height_ratio = max(item_geo["height"], line_geo["height"]) / max(
        min(item_geo["height"], line_geo["height"]), 1.0
    )
    return (
        overlap_ratio >= 0.25
        or center_delta <= avg_height * 0.85
    ) and height_ratio <= 4.0


def sort_items_reading_order(items, bbox_getter):
    """Group boxes into visual text lines, then sort each line left-to-right."""
    normalized = []
    for index, item in enumerate(items):
        geo = bbox_geometry(bbox_getter(item))
        normalized.append({"index": index, "item": item, **geo})

    lines = []
    for item in sorted(normalized, key=lambda value: (value["cy"], value["x"])):
        matched = None
        for line in lines:
            if reading_line_match(item, line):
                matched = line
                break
        if matched is None:
            lines.append(
                {
                    "bbox": item["bbox"][:],
                    "x": item["x"],
                    "y": item["y"],
                    "x2": item["x2"],
                    "y2": item["y2"],
                    "cy": item["cy"],
                    "height": item["height"],
                    "items": [item],
                }
            )
            continue

        matched["items"].append(item)
        matched["bbox"] = merge_sort_bbox(matched["bbox"], item["bbox"])
        updated = bbox_geometry(matched["bbox"])
        matched.update(updated)

    ordered = []
    for line in sorted(lines, key=lambda value: (value["y"], value["x"])):
        ordered.extend(
            value["item"]
            for value in sorted(line["items"], key=lambda item: (item["x"], item["y"]))
        )
    return ordered


def prediction_tuple_to_detection(prediction):
    bbox, class_name, confidence, text = prediction
    text = clean_ocr_text(text)
    return {
        "bbox": [float(value) for value in bbox],
        "class_name": class_name,
        "confidence": float(confidence),
        "text": text,
    }


def draw_detection_boxes(image, detections, show_boxes=True, selected_bbox=None):
    annotated = image.convert("RGB").copy()
    if not show_boxes:
        return annotated

    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
        label = f'{item.get("text", "")}  {float(item.get("confidence", 0)):.2f}'
        draw.rectangle([x1, y1, x2, y2], outline="#1f5f99", width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        label_width = text_bbox[2] - text_bbox[0]
        label_y = max(0, y1 - 24)
        draw.rectangle([x1, label_y, x1 + label_width + 10, label_y + 24], fill="#12355b")
        draw.text((x1 + 5, label_y + 4), label, fill="white", font=font)

    if selected_bbox:
        x1, y1, x2, y2 = [int(float(value)) for value in selected_bbox]
        draw.rectangle([x1, y1, x2, y2], outline="#f59e0b", width=5)
        label = "selected store name"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        label_width = text_bbox[2] - text_bbox[0]
        label_y = max(0, y1 - 26)
        draw.rectangle([x1, label_y, x1 + label_width + 12, label_y + 24], fill="#f59e0b")
        draw.text((x1 + 6, label_y + 4), label, fill="white", font=font)
    return annotated


def average_confidence(detections):
    if not detections:
        return 0.0
    return round(sum(float(item["confidence"]) for item in detections) / len(detections), 3)


def infer_services(raw_lines, store_name, phone, address):
    blocked = {str(store_name or "").lower(), str(phone or "").lower(), str(address or "").lower()}
    candidates = []
    for line in raw_lines:
        text = str(line or "").strip()
        if not text or text.lower() in blocked:
            continue
        if sum(char.isalpha() for char in text) < 3:
            continue
        if any(char.isdigit() for char in text) and sum(char.isalpha() for char in text) < 3:
            continue
        candidates.append(text)
    return ", ".join(candidates[:3])


def to_ui_parsed_fields(parsed_fields, image_name, detections):
    raw_lines = parsed_fields.get("lines") or []
    store_name = parsed_fields.get("business_name", "")
    category_key = parsed_fields.get("category_hint", "")
    address = parsed_fields.get("address_hint", "")
    phone = parsed_fields.get("phone", "")
    return {
        "image_name": image_name,
        "store_name": store_name,
        "category": CATEGORY_LABELS.get(category_key, "Chưa xác định"),
        "phone": phone,
        "address": address,
        "services": infer_services(raw_lines, store_name, phone, address),
        "raw_text": parsed_fields.get("raw_text", ""),
        "confidence": average_confidence(detections),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "",
        "store_name_bbox": parsed_fields.get("store_name_bbox", []),
        "store_name_score": parsed_fields.get("store_name_score", 0.0),
        "store_name_reason": parsed_fields.get("store_name_reason", {}),
        "store_name_candidates": parsed_fields.get("store_name_candidates", []),
        "category_key": category_key,
        "category_confidence": parsed_fields.get("category_confidence", 0.0),
        "category_evidence": parsed_fields.get("category_evidence", []),
        "category_scores": parsed_fields.get("category_scores", {}),
        "line_items": parsed_fields.get("line_items", []),
    }


@st.cache_resource(show_spinner=False)
def get_real_ocr_service():
    os.environ.setdefault("OCR_ENGINE", "vietocr")
    import ocr

    return ocr.OCRCore(
        reg_model=ocr.reg_model,
        det_model=ocr.det_model,
        ocr_engine=ocr.OCR_ENGINE,
        vietocr_config_path=str(ocr.VIETOCR_CONFIG_PATH),
        vietocr_weights_path=str(ocr.VIETOCR_WEIGHTS_PATH),
    )


def run_unmerged_real_ocr(service, image_path, image, min_confidence, sort_bbox):
    bboxes, classes, names, confs = service.text_detection(image_path)
    raw_items = []
    for bbox, cls_idx, conf in zip(bboxes, classes, confs):
        if float(conf) < min_confidence:
            continue
        raw_items.append(
            {
                "bbox": [float(value) for value in bbox],
                "class_name": names[int(cls_idx)],
                "confidence": float(conf),
            }
        )

    if sort_bbox:
        raw_items = sort_items_reading_order(raw_items, lambda item: item["bbox"])

    predictions = []
    for item in raw_items:
        x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
        crop = image.crop((x1, y1, x2, y2))
        text = service.text_recognition(crop)
        predictions.append((item["bbox"], item["class_name"], item["confidence"], text))
    return predictions


def run_real_ocr_result(image, image_name, min_confidence, merge_bbox, sort_bbox, show_boxes):
    from signboard_parser import parse_signboard_fields

    service = get_real_ocr_service()
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        if merge_bbox:
            predictions = service.process_image(temp_file_path)
            predictions = [item for item in predictions if float(item[2]) >= min_confidence]
            if not sort_bbox:
                predictions = sorted(predictions, key=lambda item: item[0][0])
        else:
            predictions = run_unmerged_real_ocr(
                service=service,
                image_path=temp_file_path,
                image=image,
                min_confidence=min_confidence,
                sort_bbox=sort_bbox,
            )

        predictions = filter_noise_predictions(clean_predictions(predictions))
        if sort_bbox:
            predictions = sort_items_reading_order(predictions, lambda item: item[0])
        detections = [prediction_tuple_to_detection(prediction) for prediction in predictions]
        parsed_fields = parse_signboard_fields(predictions)
        return {
            "processed_image": draw_detection_boxes(
                image,
                detections,
                show_boxes=show_boxes,
                selected_bbox=parsed_fields.get("store_name_bbox"),
            ),
            "detections": detections,
            "raw_text": parsed_fields.get("raw_text", ""),
            "parsed": to_ui_parsed_fields(parsed_fields, image_name, detections),
            "parser_result": parsed_fields,
            "model_mode": "real",
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def demo_ocr_result(image, image_name, show_boxes=True):
    """Lightweight UI demo path. Real model mode uses run_real_ocr_result()."""
    from signboard_parser import parse_signboard_fields

    width, height = image.size
    detections = [
        {
            "bbox": [int(width * 0.10), int(height * 0.16), int(width * 0.72), int(height * 0.28)],
            "text": "CA PHE THAO UYEN",
            "confidence": 0.91,
        },
        {
            "bbox": [int(width * 0.12), int(height * 0.34), int(width * 0.62), int(height * 0.43)],
            "text": "Cafe - Nuoc giai khat",
            "confidence": 0.86,
        },
        {
            "bbox": [int(width * 0.12), int(height * 0.50), int(width * 0.54), int(height * 0.58)],
            "text": "0909 123 456",
            "confidence": 0.94,
        },
        {
            "bbox": [int(width * 0.12), int(height * 0.64), int(width * 0.84), int(height * 0.73)],
            "text": "123 Nguyen Hue, Quan 1",
            "confidence": 0.82,
        },
    ]
    predictions = [
        (item["bbox"], item.get("class_name", "text"), item["confidence"], item["text"])
        for item in detections
    ]
    parsed_fields = parse_signboard_fields(predictions)
    raw_text = parsed_fields.get("raw_text", "")
    parsed = to_ui_parsed_fields(parsed_fields, image_name, detections)
    return {
        "processed_image": draw_detection_boxes(
            image,
            detections,
            show_boxes=show_boxes,
            selected_bbox=parsed_fields.get("store_name_bbox"),
        ),
        "detections": detections,
        "raw_text": raw_text,
        "parsed": parsed,
        "parser_result": parsed_fields,
        "model_mode": "demo",
    }


def sidebar_menu():
    st.sidebar.markdown("## Smart Signboard OCR")
    st.sidebar.caption("Vietnamese signboard text extraction")
    st.sidebar.divider()
    return st.sidebar.radio(
        "Main menu",
        ["Dashboard", "OCR Workspace", "Signboard Records", "Search", "Settings", "About"],
        label_visibility="collapsed",
    )


def render_dashboard():
    records = load_records()
    page_header("Dashboard", "System overview and saved signboard data.")

    total_records = len(records)
    total_images = records["image_name"].nunique() if not records.empty else 0
    avg_confidence = records["confidence"].mean() if not records.empty else 0
    low_confidence = len(records[records["confidence"] < 0.70]) if not records.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Processed images", total_images, "Unique image names")
    with col2:
        metric_card("Saved records", total_records, "Rows in signboards.xlsx")
    with col3:
        metric_card("Average confidence", f"{avg_confidence:.2f}", "Mean OCR confidence")
    with col4:
        metric_card("Needs review", low_confidence, "Confidence below 0.70")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Signboards by category")
        if records.empty:
            st.info("No records yet. Run OCR and save a signboard first.")
        else:
            counts = records["category"].fillna("Chưa xác định").replace("", "Chưa xác định").value_counts()
            st.bar_chart(counts)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Latest 5 records")
        if records.empty:
            st.info("No saved records.")
        else:
            latest = records.sort_values("created_at", ascending=False).head(5)
            st.dataframe(
                latest[["id", "store_name", "category", "phone", "confidence", "created_at"]],
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


def render_ocr_workspace():
    page_header("OCR Workspace", "Upload a signboard image, inspect OCR output, verify, then save to Excel.")

    controls, preview = st.columns([0.85, 1.35])

    with controls:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload signboard image", type=["jpg", "jpeg", "png"])
        model_mode = st.radio(
            "Processing mode",
            ["Real model", "Demo"],
            horizontal=True,
            help="Real model runs YOLO + VietOCR. Demo uses sample data for UI testing.",
        )
        detection_confidence = st.slider("Detection confidence threshold", 0.10, 0.95, 0.35, 0.05)
        st.markdown(
            '<div class="control-help">NgÆ°á»¡ng confidence tá»‘i thiá»ƒu cá»§a YOLO Ä‘á»ƒ cháº¥p nháº­n má»™t vÃ¹ng chá»¯. '
            "TÄƒng chá»‰ sá»‘ nÃ y giÃºp giáº£m box sai, nhÆ°ng cÃ³ thá»ƒ bá» sÃ³t chá»¯ nhá» hoáº·c má».</div>",
            unsafe_allow_html=True,
        )
        merge_bbox = st.toggle("Merge bounding boxes", value=False)
        st.markdown(
            '<div class="control-help">Gá»™p cÃ¡c box chá»¯ gáº§n nhau thÃ nh má»™t dÃ²ng hoáº·c cá»¥m chá»¯ trÆ°á»›c khi OCR. '
            "Há»¯u Ã­ch khi biá»ƒn hiá»‡u bá»‹ tÃ¡ch má»™t tá»« thÃ nh nhiá»u box nhá».</div>",
            unsafe_allow_html=True,
        )
        sort_bbox = st.toggle("Sort bounding boxes", value=True)
        st.markdown(
            '<div class="control-help">Sáº¯p xáº¿p vÃ¹ng chá»¯ theo thá»© tá»± Ä‘á»c Ä‘á»ƒ raw OCR text dá»… parser hÆ¡n.</div>',
            unsafe_allow_html=True,
        )
        show_bbox = st.toggle("Show bounding boxes", value=True)
        st.markdown(
            '<div class="control-help">Báº­t hoáº·c táº¯t khung bounding box trÃªn áº£nh káº¿t quáº£.</div>',
            unsafe_allow_html=True,
        )
        run_ocr = st.button("Run OCR", use_container_width=True)
        st.caption("Real model mode uses YOLO text detection and VietOCR text recognition. Demo mode is only for UI testing.")
        st.markdown("</div>", unsafe_allow_html=True)

    with preview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Image preview")
        if uploaded_file is None:
            st.info("Upload an image to start.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run_ocr:
        if uploaded_file is None:
            st.warning("Please upload an image before running OCR.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Running YOLO + VietOCR..." if model_mode == "Real model" else "Running demo OCR..."):
                if model_mode == "Real model":
                    st.session_state["ocr_result"] = run_real_ocr_result(
                        image=image,
                        image_name=uploaded_file.name,
                        min_confidence=detection_confidence,
                        merge_bbox=merge_bbox,
                        sort_bbox=sort_bbox,
                        show_boxes=show_bbox,
                    )
                    st.success("OCR completed with YOLO + VietOCR.")
                else:
                    st.session_state["ocr_result"] = demo_ocr_result(
                        image=image,
                        image_name=uploaded_file.name,
                        show_boxes=show_bbox,
                    )
                    st.session_state["ocr_result"]["model_mode"] = "demo"
                    st.success("OCR completed using demo_ocr_result().")

    result = st.session_state.get("ocr_result")
    if not result:
        return

    detection_tab, text_tab, parsed_tab, save_tab = st.tabs(
        ["Detection Result", "OCR Text", "Parsed Information", "Verify & Save"]
    )

    with detection_tab:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption(f"Mode: {result.get('model_mode', 'demo')}")
        st.image(result["processed_image"], use_container_width=True)
        detections_df = pd.DataFrame(result["detections"])
        st.dataframe(detections_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with text_tab:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.text_area("Raw OCR text", value=result["raw_text"], height=220)
        st.markdown("</div>", unsafe_allow_html=True)

    with parsed_tab:
        parsed = result["parsed"]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            metric_card("Store name", parsed["store_name"], "Parsed candidate")
        with col_b:
            metric_card("Category", parsed["category"], "Business type")
        with col_c:
            metric_card("Confidence", f'{parsed["confidence"]:.2f}', "Average score")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_info_grid(
            [
                ("Phone", parsed["phone"]),
                ("Address", parsed["address"]),
                ("Services", parsed["services"]),
            ]
        )
        st.divider()
        st.subheader("Parser reasoning")
        reason_col, category_col = st.columns(2)
        with reason_col:
            render_json_panel(
                "Selected store-name region",
                {
                    "store_name_bbox": parsed.get("store_name_bbox", []),
                    "store_name_score": parsed.get("store_name_score", 0.0),
                    "store_name_reason": parsed.get("store_name_reason", {}),
                },
            )
        with category_col:
            render_json_panel(
                "Business category evidence",
                {
                    "category_key": parsed.get("category_key", ""),
                    "category_confidence": parsed.get("category_confidence", 0.0),
                    "category_evidence": parsed.get("category_evidence", []),
                    "category_scores": parsed.get("category_scores", {}),
                },
            )
        candidates = parsed.get("store_name_candidates", [])
        if candidates:
            render_light_table(
                "Top store-name candidates",
                candidates,
                ["line_index", "text", "score", "bbox", "reason"],
            )
        line_items = parsed.get("line_items", [])
        if line_items:
            render_light_table(
                "Merged visual lines used by parser",
                line_items,
                ["line_index", "text", "bbox", "confidence", "word_count", "area"],
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with save_tab:
        parsed = result["parsed"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("verify_save_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                store_name = st.text_input("Store name", value=parsed["store_name"])
                category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    index=CATEGORIES.index(parsed["category"]) if parsed["category"] in CATEGORIES else len(CATEGORIES) - 1,
                )
                phone = st.text_input("Phone", value=parsed["phone"])
                confidence = st.number_input(
                    "Confidence", min_value=0.0, max_value=1.0, value=float(parsed["confidence"]), step=0.01
                )
            with col_b:
                address = st.text_input("Address", value=parsed["address"])
                services = st.text_input("Services", value=parsed["services"])
                note = st.text_input("Note", value=parsed.get("note", ""))
                image_name = st.text_input("Image name", value=parsed["image_name"])

            raw_text = st.text_area("Raw OCR text", value=parsed["raw_text"], height=160)
            submitted = st.form_submit_button("Save to signboards.xlsx")

        if submitted:
            record = {
                "image_name": image_name,
                "store_name": store_name,
                "category": category,
                "phone": phone,
                "address": address,
                "services": services,
                "raw_text": raw_text,
                "confidence": confidence,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": note,
            }
            saved_id = append_record(record)
            feedback_path = append_feedback_record(
                {
                    "created_at": record["created_at"],
                    "image_name": image_name,
                    "model_mode": result.get("model_mode", ""),
                    "predicted_store_name": parsed.get("store_name", ""),
                    "corrected_store_name": store_name,
                    "predicted_category": parsed.get("category", ""),
                    "corrected_category": category,
                    "predicted_phone": parsed.get("phone", ""),
                    "corrected_phone": phone,
                    "predicted_address": parsed.get("address", ""),
                    "corrected_address": address,
                    "predicted_services": parsed.get("services", ""),
                    "corrected_services": services,
                    "raw_text": raw_text,
                    "detected_lines": json.dumps(parsed.get("line_items", []), ensure_ascii=False),
                    "store_name_bbox": json.dumps(parsed.get("store_name_bbox", []), ensure_ascii=False),
                    "store_name_score": parsed.get("store_name_score", 0.0),
                    "category_confidence": parsed.get("category_confidence", 0.0),
                    "category_evidence": json.dumps(parsed.get("category_evidence", []), ensure_ascii=False),
                    "changed_fields": compare_changed_fields(parsed, record),
                    "note": note,
                }
            )
            st.success(f"Saved record #{saved_id} to {EXCEL_PATH}")
            st.caption(f"Feedback saved to {feedback_path}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_records():
    page_header("Signboard Records", "Review, edit, and export saved signboard records.")
    records = load_records()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    edited = st.data_editor(
        records,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "confidence": st.column_config.NumberColumn("confidence", min_value=0.0, max_value=1.0, step=0.01),
            "raw_text": st.column_config.TextColumn("raw_text", width="large"),
            "note": st.column_config.TextColumn("note", width="medium"),
        },
    )

    col_save, col_download, col_path = st.columns([1, 1, 3])
    with col_save:
        if st.button("Save changes", use_container_width=True):
            save_records(edited)
            st.success("Changes saved to signboards.xlsx")
    with col_download:
        st.download_button(
            "Download Excel",
            data=dataframe_to_excel_bytes(edited),
            file_name="signboards.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_path:
        st.caption(EXCEL_PATH)
    st.markdown("</div>", unsafe_allow_html=True)


def render_search():
    page_header("Search", "Search saved signboard records by name, phone, address, service, and confidence.")
    records = load_records()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        keyword = st.text_input("Keyword", placeholder="Store name, phone, address, services")
    with col_b:
        category_options = ["Tất cả"] + sorted([value for value in records["category"].dropna().unique() if value])
        category = st.selectbox("Category", category_options)
    with col_c:
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        st.caption("Chá»‰ hiá»ƒn thá»‹ báº£n ghi cÃ³ OCR confidence báº±ng hoáº·c cao hÆ¡n giÃ¡ trá»‹ nÃ y.")

    filtered = records.copy()
    if keyword.strip():
        keyword_l = keyword.strip().lower()
        searchable = ["store_name", "phone", "address", "services", "raw_text", "note"]
        mask = filtered[searchable].astype(str).apply(
            lambda row: any(keyword_l in value.lower() for value in row), axis=1
        )
        filtered = filtered[mask]
    if category != "Tất cả":
        filtered = filtered[filtered["category"] == category]
    filtered = filtered[pd.to_numeric(filtered["confidence"], errors="coerce").fillna(0.0) >= min_confidence]

    st.caption(f"{len(filtered)} result(s)")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_settings():
    page_header("Settings", "Model configuration UI for the YOLO + VietOCR pipeline.")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.text_input("YOLO model path", value=os.path.join(BASE_DIR, "weights", "yolo_vintext", "best.pt"))
        st.slider("Detection confidence", 0.10, 0.95, 0.35, 0.05)
        st.caption("NgÆ°á»¡ng confidence tá»‘i thiá»ƒu khi YOLO phÃ¡t hiá»‡n vÃ¹ng chá»¯.")
        st.toggle("Merge bbox", value=False)
        st.caption("Gá»™p cÃ¡c box gáº§n nhau thÃ nh vÃ¹ng chá»¯ cáº¥p dÃ²ng trÆ°á»›c khi nháº­n dáº¡ng.")
        st.toggle("Use GPU", value=True)
        st.caption("Sá»­ dá»¥ng CUDA GPU khi pipeline model tháº­t Ä‘Æ°á»£c káº¿t ná»‘i.")
    with col_b:
        st.text_input(
            "VietOCR model path",
            value=os.path.join(BASE_DIR, "weights", "vietocr_vintext", "transformerocr.pth"),
        )
        st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
        st.caption("NgÆ°á»¡ng chá»“ng láº¥p dÃ¹ng khi lá»c hoáº·c gá»™p cÃ¡c box bá»‹ trÃ¹ng.")
        st.toggle("Sort bbox", value=True)
        st.caption("Sáº¯p xáº¿p box tá»« trÃªn xuá»‘ng dÆ°á»›i, trÃ¡i sang pháº£i trÆ°á»›c khi parser.")
    st.info("OCR Workspace now uses the real YOLO + VietOCR pipeline in Real model mode. Some advanced settings remain UI-only until the next tuning step.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_about():
    page_header("About", "Project overview for Smart Signboard OCR.")
    st.markdown(
        """
        <div class="card">
            <h3>Smart Signboard OCR</h3>
            <p>
                Smart Signboard OCR is a Vietnamese signboard recognition system designed to
                detect text regions, recognize signboard content, extract useful business
                information, and store searchable records in Excel.
            </p>
            <p><b>Technology stack:</b> YOLO, VietOCR, Streamlit, OpenCV, Excel.</p>
            <p>
                <b>Main pipeline:</b>
                Upload image -> YOLO detect text -> Post-process bbox -> VietOCR recognition
                -> Parser information -> Verify/edit -> Save Excel -> Search and review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    selected_page = sidebar_menu()

    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "OCR Workspace":
        render_ocr_workspace()
    elif selected_page == "Signboard Records":
        render_records()
    elif selected_page == "Search":
        render_search()
    elif selected_page == "Settings":
        render_settings()
    elif selected_page == "About":
        render_about()


if __name__ == "__main__":
    main()
