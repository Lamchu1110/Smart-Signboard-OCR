import re
import unicodedata


PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")

CATEGORY_RULES = {
    "cafe": [
        ("ca phe", 3.0),
        ("cafe", 3.0),
        ("coffee", 2.5),
        ("tra sua", 3.0),
        ("sinh to", 2.0),
        ("nuoc ep", 2.0),
        ("giai khat", 1.8),
    ],
    "quan_an": [
        ("quan an", 3.0),
        ("nha hang", 3.0),
        ("pho", 2.7),
        ("bun", 2.4),
        ("com", 2.2),
        ("chao", 2.0),
        ("lau", 2.4),
        ("nuong", 2.3),
        ("banh mi", 2.3),
        ("hu tieu", 2.3),
        ("an vat", 2.0),
    ],
    "sua_xe": [
        ("sua xe", 3.2),
        ("rua xe", 2.8),
        ("xe may", 2.5),
        ("thay nhot", 2.5),
        ("va vo", 2.4),
        ("phu tung", 2.0),
    ],
    "spa": [
        ("spa", 3.0),
        ("massage", 2.8),
        ("tham my", 2.8),
        ("nail", 2.8),
        ("cham soc da", 2.5),
    ],
    "tiem_toc": [
        ("cat toc", 3.0),
        ("salon", 2.8),
        ("barber", 2.8),
        ("toc", 1.8),
        ("goi dau", 2.4),
        ("uon", 1.8),
        ("nhuom", 1.8),
    ],
    "nha_thuoc": [
        ("nha thuoc", 3.4),
        ("pharmacy", 3.0),
        ("duoc pham", 2.8),
        ("thuoc tay", 2.8),
    ],
    "dien_thoai": [
        ("dien thoai", 3.0),
        ("dien may", 2.8),
        ("sua dien thoai", 3.2),
        ("iphone", 2.0),
        ("laptop", 2.0),
        ("sim the", 2.0),
    ],
    "nha_khoa": [
        ("nha khoa", 3.4),
        ("dental", 3.0),
        ("rang ham mat", 2.8),
        ("rang", 1.8),
    ],
    "phong_kham": [
        ("phong kham", 3.4),
        ("clinic", 3.0),
        ("bac si", 2.5),
        ("kham benh", 2.5),
    ],
    "tap_hoa": [
        ("tap hoa", 3.0),
        ("bach hoa", 3.0),
        ("mini mart", 2.8),
        ("minimart", 2.8),
        ("sieu thi", 2.4),
    ],
    "karaoke": [
        ("karaoke", 3.2),
    ],
    "khach_san": [
        ("khach san", 3.2),
        ("nha nghi", 3.0),
        ("hotel", 2.8),
        ("motel", 2.8),
    ],
}

ADDRESS_HINTS = (
    "duong",
    "phuong",
    "quan",
    "huyen",
    "tinh",
    "tp",
    "thanh pho",
    "thi xa",
    "thi tran",
    "xa",
    "so",
    "street",
    "ward",
    "district",
    "city",
    "province",
)

STRONG_ADDRESS_HINTS = tuple(hint for hint in ADDRESS_HINTS if hint != "quan")

BUSINESS_STOPWORDS = {
    "wifi",
    "hotline",
    "phone",
    "tel",
    "open",
    "24h",
    "sale",
    "khuyen mai",
    "giam gia",
    "lien he",
    "ship",
    "free",
}

GENERIC_BUSINESS_PREFIXES = {
    "quan",
    "tiem",
    "cua hang",
    "shop",
    "nha",
    "cong ty",
    "cty",
}


def normalize_spaces(text):
    return " ".join(str(text or "").strip().split())


def strip_accents(text):
    text = str(text or "")
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_key_text(text):
    value = strip_accents(normalize_spaces(text)).lower()
    for source in (":", "-", "_", "|", "/", "\\", ".", ",", ";", "(", ")", "[", "]"):
        value = value.replace(source, " ")
    return normalize_spaces(value)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_bbox(bbox):
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [safe_float(value) for value in bbox[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, max(x2, x1 + 1.0), max(y2, y1 + 1.0)]


def bbox_features(bbox):
    x1, y1, x2, y2 = normalize_bbox(bbox)
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
        "width": width,
        "height": height,
        "area": width * height,
    }


def union_bbox(items):
    boxes = [normalize_bbox(item["bbox"]) for item in items if item.get("bbox")]
    if not boxes:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def prediction_to_words(predictions):
    words = []
    for prediction in predictions:
        if len(prediction) < 4:
            continue
        bbox, _class_name, confidence, text = prediction[:4]
        text = normalize_spaces(text)
        if not text:
            continue
        features = bbox_features(bbox)
        words.append(
            {
                **features,
                "confidence": safe_float(confidence),
                "text": text,
                "normalized": normalize_key_text(text),
            }
        )
    return sorted(words, key=lambda item: (item["y"], item["x"]))


def same_visual_line(word, line):
    overlap = min(word["y2"], line["y2"]) - max(word["y"], line["y"])
    overlap_ratio = overlap / max(min(word["height"], line["height"]), 1.0)
    center_delta = abs(word["cy"] - line["cy"])
    max_height = max(word["height"], line["height"], 1.0)
    return overlap_ratio >= 0.35 or center_delta <= max_height * 0.65


def rebuild_line(line):
    words = sorted(line["words"], key=lambda item: item["x"])
    bbox = union_bbox(words)
    features = bbox_features(bbox)
    text = normalize_spaces(" ".join(word["text"] for word in words))
    confidence = sum(word["confidence"] for word in words) / len(words)
    line.update(
        {
            **features,
            "text": text,
            "normalized": normalize_key_text(text),
            "confidence": confidence,
            "word_count": len(words),
        }
    )
    return line


def predictions_to_lines(predictions):
    words = prediction_to_words(predictions)
    lines = []

    for word in words:
        candidates = []
        for index, line in enumerate(lines):
            if same_visual_line(word, line):
                candidates.append((abs(word["cy"] - line["cy"]), index))
        if candidates:
            _, line_index = min(candidates)
            lines[line_index]["words"].append(word)
            rebuild_line(lines[line_index])
        else:
            lines.append(rebuild_line({"words": [word]}))

    lines = sorted(lines, key=lambda item: (item["y"], item["x"]))
    for index, line in enumerate(lines):
        line["line_index"] = index
    return lines


def normalize_phone(phone):
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("84") and len(digits) in (11, 12):
        return "0" + digits[2:]
    return digits


def find_phone(lines):
    candidates = []
    for index, line in enumerate(lines):
        for match in PHONE_PATTERN.finditer(line["text"]):
            phone = normalize_phone(match.group(0))
            if 9 <= len(phone) <= 11:
                candidates.append((line["confidence"], index, phone))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def contains_keyword(normalized_text, keyword):
    keyword = normalize_key_text(keyword)
    if not keyword:
        return False
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def find_category_hint(lines):
    normalized_blob = normalize_key_text("\n".join(line["text"] for line in lines))
    scores = {}
    evidence = {}

    for category, rules in CATEGORY_RULES.items():
        category_score = 0.0
        category_evidence = []
        for keyword, weight in rules:
            if contains_keyword(normalized_blob, keyword):
                category_score += weight
                category_evidence.append(keyword)
        if category_score:
            scores[category] = round(category_score, 3)
            evidence[category] = category_evidence

    if not scores:
        return {
            "category": "",
            "confidence": 0.0,
            "evidence": [],
            "scores": {},
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 2.0 or (best_score < 3.0 and best_score - second_score < 0.75):
        return {
            "category": "",
            "confidence": round(min(best_score / 5.0, 0.55), 3),
            "evidence": evidence.get(best_category, []),
            "scores": scores,
        }

    confidence = min(0.98, 0.45 + best_score / 8.0)
    if second_score:
        confidence = min(confidence, 0.75 + max(best_score - second_score, 0) / 8.0)

    return {
        "category": best_category,
        "confidence": round(confidence, 3),
        "evidence": evidence.get(best_category, []),
        "scores": scores,
    }


def find_address_hint(lines):
    candidates = []
    for index, line in enumerate(lines):
        has_address_keyword = has_address_signal(line)
        starts_with_number = re.search(r"^\s*\d+[\w\/\-]*\s+", line["text"]) is not None
        if has_address_keyword or starts_with_number:
            score = line["confidence"] + min(len(line["text"]), 80) / 80
            if starts_with_number:
                score += 0.15
            candidates.append((score, index, line["text"]))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def has_address_signal(line):
    normalized = line["normalized"]
    if any(contains_keyword(normalized, hint) for hint in STRONG_ADDRESS_HINTS):
        return True
    if re.search(r"(?<!\w)(?:q|quan)\s*\d{1,2}(?!\w)", normalized):
        return True
    if re.search(r"(?<!\w)quan\s+(?:binh|tan|go|thu|cau|ba|hai|hoan|dong|nam|bac)", normalized):
        return True
    return False


def is_phone_or_address(line):
    return PHONE_PATTERN.search(line["text"]) or has_address_signal(line)


def is_noise_line(line):
    text = line["text"]
    normalized = line["normalized"]
    if PHONE_PATTERN.search(text):
        return True
    if len(normalized) < 2:
        return True
    if any(contains_keyword(normalized, stopword) for stopword in BUSINESS_STOPWORDS):
        return True
    if has_address_signal(line):
        return True
    if sum(char.isalpha() for char in text) < 2:
        return True
    return False


def line_has_brand_signal(line):
    normalized = line["normalized"]
    tokens = [token for token in normalized.split() if len(token) >= 2]
    generic_tokens = set()
    for prefix in GENERIC_BUSINESS_PREFIXES:
        generic_tokens.update(normalize_key_text(prefix).split())
    category_tokens = set()
    for rules in CATEGORY_RULES.values():
        for keyword, _weight in rules:
            category_tokens.update(normalize_key_text(keyword).split())
    meaningful = [token for token in tokens if token not in generic_tokens and token not in category_tokens]
    return len(meaningful) >= 1


def store_name_score(line, max_area, max_height, image_width, image_height, category_info):
    text = line["text"]
    normalized = line["normalized"]
    alpha_count = sum(char.isalpha() for char in text)

    area_score = min(line["area"] / max(max_area, 1.0), 1.0) * 1.2
    height_score = min(line["height"] / max(max_height, 1.0), 1.0) * 0.8
    width_score = min(line["width"] / max(image_width, 1.0), 1.0) * 0.55
    top_score = max(0.0, 1.0 - line["y"] / max(image_height, 1.0)) * 0.45
    length_score = min(alpha_count / 24.0, 1.0) * 0.55
    confidence_score = min(max(line["confidence"], 0.0), 1.0) * 0.35
    uppercase_bonus = 0.18 if text.upper() == text and alpha_count >= 3 else 0.0
    brand_bonus = 0.18 if line_has_brand_signal(line) else 0.0

    penalty = 0.0
    if is_phone_or_address(line):
        penalty += 1.8
    if any(contains_keyword(normalized, stopword) for stopword in BUSINESS_STOPWORDS):
        penalty += 0.8
    if category_info["category"] and not line_has_brand_signal(line):
        penalty += 0.45
    if alpha_count <= 3:
        penalty += 0.6

    score = (
        area_score
        + height_score
        + width_score
        + top_score
        + length_score
        + confidence_score
        + uppercase_bonus
        + brand_bonus
        - penalty
    )
    reasons = {
        "area": round(area_score, 3),
        "height": round(height_score, 3),
        "width": round(width_score, 3),
        "top_position": round(top_score, 3),
        "text_length": round(length_score, 3),
        "detection_confidence": round(confidence_score, 3),
        "uppercase_bonus": round(uppercase_bonus, 3),
        "brand_signal": round(brand_bonus, 3),
        "penalty": round(penalty, 3),
    }
    return round(score, 3), reasons


def find_business_name(lines, category_info):
    if not lines:
        return {
            "text": "",
            "bbox": [],
            "score": 0.0,
            "reason": {},
            "candidates": [],
        }

    max_area = max(line["area"] for line in lines)
    max_height = max(line["height"] for line in lines)
    image_width = max(line["x2"] for line in lines)
    image_height = max(line["y2"] for line in lines)
    candidates = []

    for line in lines:
        if is_noise_line(line):
            continue
        score, reason = store_name_score(line, max_area, max_height, image_width, image_height, category_info)
        candidates.append(
            {
                "text": line["text"],
                "bbox": [round(value, 2) for value in line["bbox"]],
                "score": score,
                "reason": reason,
                "line_index": line["line_index"],
            }
        )

    if not candidates:
        return {
            "text": "",
            "bbox": [],
            "score": 0.0,
            "reason": {},
            "candidates": [],
        }

    candidates.sort(key=lambda item: (-item["score"], item["line_index"]))
    best = candidates[0]
    return {
        "text": best["text"],
        "bbox": best["bbox"],
        "score": best["score"],
        "reason": best["reason"],
        "candidates": candidates[:5],
    }


def line_to_debug_item(line):
    return {
        "line_index": line["line_index"],
        "text": line["text"],
        "bbox": [round(value, 2) for value in line["bbox"]],
        "confidence": round(line["confidence"], 4),
        "word_count": line["word_count"],
        "area": round(line["area"], 2),
    }


def parse_signboard_fields(predictions):
    lines = predictions_to_lines(predictions)
    raw_lines = [line["text"] for line in lines]
    category_info = find_category_hint(lines)
    store_info = find_business_name(lines, category_info)

    parsed = {
        "business_name": store_info["text"],
        "store_name_bbox": store_info["bbox"],
        "store_name_score": store_info["score"],
        "store_name_reason": store_info["reason"],
        "store_name_candidates": store_info["candidates"],
        "category_hint": category_info["category"],
        "category_confidence": category_info["confidence"],
        "category_evidence": category_info["evidence"],
        "category_scores": category_info["scores"],
        "phone": find_phone(lines),
        "address_hint": find_address_hint(lines),
        "raw_text": "\n".join(raw_lines),
        "lines": raw_lines,
        "line_items": [line_to_debug_item(line) for line in lines],
        "review_status": "needs_review",
    }
    parsed["missing_fields"] = [
        field
        for field in ("business_name", "category_hint", "phone", "address_hint")
        if not parsed[field]
    ]
    return parsed
