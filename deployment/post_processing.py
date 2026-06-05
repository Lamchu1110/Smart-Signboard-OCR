import os


CONFIDENCE_THRESHOLD = float(os.getenv("OCR_DET_CONF_THRESHOLD", "0.15"))
DUPLICATE_IOU_THRESHOLD = float(os.getenv("OCR_DUPLICATE_IOU_THRESHOLD", "0.85"))
LINE_CENTER_Y_RATIO = float(os.getenv("OCR_LINE_CENTER_Y_RATIO", "0.95"))
LINE_MIN_VERTICAL_OVERLAP = float(os.getenv("OCR_LINE_MIN_VERTICAL_OVERLAP", "0.15"))
MERGE_MAX_GAP_RATIO = float(os.getenv("OCR_MERGE_MAX_GAP_RATIO", "2.8"))
MERGE_MIN_GAP = float(os.getenv("OCR_MERGE_MIN_GAP", "34"))
BOX_PADDING_X_RATIO = float(os.getenv("OCR_BOX_PADDING_X_RATIO", "0.22"))
BOX_PADDING_Y_RATIO = float(os.getenv("OCR_BOX_PADDING_Y_RATIO", "0.16"))
BOX_MIN_PADDING = float(os.getenv("OCR_BOX_MIN_PADDING", "5"))
MIN_BOX_HEIGHT_RATIO = float(os.getenv("OCR_MIN_BOX_HEIGHT_RATIO", "0.006"))
MIN_BOX_WIDTH_RATIO = float(os.getenv("OCR_MIN_BOX_WIDTH_RATIO", "0.008"))
MAX_BOX_ASPECT_RATIO = float(os.getenv("OCR_MAX_BOX_ASPECT_RATIO", "28"))


def bbox_width(bbox):
    return max(float(bbox[2]) - float(bbox[0]), 1.0)


def bbox_height(bbox):
    return max(float(bbox[3]) - float(bbox[1]), 1.0)


def bbox_area(bbox):
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center_y(bbox):
    return (float(bbox[1]) + float(bbox[3])) / 2


def vertical_overlap_ratio(box_a, box_b):
    overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return overlap / max(min(bbox_height(box_a), bbox_height(box_b)), 1.0)


def bbox_iou(box_a, box_b):
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = bbox_area(box_a) + bbox_area(box_b) - inter_area
    return inter_area / max(union_area, 1.0)


def merge_bbox(box_a, box_b):
    return [
        min(float(box_a[0]), float(box_b[0])),
        min(float(box_a[1]), float(box_b[1])),
        max(float(box_a[2]), float(box_b[2])),
        max(float(box_a[3]), float(box_b[3])),
    ]


def clip_and_pad_bbox(bbox, image_width, image_height):
    height = bbox_height(bbox)
    pad_x = max(BOX_MIN_PADDING, height * BOX_PADDING_X_RATIO)
    pad_y = max(BOX_MIN_PADDING, height * BOX_PADDING_Y_RATIO)
    return [
        max(0.0, float(bbox[0]) - pad_x),
        max(0.0, float(bbox[1]) - pad_y),
        min(float(image_width), float(bbox[2]) + pad_x),
        min(float(image_height), float(bbox[3]) + pad_y),
    ]


def is_reasonable_text_box(bbox, image_width, image_height):
    width = bbox_width(bbox)
    height = bbox_height(bbox)
    min_width = max(4.0, float(image_width) * MIN_BOX_WIDTH_RATIO)
    min_height = max(4.0, float(image_height) * MIN_BOX_HEIGHT_RATIO)
    aspect_ratio = width / max(height, 1.0)
    return width >= min_width and height >= min_height and aspect_ratio <= MAX_BOX_ASPECT_RATIO


def normalize_detection(item, image_width, image_height):
    bbox = clip_and_pad_bbox(item["bbox"], image_width, image_height)
    return {
        "bbox": bbox,
        "class_name": item.get("class_name", "text"),
        "confidence": float(item.get("confidence", 0.0)),
        "count": int(item.get("count", 1)),
    }


def suppress_duplicate_boxes(detections):
    kept = []
    for item in sorted(detections, key=lambda value: value["confidence"], reverse=True):
        if any(bbox_iou(item["bbox"], kept_item["bbox"]) >= DUPLICATE_IOU_THRESHOLD for kept_item in kept):
            continue
        kept.append(item)
    return kept


def belongs_to_line(line_bbox, bbox):
    avg_height = (bbox_height(line_bbox) + bbox_height(bbox)) / 2
    center_y_diff = abs(bbox_center_y(line_bbox) - bbox_center_y(bbox))
    line_height_ratio = max(bbox_height(line_bbox), bbox_height(bbox)) / max(
        min(bbox_height(line_bbox), bbox_height(bbox)), 1.0
    )
    return (
        center_y_diff <= avg_height * LINE_CENTER_Y_RATIO
        or vertical_overlap_ratio(line_bbox, bbox) >= LINE_MIN_VERTICAL_OVERLAP
    ) and line_height_ratio <= 4.0


def group_into_lines(detections):
    lines = []
    for item in sorted(detections, key=lambda value: (value["bbox"][1], value["bbox"][0])):
        matching_line = None
        for line in lines:
            if belongs_to_line(line["bbox"], item["bbox"]):
                matching_line = line
                break

        if matching_line is None:
            lines.append({"bbox": item["bbox"][:], "items": [item]})
            continue

        matching_line["items"].append(item)
        matching_line["bbox"] = merge_bbox(matching_line["bbox"], item["bbox"])

    return sorted(lines, key=lambda value: (value["bbox"][1], value["bbox"][0]))


def should_merge_in_line(current_bbox, next_bbox):
    current_height = bbox_height(current_bbox)
    next_height = bbox_height(next_bbox)
    avg_height = (current_height + next_height) / 2
    horizontal_gap = float(next_bbox[0]) - float(current_bbox[2])
    max_gap = max(MERGE_MIN_GAP, avg_height * MERGE_MAX_GAP_RATIO)
    height_ratio = max(current_height, next_height) / max(min(current_height, next_height), 1.0)

    return (
        horizontal_gap <= max_gap
        and height_ratio <= 3.0
        and (
            vertical_overlap_ratio(current_bbox, next_bbox) >= LINE_MIN_VERTICAL_OVERLAP
            or abs(bbox_center_y(current_bbox) - bbox_center_y(next_bbox)) <= avg_height * LINE_CENTER_Y_RATIO
        )
    )


def merge_line_items(items):
    sorted_items = sorted(items, key=lambda value: value["bbox"][0])
    if not sorted_items:
        return []

    merged = []
    current = {
        "bbox": sorted_items[0]["bbox"][:],
        "class_name": sorted_items[0]["class_name"],
        "confidence": sorted_items[0]["confidence"],
        "count": sorted_items[0]["count"],
    }

    for item in sorted_items[1:]:
        if should_merge_in_line(current["bbox"], item["bbox"]):
            current["bbox"] = merge_bbox(current["bbox"], item["bbox"])
            current["confidence"] = max(current["confidence"], item["confidence"])
            current["count"] += item["count"]
        else:
            merged.append(current)
            current = {
                "bbox": item["bbox"][:],
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "count": item["count"],
            }

    merged.append(current)
    return merged


def post_process_text_detections(detections, image_size):
    image_width, image_height = image_size
    normalized = [
        normalize_detection(item, image_width, image_height)
        for item in detections
        if float(item.get("confidence", 0.0)) >= CONFIDENCE_THRESHOLD
        and is_reasonable_text_box(item["bbox"], image_width, image_height)
    ]
    deduped = suppress_duplicate_boxes(normalized)
    lines = group_into_lines(deduped)

    processed = []
    for line_index, line in enumerate(lines):
        for order_index, item in enumerate(merge_line_items(line["items"])):
            item["line_index"] = line_index
            item["order_index"] = order_index
            processed.append(item)

    return sorted(
        processed,
        key=lambda value: (value["line_index"], value["order_index"], value["bbox"][1], value["bbox"][0]),
    )
