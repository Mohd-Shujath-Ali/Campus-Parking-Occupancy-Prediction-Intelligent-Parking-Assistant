"""
LPU Campus Parking - Dual YOLO Detection & ROI Analysis Engine
Supports:
  1. Custom Fine-Tuned Model: best.pt (class 0 = 'Car')
  2. Pretrained Large Model: yolo26l.pt (vehicle classes: car, motorcycle, bus, truck)
  3. Interactive Region of Interest (ROI) filtering via ROIManager
  4. Model benchmarking and side-by-side comparative analytics
"""

import os
import io
import time
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

# Set environment variables to avoid PyTorch inductor/temp permission issues
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), 'torch_inductor')
os.environ['TORCH_HOME'] = os.path.join(tempfile.gettempdir(), 'torch')
os.environ['YOLO_CONFIG_DIR'] = os.path.join(tempfile.gettempdir(), 'Ultralytics')

from computer_vision.roi_manager import ROIManager

logger = logging.getLogger(__name__)

# Model candidate paths — priority order: env var → models/weights/ → collab backup → cwd fallback
BEST_CANDIDATE_PATHS = [
    Path(os.environ.get("BEST_MODEL_PATH", "")),
    Path("models/weights/best.pt").resolve(),
    Path("collab_content_backup/runs/detect/train/weights/best.pt").resolve(),
    Path("best.pt").resolve(),
]

YOLO26L_CANDIDATE_PATHS = [
    Path(os.environ.get("YOLO26L_MODEL_PATH", "")),
    Path("models/weights/yolo26l.pt").resolve(),
    Path("collab_content_backup/yolo26l.pt").resolve(),
    Path("yolo26l.pt").resolve(),
]

BEST_WEIGHTS_PATH = next((p for p in BEST_CANDIDATE_PATHS if p.name == "best.pt" and p.exists()), Path("models/weights/best.pt"))
YOLO26L_WEIGHTS_PATH = next((p for p in YOLO26L_CANDIDATE_PATHS if p.name == "yolo26l.pt" and p.exists()), Path("models/weights/yolo26l.pt"))

# Inference parameters
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_MAX_SIZE = 1280

# Class filters
BEST_CAR_CLASS = [0]                 # best.pt class 0 = 'Car'
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # yolo26l: 2=car, 3=motorcycle, 5=bus, 7=truck

# Global models cached in memory
_model_best = None
_model_yolo26l = None
_load_errors = {}


def _get_best_model():
    global _model_best, _load_errors
    if _model_best is not None:
        return _model_best
    try:
        from ultralytics import YOLO
        logger.info(f"Loading custom fine-tuned model: {BEST_WEIGHTS_PATH}")
        _model_best = YOLO(str(BEST_WEIGHTS_PATH))
        return _model_best
    except Exception as e:
        _load_errors["best"] = str(e)
        logger.error(f"Failed to load best.pt: {e}")
        return None


def _get_yolo26l_model():
    global _model_yolo26l, _load_errors
    if _model_yolo26l is not None:
        return _model_yolo26l
    try:
        from ultralytics import YOLO
        logger.info(f"Loading pretrained yolo26l model: {YOLO26L_WEIGHTS_PATH}")
        _model_yolo26l = YOLO(str(YOLO26L_WEIGHTS_PATH))
        return _model_yolo26l
    except Exception as e:
        _load_errors["yolo26l"] = str(e)
        logger.error(f"Failed to load yolo26l.pt: {e}")
        return None


def _resize_keep_aspect(img: np.ndarray, max_size: int = IMG_MAX_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _img_to_base64(img: np.ndarray, quality: int = 85) -> str:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", img, encode_params)
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def _draw_model_detections(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    model_title: str,
    roi_polygon: Optional[List[List[int]]] = None
) -> np.ndarray:
    """
    Renders bounding boxes, labels, and ROI boundaries on a copy of img.
    """
    annotated = img.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Draw ROI if provided
    has_roi = roi_polygon and len(roi_polygon) >= 3
    if has_roi:
        pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
        
        # Semi-transparent overlay for ROI zone
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], (255, 230, 0)) # Cyan/yellow tint in BGR
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        
        # Glowing border around ROI
        cv2.polylines(annotated, [pts], isClosed=True, color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # ROI Label Tag
        rx, ry = pts[0][0]
        cv2.putText(annotated, "SELECTED ROI ZONE", (max(10, rx), max(25, ry - 8)),
                    font, 0.55, (255, 240, 0), 2, cv2.LINE_AA)

    # 2. Draw Detections
    roi_count = 0
    total_count = len(detections)

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        cname = det["class_name"]
        in_roi = det.get("in_roi", True)

        if in_roi:
            roi_count += 1
            # Bright emerald green for ROI detections
            box_color = (0, 220, 80)
            tag = " [ROI]" if has_roi else ""
            line_w = 2
        else:
            # Subtle muted color for outside detections
            box_color = (160, 160, 160)
            tag = " [OUT]"
            line_w = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_w, cv2.LINE_AA)
        label = f"{cname} {conf:.0%}{tag}"
        (lw, lh), baseline = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - lh - baseline - 4)), (x1 + lw + 2, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 1, max(lh + 2, y1 - baseline - 2)),
                    font, 0.45, (0, 0, 0) if in_roi else (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Top Header Banner
    header_h = 42
    cv2.rectangle(annotated, (0, 0), (w, header_h), (20, 24, 30), -1)
    
    title_text = f"{model_title}"
    cv2.putText(annotated, title_text, (12, 26), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    if has_roi:
        stat_text = f"In ROI: {roi_count} | Total: {total_count}"
        cv2.putText(annotated, stat_text, (w - 240, 26), font, 0.6, (0, 240, 100), 2, cv2.LINE_AA)
    else:
        stat_text = f"Detected: {total_count} Cars"
        cv2.putText(annotated, stat_text, (w - 200, 26), font, 0.6, (0, 240, 100), 2, cv2.LINE_AA)

    return annotated


def assess_image_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates image quality metrics to diagnose camera degradation before inference:
    detects extreme underexposure (night/lighting failure), glare/overexposure, blur,
    and low contrast (fog/smudged lens).
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"is_degraded": True, "reasons": ["Empty or invalid image frame"]}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []
    if mean_val < 35.0:
        reasons.append("Severe underexposure / night scene without IR illumination")
    elif mean_val > 225.0:
        reasons.append("Severe glare / sensor overexposure")

    if laplacian_var < 50.0:
        reasons.append(f"Significant image blur / defocus (Laplacian variance = {laplacian_var:.1f})")

    if std_val < 18.0:
        reasons.append(f"Low contrast / dense fog / obstructed lens (std = {std_val:.1f})")

    return {
        "is_degraded": len(reasons) > 0,
        "reasons": reasons,
        "mean_brightness": round(mean_val, 1),
        "contrast_std": round(std_val, 1),
        "blur_laplacian_var": round(laplacian_var, 1),
    }


def _run_single_model(
    model,
    img: np.ndarray,
    model_name: str,
    classes: Optional[List[int]] = None,
    roi_polygon: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Executes inference for a single model and filters via ROI.
    """
    quality = assess_image_quality(img)
    if model is None:
        return {
            "model_name": model_name,
            "vehicle_count_total": 0,
            "vehicle_count_roi": 0,
            "vehicle_count_outside": 0,
            "avg_confidence": 0.0,
            "inference_time_ms": 0.0,
            "detections": [],
            "annotated_image_base64": "",
            "image_quality": quality,
            "error": _load_errors.get(model_name.lower(), "Model not loaded")
        }

    t0 = time.perf_counter()
    predict_kwargs = {
        "source": img,
        "conf": CONFIDENCE_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "verbose": False,
        "stream": False
    }
    if classes is not None:
        predict_kwargs["classes"] = classes

    results = model.predict(**predict_kwargs)
    inf_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Parse detections
    raw_detections = []
    has_roi = roi_polygon is not None and len(roi_polygon) >= 3
    roi_mgr = ROIManager(roi_polygon) if has_roi else None

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names or {}
        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cname = names.get(cls_id, f"cls_{cls_id}")

            # Determine whether vehicle is in ROI
            if roi_mgr:
                contact_pt = roi_mgr.get_ground_contact_point([x1, y1, x2, y2])
                in_roi = roi_mgr.is_inside_roi(contact_pt)
            else:
                in_roi = True

            raw_detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(conf, 4),
                "class_name": cname,
                "class_id": cls_id,
                "in_roi": in_roi
            })

    roi_dets = [d for d in raw_detections if d["in_roi"]]
    out_dets = [d for d in raw_detections if not d["in_roi"]]
    avg_conf = round(float(np.mean([d["confidence"] for d in raw_detections])), 3) if raw_detections else 0.0

    # Draw annotated image
    annotated = _draw_model_detections(img, raw_detections, model_name, roi_polygon)
    b64_img = _img_to_base64(annotated)

    return {
        "model_name": model_name,
        "vehicle_count_total": len(raw_detections),
        "vehicle_count_roi": len(roi_dets),
        "vehicle_count_outside": len(out_dets),
        "avg_confidence": avg_conf,
        "inference_time_ms": inf_time_ms,
        "detections": raw_detections,
        "detections_roi": roi_dets,
        "detections_outside": out_dets,
        "annotated_image_base64": b64_img,
        "image_quality": quality,
        "error": None
    }


def compare_models_from_bytes(
    image_bytes: bytes,
    roi_polygon: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Runs both best.pt and yolo26l.pt on the provided image and returns
    full detection results and comparative analysis.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "error": "Could not decode image bytes. Please ensure it is a valid JPEG/PNG."
        }

    img = _resize_keep_aspect(img)

    # 1. Model A: Fine-Tuned Custom LPU Model (best.pt)
    m_best = _get_best_model()
    res_best = _run_single_model(
        m_best, img,
        model_name="Custom LPU Fine-Tuned (best.pt)",
        classes=BEST_CAR_CLASS,
        roi_polygon=roi_polygon
    )

    # 2. Model B: Pretrained Large Model (yolo26l.pt)
    m_yolo26l = _get_yolo26l_model()
    res_yolo26l = _run_single_model(
        m_yolo26l, img,
        model_name="Pretrained YOLO26 Large (yolo26l.pt)",
        classes=COCO_VEHICLE_CLASSES,
        roi_polygon=roi_polygon
    )

    # 3. Compute Comparative Metrics
    has_roi = roi_polygon is not None and len(roi_polygon) >= 3
    count_key = "vehicle_count_roi" if has_roi else "vehicle_count_total"

    count_best = res_best[count_key]
    count_yolo26l = res_yolo26l[count_key]
    count_delta = count_best - count_yolo26l

    time_best = res_best["inference_time_ms"]
    time_yolo26l = res_yolo26l["inference_time_ms"]

    if time_best > 0 and time_yolo26l > 0:
        if time_best <= time_yolo26l:
            speed_text = f"best.pt is {round(time_yolo26l / time_best, 1)}x faster ({time_best}ms vs {time_yolo26l}ms)"
        else:
            speed_text = f"yolo26l.pt is {round(time_best / time_yolo26l, 1)}x faster ({time_yolo26l}ms vs {time_best}ms)"
    else:
        speed_text = "Latency measured"

    analysis_bullets = [
        f"Custom model detected {count_best} vehicles in target region (avg conf: {res_best['avg_confidence']:.1%}).",
        f"Pretrained baseline detected {count_yolo26l} vehicles (avg conf: {res_yolo26l['avg_confidence']:.1%}).",
        f"Speed performance: {speed_text}."
    ]

    if count_best > count_yolo26l:
        analysis_bullets.append(
            f"Fine-tuned best.pt captured {count_delta} additional vehicle(s) that general pretrained weights missed due to domain adaptation on campus parking layouts."
        )
    elif count_best == count_yolo26l:
        analysis_bullets.append(
            "Both models achieved consistent vehicle counts in the region of interest."
        )

    return {
        "roi_defined": has_roi,
        "roi_polygon": roi_polygon,
        "best_model": res_best,
        "yolo26l_model": res_yolo26l,
        "comparison": {
            "target_region": "Selected ROI" if has_roi else "Full Lot",
            "best_count": count_best,
            "yolo26l_count": count_yolo26l,
            "count_delta": count_delta,
            "best_latency_ms": time_best,
            "yolo26l_latency_ms": time_yolo26l,
            "speed_summary": speed_text,
            "analysis_bullets": analysis_bullets
        }
    }


def detect_from_bytes(
    image_bytes: bytes,
    roi_polygon: Optional[List[List[int]]] = None,
    model_choice: str = "both"
) -> Dict[str, Any]:
    """
    Public API wrapper. If model_choice is 'both', returns full comparison.
    If 'best', returns best.pt results. If 'yolo26l', returns yolo26l.pt results.
    """
    if model_choice == "both":
        return compare_models_from_bytes(image_bytes, roi_polygon)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}
    img = _resize_keep_aspect(img)

    if model_choice == "yolo26l":
        m = _get_yolo26l_model()
        return _run_single_model(m, img, "Pretrained YOLO26 Large (yolo26l.pt)", COCO_VEHICLE_CLASSES, roi_polygon)
    else:
        m = _get_best_model()
        return _run_single_model(m, img, "Custom LPU Fine-Tuned (best.pt)", BEST_CAR_CLASS, roi_polygon)


def model_status() -> dict:
    """Return status of both models."""
    return {
        "best_model": {
            "loaded": _model_best is not None,
            "weights_path": str(BEST_WEIGHTS_PATH),
            "weights_exist": BEST_WEIGHTS_PATH.exists()
        },
        "yolo26l_model": {
            "loaded": _model_yolo26l is not None,
            "weights_path": str(YOLO26L_WEIGHTS_PATH),
            "weights_exist": YOLO26L_WEIGHTS_PATH.exists()
        }
    }
