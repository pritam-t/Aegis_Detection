import cv2
import re
import numpy as np
import os
import json
import time
import queue
import threading
import torch
import easyocr
from ultralytics import YOLO
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = "best.pt"
INPUT_VIDEO  = "resources/test5.mp4"
REPORT_PATH  = "aegis_report.json"
PLATES_DIR   = "aegis_plates"
REVIEW_DIR   = "aegis_review"

# Detection confidence thresholds
HELMET_OFF_CONF = 0.35
PLATE_CONF_MIN  = 0.40
HARD_NEG_CONF   = 0.70

# Strict Indian plate: 2L + 2N + 1-3L + 4N  e.g. MH03EJ7565, DL1CAB1234
PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

# Pipeline
FRAME_SKIP       = 1
CAPTURE_QUEUE_SZ = 8
RESULT_QUEUE_SZ  = 4

# Frame enhancement — applied to whole frame for YOLO inference ONLY
# The raw (unenhanced) frame is always used for plate crops sent to OCR
ENABLE_FRAME_ENHANCE = True
ENABLE_DEBLUR        = True
ENABLE_GAMMA         = True
GAMMA_VALUE          = 1.2

# Tracking
USE_TRACKER = True
TRACKER_CFG = "bytetrack.yaml"

# Rider lifecycle
RIDER_TIMEOUT    = 60
HIGH_CONF_COMMIT = 0.85
MAX_OCR_ATTEMPTS = 8

# Fuzzy dedup
FUZZY_THRESHOLD = 3

DEBUG_OCR = False

# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════════════
_ocr_reader  = None
_gamma_table = None


# ══════════════════════════════════════════════════════════════════════════════
# PLATE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def is_valid_plate(text: str) -> bool:
    return bool(PLATE_RE.match(text.strip().upper()))


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY DEDUP
# ══════════════════════════════════════════════════════════════════════════════
def edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > FUZZY_THRESHOLD:
        return FUZZY_THRESHOLD + 1
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def find_matching_plate(plate_text: str, vehicles: dict):
    for key in vehicles:
        if edit_distance(plate_text, key) <= FUZZY_THRESHOLD:
            return key
    return None


# ══════════════════════════════════════════════════════════════════════════════
# IOU
# ══════════════════════════════════════════════════════════════════════════════
def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


# ══════════════════════════════════════════════════════════════════════════════
# GAMMA LUT
# ══════════════════════════════════════════════════════════════════════════════
def _build_gamma_table(gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# FRAME ENHANCEMENT — for YOLO inference only, never used on plate crops
# ══════════════════════════════════════════════════════════════════════════════
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    global _gamma_table
    if ENABLE_GAMMA and _gamma_table is not None:
        frame = cv2.LUT(frame, _gamma_table)
    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if ENABLE_DEBLUR:
        blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=2)
        frame = cv2.addWeighted(frame, 1.4, blur, -0.4, 0)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# EASYOCR INIT
# ══════════════════════════════════════════════════════════════════════════════
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        print(f"[AEGIS] EasyOCR ready (gpu={gpu})")
    return _ocr_reader


# ══════════════════════════════════════════════════════════════════════════════
# PLATE CROP PREPROCESSING
# Applied to the RAW unenhanced crop only — no double processing.
# Mirrors the original Tesseract pipeline: upscale → grayscale → threshold.
# EasyOCR reads clean binarised text far better than over-processed BGR.
# ══════════════════════════════════════════════════════════════════════════════
def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    # Upscale — OCR accuracy rises sharply above ~150px height
    scale = max(2, 150 // max(h, 1))
    crop  = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Grayscale + light blur to suppress jpeg noise
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarisation — works well for clear contrast plates
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold — handles uneven lighting / shadows
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )

    # Pick whichever has more contrast (higher stddev = cleaner binarisation)
    best = otsu if np.std(otsu) >= np.std(adapt) else adapt

    return cv2.merge([best, best, best])


# ══════════════════════════════════════════════════════════════════════════════
# OCR CORE — single image
# ══════════════════════════════════════════════════════════════════════════════
def _run_ocr_on_image(crop: np.ndarray) -> str:
    preprocessed = _preprocess_plate_crop(crop)
    reader       = get_ocr_reader()

    raw_results = reader.readtext(
        preprocessed,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        detail=1,
        paragraph=False,
        text_threshold=0.4,
        low_text=0.3,
    )

    if DEBUG_OCR:
        print(f"    [OCR] raw={[(r[1], round(r[2], 2)) for r in raw_results]}")

    if not raw_results:
        return ""

    # Sort top-to-bottom so two-line plates concatenate in correct order
    raw_results.sort(key=lambda r: r[0][0][1])

    # Attempt 1: full concatenation (catches two-line plates)
    concat = "".join(
        "".join(c for c in r[1] if c.isalnum()).upper()
        for r in raw_results
    )
    if DEBUG_OCR:
        print(f"    [OCR] concat='{concat}'  valid={is_valid_plate(concat)}")
    if is_valid_plate(concat):
        return concat

    # Attempt 2: best-confidence single segment
    for _, text, _ in sorted(raw_results, key=lambda r: r[2], reverse=True):
        cleaned = "".join(c for c in text if c.isalnum()).upper()
        if is_valid_plate(cleaned):
            if DEBUG_OCR:
                print(f"    [OCR] single='{cleaned}'")
            return cleaned

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# OCR WITH ROTATION RETRY
# ══════════════════════════════════════════════════════════════════════════════
def ocr_plate_crop(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""

    result = _run_ocr_on_image(crop)
    if result:
        return result

    if DEBUG_OCR:
        print("    [OCR] retrying with 90° rotation")
    rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return _run_ocr_on_image(rotated) or ""


# ══════════════════════════════════════════════════════════════════════════════
# RIDER RECORD
# ══════════════════════════════════════════════════════════════════════════════
class RiderRecord:
    __slots__ = (
        "track_key",
        "best_helmet_conf", "best_helmet_frame",
        "best_plate_conf",  "best_plate_crop",
        "last_seen_frame",
        "ocr_attempts",
        "confirmed_plate",
    )

    def __init__(self, track_key):
        self.track_key         = track_key
        self.best_helmet_conf  = 0.0
        self.best_helmet_frame = None
        self.best_plate_conf   = 0.0
        self.best_plate_crop   = None
        self.last_seen_frame   = 0
        self.ocr_attempts      = 0
        self.confirmed_plate   = None

    def update(self, helmet_conf: float, helmet_frame: np.ndarray,
               plate_conf: float, plate_crop: np.ndarray, frame_no: int):
        if helmet_conf > self.best_helmet_conf:
            self.best_helmet_conf  = helmet_conf
            self.best_helmet_frame = helmet_frame.copy()
        if plate_conf > self.best_plate_conf:
            self.best_plate_conf = plate_conf
            self.best_plate_crop = plate_crop.copy()
        self.last_seen_frame = frame_no

    def ready_for_ocr(self) -> bool:
        return (
            self.best_plate_crop   is not None and
            self.best_helmet_frame is not None
        )


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 1 — CAPTURE
# ══════════════════════════════════════════════════════════════════════════════
def capture_thread(cap: cv2.VideoCapture,
                   frame_queue: queue.Queue,
                   stop_event: threading.Event):
    frame_no = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if FRAME_SKIP > 1 and frame_no % FRAME_SKIP != 0:
            continue
        frame_queue.put((frame_no, frame))
    frame_queue.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 2 — INFERENCE
# Passes both raw_frame (for cropping) and runs YOLO on enhanced frame.
# The bounding box coordinates come from the enhanced frame but are valid
# for the raw frame too since both have identical dimensions.
# ══════════════════════════════════════════════════════════════════════════════
def inference_thread(model: YOLO,
                     names: dict,
                     frame_queue: queue.Queue,
                     result_queue: queue.Queue,
                     use_fp16: bool):
    while True:
        item = frame_queue.get()
        if item is None:
            result_queue.put(None)
            break

        frame_no, raw_frame = item

        # Enhanced frame used for YOLO only — raw_frame passed downstream for crops
        inf_frame = enhance_frame(raw_frame) if ENABLE_FRAME_ENHANCE else raw_frame

        if USE_TRACKER:
            results = model.track(
                inf_frame, verbose=False, persist=True,
                tracker=TRACKER_CFG, half=use_fp16
            )[0]
        else:
            results = model(inf_frame, verbose=False, half=use_fp16)[0]

        helmet_off_boxes = []
        plate_boxes      = []
        rider_boxes      = []

        for box in results.boxes:
            cls      = int(box.cls[0])
            conf     = float(box.conf[0])
            label    = names[cls]
            coords   = tuple(map(int, box.xyxy[0]))
            track_id = int(box.id[0]) if (USE_TRACKER and box.id is not None) else None

            if label == "helmet-off" and conf >= HELMET_OFF_CONF:
                helmet_off_boxes.append((*coords, conf, track_id))
            elif label == "numer_plate" and conf >= PLATE_CONF_MIN:
                plate_boxes.append((*coords, conf, track_id))
            elif label == "rider":
                rider_boxes.append((*coords, conf, track_id))

        result_queue.put({
            "frame_no":   frame_no,
            "raw_frame":  raw_frame,    # unenhanced — used for plate crops
            "helmet_off": helmet_off_boxes,
            "plates":     plate_boxes,
            "riders":     rider_boxes,
        })


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 3 — WRITER
# ══════════════════════════════════════════════════════════════════════════════
def writer_thread(result_queue: queue.Queue,
                  vehicles: dict,
                  fps: float,
                  total_frames: int,
                  counters: dict):

    active_riders: dict = {}

    def _commit_rider(record: RiderRecord, frame_no: int):
        if not record.ready_for_ocr():
            return

        plate_text = (record.confirmed_plate
                      or ocr_plate_crop(record.best_plate_crop))

        if not plate_text:
            counters["noise"] += 1
            if record.best_plate_conf >= HARD_NEG_CONF:
                counters["hard_neg"] += 1
                hn_path = os.path.join(
                    REVIEW_DIR,
                    f"hn_track{record.track_key}_f{frame_no}"
                    f"_pc{int(record.best_plate_conf * 100)}.jpg"
                )
                cv2.imwrite(hn_path, record.best_plate_crop)
            return

        _register_violation(plate_text, record, frame_no)

    def _register_violation(plate_text: str, record: RiderRecord, frame_no: int):
        match_key = find_matching_plate(plate_text, vehicles)

        if match_key is None:
            img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
            frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
            cv2.imwrite(img_path,   record.best_plate_crop)
            cv2.imwrite(frame_path, record.best_helmet_frame)
            ts = round(frame_no / fps, 2)
            vehicles[plate_text] = {
                "plate_text":      plate_text,
                "helmet_off_conf": round(record.best_helmet_conf, 4),
                "plate_conf":      round(record.best_plate_conf,  4),
                "frame":           frame_no,
                "timestamp_sec":   ts,
                "plate_image":     img_path,
                "frame_image":     frame_path,
            }
            print(f"  🚨 VIOLATION | Frame {frame_no:>5} ({ts:.1f}s) | "
                  f"Plate: {plate_text:<14} "
                  f"helmet={record.best_helmet_conf:.2%}  "
                  f"plate={record.best_plate_conf:.2%}")
        else:
            existing     = vehicles[match_key]
            new_combined = record.best_helmet_conf + record.best_plate_conf
            old_combined = existing["helmet_off_conf"] + existing["plate_conf"]

            if new_combined > old_combined:
                old_text   = existing["plate_text"]
                img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
                frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
                cv2.imwrite(img_path,   record.best_plate_crop)
                cv2.imwrite(frame_path, record.best_helmet_frame)

                if old_text != plate_text:
                    for old_f in [existing["plate_image"], existing["frame_image"]]:
                        if os.path.exists(old_f):
                            os.remove(old_f)
                    del vehicles[match_key]
                    match_key = plate_text

                ts = round(frame_no / fps, 2)
                vehicles[match_key] = {
                    "plate_text":      plate_text,
                    "helmet_off_conf": round(record.best_helmet_conf, 4),
                    "plate_conf":      round(record.best_plate_conf,  4),
                    "frame":           frame_no,
                    "timestamp_sec":   ts,
                    "plate_image":     img_path,
                    "frame_image":     frame_path,
                }
                print(f"  ⬆️  UPGRADE   | Frame {frame_no:>5} ({ts:.1f}s) | "
                      f"Plate: {plate_text:<14} "
                      f"combined: {old_combined:.2%} → {new_combined:.2%}")

    # ── Main writer loop ───────────────────────────────────────────────────────
    last_frame_no = 0

    while True:
        item = result_queue.get()
        if item is None:
            for record in list(active_riders.values()):
                _commit_rider(record, last_frame_no)
            break

        frame_no    = item["frame_no"]
        raw_frame   = item["raw_frame"]
        helmet_offs = item["helmet_off"]
        plate_boxes = item["plates"]
        rider_boxes = item["riders"]
        last_frame_no = frame_no

        if frame_no % 100 == 0:
            _progress(frame_no, total_frames, counters["t_start"], len(vehicles))

        # Expire stale riders
        expired_keys = [
            k for k, r in active_riders.items()
            if frame_no - r.last_seen_frame > RIDER_TIMEOUT
        ]
        for k in expired_keys:
            _commit_rider(active_riders[k], frame_no)
            del active_riders[k]

        if not helmet_offs or not plate_boxes:
            continue

        best_helmet_off = max(helmet_offs, key=lambda b: b[4])
        best_plate      = max(plate_boxes, key=lambda b: b[4])

        hx1, hy1, hx2, hy2, h_conf, h_tid = best_helmet_off
        px1, py1, px2, py2, p_conf, p_tid = best_plate

        rider_box  = None
        rider_tid  = None
        best_iou_v = 0.3
        for rb in rider_boxes:
            overlap = iou((hx1, hy1, hx2, hy2), rb[:4])
            if overlap > best_iou_v:
                best_iou_v = overlap
                rider_box  = rb[:4]
                rider_tid  = rb[5]

        if rider_box is None:
            rider_box = (hx1, hy1, hx2, hy2)

        # Prefer ByteTrack ID; fall back to IOU-matched box tuple
        if rider_tid is not None:
            track_key = rider_tid
        else:
            track_key = None
            for k, r in active_riders.items():
                if isinstance(k, tuple) and iou(rider_box, k) > 0.4:
                    track_key = k
                    break
            if track_key is None:
                track_key = rider_box

        # ── CRITICAL: crop from RAW frame — no double-enhancement ──────────
        plate_crop = raw_frame[py1:py2, px1:px2]
        if plate_crop.size == 0:
            continue

        if track_key not in active_riders:
            active_riders[track_key] = RiderRecord(track_key)

        record = active_riders[track_key]

        if record.confirmed_plate:
            record.last_seen_frame = frame_no
            continue

        record.update(h_conf, raw_frame, p_conf, plate_crop, frame_no)

        # Eager per-frame OCR — commit on first valid read
        if record.ocr_attempts < MAX_OCR_ATTEMPTS:
            record.ocr_attempts += 1
            plate_text = ocr_plate_crop(plate_crop)
            if plate_text:
                record.confirmed_plate = plate_text
                _commit_rider(record, frame_no)

        # Early commit on very high confidence
        if (record.best_helmet_conf >= HIGH_CONF_COMMIT and
                record.best_plate_conf  >= HIGH_CONF_COMMIT and
                record.confirmed_plate):
            del active_riders[track_key]


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
def _progress(frame_no: int, total: int, t_start: float, vehicle_count: int):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  "
          f"FPS: {fps_d:.1f}  Violations so far: {vehicle_count}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    global _gamma_table

    if ENABLE_GAMMA:
        _gamma_table = _build_gamma_table(GAMMA_VALUE)

    get_ocr_reader()

    os.makedirs(PLATES_DIR, exist_ok=True)
    os.makedirs(REVIEW_DIR, exist_ok=True)
    for folder in [PLATES_DIR, REVIEW_DIR]:
        for f in os.listdir(folder):
            if f.endswith(".jpg"):
                os.remove(os.path.join(folder, f))

    gpu_available = torch.cuda.is_available()
    use_fp16      = gpu_available
    device_name   = torch.cuda.get_device_name(0) if gpu_available else "CPU"

    print(f"[AEGIS] Loading model  : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    if gpu_available:
        model.to("cuda")
    names = model.names

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"[AEGIS] Video          : {int(cap.get(3))}x{int(cap.get(4))} "
          f"@ {fps:.0f}fps | {total_frames} frames")
    print(f"[AEGIS] Hardware       : {device_name}  |  FP16={use_fp16}")
    print(f"[AEGIS] Tracking       : {'ByteTrack' if USE_TRACKER else 'IOU-manual'}")
    print(f"[AEGIS] Frame skip     : {FRAME_SKIP}")
    print(f"[AEGIS] Plate format   : XX YY X{{1-3}} YYYY")
    print(f"[AEGIS] OCR source     : RAW frame crops (enhancement for YOLO only)")
    print(f"[AEGIS] OCR strategy   : Eager per-frame (max {MAX_OCR_ATTEMPTS}) + timeout fallback")
    print(f"[AEGIS] Rider timeout  : {RIDER_TIMEOUT} frames")
    print(f"[AEGIS] Fuzzy threshold: edit distance <= {FUZZY_THRESHOLD}")
    print("-" * 65)

    vehicles = {}
    counters = {"noise": 0, "hard_neg": 0, "t_start": time.time()}
    stop_evt = threading.Event()

    frame_queue  = queue.Queue(maxsize=CAPTURE_QUEUE_SZ)
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_SZ)

    t_capture = threading.Thread(
        target=capture_thread,
        args=(cap, frame_queue, stop_evt),
        daemon=True, name="Capture"
    )
    t_infer = threading.Thread(
        target=inference_thread,
        args=(model, names, frame_queue, result_queue, use_fp16),
        daemon=True, name="Inference"
    )
    t_writer = threading.Thread(
        target=writer_thread,
        args=(result_queue, vehicles, fps, total_frames, counters),
        daemon=True, name="Writer"
    )

    t_capture.start()
    t_infer.start()
    t_writer.start()

    t_capture.join()
    t_infer.join()
    t_writer.join()

    cap.release()

    final = sorted(vehicles.values(), key=lambda r: r["frame"])
    for i, rec in enumerate(final, 1):
        rec["violation_no"] = i

    elapsed_total = time.time() - counters["t_start"]
    processed     = total_frames // max(FRAME_SKIP, 1)

    report = {
        "run_time":          datetime.now().isoformat(),
        "model":             MODEL_PATH,
        "input_video":       INPUT_VIDEO,
        "ocr_engine":        "easyocr",
        "plate_format":      "XX YY X{1-3} YYYY",
        "total_frames":      total_frames,
        "frames_processed":  processed,
        "frame_skip":        FRAME_SKIP,
        "avg_fps_processed": round(processed / elapsed_total, 1),
        "elapsed_seconds":   round(elapsed_total, 1),
        "hardware":          device_name,
        "fp16":              use_fp16,
        "tracker":           "ByteTrack" if USE_TRACKER else "IOU-manual",
        "noise_discarded":   counters["noise"],
        "hard_negatives":    counters["hard_neg"],
        "total_violations":  len(final),
        "violations":        final,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  ✅  {len(final)} unique violation(s)  "
          f"({counters['noise']} OCR misreads  |  "
          f"{counters['hard_neg']} saved for review)")
    for v in final:
        print(f"     #{v['violation_no']}  {v['plate_text']:<14} "
              f"helmet={v['helmet_off_conf']:.2%}  plate={v['plate_conf']:.2%}  "
              f"@ {v['timestamp_sec']}s")
    print(f"\n  Processed {processed}/{total_frames} frames in {elapsed_total:.1f}s "
          f"({processed / elapsed_total:.1f} fps)")
    print(f"  📄  Report       → {REPORT_PATH}")
    print(f"  🖼️   Plate crops  → {PLATES_DIR}/")
    print(f"  🔍  Review crops → {REVIEW_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    run()