import cv2
import re
import numpy as np
import pytesseract
import os
import json
import time
from ultralytics import YOLO
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH      = "best.pt"
INPUT_VIDEO     = "test2.mp4"
REPORT_PATH     = "aegis_report.json"
PLATES_DIR      = "aegis_plates"

TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

HELMET_OFF_CONF = 0.50   # Min confidence to flag helmet-off
PLATE_CONF_MIN  = 0.60   # Min YOLO plate-box confidence to attempt OCR
FUZZY_THRESHOLD = 3      # Max edit distance to consider two plates the same vehicle

# Strict Indian plate: exactly 2L + 2N + 1-3L + 4N, nothing more, nothing less
PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def is_valid_plate(text):
    return bool(PLATE_RE.match(text.strip().upper()))


def edit_distance(a, b):
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


def find_matching_vehicle(plate_text, vehicles):
    for key in vehicles:
        if edit_distance(plate_text, key) <= FUZZY_THRESHOLD:
            return key
    return None


def iou(boxA, boxB):
    """Intersection over Union for two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


def read_plate(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = max(2, int(150 / h))
    crop  = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt   = cv2.adaptiveThreshold(blur, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )
    best = ""
    for img in [otsu, adapt, cv2.bitwise_not(adapt)]:
        try:
            raw     = pytesseract.image_to_string(img, config=config)
            cleaned = "".join(c for c in raw if c.isalnum()).upper()
            if len(cleaned) > len(best):
                best = cleaned
        except Exception:
            continue
    return best


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    os.makedirs(PLATES_DIR, exist_ok=True)

    # Clear old plate images so only current-run saves remain
    for f in os.listdir(PLATES_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(PLATES_DIR, f))

    print(f"[AEGIS] Loading model  : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    names = model.names

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[AEGIS] Video          : {int(cap.get(3))}x{int(cap.get(4))} "
          f"@ {fps:.0f}fps | {total_frames} frames")
    print(f"[AEGIS] Plate format   : Indian strict (XX##XX####)")
    print(f"[AEGIS] Fuzzy tolerance: edit distance <= {FUZZY_THRESHOLD}")
    print("-" * 55)

    vehicles      = {}    # { canonical_plate -> record }
    noise_count   = 0
    frame_no      = 0
    t_start       = time.time()

    # Track which rider boxes we've already committed a read for this "event"
    # Key: rider box tuple, Value: plate text already saved for this rider
    # We reset this when rider disappears (IOU < threshold across frames)
    active_riders = {}    # { rider_box -> plate_text_saved }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        results = model(frame, verbose=False)[0]

        # Collect all detections this frame
        helmet_off_boxes = []   # list of (x1,y1,x2,y2, conf)
        plate_boxes      = []   # list of (x1,y1,x2,y2, conf)
        rider_boxes      = []   # list of (x1,y1,x2,y2, conf)

        for box in results.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = names[cls]
            coords = tuple(map(int, box.xyxy[0]))

            if label == "helmet-off" and conf >= HELMET_OFF_CONF:
                helmet_off_boxes.append((*coords, conf))
            elif label == "numer_plate" and conf >= PLATE_CONF_MIN:
                plate_boxes.append((*coords, conf))
            elif label == "rider":
                rider_boxes.append((*coords, conf))

        # No violations this frame
        if not helmet_off_boxes or not plate_boxes:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start, len(vehicles))
            continue

        # For each helmet-off, find the spatially closest plate
        # (both should be near the same rider)
        best_plate = max(plate_boxes, key=lambda b: b[4])
        px1, py1, px2, py2, p_conf = best_plate

        # Find the rider box that contains/overlaps the helmet-off region
        # Use the largest overlapping rider box as the identity anchor
        best_helmet_off = max(helmet_off_boxes, key=lambda b: b[4])
        hx1, hy1, hx2, hy2, _ = best_helmet_off

        rider_box = None
        best_iou  = 0.3     # minimum overlap threshold
        for rb in rider_boxes:
            overlap = iou((hx1, hy1, hx2, hy2), rb[:4])
            if overlap > best_iou:
                best_iou  = overlap
                rider_box = rb[:4]

        # If no rider box found, use helmet-off box as identity anchor
        if rider_box is None:
            rider_box = (hx1, hy1, hx2, hy2)

        # Check if this rider already had a valid plate saved
        # by comparing IOU with previously tracked riders
        already_tracked_key = None
        for tracked_box in active_riders:
            if iou(rider_box, tracked_box) > 0.4:
                already_tracked_key = tracked_box
                break

        if already_tracked_key is not None:
            # We've seen this rider — only upgrade if better plate conf arrives
            existing_plate = active_riders[already_tracked_key]
            match_key = find_matching_vehicle(existing_plate, vehicles)
            if match_key and vehicles[match_key]["plate_conf"] >= p_conf:
                # Already have a better or equal read → skip OCR entirely
                if frame_no % 100 == 0:
                    _progress(frame_no, total_frames, t_start, len(vehicles))
                continue

        # Attempt OCR
        plate_crop = frame[py1:py2, px1:px2]
        raw_text   = read_plate(plate_crop)

        # Gate: strict format validation — EXACT match only, no substring tricks
        if not is_valid_plate(raw_text):
            noise_count += 1
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start, len(vehicles))
            continue

        plate_text = raw_text.strip().upper()

        # Fuzzy dedup across all known vehicles
        match_key = find_matching_vehicle(plate_text, vehicles)

        if match_key is None:
            # New vehicle
            img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
            frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
            cv2.imwrite(img_path, plate_crop)
            cv2.imwrite(frame_path, frame)
            vehicles[plate_text] = {
                "frame":         frame_no,
                "timestamp_sec": round(frame_no / fps, 2),
                "plate_text":    plate_text,
                "plate_conf":    round(p_conf, 4),
                "plate_image":   img_path,
                "frame_image":   frame_path,
            }
            active_riders[rider_box] = plate_text
            print(f"  🚨 NEW     | Frame {frame_no:>4} ({frame_no/fps:.1f}s) | "
                  f"Plate: {plate_text:<12} | Conf: {p_conf:.2%}")

        else:
            existing = vehicles[match_key]
            if p_conf > existing["plate_conf"]:
                # Better read for same vehicle — replace both images
                old_conf   = existing["plate_conf"]
                img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
                frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
                cv2.imwrite(img_path, plate_crop)
                cv2.imwrite(frame_path, frame)

                if existing["plate_text"] != plate_text:
                    for old_file in [existing["plate_image"], existing["frame_image"]]:
                        if os.path.exists(old_file):
                            os.remove(old_file)
                    del vehicles[match_key]
                    match_key = plate_text

                vehicles[match_key] = {
                    "frame":         frame_no,
                    "timestamp_sec": round(frame_no / fps, 2),
                    "plate_text":    plate_text,
                    "plate_conf":    round(p_conf, 4),
                    "plate_image":   img_path,
                    "frame_image":   frame_path,
                }
                active_riders[rider_box] = plate_text
                print(f"  ⬆️  UPGRADE | Frame {frame_no:>4} ({frame_no/fps:.1f}s) | "
                      f"Plate: {plate_text:<12} | Conf: {old_conf:.2%} → {p_conf:.2%}")

        if frame_no % 100 == 0:
            _progress(frame_no, total_frames, t_start, len(vehicles))

    cap.release()

    final = sorted(vehicles.values(), key=lambda r: r["frame"])
    for i, rec in enumerate(final, 1):
        rec["violation_no"] = i

    report = {
        "run_time":         datetime.now().isoformat(),
        "model":            MODEL_PATH,
        "input_video":      INPUT_VIDEO,
        "total_frames":     frame_no,
        "noise_discarded":  noise_count,
        "total_violations": len(final),
        "violations":       final,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  ✅  {len(final)} unique vehicle(s)  "
          f"({noise_count} invalid format reads discarded)")
    for v in final:
        print(f"     #{v['violation_no']}  {v['plate_text']:<14} "
              f"conf: {v['plate_conf']:.2%}  @ {v['timestamp_sec']}s")
    print(f"\n  📄  Report      → {REPORT_PATH}")
    print(f"  🖼️   Plate crops → {PLATES_DIR}/")
    print("=" * 55)


def _progress(frame_no, total, t_start, vehicle_count):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  "
          f"FPS: {fps_d:.1f}  Vehicles: {vehicle_count}")


if __name__ == "__main__":
    run()