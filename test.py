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
INPUT_VIDEO     = "test3.mp4"
REPORT_PATH     = "aegis_report.json"
PLATES_DIR      = "aegis_plates"

TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

HELMET_OFF_CONF = 0.50
PLATE_CONF_MIN  = 0.60
FUZZY_THRESHOLD = 3

PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

# ── NEW: Frame pre-processing toggles ──────────
ENABLE_FRAME_ENHANCE  = True   # apply to every frame before YOLO
ENABLE_DEBLUR         = True   # motion blur correction (can slow things down)
FRAME_SCALE           = 1.0    # upscale before YOLO: 1.0 = no change, 1.5 = 50% larger


# ──────────────────────────────────────────────
# FRAME ENHANCEMENT  (applied before YOLO)
# ──────────────────────────────────────────────
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Light, fast pre-processing applied to every video frame.
    Goal: improve YOLO detection confidence, not visual beauty.
    """
    if FRAME_SCALE != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame,
                           (int(w * FRAME_SCALE), int(h * FRAME_SCALE)),
                           interpolation=cv2.INTER_LINEAR)

    # 1. Bilateral denoise — blurs noise but preserves hard edges (helmets/plates)
    #    d=5 is a good speed/quality trade-off; d=9 is slower but stronger.
    frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE on the luminance channel — boosts local contrast without blowing out
    #    highlights (important for shiny helmets and reflective plates in sunlight).
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Gentle unsharp mask — sharpens edges just enough to help YOLO find
    #    bounding-box borders without introducing ringing artifacts.
    if ENABLE_DEBLUR:
        blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=2)
        frame = cv2.addWeighted(frame, 1.4, blur, -0.4, 0)

    return frame


# ──────────────────────────────────────────────
# HELPERS (unchanged)
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
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


# ──────────────────────────────────────────────
# PLATE OCR  (enhanced version of read_plate)
# ──────────────────────────────────────────────
def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct small rotation angles in the plate crop using Hough lines.
    A tilted plate causes Tesseract to mis-read characters — fixing ±15° helps a lot.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(20, gray.shape[1] // 4))
    if lines is None:
        return gray

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < 15:          # only trust near-horizontal lines
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:      # ignore sub-half-degree noise
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def read_plate(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""

    # ── Step 1: upscale ──────────────────────────────────────────────────
    # Aim for a plate height of ~60px — sweet spot for Tesseract PSM 7.
    target_h = 60
    scale    = max(2, int(target_h / h))
    crop     = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # ── Step 2: CLAHE on the crop (local contrast) ───────────────────────
    lab   = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l     = clahe.apply(l)
    crop  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── Step 3: grayscale + bilateral denoise ────────────────────────────
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)

    # ── Step 4: unsharp mask (stronger than on full frame) ───────────────
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    sharp   = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)

    # ── Step 5: deskew ───────────────────────────────────────────────────
    sharp = _deskew(sharp)

    # ── Step 6: binarise (Otsu + adaptive + inverted adaptive) ──────────
    blur_for_thresh = cv2.GaussianBlur(sharp, (3, 3), 0)
    _, otsu  = cv2.threshold(blur_for_thresh, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt    = cv2.adaptiveThreshold(blur_for_thresh, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 4)
    adapt_inv = cv2.bitwise_not(adapt)

    # ── Step 7: morphological cleanup ────────────────────────────────────
    # Close small gaps inside characters (broken strokes), then open tiny blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_variants = []
    for img in [otsu, adapt, adapt_inv]:
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)
        cleaned_variants.append(opened)

    # ── Step 8: Tesseract with multiple variants ─────────────────────────
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )
    best = ""
    for img in cleaned_variants:
        try:
            raw     = pytesseract.image_to_string(img, config=config)
            cleaned = "".join(c for c in raw if c.isalnum()).upper()
            if is_valid_plate(cleaned):          # prefer a format-valid read
                return cleaned
            if len(cleaned) > len(best):
                best = cleaned
        except Exception:
            continue

    return best


# ──────────────────────────────────────────────
# MAIN  (only the frame-reading section changes)
# ──────────────────────────────────────────────
def run():
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    os.makedirs(PLATES_DIR, exist_ok=True)
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
    print(f"[AEGIS] Frame enhance  : {ENABLE_FRAME_ENHANCE}  |  "
          f"Deblur: {ENABLE_DEBLUR}  |  Scale: {FRAME_SCALE}x")
    print("-" * 55)

    vehicles      = {}
    noise_count   = 0
    frame_no      = 0
    t_start       = time.time()
    active_riders = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # ── NEW: enhance before detection ──────────────────────────────
        display_frame = frame.copy()          # keep original for saving screenshots
        if ENABLE_FRAME_ENHANCE:
            frame = enhance_frame(frame)

        results = model(frame, verbose=False)[0]

        helmet_off_boxes = []
        plate_boxes      = []
        rider_boxes      = []

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

        if not helmet_off_boxes or not plate_boxes:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start, len(vehicles))
            continue

        best_plate = max(plate_boxes, key=lambda b: b[4])
        px1, py1, px2, py2, p_conf = best_plate

        best_helmet_off = max(helmet_off_boxes, key=lambda b: b[4])
        hx1, hy1, hx2, hy2, _ = best_helmet_off

        rider_box = None
        best_iou  = 0.3
        for rb in rider_boxes:
            overlap = iou((hx1, hy1, hx2, hy2), rb[:4])
            if overlap > best_iou:
                best_iou  = overlap
                rider_box = rb[:4]

        if rider_box is None:
            rider_box = (hx1, hy1, hx2, hy2)

        already_tracked_key = None
        for tracked_box in active_riders:
            if iou(rider_box, tracked_box) > 0.4:
                already_tracked_key = tracked_box
                break

        if already_tracked_key is not None:
            existing_plate = active_riders[already_tracked_key]
            match_key = find_matching_vehicle(existing_plate, vehicles)
            if match_key and vehicles[match_key]["plate_conf"] >= p_conf:
                if frame_no % 100 == 0:
                    _progress(frame_no, total_frames, t_start, len(vehicles))
                continue

        # ── Plate OCR on the ENHANCED frame crop ─────────────────────
        plate_crop = frame[py1:py2, px1:px2]
        raw_text   = read_plate(plate_crop)

        if not is_valid_plate(raw_text):
            noise_count += 1
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start, len(vehicles))
            continue

        plate_text = raw_text.strip().upper()
        match_key  = find_matching_vehicle(plate_text, vehicles)

        if match_key is None:
            img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
            frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
            cv2.imwrite(img_path, plate_crop)
            cv2.imwrite(frame_path, display_frame)   # save the original (unenhanced) frame
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
                old_conf   = existing["plate_conf"]
                img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
                frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
                cv2.imwrite(img_path, plate_crop)
                cv2.imwrite(frame_path, display_frame)

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