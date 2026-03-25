
import cv2
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
PLATE_CONF_MIN  = 0.60   # Min plate confidence to attempt OCR
                         # NOTE: This model outputs plate conf in 0.60-0.75 range.
                         #       Raise to 0.70+ once you retrain on more data.
MIN_PLATE_CHARS = 4      # OCR must return at least this many chars to count


# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
def read_plate(crop):
    """Preprocess and OCR a plate crop. Returns cleaned alphanumeric string."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""

    # Upscale so Tesseract has enough pixels
    scale = max(2, int(150 / h))
    crop  = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt   = cv2.adaptiveThreshold(blur, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY, 15, 4)

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

    print(f"[AEGIS] Loading model : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    names = model.names
    print(f"[AEGIS] Classes       : {names}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[AEGIS] Video         : {int(cap.get(3))}x{int(cap.get(4))} "
          f"@ {fps:.0f}fps | {total_frames} frames")
    print(f"[AEGIS] Thresholds    : helmet-off >= {HELMET_OFF_CONF:.0%} | "
          f"plate >= {PLATE_CONF_MIN:.0%}")
    print("-" * 55)

    saved_plates = set()   # plates already saved — dedup
    violations   = []
    frame_no     = 0
    t_start      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        results = model(frame, verbose=False)[0]

        helmet_off = False
        best_plate = None   # (x1,y1,x2,y2, conf) — highest conf plate this frame

        for box in results.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "helmet-off" and conf >= HELMET_OFF_CONF:
                helmet_off = True

            if label == "numer_plate":
                if best_plate is None or conf > best_plate[4]:
                    best_plate = (x1, y1, x2, y2, conf)

        # Helmet worn (or no rider) → nothing to do
        if not helmet_off:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start)
            continue

        # Helmet off but no plate found → skip
        if best_plate is None:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start)
            continue

        px1, py1, px2, py2, p_conf = best_plate

        # Plate confidence below threshold → skip
        if p_conf < PLATE_CONF_MIN:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start)
            continue

        # OCR the plate
        plate_crop = frame[py1:py2, px1:px2]
        plate_text = read_plate(plate_crop)

        # OCR returned too little → skip (plate not readable yet)
        if len(plate_text) < MIN_PLATE_CHARS:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start)
            continue

        # Already saved this plate → skip
        if plate_text in saved_plates:
            if frame_no % 100 == 0:
                _progress(frame_no, total_frames, t_start)
            continue

        # ── New unique violation — save it ───────
        saved_plates.add(plate_text)
        img_path = os.path.join(PLATES_DIR,
                                f"plate_{len(violations)+1:03d}_{plate_text}.jpg")
        cv2.imwrite(img_path, plate_crop)

        record = {
            "violation_no":  len(violations) + 1,
            "frame":         frame_no,
            "timestamp_sec": round(frame_no / fps, 2),
            "plate_text":    plate_text,
            "plate_conf":    round(p_conf, 4),
            "plate_image":   img_path,
        }
        violations.append(record)

        print(f"  🚨 Violation #{record['violation_no']} | "
              f"Frame {frame_no} ({record['timestamp_sec']}s) | "
              f"Plate: {plate_text} | Conf: {p_conf:.2%}")

        if frame_no % 100 == 0:
            _progress(frame_no, total_frames, t_start)

    cap.release()

    # Save report
    report = {
        "run_time":         datetime.now().isoformat(),
        "model":            MODEL_PATH,
        "input_video":      INPUT_VIDEO,
        "total_frames":     frame_no,
        "total_violations": len(violations),
        "violations":       violations,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  ✅  {len(violations)} unique violation(s) saved.")
    print(f"  📄  Report      → {REPORT_PATH}")
    print(f"  🖼️   Plate crops → {PLATES_DIR}/")
    print("=" * 55)


def _progress(frame_no, total, t_start):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  FPS: {fps_d:.1f}")


if __name__ == "__main__":
    run()