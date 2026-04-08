import cv2
import re
import numpy as np
import pytesseract
import os
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
import os
from dotenv import load_dotenv

start = time.perf_counter()


# ──────────────────────────────────────────────
# SUPABASE — pip install supabase
# ──────────────────────────────────────────────
from supabase import create_client, Client

load_dotenv()   

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH      = "best.pt"
INPUT_VIDEO     = r"resources\test3.mp4"
REPORT_PATH     = "aegis_report.json"
PLATES_DIR      = "aegis_plates"   # temp local storage during processing

TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

HELMET_OFF_CONF = 0.50
PLATE_CONF_MIN  = 0.60
FUZZY_THRESHOLD = 3

PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

# ── Supabase credentials ───────────────────────
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")

# ── Supabase Storage bucket names ─────────────
BUCKET_PLATES = "number_plate"
BUCKET_FRAMES = "violation_image"

# ── Violation metadata ─────────────────────────
VIOLATION_TYPE = "No Helmet"
VIOLATION_FINE = 1000   # INR

# ── Location (Navi Mumbai, Nerul) ─────────────
LOCATION = "Sector 19, Nerul, Navi Mumbai, Maharashtra 400706, India"

# ── IST timezone ──────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


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
# SUPABASE BATCH PUSH  (called once after video)
# ──────────────────────────────────────────────
def push_to_supabase(vehicles: dict) -> dict:
    """
    For each unique vehicle:
      1. Read the best local image files saved during processing
      2. Upload plate crop  → BUCKET_PLATES
      3. Upload full frame  → BUCKET_FRAMES
      4. Insert one row     → violation_event table
      5. Delete local files only after confirmed success

    Returns mapping: plate_text → { plate_url, frame_url, row_id, timestamp }
    """
    print("\n[AEGIS] ── Connecting to Supabase ──────────────────")
    sb      = create_client(SUPABASE_URL, SUPABASE_KEY)
    results = {}   # plate_text → supabase metadata
    failed  = []
    now_ist = datetime.now(IST)   # single consistent timestamp for entire batch
    total   = len(vehicles)

    for idx, (plate_text, rec) in enumerate(vehicles.items(), 1):
        print(f"  [{idx}/{total}] {plate_text} — uploading ...", end=" ", flush=True)

        plate_img = cv2.imread(rec["plate_image"])
        frame_img = cv2.imread(rec["frame_image"])

        if plate_img is None or frame_img is None:
            print("❌  local file missing — skipped")
            failed.append(plate_text)
            continue

        # Encode to JPEG bytes in memory
        _, plate_buf = cv2.imencode(".jpg", plate_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, frame_buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        plate_filename = f"plate_{plate_text}.jpg"
        frame_filename = f"frame_{plate_text}.jpg"

        # ── Upload to Storage buckets ──────────────
        try:
            sb.storage.from_(BUCKET_PLATES).upload(
                plate_filename, plate_buf.tobytes(),
                {"content-type": "image/jpeg", "upsert": "true"},
            )
            sb.storage.from_(BUCKET_FRAMES).upload(
                frame_filename, frame_buf.tobytes(),
                {"content-type": "image/jpeg", "upsert": "true"},
            )
            plate_url = sb.storage.from_(BUCKET_PLATES).get_public_url(plate_filename)
            frame_url = sb.storage.from_(BUCKET_FRAMES).get_public_url(frame_filename)
        except Exception as e:
            print(f"❌  storage upload failed: {e}")
            failed.append(plate_text)
            continue   # keep local files — do NOT delete on failure

        # ── Insert row into violation_event ────────
        row_id = str(uuid.uuid4())
        try:
            sb.table("violation_event").insert({
                "id":               row_id,
                "number_plate":     plate_text,
                "timestamp":        now_ist.isoformat(),
                "location":         LOCATION,
                "confidence":       rec["plate_conf"],
                "violation_type":   VIOLATION_TYPE,
                "violation_fine":   VIOLATION_FINE,
                "fine_paid":        False,
                "number_plate_img": plate_url,
                "violation_img":    frame_url,
            }).execute()
        except Exception as e:
            print(f"❌  DB insert failed: {e}")
            failed.append(plate_text)
            continue   # keep local files on DB failure too

        print("✅")

        # ── Only delete local files after full success ──
        for path in [rec["plate_image"], rec["frame_image"]]:
            try:
                os.remove(path)
            except OSError as e:
                print(f"  [WARN] Could not delete {path}: {e}")

        results[plate_text] = {
            "plate_url": plate_url,
            "frame_url": frame_url,
            "row_id":    row_id,
            "timestamp": now_ist.isoformat(),
        }

    # ── Cleanup empty plates dir if all files removed ──
    remaining = [f for f in os.listdir(PLATES_DIR) if f.endswith(".jpg")]
    if not remaining:
        try:
            os.rmdir(PLATES_DIR)
            print(f"[AEGIS] Local temp folder '{PLATES_DIR}' removed (all files pushed)")
        except OSError:
            pass
    elif failed:
        print(f"[AEGIS] ⚠️  {len(remaining)} file(s) kept in '{PLATES_DIR}' for failed plates")

    pushed = len(results)
    print(f"[AEGIS] Push complete: {pushed}/{total} succeeded"
          + (f", {len(failed)} failed: {failed}" if failed else ""))

    return results


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    os.makedirs(PLATES_DIR, exist_ok=True)
    # Clear leftover files from any previous interrupted run
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
    print(f"[AEGIS] Location       : {LOCATION}")
    print(f"[AEGIS] Violation fine : ₹{VIOLATION_FINE}")
    print(f"[AEGIS] Mode           : local-first → batch push after video ends")
    print("-" * 55)

    vehicles      = {}   # canonical_plate → record
    noise_count   = 0
    frame_no      = 0
    t_start       = time.time()
    active_riders = {}   # rider_box → plate_text

    # ══════════════════════════════════════════════════════
    # PHASE 1 — process entire video, save best images locally
    # No network calls happen here at all
    # ══════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        results = model(frame, verbose=False)[0]

        helmet_off_boxes = []
        plate_boxes      = []
        rider_boxes      = []

        for box in results.boxes:
            cls    = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names[cls]
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

        # Skip if this rider already has an equal or better read
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
            # New unique violation — write best images to disk
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
                  f"Plate: {plate_text:<12} | Conf: {p_conf:.2%} | 💾 Saved locally")

        else:
            existing      = vehicles[match_key]
            old_conf      = existing["plate_conf"]
            plate_changed = existing["plate_text"] != plate_text

            # Only overwrite if this frame gives a strictly better read
            if p_conf <= old_conf and not plate_changed:
                if frame_no % 100 == 0:
                    _progress(frame_no, total_frames, t_start, len(vehicles))
                continue

            if p_conf > old_conf:
                img_path   = existing["plate_image"]
                frame_path = existing["frame_image"]

                if plate_changed:
                    # Plate text corrected — rename local files
                    img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
                    frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
                    for old_f in [existing["plate_image"], existing["frame_image"]]:
                        if os.path.exists(old_f):
                            os.remove(old_f)
                    del vehicles[match_key]
                    match_key = plate_text

                # Overwrite local files with the better crop/frame
                cv2.imwrite(img_path, plate_crop)
                cv2.imwrite(frame_path, frame)

                vehicles[match_key] = {
                    "frame":         frame_no,
                    "timestamp_sec": round(frame_no / fps, 2),
                    "plate_text":    plate_text,
                    "plate_conf":    round(p_conf, 4),
                    "plate_image":   img_path,
                    "frame_image":   frame_path,
                }
                active_riders[rider_box] = plate_text
                tag = " (text corrected)" if plate_changed else ""
                print(f"  ⬆️  UPGRADE | Frame {frame_no:>4} ({frame_no/fps:.1f}s) | "
                      f"Plate: {plate_text:<12} | Conf: {old_conf:.2%} → {p_conf:.2%}"
                      f" | 💾 Local overwrite{tag}")

        if frame_no % 100 == 0:
            _progress(frame_no, total_frames, t_start, len(vehicles))

    cap.release()
    elapsed = time.time() - t_start
    print(f"\n[AEGIS] ✅ Video complete in {elapsed:.1f}s  |  "
          f"{len(vehicles)} unique violation(s) ready to push")

    # ══════════════════════════════════════════════════════
    # PHASE 2 — batch push all results to Supabase at once
    #           local files deleted only after confirmed push
    # ══════════════════════════════════════════════════════
    supabase_results = push_to_supabase(vehicles)

    # Merge Supabase metadata back into records for the report
    final = sorted(vehicles.values(), key=lambda r: r["frame"])
    for i, rec in enumerate(final, 1):
        rec["violation_no"] = i
        sb_data = supabase_results.get(rec["plate_text"], {})
        rec["plate_url"]       = sb_data.get("plate_url", "")
        rec["frame_url"]       = sb_data.get("frame_url", "")
        rec["supabase_row_id"] = sb_data.get("row_id", "")
        rec["pushed_at"]       = sb_data.get("timestamp", "")

    report = {
        "run_time":         datetime.now(IST).isoformat(),
        "model":            MODEL_PATH,
        "input_video":      INPUT_VIDEO,
        "location":         LOCATION,
        "total_frames":     frame_no,
        "noise_discarded":  noise_count,
        "total_violations": len(final),
        "violations":       final,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  ✅  {len(final)} unique vehicle(s)  "
          f"({noise_count} invalid reads discarded)")
    for v in final:
        pushed_icon = "☁️ " if v.get("supabase_row_id") else "⚠️  local only"
        print(f"     #{v['violation_no']}  {v['plate_text']:<14} "
              f"conf: {v['plate_conf']:.2%}  @ {v['timestamp_sec']}s  {pushed_icon}")
    print(f"\n  📄  Report → {REPORT_PATH}")
    print(f"  ☁️   Supabase → {SUPABASE_URL}")
    print("=" * 55)
    


def _progress(frame_no, total, t_start, vehicle_count):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  "
          f"FPS: {fps_d:.1f}  Vehicles: {vehicle_count}")



if __name__ == "__main__":
    run()