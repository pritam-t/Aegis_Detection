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
from dotenv import load_dotenv

import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from supabase import create_client, Client

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH      = "best.pt"
INPUT_VIDEO     = r"resources\test3.mp4"
REPORT_PATH     = "aegis_report.json"
PLATES_DIR      = "aegis_plates"

TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

HELMET_OFF_CONF = 0.50
PLATE_CONF_MIN  = 0.60
FUZZY_THRESHOLD = 3

PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")

BUCKET_PLATES = "number_plate"
BUCKET_FRAMES = "violation_image"

VIOLATION_TYPE = "No Helmet"
VIOLATION_FINE = 1000

LOCATION = "Sector 19, Nerul, Navi Mumbai, Maharashtra 400706, India"
IST      = timezone(timedelta(hours=5, minutes=30))

# ──────────────────────────────────────────────
# PARALLEL TUNING
# ──────────────────────────────────────────────
import torch

# ── Frame reader ───────────────────────────────
# Buffer this many decoded frames ahead of the GPU worker.
# Large = more RAM but smoother GPU feeding. 64–128 is good.
FRAME_QUEUE_SIZE = 64

# ── GPU inference ──────────────────────────────
# Always 1 on a single GPU — adding more just causes CUDA contention.
GPU_WORKERS = 1

# ── OCR workers ────────────────────────────────
# Tesseract is CPU-bound and has ZERO GPU benefit.
# Use (physical core count) for best throughput.
# We leave 1 core free for the reader + collector.
OCR_WORKERS = max(2, (os.cpu_count() or 4) - 1)

# Buffer between GPU worker and OCR workers.
# Each slot holds a plate crop + metadata (cheap).
OCR_QUEUE_SIZE = 256

# ── Frame skip ─────────────────────────────────
# 0 = every frame, 2 = every 3rd frame.
# At 30 fps, FRAME_SKIP=2 → effective 10 fps (plenty for traffic cams).
# Raise to 4–5 if the GPU can't keep up.
FRAME_SKIP = 2

# ── YOLO inference batch size ──────────────────
# Send N frames to the GPU in one call — dramatically improves GPU utilisation.
# 4–8 is sweet spot for most cards. Raise if VRAM > 6 GB.
INFERENCE_BATCH = 4

# ── Half-precision ─────────────────────────────
# FP16 inference: ~2× faster on NVIDIA RTX/GTX. Safe for detection models.
USE_FP16 = torch.cuda.is_available()

print(f"[AEGIS] OCR workers    : {OCR_WORKERS}")
print(f"[AEGIS] Inference batch: {INFERENCE_BATCH}  |  FP16: {USE_FP16}")

_SENTINEL = None


# ──────────────────────────────────────────────
# HELPERS  (unchanged)
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


def correct_plate(text: str) -> str:
    text = text.upper()
    if len(text) < 8:
        return text
    corrected = list(text)
    for i, ch in enumerate(corrected):
        if i in (0, 1, 4, 5):
            if ch.isdigit():
                corrected[i] = {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G'}.get(ch, ch)
        elif i in (2, 3, 6, 7, 8, 9):
            if ch.isalpha():
                corrected[i] = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6'}.get(ch, ch)
    corrected[0] = 'M'
    corrected[1] = 'H'
    return "".join(corrected)


def read_plate(crop):
    """OCR one plate crop. Called from a thread pool — purely CPU."""
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
    return correct_plate(best)


# ──────────────────────────────────────────────
# SUPABASE BATCH PUSH  (unchanged)
# ──────────────────────────────────────────────
def push_to_supabase(vehicles: dict) -> dict:
    print("\n[AEGIS] ── Connecting to Supabase ──────────────────")
    sb      = create_client(SUPABASE_URL, SUPABASE_KEY)
    results = {}
    failed  = []
    now_ist = datetime.now(IST)
    total   = len(vehicles)

    for idx, (plate_text, rec) in enumerate(vehicles.items(), 1):
        print(f"  [{idx}/{total}] {plate_text} — uploading ...", end=" ", flush=True)

        plate_img = cv2.imread(rec["plate_image"])
        frame_img = cv2.imread(rec["frame_image"])
        if plate_img is None or frame_img is None:
            print("❌  local file missing — skipped")
            failed.append(plate_text)
            continue

        _, plate_buf = cv2.imencode(".jpg", plate_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, frame_buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        plate_filename = f"plate_{plate_text}.jpg"
        frame_filename = f"frame_{plate_text}.jpg"

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
            continue

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
            continue

        print("✅")
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

    remaining = [f for f in os.listdir(PLATES_DIR) if f.endswith(".jpg")]
    if not remaining:
        try:
            os.rmdir(PLATES_DIR)
            print(f"[AEGIS] Local temp folder '{PLATES_DIR}' removed")
        except OSError:
            pass
    elif failed:
        print(f"[AEGIS] ⚠️  {len(remaining)} file(s) kept for failed plates")

    pushed = len(results)
    print(f"[AEGIS] Push complete: {pushed}/{total} succeeded"
          + (f", {len(failed)} failed: {failed}" if failed else ""))
    return results


# ──────────────────────────────────────────────
# OPTIMISED PARALLEL PIPELINE
# ──────────────────────────────────────────────
#
#  OLD (serial per-frame):
#    Reader ──► [GPU infer + OCR] × 1 worker ──► Collector
#
#  NEW (fully decoupled):
#    Reader ──► frame_q ──► GPU worker (batch infer) ──► ocr_q ──► OCR pool × N ──► result_q ──► Collector
#
#  Key wins:
#  1. GPU worker sends INFERENCE_BATCH frames at once → higher GPU utilisation
#  2. OCR runs on ALL CPU cores concurrently, never blocking the GPU
#  3. FP16 inference halves GPU compute time on NVIDIA cards
#  4. Frame queue is large enough that the reader never stalls the GPU
#  5. cv2.imencode (disk writes) happen in the collector on main thread — I/O is fast
#     relative to inference+OCR so this is not a bottleneck


def _reader_thread(cap, frame_q: queue.Queue, total_frames, fps):
    """Reads and buffers frames. One sentinel per GPU worker."""
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if (frame_no - 1) % (FRAME_SKIP + 1) != 0:
            continue
        frame_q.put((frame_no, frame))
    for _ in range(GPU_WORKERS):
        frame_q.put(_SENTINEL)


def _gpu_worker(worker_id, model_path, frame_q: queue.Queue, ocr_q: queue.Queue):
    """
    Pulls frames in batches, runs YOLO inference on GPU, and pushes individual
    plate crops + metadata to the OCR queue.
    One sentinel per OCR worker pushed when done.
    """
    model = YOLO(model_path)
    if USE_FP16:
        model.model.half()              # FP16 — 2× faster on RTX cards
    names = model.names

    batch_frames = []   # list of (frame_no, frame)
    done = False

    def _flush(batch):
        """Run inference on a batch and push results to ocr_q."""
        if not batch:
            return
        imgs     = [b[1] for b in batch]
        all_res  = model(imgs, verbose=False, half=USE_FP16)

        for (frame_no, frame), results in zip(batch, all_res):
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
                ocr_q.put({"frame_no": frame_no, "violation": False})
                continue

            best_plate      = max(plate_boxes, key=lambda b: b[4])
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

            plate_crop = frame[py1:py2, px1:px2]

            # Push to OCR queue — crop is cheap (numpy slice)
            ocr_q.put({
                "frame_no":   frame_no,
                "violation":  True,
                "plate_conf": p_conf,
                "rider_box":  rider_box,
                "plate_crop": plate_crop,
                "frame":      frame,
            })

    while not done:
        try:
            item = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is _SENTINEL:
            done = True
        else:
            batch_frames.append(item)

        # Flush when batch is full OR we've seen the sentinel
        if len(batch_frames) >= INFERENCE_BATCH or (done and batch_frames):
            _flush(batch_frames)
            batch_frames = []

    # Signal all OCR workers
    for _ in range(OCR_WORKERS):
        ocr_q.put(_SENTINEL)


def _ocr_worker(ocr_q: queue.Queue, result_q: queue.Queue):
    """
    Pure CPU thread. Pulls plate crops, runs Tesseract, pushes final result.
    Many of these run in parallel — one per physical core.
    """
    while True:
        item = ocr_q.get()
        if item is _SENTINEL:
            result_q.put(_SENTINEL)
            break

        if not item["violation"]:
            result_q.put(item)
            continue

        plate_text = read_plate(item["plate_crop"])   # ← the slow part, now parallelised
        result_q.put({
            "frame_no":   item["frame_no"],
            "violation":  True,
            "plate_text": plate_text,
            "plate_conf": item["plate_conf"],
            "rider_box":  item["rider_box"],
            "plate_crop": item["plate_crop"],
            "frame":      item["frame"],
        })


def _collector(result_q: queue.Queue, fps, total_frames, t_start):
    """Unchanged logic. Runs in main thread — no locks needed."""
    vehicles       = {}
    active_riders  = {}
    noise_count    = 0
    sentinels_seen = 0
    last_frame_no  = 0

    while sentinels_seen < OCR_WORKERS:
        item = result_q.get()

        if item is _SENTINEL:
            sentinels_seen += 1
            continue

        frame_no = item["frame_no"]
        last_frame_no = max(last_frame_no, frame_no)

        if frame_no % 100 == 0:
            _progress(frame_no, total_frames, t_start, len(vehicles))

        if not item["violation"]:
            continue

        plate_text_raw = item["plate_text"]
        if not is_valid_plate(plate_text_raw):
            noise_count += 1
            continue

        plate_text = plate_text_raw.strip().upper()
        p_conf     = item["plate_conf"]
        rider_box  = item["rider_box"]
        plate_crop = item["plate_crop"]
        frame      = item["frame"]

        already_tracked_key = None
        for tracked_box in active_riders:
            if iou(rider_box, tracked_box) > 0.4:
                already_tracked_key = tracked_box
                break

        if already_tracked_key is not None:
            existing_plate = active_riders[already_tracked_key]
            match_key = find_matching_vehicle(existing_plate, vehicles)
            if match_key and vehicles[match_key]["plate_conf"] >= p_conf:
                continue

        match_key = find_matching_vehicle(plate_text, vehicles)

        if match_key is None:
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

            if p_conf <= old_conf and not plate_changed:
                continue

            if p_conf > old_conf:
                img_path   = existing["plate_image"]
                frame_path = existing["frame_image"]

                if plate_changed:
                    img_path   = os.path.join(PLATES_DIR, f"plate_{plate_text}.jpg")
                    frame_path = os.path.join(PLATES_DIR, f"frame_{plate_text}.jpg")
                    for old_f in [existing["plate_image"], existing["frame_image"]]:
                        if os.path.exists(old_f):
                            os.remove(old_f)
                    del vehicles[match_key]
                    match_key = plate_text

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

    return vehicles, noise_count, last_frame_no


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    os.makedirs(PLATES_DIR, exist_ok=True)
    for f in os.listdir(PLATES_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(PLATES_DIR, f))

    print(f"[AEGIS] Loading model  : {MODEL_PATH}")
    print(f"[AEGIS] GPU workers    : {GPU_WORKERS}  |  batch={INFERENCE_BATCH}  FP16={USE_FP16}")
    print(f"[AEGIS] OCR workers    : {OCR_WORKERS}  (FRAME_SKIP={FRAME_SKIP})")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[AEGIS] Video          : {int(cap.get(3))}x{int(cap.get(4))} "
          f"@ {fps:.0f}fps | {total_frames} frames")
    print(f"[AEGIS] Effective FPS  : {fps / (FRAME_SKIP + 1):.1f} (after skip)")
    print(f"[AEGIS] Location       : {LOCATION}")
    print(f"[AEGIS] Violation fine : ₹{VIOLATION_FINE}")
    print(f"[AEGIS] Mode           : GPU-batch infer → parallel OCR → batch push")
    print("-" * 55)

    frame_q = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    ocr_q   = queue.Queue(maxsize=OCR_QUEUE_SIZE)
    result_q = queue.Queue()

    t_start = time.time()

    # ── Reader ────────────────────────────────
    reader = threading.Thread(
        target=_reader_thread,
        args=(cap, frame_q, total_frames, fps),
        daemon=True, name="FrameReader",
    )
    reader.start()

    # ── GPU worker(s) ─────────────────────────
    gpu_threads = []
    for wid in range(GPU_WORKERS):
        t = threading.Thread(
            target=_gpu_worker,
            args=(wid, MODEL_PATH, frame_q, ocr_q),
            daemon=True, name=f"GPU-{wid}",
        )
        t.start()
        gpu_threads.append(t)

    # ── OCR workers ───────────────────────────
    ocr_threads = []
    for wid in range(OCR_WORKERS):
        t = threading.Thread(
            target=_ocr_worker,
            args=(ocr_q, result_q),
            daemon=True, name=f"OCR-{wid}",
        )
        t.start()
        ocr_threads.append(t)

    # ── Collector (main thread) ───────────────
    vehicles, noise_count, frame_no = _collector(result_q, fps, total_frames, t_start)

    reader.join()
    for t in gpu_threads + ocr_threads:
        t.join()
    cap.release()

    elapsed = time.time() - t_start
    print(f"\n[AEGIS] ✅ Video complete in {elapsed:.1f}s  |  "
          f"{len(vehicles)} unique violation(s) ready to push")

    # ── PHASE 2 — batch push ──────────────────
    supabase_results = push_to_supabase(vehicles)

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