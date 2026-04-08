import cv2
import re
import numpy as np
import os
import json
import time
import uuid
import hashlib
import threading
import queue
from datetime import datetime, timezone, timedelta

import torch
from ultralytics import YOLO
from dotenv import load_dotenv
from supabase import create_client

# ── PaddleOCR (lazy-initialised per worker to avoid pickling issues) ──────────
from paddleocr import PaddleOCR

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG--updated
# ──────────────────────────────────────────────
MODEL_PATH      = "best.pt"
INPUT_VIDEO     = r"resources\test3.mp4"
REPORT_PATH     = "aegis_report.json"
PLATES_DIR      = "aegis_plates"

HELMET_OFF_CONF = 0.50
PLATE_CONF_MIN  = 0.60
FUZZY_THRESHOLD = 3

PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")

BUCKET_PLATES   = "number_plate"
BUCKET_FRAMES   = "violation_image"
VIOLATION_TYPE  = "No Helmet"
VIOLATION_FINE  = 1000
LOCATION        = "Sector 19, Nerul, Navi Mumbai, Maharashtra 400706, India"
IST             = timezone(timedelta(hours=5, minutes=30))

# ──────────────────────────────────────────────
# PERFORMANCE TUNING
# ──────────────────────────────────────────────
_GPU_AVAILABLE     = torch.cuda.is_available()

# Frame reader
FRAME_QUEUE_SIZE   = 64        # decoded frames buffered ahead of GPU

# GPU inference
INFERENCE_BATCH    = 8         # frames per YOLO call; raise to 16 if VRAM > 8 GB
INFER_IMG_SIZE     = 640       # resize input to this before sending to GPU
USE_FP16           = False   # ~2× faster on NVIDIA RTX/GTX

# OCR
# PaddleOCR is thread-safe but each thread gets its own instance to avoid
# internal lock contention.  1 is usually enough because GPU OCR is fast;
# raise to 2–4 if you see the ocr_q growing.
OCR_WORKERS        = 2 if _GPU_AVAILABLE else max(2, (os.cpu_count() or 4) - 1)
OCR_QUEUE_SIZE     = 512

# Frame skip: 5 → process every 6th frame.
# At 30 fps → 5 effective fps, enough for any traffic cam.
# Lower to 2–3 if you're missing fast-moving bikes.
FRAME_SKIP         = 5

_SENTINEL = None   # poison pill


# ──────────────────────────────────────────────
# FAST VIDEO CAPTURE  (background decode thread)
# ──────────────────────────────────────────────
class FastCapture:
    """
    Wraps cv2.VideoCapture with a background reader thread so decoded frames
    are always ready — GPU worker never stalls waiting for disk I/O.
    """
    def __init__(self, path: str, queue_size: int = 128):
        self._cap   = cv2.VideoCapture(path)
        self._q     = queue.Queue(maxsize=queue_size)
        self._stop  = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="FastCap")
        self._thread.start()

    def _run(self):
        while not self._stop:
            ret, frame = self._cap.read()
            if not ret:
                self._q.put(None)   # EOF
                break
            # Drop if consumer is too slow (keeps memory bounded)
            try:
                self._q.put(frame, timeout=0.5)
            except queue.Full:
                pass

    # VideoCapture-compatible interface
    def read(self):
        frame = self._q.get()
        return frame is not None, frame

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._stop = True
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def is_valid_plate(text: str) -> bool:
    return bool(PLATE_RE.match(text.strip().upper()))


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


def find_matching_vehicle(plate_text: str, vehicles: dict):
    for key in vehicles:
        if edit_distance(plate_text, key) <= FUZZY_THRESHOLD:
            return key
    return None


def iou(boxA, boxB) -> float:
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
    # Strip common noise suffixes
    for suffix in ["IND", "IN", "ND"]:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    
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


# ── Plate OCR with GPU + crop-level cache ────────────────────────────────────
_ocr_cache: dict = {}           # hash → plate_text
_cache_lock = threading.Lock()
_ocr_engine_tls = threading.local()   # one PaddleOCR instance per thread


def _get_ocr_engine() -> PaddleOCR:
    if not hasattr(_ocr_engine_tls, "engine"):
        _ocr_engine_tls.engine = PaddleOCR(
            use_textline_orientation=False,
            lang="en",
            use_gpu=False,  # Force CPU — avoids cuDNN missing DLL error
        )
    return _ocr_engine_tls.engine


def _crop_hash(crop: np.ndarray) -> str:
    """Cheap perceptual hash — identical crops return same key."""
    small = cv2.resize(crop, (16, 8))
    return hashlib.md5(small.tobytes()).hexdigest()


def read_plate(crop: np.ndarray) -> str:
    """
    OCR a plate crop.
    1. Check cache first — same crop seen before → instant return.
    2. Run PaddleOCR (GPU-accelerated).
    3. Apply positional correction + state-code override.
    """
    if crop is None or crop.size == 0:
        return ""

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""

    key = _crop_hash(crop)
    with _cache_lock:
        if key in _ocr_cache:
            return _ocr_cache[key]

    # Upscale small crops so OCR has enough pixels
    if h < 40:
        scale = max(2, int(60 / h))
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    try:
        engine  = _get_ocr_engine()
        results = engine.ocr(crop, cls=False)
        if results and results[0]:
            raw = "".join(line[1][0] for line in results[0])
        else:
            raw = ""
    except Exception:
        raw = ""

    cleaned = "".join(c for c in raw if c.isalnum()).upper()
    text    = correct_plate(cleaned)

    with _cache_lock:
        _ocr_cache[key] = text

    return text


# ──────────────────────────────────────────────
# SUPABASE BATCH PUSH
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
            except OSError:
                pass

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
        except OSError:
            pass

    pushed = len(results)
    print(f"[AEGIS] Push complete: {pushed}/{total} succeeded"
          + (f", {len(failed)} failed: {failed}" if failed else ""))
    return results


# ──────────────────────────────────────────────
# PIPELINE STAGES
# ──────────────────────────────────────────────

def _reader_thread(cap: FastCapture, frame_q: queue.Queue):
    """
    Pulls frames from FastCapture (already decoded in background) and forwards
    them to the GPU worker, skipping according to FRAME_SKIP.
    """
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if (frame_no - 1) % (FRAME_SKIP + 1) != 0:
            continue
        frame_q.put((frame_no, frame))

    frame_q.put(_SENTINEL)   # one GPU worker → one sentinel


def _gpu_worker(model_path: str, frame_q: queue.Queue, ocr_q: queue.Queue):
    """
    Batched YOLO inference (FP16) on GPU.
    Extracts plate crops and metadata, pushes to ocr_q.
    Never calls Tesseract/PaddleOCR — stays on GPU full-time.
    """
    model = YOLO(model_path)
    if USE_FP16:
        model.model.half()
    names = model.names

    batch: list = []
    done  = False

    def flush(batch):
        if not batch:
            return

        # Pre-resize all frames to INFER_IMG_SIZE before GPU transfer
        # This is faster than letting YOLO do it internally in Python
        resized = [cv2.resize(f, (INFER_IMG_SIZE, INFER_IMG_SIZE)) for _, f in batch]
        h_orig, w_orig = batch[0][1].shape[:2]
        sx = w_orig / INFER_IMG_SIZE
        sy = h_orig / INFER_IMG_SIZE

        all_results = model(resized, verbose=False, half=USE_FP16)

        for (frame_no, orig_frame), result in zip(batch, all_results):
            helmet_off_boxes, plate_boxes, rider_boxes = [], [], []

            for box in result.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = names[cls]
                # Scale coords back to original frame size
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                x1 = int(x1 * sx); x2 = int(x2 * sx)
                y1 = int(y1 * sy); y2 = int(y2 * sy)

                if label == "helmet-off" and conf >= HELMET_OFF_CONF:
                    helmet_off_boxes.append((x1, y1, x2, y2, conf))
                elif label == "numer_plate" and conf >= PLATE_CONF_MIN:
                    plate_boxes.append((x1, y1, x2, y2, conf))
                elif label == "rider":
                    rider_boxes.append((x1, y1, x2, y2, conf))

            if not helmet_off_boxes or not plate_boxes:
                ocr_q.put({"frame_no": frame_no, "violation": False})
                continue

            best_plate              = max(plate_boxes, key=lambda b: b[4])
            px1, py1, px2, py2, pc  = best_plate
            best_ho                 = max(helmet_off_boxes, key=lambda b: b[4])
            hx1, hy1, hx2, hy2, _  = best_ho

            rider_box = None
            best_iou  = 0.3
            for rb in rider_boxes:
                ov = iou((hx1, hy1, hx2, hy2), rb[:4])
                if ov > best_iou:
                    best_iou  = ov
                    rider_box = rb[:4]
            if rider_box is None:
                rider_box = (hx1, hy1, hx2, hy2)

            plate_crop = orig_frame[py1:py2, px1:px2]
            ocr_q.put({
                "frame_no":   frame_no,
                "violation":  True,
                "plate_conf": pc,
                "rider_box":  rider_box,
                "plate_crop": plate_crop,
                "frame":      orig_frame,
            })

    while not done:
        try:
            item = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is _SENTINEL:
            done = True
        else:
            batch.append(item)

        if len(batch) >= INFERENCE_BATCH or (done and batch):
            flush(batch)
            batch = []

    # Signal all OCR workers
    for _ in range(OCR_WORKERS):
        ocr_q.put(_SENTINEL)


def _ocr_worker(ocr_q: queue.Queue, result_q: queue.Queue):
    """
    GPU-accelerated PaddleOCR + crop cache.
    Runs on a dedicated thread — multiple workers saturate PaddleOCR throughput.
    """
    while True:
        item = ocr_q.get()
        if item is _SENTINEL:
            result_q.put(_SENTINEL)
            break

        if not item["violation"]:
            result_q.put(item)
            continue

        plate_text = read_plate(item["plate_crop"])
        result_q.put({
            "frame_no":   item["frame_no"],
            "violation":  True,
            "plate_text": plate_text,
            "plate_conf": item["plate_conf"],
            "rider_box":  item["rider_box"],
            "plate_crop": item["plate_crop"],
            "frame":      item["frame"],
        })


def _collector(result_q: queue.Queue, fps: float, total_frames: int, t_start: float):
    """Deduplication + best-frame logic. Runs in main thread — no locks needed."""
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

        frame_no      = item["frame_no"]
        last_frame_no = max(last_frame_no, frame_no)

        if frame_no % 50 == 0:
            _progress(frame_no, total_frames, t_start, len(vehicles))

        if not item["violation"]:
            continue

        plate_text_raw = item["plate_text"]
        if not is_valid_plate(plate_text_raw):
            noise_count += 1
            print(f"  [NOISE] Raw OCR: '{plate_text_raw}'") 
            continue

        plate_text = plate_text_raw.strip().upper()
        p_conf     = item["plate_conf"]
        rider_box  = item["rider_box"]
        plate_crop = item["plate_crop"]
        frame      = item["frame"]

        # Skip if this rider already has a better read
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
                  f"Plate: {plate_text:<12} | Conf: {p_conf:.2%} | 💾 Saved")

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
                tag = " (corrected)" if plate_changed else ""
                print(f"  ⬆️  UPGRADE | Frame {frame_no:>4} ({frame_no/fps:.1f}s) | "
                      f"Plate: {plate_text:<12} | {old_conf:.2%} → {p_conf:.2%}{tag}")


    return vehicles, noise_count, last_frame_no


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    os.makedirs(PLATES_DIR, exist_ok=True)
    for f in os.listdir(PLATES_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(PLATES_DIR, f))

    print(f"[AEGIS] ⚡ TURBO MODE")
    print(f"[AEGIS] Model          : {MODEL_PATH}")
    print(f"[AEGIS] GPU            : {'✅ ' + torch.cuda.get_device_name(0) if _GPU_AVAILABLE else '❌ CPU only'}")
    print(f"[AEGIS] FP16           : {USE_FP16}")
    print(f"[AEGIS] Infer batch    : {INFERENCE_BATCH}  @  {INFER_IMG_SIZE}px")
    print(f"[AEGIS] OCR engine     : PaddleOCR (GPU={_GPU_AVAILABLE})  ×{OCR_WORKERS} workers")
    print(f"[AEGIS] Frame skip     : {FRAME_SKIP} (every {FRAME_SKIP+1}th frame)")

    cap = FastCapture(INPUT_VIDEO, queue_size=FRAME_QUEUE_SIZE * 2)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    eff_fps      = fps / (FRAME_SKIP + 1)
    print(f"[AEGIS] Video          : @ {fps:.0f}fps | {total_frames} frames "
          f"→ effective {eff_fps:.1f}fps")
    print(f"[AEGIS] Location       : {LOCATION}")
    print(f"[AEGIS] Violation fine : ₹{VIOLATION_FINE}")
    print("-" * 60)

    frame_q  = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    ocr_q    = queue.Queue(maxsize=OCR_QUEUE_SIZE)
    result_q = queue.Queue()

    t_start = time.time()

    # Reader thread
    reader = threading.Thread(
        target=_reader_thread, args=(cap, frame_q),
        daemon=True, name="Reader",
    )
    reader.start()

    # GPU worker (single thread — GPU is already saturated by batching)
    gpu_thread = threading.Thread(
        target=_gpu_worker, args=(MODEL_PATH, frame_q, ocr_q),
        daemon=True, name="GPU",
    )
    gpu_thread.start()

    # OCR worker pool
    ocr_threads = [
        threading.Thread(
            target=_ocr_worker, args=(ocr_q, result_q),
            daemon=True, name=f"OCR-{i}",
        )
        for i in range(OCR_WORKERS)
    ]
    for t in ocr_threads:
        t.start()

    # Collector (main thread)
    vehicles, noise_count, frame_no = _collector(result_q, fps, total_frames, t_start)

    reader.join()
    gpu_thread.join()
    for t in ocr_threads:
        t.join()
    cap.release()

    elapsed     = time.time() - t_start
    cache_hits  = sum(1 for _ in _ocr_cache)   # approximate
    print(f"\n[AEGIS] ✅ Done in {elapsed:.1f}s  |  {len(vehicles)} violation(s)  |  "
          f"OCR cache size: {len(_ocr_cache)}")

    # ── PHASE 2 — batch push ──────────────────
    supabase_results = push_to_supabase(vehicles)

    final = sorted(vehicles.values(), key=lambda r: r["frame"])
    for i, rec in enumerate(final, 1):
        rec["violation_no"] = i
        sb = supabase_results.get(rec["plate_text"], {})
        rec["plate_url"]       = sb.get("plate_url", "")
        rec["frame_url"]       = sb.get("frame_url", "")
        rec["supabase_row_id"] = sb.get("row_id", "")
        rec["pushed_at"]       = sb.get("timestamp", "")

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

    print("\n" + "=" * 60)
    print(f"  ✅  {len(final)} unique vehicle(s)  "
          f"({noise_count} invalid reads discarded)")
    for v in final:
        icon = "☁️ " if v.get("supabase_row_id") else "⚠️  local only"
        print(f"     #{v['violation_no']}  {v['plate_text']:<14} "
              f"conf: {v['plate_conf']:.2%}  @ {v['timestamp_sec']}s  {icon}")
    print(f"\n  📄  Report  → {REPORT_PATH}")
    print(f"  ☁️   Supabase → {SUPABASE_URL}")
    print(f"  ⏱   Total   → {elapsed:.1f}s")
    print("=" * 60)


def _progress(frame_no, total, t_start, vehicle_count):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  "
          f"Speed: {fps_d:.1f} fps  Violations: {vehicle_count}")


if __name__ == "__main__":
    run()