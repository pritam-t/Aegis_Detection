import cv2
import re
import numpy as np
import os
import json
import time
import uuid
import hashlib
import shutil
import threading
import queue
from datetime import datetime, timezone, timedelta

import torch
from ultralytics import YOLO
from dotenv import load_dotenv
from supabase import create_client

from paddleocr import PaddleOCR

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH      = "bestv8.pt"
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
# PERFORMANCE TUNING — RTX 3050 + CUDA 12
# ──────────────────────────────────────────────
_GPU_AVAILABLE  = torch.cuda.is_available()

FRAME_QUEUE_SIZE  = 128
INFERENCE_BATCH   = 12
INFER_IMG_SIZE    = 640
USE_FP16          = True      # FIX: managed as a module-level variable, not mutated via globals()

OCR_WORKERS       = 2
OCR_QUEUE_SIZE    = 512

FRAME_SKIP        = 0

PENDING_TTL_FRAMES = 90

# FIX: Use a dedicated sentinel object instead of None to avoid false matches
# with legitimate None values (e.g. a failed frame read).
_SENTINEL = object()


# ──────────────────────────────────────────────
# FAST VIDEO CAPTURE
# ──────────────────────────────────────────────
class FastCapture:
    def __init__(self, path: str, queue_size: int = 256):
        self._cap    = cv2.VideoCapture(path)
        self._q      = queue.Queue(maxsize=queue_size)
        self._stop   = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="FastCap")
        self._thread.start()

    def _run(self):
        while not self._stop:
            ret, frame = self._cap.read()
            if not ret:
                self._q.put(None)
                break
            try:
                self._q.put(frame, timeout=0.5)
            except queue.Full:
                pass

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
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def find_overlapping_box(target_box, box_dict: dict, threshold: float = 0.35):
    best_key, best_iou = None, threshold
    for key in box_dict:
        ov = iou(target_box, key)
        if ov > best_iou:
            best_iou = ov
            best_key = key
    return best_key


def correct_plate(text: str) -> str:
    text = text.upper()
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


# ── Plate OCR ────────────────────────────────────────────────────────────────
_ocr_cache: dict      = {}
_cache_lock           = threading.Lock()
_ocr_engine_tls       = threading.local()


def _get_ocr_engine() -> PaddleOCR:
    if not hasattr(_ocr_engine_tls, "engine"):
        # FIX: `use_textline_orientation` was removed in PaddleOCR >= 2.7 (which
        # supports Python 3.12). Use `use_angle_cls` instead, which is the correct
        # equivalent parameter name across all supported versions.
        _ocr_engine_tls.engine = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
        )
    return _ocr_engine_tls.engine


def _crop_hash(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (16, 8))
    return hashlib.md5(small.tobytes()).hexdigest()


def read_plate(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""

    key = _crop_hash(crop)
    with _cache_lock:
        if key in _ocr_cache:
            return _ocr_cache[key]

    if h < 40:
        scale = max(2, int(60 / h))
        crop  = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    try:
        engine  = _get_ocr_engine()
        results = engine.ocr(crop, cls=False)
        # FIX: PaddleOCR >= 2.7 may return None instead of [] for empty results.
        # Guard against both None and empty list before indexing.
        raw = ""
        if results and results[0]:
            raw = "".join(
                line[1][0] for line in results[0]
                if line and len(line) > 1 and line[1]
            )
    except Exception as e:
        print(f"  [OCR ERROR] {e}")
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

    # FIX: shutil.rmtree is safe even if dir has leftover files;
    # os.rmdir raises OSError on non-empty dirs even with try/except in 3.12
    # due to stricter OSError propagation semantics.
    try:
        remaining = [f for f in os.listdir(PLATES_DIR) if f.endswith(".jpg")]
        if not remaining:
            shutil.rmtree(PLATES_DIR, ignore_errors=True)
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
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if FRAME_SKIP > 0 and (frame_no - 1) % (FRAME_SKIP + 1) != 0:
            continue
        frame_q.put((frame_no, frame))

    frame_q.put(_SENTINEL)


def _gpu_worker(model_path: str, frame_q: queue.Queue, ocr_q: queue.Queue, use_fp16: bool):
    # FIX: Accept use_fp16 as a parameter instead of reading the global.
    # Python 3.12 optimizes module globals more aggressively, so mutations
    # via globals() from other threads are not reliably visible here.
    model = YOLO(model_path)

    if _GPU_AVAILABLE:
        model.to("cuda")
        print(f"[GPU]  Running on: {torch.cuda.get_device_name(0)}  |  "
              f"FP16: {use_fp16}  |  Batch: {INFERENCE_BATCH}")
    else:
        print("[GPU]  ⚠️  No CUDA device — running on CPU")

    names = model.names
    batch: list = []
    done  = False

    def flush(batch):
        if not batch:
            return

        resized  = [cv2.resize(f, (INFER_IMG_SIZE, INFER_IMG_SIZE)) for _, f in batch]
        h_orig, w_orig = batch[0][1].shape[:2]
        sx = w_orig / INFER_IMG_SIZE
        sy = h_orig / INFER_IMG_SIZE

        all_results = model(resized, verbose=False, half=use_fp16)

        for (frame_no, orig_frame), result in zip(batch, all_results):
            helmet_off_boxes, plate_boxes, rider_boxes = [], [], []

            for box in result.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = names[cls]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                x1 = int(x1 * sx); x2 = int(x2 * sx)
                y1 = int(y1 * sy); y2 = int(y2 * sy)

                if label == "helmet-off" and conf >= HELMET_OFF_CONF:
                    helmet_off_boxes.append((x1, y1, x2, y2, conf))
                elif label == "numer_plate" and conf >= PLATE_CONF_MIN:
                    plate_boxes.append((x1, y1, x2, y2, conf))
                elif label == "rider":
                    rider_boxes.append((x1, y1, x2, y2, conf))

            ocr_q.put({
                "frame_no":         frame_no,
                "helmet_off_boxes": helmet_off_boxes,
                "plate_boxes":      plate_boxes,
                "rider_boxes":      rider_boxes,
                "frame":            orig_frame,
            })

    while not done:
        try:
            item = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # FIX: Compare against the sentinel object identity, not None.
        if item is _SENTINEL:
            done = True
        else:
            batch.append(item)

        if len(batch) >= INFERENCE_BATCH or (done and batch):
            flush(batch)
            batch = []

    for _ in range(OCR_WORKERS):
        ocr_q.put(_SENTINEL)


def _ocr_worker(ocr_q: queue.Queue, result_q: queue.Queue):
    while True:
        item = ocr_q.get()
        # FIX: sentinel identity check (not None check)
        if item is _SENTINEL:
            result_q.put(_SENTINEL)
            break

        plate_boxes = item["plate_boxes"]

        ocr_results = []
        for (px1, py1, px2, py2, pconf) in plate_boxes:
            crop = item["frame"][py1:py2, px1:px2]
            text = read_plate(crop)
            ocr_results.append({
                "box":        (px1, py1, px2, py2),
                "conf":       pconf,
                "plate_text": text,
                "plate_crop": crop,
            })

        result_q.put({
            "frame_no":         item["frame_no"],
            "helmet_off_boxes": item["helmet_off_boxes"],
            "rider_boxes":      item["rider_boxes"],
            "plate_results":    ocr_results,
            "frame":            item["frame"],
        })


# ──────────────────────────────────────────────
# TWO-STAGE COLLECTOR
# ──────────────────────────────────────────────
def _collector(result_q: queue.Queue, fps: float, total_frames: int, t_start: float):
    pending_violators: dict = {}
    vehicles: dict  = {}

    noise_count    = 0
    sentinels_seen = 0
    last_frame_no  = 0

    while sentinels_seen < OCR_WORKERS:
        item = result_q.get()

        # FIX: sentinel identity check
        if item is _SENTINEL:
            sentinels_seen += 1
            continue

        frame_no       = item["frame_no"]
        last_frame_no  = max(last_frame_no, frame_no)

        helmet_off_boxes = item["helmet_off_boxes"]
        rider_boxes      = item["rider_boxes"]
        plate_results    = item["plate_results"]
        frame            = item["frame"]

        if frame_no % 60 == 0:
            _progress(frame_no, total_frames, t_start, len(vehicles), len(pending_violators))

        stale = [
            rb for rb, data in pending_violators.items()
            if frame_no - data["last_seen_frame"] > PENDING_TTL_FRAMES
        ]
        for rb in stale:
            print(f"  ⏰ EXPIRED | Rider @ {rb} — no plate found within TTL")
            del pending_violators[rb]

        for (hx1, hy1, hx2, hy2, hconf) in helmet_off_boxes:
            ho_box = (hx1, hy1, hx2, hy2)

            rider_box = ho_box
            best_r_iou = 0.25
            for (rx1, ry1, rx2, ry2, _) in rider_boxes:
                ov = iou(ho_box, (rx1, ry1, rx2, ry2))
                if ov > best_r_iou:
                    best_r_iou = ov
                    rider_box  = (rx1, ry1, rx2, ry2)

            matched_pending = find_overlapping_box(rider_box, pending_violators, threshold=0.35)

            already_confirmed = False
            for v in vehicles.values():
                if iou(rider_box, v.get("rider_box", (0,0,0,0))) > 0.35:
                    already_confirmed = True
                    break

            if already_confirmed:
                continue

            if matched_pending is None:
                pending_violators[rider_box] = {
                    "first_frame":    frame_no,
                    "last_seen_frame": frame_no,
                    "best_plate_conf": 0.0,
                    "best_plate_text": "",
                    "best_plate_crop": None,
                    "best_frame":      None,
                    "rider_box":       rider_box,
                }
                print(f"  🔍 PENDING  | Frame {frame_no:>5} | New helmet-off rider detected @ {rider_box}")
            else:
                pending_violators[matched_pending]["last_seen_frame"] = frame_no

        for plate_data in plate_results:
            px1, py1, px2, py2 = plate_data["box"]
            pconf              = plate_data["conf"]
            plate_text_raw     = plate_data["plate_text"]
            plate_crop         = plate_data["plate_crop"]

            if not is_valid_plate(plate_text_raw):
                noise_count += 1
                if plate_text_raw:
                    print(f"  [NOISE] '{plate_text_raw}'")
                continue

            plate_text = plate_text_raw.strip().upper()

            best_pending_key  = None
            best_pending_score = 0.0

            for rb, data in pending_violators.items():
                rx1, ry1, rx2, ry2 = rb
                horiz_overlap = max(0, min(px2, rx2) - max(px1, rx1))
                rider_width   = max(1, rx2 - rx1)
                horiz_score   = horiz_overlap / rider_width

                rider_height  = max(1, ry2 - ry1)
                vert_dist     = py1 - ry2
                vert_score    = 1.0 if 0 <= vert_dist <= rider_height * 2 else (
                                0.5 if -rider_height <= vert_dist < 0 else 0.0
                )

                score = horiz_score * 0.6 + vert_score * 0.4
                if score > best_pending_score and score > 0.3:
                    best_pending_score = score
                    best_pending_key   = rb

            if best_pending_key is None:
                continue

            pending = pending_violators[best_pending_key]

            if pconf > pending["best_plate_conf"]:
                pending["best_plate_conf"] = pconf
                pending["best_plate_text"] = plate_text
                pending["best_plate_crop"] = plate_crop
                pending["best_frame"]      = frame
                pending["last_seen_frame"] = frame_no
                print(f"  📋 PLATE   | Frame {frame_no:>5} | Pending rider → '{plate_text}' "
                      f"conf: {pconf:.2%}")

            if pending["best_plate_conf"] >= PLATE_CONF_MIN:
                pt   = pending["best_plate_text"]
                pc   = pending["best_plate_conf"]
                crop = pending["best_plate_crop"]
                frm  = pending["best_frame"]

                match_key = find_matching_vehicle(pt, vehicles)

                if match_key is None:
                    img_path   = os.path.join(PLATES_DIR, f"plate_{pt}.jpg")
                    frame_path = os.path.join(PLATES_DIR, f"frame_{pt}.jpg")
                    cv2.imwrite(img_path, crop)
                    cv2.imwrite(frame_path, frm)

                    vehicles[pt] = {
                        "frame":         frame_no,
                        "timestamp_sec": round(frame_no / fps, 2),
                        "plate_text":    pt,
                        "plate_conf":    round(pc, 4),
                        "plate_image":   img_path,
                        "frame_image":   frame_path,
                        "rider_box":     best_pending_key,
                    }
                    del pending_violators[best_pending_key]

                    print(f"  🚨 CONFIRMED | Frame {frame_no:>5} ({frame_no/fps:.1f}s) | "
                          f"Plate: {pt:<12} | Conf: {pc:.2%} | 💾 Saved")

                else:
                    existing = vehicles[match_key]
                    if pc > existing["plate_conf"]:
                        old_conf      = existing["plate_conf"]
                        plate_changed = existing["plate_text"] != pt

                        img_path   = existing["plate_image"]
                        frame_path = existing["frame_image"]

                        if plate_changed:
                            img_path   = os.path.join(PLATES_DIR, f"plate_{pt}.jpg")
                            frame_path = os.path.join(PLATES_DIR, f"frame_{pt}.jpg")
                            for old_f in [existing["plate_image"], existing["frame_image"]]:
                                if os.path.exists(old_f):
                                    os.remove(old_f)
                            del vehicles[match_key]
                            match_key = pt

                        cv2.imwrite(img_path, crop)
                        cv2.imwrite(frame_path, frm)

                        vehicles[match_key] = {
                            "frame":         frame_no,
                            "timestamp_sec": round(frame_no / fps, 2),
                            "plate_text":    pt,
                            "plate_conf":    round(pc, 4),
                            "plate_image":   img_path,
                            "frame_image":   frame_path,
                            "rider_box":     best_pending_key,
                        }
                        tag = " (corrected)" if plate_changed else ""
                        print(f"  ⬆️  UPGRADE  | Frame {frame_no:>5} ({frame_no/fps:.1f}s) | "
                              f"Plate: {pt:<12} | {old_conf:.2%} → {pc:.2%}{tag}")

                    if best_pending_key in pending_violators:
                        del pending_violators[best_pending_key]

    if pending_violators:
        print(f"\n[AEGIS] ⚠️  {len(pending_violators)} rider(s) had helmet-off "
              f"but no valid plate was found:")
        for rb, data in pending_violators.items():
            print(f"         Rider @ {rb}  |  First seen: frame {data['first_frame']}  "
                  f"|  Best OCR: '{data['best_plate_text']}'  "
                  f"conf: {data['best_plate_conf']:.2%}")

    return vehicles, noise_count, last_frame_no


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def run():
    os.makedirs(PLATES_DIR, exist_ok=True)
    for f in os.listdir(PLATES_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(PLATES_DIR, f))

    # FIX: use a local variable for fp16 decision; do NOT mutate the global
    # via globals() — unreliable in Python 3.12 across threads.
    use_fp16 = USE_FP16

    if _GPU_AVAILABLE:
        # FIX: torch.version.cuda can be None on CPU-only torch builds.
        # Always guard it before string formatting.
        cuda_ver = torch.version.cuda or "unknown"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[AEGIS] ✅ GPU : {gpu_name}")
        print(f"[AEGIS]    CUDA: {cuda_ver}  |  VRAM: {vram_gb:.1f} GB")
        if use_fp16:
            try:
                _test = torch.zeros(1, device="cuda", dtype=torch.float16)
                del _test
                print(f"[AEGIS]    FP16: ✅ verified")
            except Exception as e:
                print(f"[AEGIS]    FP16: ❌ {e} — falling back to FP32")
                use_fp16 = False
    else:
        print("[AEGIS] ⚠️  No CUDA GPU found — running on CPU (slow)")

    print(f"[AEGIS] ⚡ TURBO MODE")
    print(f"[AEGIS] Model          : {MODEL_PATH}")
    print(f"[AEGIS] FP16           : {use_fp16}")
    print(f"[AEGIS] Infer batch    : {INFERENCE_BATCH}  @  {INFER_IMG_SIZE}px")
    print(f"[AEGIS] OCR engine     : PaddleOCR (GPU={_GPU_AVAILABLE})  ×{OCR_WORKERS} workers")
    print(f"[AEGIS] Frame skip     : {FRAME_SKIP} ({'every frame' if FRAME_SKIP == 0 else f'every {FRAME_SKIP+1}th frame'})")
    print(f"[AEGIS] Pending TTL    : {PENDING_TTL_FRAMES} frames (~{PENDING_TTL_FRAMES/30:.1f}s @ 30fps)")

    cap = FastCapture(INPUT_VIDEO, queue_size=FRAME_QUEUE_SIZE * 2)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    eff_fps      = fps / (FRAME_SKIP + 1) if FRAME_SKIP > 0 else fps
    print(f"[AEGIS] Video          : @ {fps:.0f}fps | {total_frames} frames "
          f"→ effective {eff_fps:.1f}fps")
    print(f"[AEGIS] Location       : {LOCATION}")
    print(f"[AEGIS] Violation fine : ₹{VIOLATION_FINE}")
    print("-" * 60)

    frame_q  = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    ocr_q    = queue.Queue(maxsize=OCR_QUEUE_SIZE)
    result_q = queue.Queue()

    t_start = time.time()

    reader = threading.Thread(
        target=_reader_thread, args=(cap, frame_q),
        daemon=True, name="Reader",
    )
    reader.start()

    # FIX: pass use_fp16 explicitly so the GPU thread doesn't need to read globals
    gpu_thread = threading.Thread(
        target=_gpu_worker, args=(MODEL_PATH, frame_q, ocr_q, use_fp16),
        daemon=True, name="GPU",
    )
    gpu_thread.start()

    ocr_threads = [
        threading.Thread(
            target=_ocr_worker, args=(ocr_q, result_q),
            daemon=True, name=f"OCR-{i}",
        )
        for i in range(OCR_WORKERS)
    ]
    for t in ocr_threads:
        t.start()

    vehicles, noise_count, frame_no = _collector(result_q, fps, total_frames, t_start)

    reader.join()
    gpu_thread.join()
    for t in ocr_threads:
        t.join()
    cap.release()

    elapsed = time.time() - t_start
    print(f"\n[AEGIS] ✅ Done in {elapsed:.1f}s  |  {len(vehicles)} violation(s)  |  "
          f"OCR cache size: {len(_ocr_cache)}")

    supabase_results = push_to_supabase(vehicles)

    final = sorted(vehicles.values(), key=lambda r: r["frame"])
    for i, rec in enumerate(final, 1):
        rec["violation_no"] = i
        sb = supabase_results.get(rec["plate_text"], {})
        rec["plate_url"]       = sb.get("plate_url", "")
        rec["frame_url"]       = sb.get("frame_url", "")
        rec["supabase_row_id"] = sb.get("row_id", "")
        rec["pushed_at"]       = sb.get("timestamp", "")
        rec.pop("rider_box", None)

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


def _progress(frame_no, total, t_start, vehicle_count, pending_count):
    elapsed = time.time() - t_start
    fps_d   = frame_no / elapsed if elapsed > 0 else 0
    pct     = frame_no / total * 100 if total else 0
    print(f"  [Progress] {frame_no}/{total} ({pct:.1f}%)  "
          f"Speed: {fps_d:.1f} fps  |  Confirmed: {vehicle_count}  "
          f"Pending: {pending_count}")


if __name__ == "__main__":
    run()