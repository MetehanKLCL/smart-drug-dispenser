import cv2
import time
import math
import os
import urllib.request
import requests
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()

import logging

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

TIMEOUT_SECONDS = 60


def _float_env(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Ağız: histerezis + süre — el ağıza gelince MAR kısa süre oynayıp anında “alındı” olmasın
MAR_OPEN = _float_env("SWALLOW_MAR_OPEN", 0.18)
MAR_CLOSE = _float_env("SWALLOW_MAR_CLOSE", 0.11)
MOUTH_OPEN_HOLD_SEC = _float_env("SWALLOW_MOUTH_OPEN_HOLD", 0.40)
MOUTH_CLOSE_HOLD_SEC = _float_env("SWALLOW_MOUTH_CLOSE_HOLD", 0.35)
MIN_OPEN_TO_CLOSE_SEC = _float_env("SWALLOW_MIN_OPEN_TO_CLOSE", 0.50)

# Burun / yüz merkezine yakınlık (normalize mesafe)
HAND_FACE_DIST = _float_env("SWALLOW_HAND_FACE_DIST", 0.42)
# İlaç götürürken daha anlamlı: işaret parmağı ucu + başparmak vs. ağız merkezi
HAND_MOUTH_DIST = _float_env("SWALLOW_HAND_MOUTH_DIST", 0.36)
# Klasik solutions.hands — Pi’de düşük eşik + static_image genelde daha iyi yakalar
MIN_HAND_DETECTION_CONF = _float_env("SWALLOW_MIN_HAND_DETECTION", 0.20)
MIN_HAND_TRACKING_CONF = _float_env("SWALLOW_MIN_HAND_TRACKING", 0.20)


def _legacy_hand_model_complexity() -> int:
    raw = os.environ.get("SWALLOW_HAND_MODEL_COMPLEXITY", "0").strip()
    try:
        v = int(raw)
        return 1 if v >= 1 else 0
    except ValueError:
        return 0


WIN_DISPENSE = "Swallow Monitor"


def ensure_hand_model() -> None:
    """
    El takibi için gerekenler:
    - Python: `mediapipe` paketi (yüz ile aynı; requirements-pi.txt'te zaten var).
    - Model dosyası: `hand_landmarker.task` — pip ile gelmez; bu fonksiyon yoksa Google'dan indirir.

    Dosya yolu proje kökünde: face_authentication'dan iki üst dizin (örn. ~/dispenser/hand_landmarker.task).
    """
    path = os.path.abspath(HAND_MODEL_PATH)
    min_bytes = 50_000  # bozuk/kısmi indirmeyi yakalamak için kabaca eşik

    if os.path.isfile(path) and os.path.getsize(path) >= min_bytes:
        log.info("Hand Landmarker model OK: %s (%d bytes)", path, os.path.getsize(path))
        return

    if os.path.isfile(path) and os.path.getsize(path) < min_bytes:
        log.warning("hand_landmarker.task çok küçük/bozuk görünüyor — yeniden indiriliyor: %s", path)
        try:
            os.remove(path)
        except OSError:
            pass

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info("Hand Landmarker indiriliyor → %s", path)
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    except Exception as e:
        log.exception(
            "hand_landmarker.task indirilemedi (ağ?). Elle: wget -O %s %s",
            path,
            HAND_MODEL_URL,
        )
        raise RuntimeError(
            f"El modeli yok/indirilemedi: {path}. mediapipe kurulu olmalı; .task dosyası ayrıdır."
        ) from e

    if not os.path.isfile(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"hand_landmarker.task geçersiz veya boş: {path}")

    log.info("Hand Landmarker kaydedildi: %s (%d bytes)", path, os.path.getsize(path))


def get_mar(landmarks):
    """Mouth Aspect Ratio."""
    vertical = abs(landmarks[LOWER_LIP].y - landmarks[UPPER_LIP].y)
    horizontal = abs(landmarks[RIGHT_MOUTH].x - landmarks[LEFT_MOUTH].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def min_hand_to_face_distance(hand_landmarks, face_landmarks):
    """Burun (landmark 4) ile el arası en kısa mesafe — parmak tabanları dahil."""
    nose = face_landmarks[4]
    best = 2.0
    for idx in (0, 5, 8, 9, 12, 16, 20, 4):  # bilek, MCP’ler, parmak uçları, başparmak ucu
        p = hand_landmarks[idx]
        d = math.hypot(p.x - nose.x, p.y - nose.y)
        if d < best:
            best = d
    return best


def min_hand_to_mouth_distance(hand_landmarks, face_landmarks):
    """Ağız merkezine (dudak kutusu) — ilacı ağza götürme hareketi ile uyumlu."""
    mx = (face_landmarks[LEFT_MOUTH].x + face_landmarks[RIGHT_MOUTH].x) * 0.5
    my = (face_landmarks[UPPER_LIP].y + face_landmarks[LOWER_LIP].y) * 0.5
    best = 2.0
    for idx in (8, 4, 12, 16, 20, 5, 0):  # index tip, thumb tip, diğer uçlar, bilek
        p = hand_landmarks[idx]
        best = min(best, math.hypot(p.x - mx, p.y - my))
    return best


def detect_gesture(hand, is_right_hand=True):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand[tips[0]].y < hand[tips[0] - 2].y:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if hand[tips[i]].y < hand[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)
    if fingers == [0, 1, 0, 0, 0]:
        return "One"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "High Five"
    elif total == 0:
        return "Fist"
    else:
        return f"Fingers: {total}"


def draw_mouth_landmarks(frame, face_landmarks, w, h):
    """Draw mouth landmark points on frame."""
    mouth_indices = [
        UPPER_LIP,
        LOWER_LIP,
        LEFT_MOUTH,
        RIGHT_MOUTH,
        78,
        308,
        191,
        80,
        81,
        82,
        87,
        88,
        95,
        178,
        317,
        402,
        310,
        311,
        312,
        415,
        324,
        318,
    ]

    for lm in face_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    for idx in mouth_indices:
        lm = face_landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 165, 255), -1)


def draw_hand_landmarks(frame, hand_landmarks, w, h):
    """OpenCV ile iskelet — drawing_utils bazı ortamlarda sessizce uyumsuz; Pi’de garanti."""
    for conn in mp.solutions.hands.HAND_CONNECTIONS:
        a, b = conn
        pa = hand_landmarks[a]
        pb = hand_landmarks[b]
        cv2.line(
            frame,
            (int(pa.x * w), int(pa.y * h)),
            (int(pb.x * w), int(pb.y * h)),
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 200, 255), -1, cv2.LINE_AA)
    w0 = hand_landmarks[0]
    cv2.circle(frame, (int(w0.x * w), int(w0.y * h)), 10, (255, 128, 0), 2, cv2.LINE_AA)


def log_intake(patient_id: str, status: str, schedule_id: str = None):
    """Sends dispensing log to AWS API."""
    api = os.getenv("API_URL")
    if not api:
        log.debug("API_URL not set; skip remote intake log")
        return
    try:
        requests.post(
            f"{api.rstrip('/')}/dispensing-logs",
            json={
                "patient_id": patient_id,
                "schedule_id": schedule_id,
                "status": status,
                "device_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            timeout=10,
        )
        log.info("Intake log sent → status=%s", status)
    except Exception as e:
        log.warning("Intake log failed: %s", e)


def _resolve_reader(camera_or_reader):
    """Return (read_fn, cleanup_cap_or_none). read_fn -> (ok, bgr)."""
    if callable(camera_or_reader):
        return camera_or_reader, None
    cap = camera_or_reader

    def _read():
        from face_authentication.camera import read_frame

        return read_frame(cap)

    return _read, cap


def monitor_intake(
    camera_or_reader,
    patient_id: str,
    schedule_id: str = None,
    *,
    use_opencv_window: bool = False,
    on_frame=None,
    timeout_seconds: int | None = None,
    log_to_api: bool = True,
):
    """
    Monitor intake: hand/cup near face + mouth open/close → "taken".

    camera_or_reader: OpenCV cap, Picamera2, or callable () -> (bool, bgr_frame)
    use_opencv_window: cv2.imshow (Mac/debug). Kiosk should use on_frame + pygame instead.
    on_frame: optional (frame_bgr, meta_dict) per iteration for kiosk UI.
    log_to_api: set False when the caller handles logging (e.g. kiosk_app via sync_queue)
                to avoid double-logging to AWS.
    """
    timeout = TIMEOUT_SECONDS if timeout_seconds is None else int(timeout_seconds)
    read_fn, _cap = _resolve_reader(camera_or_reader)

    # Load patient face encodings for identity check during swallow monitoring.
    # If encodings can't be loaded, face check is skipped (graceful fallback).
    _patient_encodings = []
    try:
        import sqlite3 as _sq
        import sys as _sys
        _pi_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _pi_dir not in _sys.path:
            _sys.path.insert(0, _pi_dir)
        from state_machine import local_db_path as _ldb
        _conn = _sq.connect(_ldb())
        _rows = _conn.execute(
            "SELECT vector FROM face_samples WHERE patient_id=?", (patient_id,)
        ).fetchall()
        if not _rows:
            _row = _conn.execute(
                "SELECT vector FROM local_users WHERE patient_id=?", (patient_id,)
            ).fetchone()
            if _row:
                _rows = [_row]
        _conn.close()
        _patient_encodings = [np.frombuffer(r[0], dtype=np.float32) for r in _rows]
        log.info("Swallow: %d yüz şablonu yüklendi (patient=%s)", len(_patient_encodings), patient_id)
    except Exception as _fe:
        log.warning("Swallow: yüz şablonu yüklenemedi (%s) — kimlik doğrulama atlanıyor", _fe)

    _FACE_VERIFY_THRESHOLD = 0.55   # distance; closer = stricter
    _FACE_VERIFY_INTERVAL  = 15     # her N karede bir doğrulama başlat
    _FACE_VERIFIED_TTL     = 4.0    # son doğrulamadan bu kadar saniye geçmişse geçerli say
    _face_verify_tick = 0
    _face_verified_at: float | None = None  # son başarılı doğrulama zamanı

    # Yüz doğrulama arka planda thread'de çalışır — ana döngüyü bloklamaz
    import threading as _threading
    _verify_lock = _threading.Lock()
    _verify_running = [False]

    if use_opencv_window:
        cv2.destroyAllWindows()

    # VIDEO mode: MediaPipe face landmarker uses temporal smoothing → faster per frame
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    mp_hands = mp.solutions.hands

    hand_near_face = False
    mouth_opened = False
    mouth_closed = False
    taken = False
    start_time = time.time()
    cached_hand_landmarks = None
    cached_handedness = True
    outcome = None
    _mar_high_since = None
    _mar_low_since = None
    mouth_opened_at = None

    log.info("Swallow monitor started (timeout=%ss)", timeout)

    with vision.FaceLandmarker.create_from_options(face_options) as face_detector:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=_legacy_hand_model_complexity(),
            min_detection_confidence=MIN_HAND_DETECTION_CONF,
            min_tracking_confidence=MIN_HAND_TRACKING_CONF,
        ) as hands_sol:
            while outcome is None:
                if use_opencv_window and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                ret, frame = read_fn()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)

                now = time.time()

                if now - start_time > timeout:
                    log.info("Swallow monitor timeout — missed")
                    if log_to_api:
                        log_intake(patient_id, "missed", schedule_id)
                    outcome = "missed"
                    break

                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                h, w, _ = frame.shape
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                face_result = face_detector.detect_for_video(mp_img, int(now * 1000))
                face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None

                hr = hands_sol.process(rgb)
                n_hands_raw = len(hr.multi_hand_landmarks) if hr.multi_hand_landmarks else 0
                cached_hand_landmarks = None
                cached_handedness = True
                if hr.multi_hand_landmarks:
                    lm_lists = list(hr.multi_hand_landmarks)
                    rights = []
                    if hr.multi_handedness:
                        for hdat in hr.multi_handedness:
                            try:
                                lab = hdat.classification[0].label
                                rights.append(lab == "Right")
                            except (IndexError, AttributeError):
                                rights.append(True)
                    while len(rights) < len(lm_lists):
                        rights.append(True)

                    if face_landmarks:
                        best_lm = None
                        best_right = True
                        best_d = 2.0
                        for i, hlm in enumerate(lm_lists):
                            lm = hlm.landmark
                            d = min(
                                min_hand_to_face_distance(lm, face_landmarks),
                                min_hand_to_mouth_distance(lm, face_landmarks),
                            )
                            if d < best_d:
                                best_d = d
                                best_lm = lm
                                best_right = rights[i]
                        cached_hand_landmarks = best_lm
                        cached_handedness = best_right
                    else:
                        cached_hand_landmarks = lm_lists[0].landmark
                        cached_handedness = rights[0]

                hand_landmarks = cached_hand_landmarks

                gesture = None

                if hand_landmarks:
                    draw_hand_landmarks(frame, hand_landmarks, w, h)
                    gesture = detect_gesture(hand_landmarks, cached_handedness)
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 165, 0),
                        2,
                    )

                # ── Yüz kimliği doğrulama (arka plan thread — ana döngüyü bloklamaz) ──
                # FaceLandmarker'dan gelen bbox kullanılır → yavaş HOG detection atlanır.
                if _patient_encodings and face_landmarks:
                    _face_verify_tick += 1
                    if _face_verify_tick % _FACE_VERIFY_INTERVAL == 0 and not _verify_running[0]:
                        # MediaPipe landmark'lardan yüz bbox'ı hesapla
                        _xs = [lm.x for lm in face_landmarks]
                        _ys = [lm.y for lm in face_landmarks]
                        _pad = 20
                        _bbox = (
                            max(0, int(min(_ys) * h) - _pad),   # top
                            min(w, int(max(_xs) * w) + _pad),   # right
                            min(h, int(max(_ys) * h) + _pad),   # bottom
                            max(0, int(min(_xs) * w) - _pad),   # left
                        )
                        _snap = rgb.copy()
                        _encs_ref = _patient_encodings
                        _thr = _FACE_VERIFY_THRESHOLD

                        def _verify_worker(snap, bbox, encs_ref, thr):
                            try:
                                import face_recognition as _fr
                                _e = _fr.face_encodings(snap, [bbox])
                                if _e:
                                    _d = float(np.min(_fr.face_distance(encs_ref, _e[0])))
                                    with _verify_lock:
                                        nonlocal _face_verified_at
                                        if _d <= thr:
                                            _face_verified_at = time.time()
                                        else:
                                            log.warning(
                                                "Swallow: yanlış yüz (dist=%.3f) — alındı engellendi", _d
                                            )
                                            _face_verified_at = None
                            except Exception as _ve:
                                log.debug("Swallow face verify error: %s", _ve)
                            finally:
                                _verify_running[0] = False

                        _verify_running[0] = True
                        _threading.Thread(
                            target=_verify_worker,
                            args=(_snap, _bbox, _encs_ref, _thr),
                            daemon=True,
                        ).start()

                # Yüz doğrulandı mı? (son TTL saniye içinde)
                _face_ok = (
                    not _patient_encodings  # şablon yoksa atla
                    or (_face_verified_at is not None and now - _face_verified_at < _FACE_VERIFIED_TTL)
                )

                if face_landmarks and hand_landmarks:
                    dn = min_hand_to_face_distance(hand_landmarks, face_landmarks)
                    dm = min_hand_to_mouth_distance(hand_landmarks, face_landmarks)
                    if dn < HAND_FACE_DIST or dm < HAND_MOUTH_DIST:
                        hand_near_face = True

                if face_landmarks:
                    draw_mouth_landmarks(frame, face_landmarks, w, h)
                    mar = get_mar(face_landmarks)
                    trigger = hand_near_face

                    if (trigger or mouth_opened) and _face_ok:
                        if mar > MAR_OPEN:
                            _mar_low_since = None
                            if not mouth_opened:
                                if _mar_high_since is None:
                                    _mar_high_since = now
                                elif (now - _mar_high_since) >= MOUTH_OPEN_HOLD_SEC:
                                    mouth_opened = True
                                    mouth_opened_at = now
                                    _mar_high_since = None
                                    log.debug(
                                        "Swallow: ağız açık onayı (%.2fs MAR>%.2f)",
                                        MOUTH_OPEN_HOLD_SEC,
                                        MAR_OPEN,
                                    )
                        else:
                            _mar_high_since = None
                            if mouth_opened and not mouth_closed:
                                if mar < MAR_CLOSE:
                                    if _mar_low_since is None:
                                        _mar_low_since = now
                                    elif (now - _mar_low_since) >= MOUTH_CLOSE_HOLD_SEC:
                                        if (
                                            mouth_opened_at is not None
                                            and (now - mouth_opened_at) >= MIN_OPEN_TO_CLOSE_SEC
                                        ):
                                                mouth_closed = True
                                                taken = True
                                                log.info("Swallow: ağız kapandı — alındı")
                                        else:
                                            _mar_low_since = None
                                else:
                                    _mar_low_since = None
                    else:
                        _mar_high_since = None
                        _mar_low_since = None

                remaining = max(0.0, timeout - (now - start_time))
                meta = {
                    "remaining": remaining,
                    "hand_near": hand_near_face,
                    "mouth_opened": mouth_opened,
                    "mouth_closed": mouth_closed,
                    "taken": taken,
                    "gesture": gesture,
                }

                if taken:
                    if log_to_api:
                        import threading as _thr
                        _thr.Thread(
                            target=log_intake,
                            args=(patient_id, "taken", schedule_id),
                            daemon=True,
                        ).start()
                    cv2.putText(
                        frame,
                        "MEDICATION TAKEN",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    if use_opencv_window:
                        cv2.imshow(WIN_DISPENSE, frame)
                        cv2.waitKey(2000)
                    elif on_frame is not None:
                        on_frame(frame, meta)
                    outcome = "taken"
                    break

                parts = []
                parts.append("Hand: OK" if hand_near_face else "Hand: -")
                parts.append("Mouth: OK" if mouth_opened else "Mouth: -")
                cv2.putText(
                    frame,
                    "  ".join(parts),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{remaining:.0f}s",
                    (w - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (200, 200, 200),
                    2,
                )
                dbg = f"hands={n_hands_raw} det={MIN_HAND_DETECTION_CONF:.2f}"
                cv2.putText(
                    frame,
                    dbg,
                    (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 255, 180) if n_hands_raw else (100, 100, 255),
                    1,
                    cv2.LINE_AA,
                )

                if use_opencv_window:
                    cv2.imshow(WIN_DISPENSE, frame)
                if on_frame is not None:
                    on_frame(frame, meta)

    if use_opencv_window:
        cv2.destroyAllWindows()
    final = outcome if outcome else "missed"
    log.info("Swallow monitor finished: %s", final)
    return final


if __name__ == "__main__":
    from face_authentication.camera import get_camera, release_camera

    logging.basicConfig(level=logging.INFO)
    cap = get_camera()
    if cap is None:
        print("Camera cannot be opened.")
    else:
        try:
            monitor_intake(cap, patient_id="test", use_opencv_window=True)
        finally:
            release_camera(cap)
