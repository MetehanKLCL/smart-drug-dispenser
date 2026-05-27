"""
Headless yüz doğrulama (durum makinesi / BLE): canlılık + çoklu örnek eşleştirme.

Skor = 1.0 - face_distance. DB: local_users + face_samples.
"""

from __future__ import annotations

import os
import sqlite3
import time

import cv2
import face_recognition
import numpy as np

from .facade import FaceCamera

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
_PI_BACKEND = os.path.join(_REPO_ROOT, "pi_backend")

MODEL_PATH = os.path.join(_REPO_ROOT, "face_landmarker.task")
if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = os.path.join(_PI_BACKEND, "face_landmarker.task")

DB_PATH = os.environ.get("FACES_DB", os.path.join(_PI_BACKEND, "faces.db"))

DISTANCE_THRESHOLD = 0.55
LIVENESS_TIME_LIMIT = 5.0
EAR_THRESHOLD = 0.5
BLINK_MIN_EAR = 0.25
BLINK_MIN_DURATION = 0.05
BLINK_MAX_DURATION = 0.4
MAR_THRESHOLD = 0.15
MOUTH_MIN_DURATION = 0.1
MOUTH_MAX_DURATION = 0.6

LEFT_EYE = {"top": 159, "bot": 145, "left": 33, "right": 133}
RIGHT_EYE = {"top": 386, "bot": 374, "left": 362, "right": 263}
UPPER_LIP, LOWER_LIP = 13, 14
LEFT_MOUTH, RIGHT_MOUTH = 61, 291
NOSE_TIP = 4


def _ensure_local_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS local_users (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            vector BLOB NOT NULL
        )
        """
    )
    conn.commit()


def _load_users():
    conn = sqlite3.connect(DB_PATH)
    _ensure_local_users_table(conn)

    rows = conn.execute(
        "SELECT patient_id, first_name, last_name, vector FROM local_users"
    ).fetchall()

    has_samples_table = False
    try:
        conn.execute("SELECT 1 FROM face_samples LIMIT 1")
        has_samples_table = True
    except sqlite3.OperationalError:
        pass

    users = []
    for pid, fn, ln, avg_blob in rows:
        name = f"{fn} {ln}"
        vectors = []

        if has_samples_table:
            sample_rows = conn.execute(
                "SELECT vector FROM face_samples WHERE patient_id = ?", (pid,)
            ).fetchall()
            vectors = [np.frombuffer(r[0], dtype=np.float32) for r in sample_rows]

        if not vectors:
            vectors = [np.frombuffer(avg_blob, dtype=np.float32)]

        users.append((pid, name, vectors))

    conn.close()
    return users


def _get_ear(landmarks, eye):
    v = abs(landmarks[eye["top"]].y - landmarks[eye["bot"]].y)
    h = abs(landmarks[eye["right"]].x - landmarks[eye["left"]].x)
    return v / h if h > 0 else 0.0


def _get_mar(landmarks):
    v = abs(landmarks[LOWER_LIP].y - landmarks[UPPER_LIP].y)
    h = abs(landmarks[RIGHT_MOUTH].x - landmarks[LEFT_MOUTH].x)
    return v / h if h > 0 else 0.0


def _run_liveness(cap, landmarker) -> bool:
    blink_detected = False
    eye_was_open = True
    eye_closed_since = None
    blink_min_ear = 1.0

    mouth_detected = False
    mouth_was_closed = True
    mouth_open_since = None

    head_detected = False
    turned_left = False
    turned_right = False

    start = time.time()

    while time.time() - start < LIVENESS_TIME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(img)

        if not result.face_landmarks:
            continue

        lm = result.face_landmarks[0]
        ear = (_get_ear(lm, LEFT_EYE) + _get_ear(lm, RIGHT_EYE)) / 2.0
        mar = _get_mar(lm)
        eye_closed = ear < EAR_THRESHOLD
        mouth_open = mar > MAR_THRESHOLD
        now = time.time()

        if not blink_detected:
            if eye_closed and eye_was_open:
                eye_closed_since = now
                blink_min_ear = ear
            if eye_closed and not eye_was_open:
                blink_min_ear = min(blink_min_ear, ear)
            if not eye_closed and not eye_was_open and eye_closed_since:
                dur = now - eye_closed_since
                if BLINK_MIN_DURATION <= dur <= BLINK_MAX_DURATION and blink_min_ear < BLINK_MIN_EAR:
                    blink_detected = True
                eye_closed_since = None
                blink_min_ear = 1.0
            eye_was_open = not eye_closed

        if not mouth_detected:
            if mouth_open and mouth_was_closed:
                mouth_open_since = now
            if not mouth_open and not mouth_was_closed and mouth_open_since:
                dur = now - mouth_open_since
                if MOUTH_MIN_DURATION <= dur <= MOUTH_MAX_DURATION:
                    mouth_detected = True
                mouth_open_since = None
            mouth_was_closed = not mouth_open

        if not head_detected:
            left_face = lm[234].x
            right_face = lm[454].x
            fw = abs(right_face - left_face)
            if fw > 0:
                nn = (lm[NOSE_TIP].x - left_face) / fw
                if nn < 0.45:
                    turned_right = True
                if nn > 0.55:
                    turned_left = True
                if turned_left and turned_right:
                    head_detected = True

        if blink_detected or mouth_detected or head_detected:
            elapsed = time.time() - start
            signals = []
            if blink_detected:
                signals.append("blink")
            if mouth_detected:
                signals.append("mouth")
            if head_detected:
                signals.append("head")
            print(f"[FaceAuth] Liveness passed in {elapsed:.1f}s: {', '.join(signals)}", flush=True)
            return True

    print("[FaceAuth] Liveness FAILED — no signal detected within time limit", flush=True)
    return False


def _capture_and_match(cam, users, expected_patient_id: str | None = None):
    encodings = []
    for _ in range(5):
        ret, rgb = cam.read_rgb()
        if not ret:
            continue
        encs = face_recognition.face_encodings(rgb)
        if encs:
            encodings.append(encs[0].astype(np.float32))
        if len(encodings) >= 3:
            break

    if not encodings:
        print("[FaceAuth] No face encoding captured", flush=True)
        return None

    avg = np.mean(encodings, axis=0).astype(np.float32)

    # If we know the expected patient, only match against them.
    # Otherwise fall back to best-of-all matching.
    candidates = users
    if expected_patient_id:
        candidates = [u for u in users if u[0] == expected_patient_id]
        if not candidates:
            print(f"[FaceAuth] Expected patient {expected_patient_id[:8]} not in DB — falling back to all", flush=True)
            candidates = users

    best_pid = None
    best_name = None
    best_dist = float("inf")

    for pid, name, sample_vectors in candidates:
        distances = face_recognition.face_distance(sample_vectors, avg)
        min_dist = float(np.min(distances))
        print(f"[FaceAuth]   {name}: best_dist={min_dist:.4f} ({len(sample_vectors)} samples)", flush=True)
        if min_dist < best_dist:
            best_dist = min_dist
            best_pid = pid
            best_name = name

    score = 1.0 - best_dist
    print(f"[FaceAuth] Best match: {best_name} -> score={score:.2f} (dist={best_dist:.4f}, threshold={DISTANCE_THRESHOLD})", flush=True)

    if best_dist <= DISTANCE_THRESHOLD:
        return (best_pid, best_name, score)
    return None


def authenticate_user(expected_patient_id: str | None = None):
    """
    Durum makinesi tarafından thread içinde çağrılır; bloklar.
    expected_patient_id: sadece bu hastayı kontrol et (daha hızlı ve doğru).
    """
    users = _load_users()
    if not users:
        print("[FaceAuth] No registered users in faces.db", flush=True)
        return {"status": "failed", "reason": "no_registered_users"}

    print(f"[FaceAuth] {len(users)} registered user(s) loaded, expecting={expected_patient_id}", flush=True)

    cam = FaceCamera()
    if not cam.open():
        print("[FaceAuth] Camera could not be opened", flush=True)
        return {"status": "failed", "reason": "camera_unavailable"}

    print(f"[FaceAuth] Camera backend: {cam.backend}", flush=True)

    try:
        if HAS_MEDIAPIPE and os.path.exists(MODEL_PATH):
            opts = vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            with vision.FaceLandmarker.create_from_options(opts) as landmarker:
                liveness_ok = _run_liveness(cam, landmarker)
        else:
            print("[FaceAuth] MediaPipe not available — skipping liveness check", flush=True)
            liveness_ok = True

        if not liveness_ok:
            return {"status": "failed", "reason": "liveness_failed"}

        match = _capture_and_match(cam, users, expected_patient_id=expected_patient_id)
        if match is not None:
            pid, name, score = match
            return {
                "status": "success",
                "patient_id": pid,
                "name": name,
                "score": score,
            }
        return {"status": "failed", "reason": "no_face_detected"}
    finally:
        cam.release()
