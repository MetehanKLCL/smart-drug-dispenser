# ⚠️  LEGACY SCRIPT — KULLANMA!
# Bu dosya ESKİ auth sistemine aittir:
#   DB_PATH = ~/dispenser/faces.db   (kök dizin, ESKİ)
#
# Yeni sistem ~/dispenser/pi_backend/faces.db kullanır + face_samples tablosu ile
# multi-sample matching yapar. Bu script yalnızca local_users ortalamasını kullanır
# ve yeni kayıtları GÖRMEZ.
#
# Auth için: kiosk servisi (medidispense.service) veya headless_auth.py kullanın.

import cv2
import time
import sqlite3
import requests
import os
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
import face_recognition
from mediapipe.tasks.python import vision
from camera import get_camera, read_frame, release_camera

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_landmarker.task")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faces.db")

DISTANCE_THRESHOLD   = 0.5
EAR_THRESHOLD        = 0.5
BLINK_MIN_EAR        = 0.25   # EAR must drop below this during blink — filters photo shaking
MAR_THRESHOLD        = 0.15
TIME_LIMIT           = 5
BLINK_MIN_DURATION   = 0.05
BLINK_MAX_DURATION   = 0.4
MOUTH_MIN_DURATION   = 0.1
MOUTH_MAX_DURATION   = 0.6
HEAD_TURN_THRESHOLD  = 0.08

# Eye landmarks
LEFT_EYE  = {"top": 159, "bot": 145, "left": 33,  "right": 133}
RIGHT_EYE = {"top": 386, "bot": 374, "left": 362, "right": 263}

# Mouth landmarks
UPPER_LIP   = 13
LOWER_LIP   = 14
LEFT_MOUTH  = 61
RIGHT_MOUTH = 291

# Head turn landmark
NOSE_TIP = 4


def load_users(conn):
    rows = conn.execute("SELECT patient_id, first_name, last_name, vector FROM local_users").fetchall()
    users = []
    for patient_id, first_name, last_name, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        users.append((patient_id, first_name, last_name, vec))
    return users


def frame_to_encoding(frame):
    """BGR frame → face_recognition 128-dim embedding."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None
    return encodings[0].astype(np.float32)


def get_ear(landmarks, eye):
    """Eye Aspect Ratio — normalized, distance-independent."""
    vertical   = abs(landmarks[eye["top"]].y - landmarks[eye["bot"]].y)
    horizontal = abs(landmarks[eye["right"]].x - landmarks[eye["left"]].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def get_mar(landmarks):
    """Mouth Aspect Ratio — normalized, distance-independent."""
    vertical   = abs(landmarks[LOWER_LIP].y - landmarks[UPPER_LIP].y)
    horizontal = abs(landmarks[RIGHT_MOUTH].x - landmarks[LEFT_MOUTH].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def find_best_match(users, query_vec):
    """Find the user with the lowest face distance."""
    best_patient_id = None
    best_name       = None
    best_distance   = float("inf")
    for patient_id, first_name, last_name, vec in users:
        dist = face_recognition.face_distance([vec], query_vec)[0]
        print(f"{first_name} {last_name} distance: {dist:.4f}")
        if dist < best_distance:
            best_distance   = dist
            best_patient_id = patient_id
            best_name       = f"{first_name} {last_name}"
    return best_patient_id, best_name, best_distance


def main(cap):
    conn  = sqlite3.connect(DB_PATH)
    users = load_users(conn)

    if not users:
        print("No registered users. Run register.py first.")
        return

    print(f"{len(users)} users loaded.")

    if cap is None:
        print("Camera cannot be opened.")
        return

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    STATE_WAITING  = "waiting"
    STATE_LIVENESS = "liveness"
    STATE_RESULT   = "result"

    state            = STATE_WAITING
    # Blink tracking
    blink_detected   = False
    eye_was_open     = True
    eye_closed_since = None
    blink_min_ear    = 1.0    # Track lowest EAR during closed phase
    # Mouth tracking
    mouth_detected   = False
    mouth_was_closed = True
    mouth_open_since = None
    # Head turn tracking
    head_detected    = False
    turned_left      = False
    turned_right     = False

    start_time       = None
    result_text      = ""
    result_color     = (255, 255, 255)
    result_until     = None

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = read_frame(cap)
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(img)

            h, w, _ = frame.shape
            now = time.time()

            if state == STATE_RESULT and now > result_until:
                state            = STATE_WAITING
                blink_detected   = False
                eye_was_open     = True
                eye_closed_since = None
                blink_min_ear    = 1.0
                mouth_detected   = False
                mouth_was_closed = True
                mouth_open_since = None
                head_detected    = False
                turned_left      = False
                turned_right     = False

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

                ear        = (get_ear(landmarks, LEFT_EYE) + get_ear(landmarks, RIGHT_EYE)) / 2.0
                mar        = get_mar(landmarks)
                nose_x     = landmarks[NOSE_TIP].x
                eye_closed = ear < EAR_THRESHOLD
                mouth_open = mar > MAR_THRESHOLD

                if state == STATE_WAITING:
                    start_time       = now
                    blink_detected   = False
                    eye_was_open     = True
                    eye_closed_since = None
                    blink_min_ear    = 1.0
                    mouth_detected   = False
                    mouth_was_closed = True
                    mouth_open_since = None
                    head_detected    = False
                    turned_left      = False
                    turned_right     = False
                    state            = STATE_LIVENESS

                elif state == STATE_LIVENESS:

                    # --- Blink detection with duration + min EAR check ---
                    if not blink_detected:
                        if eye_closed and eye_was_open:
                            # Eye just closed — start tracking
                            eye_closed_since = now
                            blink_min_ear    = ear
                        if eye_closed and not eye_was_open:
                            # Eye still closed — track lowest EAR
                            blink_min_ear = min(blink_min_ear, ear)
                        if not eye_closed and not eye_was_open:
                            # Eye just opened — evaluate blink
                            if eye_closed_since is not None:
                                duration = now - eye_closed_since
                                if (BLINK_MIN_DURATION <= duration <= BLINK_MAX_DURATION
                                        and blink_min_ear < BLINK_MIN_EAR):
                                    blink_detected = True
                                eye_closed_since = None
                                blink_min_ear    = 1.0
                        eye_was_open = not eye_closed

                    # --- Mouth open/close detection with duration check ---
                    if not mouth_detected:
                        if mouth_open and mouth_was_closed:
                            mouth_open_since = now
                        if not mouth_open and not mouth_was_closed:
                            if mouth_open_since is not None:
                                duration = now - mouth_open_since
                                if MOUTH_MIN_DURATION <= duration <= MOUTH_MAX_DURATION:
                                    mouth_detected = True
                                mouth_open_since = None
                        mouth_was_closed = not mouth_open

                    # --- Head turn detection (nose X relative to face width) ---
                    if not head_detected:
                        left_face  = landmarks[234].x
                        right_face = landmarks[454].x
                        face_width = abs(right_face - left_face)
                        if face_width > 0:
                            nose_norm = (nose_x - left_face) / face_width
                            if nose_norm < 0.45:
                                turned_right = True
                            if nose_norm > 0.55:
                                turned_left = True
                            if turned_left and turned_right:
                                head_detected = True

                    # Liveness passes if ANY ONE of the three is detected
                    liveness_passed = blink_detected or mouth_detected or head_detected

                    remaining = max(0, TIME_LIMIT - (now - start_time))
                    cv2.putText(frame, f"{remaining:.1f}s",
                                (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
                    cv2.putText(frame, f"Blink: {'OK' if blink_detected else '-'}  "
                                       f"Mouth: {'OK' if mouth_detected else '-'}  "
                                       f"Head: {'OK' if head_detected else '-'}",
                                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if now - start_time >= TIME_LIMIT:
                        if liveness_passed:
                            encodings = []
                            for _ in range(3):
                                ret2, frame2 = read_frame(cap)
                                if not ret2:
                                    break
                                enc = frame_to_encoding(frame2)
                                if enc is not None:
                                    encodings.append(enc)

                            if encodings:
                                avg_enc                    = np.mean(encodings, axis=0).astype(np.float32)
                                patient_id, name, dist     = find_best_match(users, avg_enc)
                                if dist <= DISTANCE_THRESHOLD:
                                    result_text  = f"ACCESS GRANTED: {name}"
                                    result_color = (0, 255, 0)
                                    # → GPIO: open servo

                                    try:
                                        requests.post(
                                            f"{os.getenv('API_URL')}/dispensing-logs",
                                            json={
                                                "patient_id": patient_id,
                                                "status": "dispensed",
                                                "face_auth_score": float(dist),
                                                "device_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                            },
                                            timeout = 3
                                        )
                                        print(f"SUCCESS: Dispensing log for {name} sent to API")
                                    except Exception as e:
                                        print(f"ERROR: Failed to send dispensing log to API: {e}")

                                else:
                                    result_text  = f"IDENTITY VERIFICATION FAILED ({dist:.2f})"
                                    result_color = (0, 0, 255)
                            else:
                                result_text  = "FACE DETECTION FAILED"
                                result_color = (0, 0, 255)
                        else:
                            result_text  = "LIVENESS DETECTION FAILED"
                            result_color = (0, 0, 255)

                        result_until = now + 3
                        state        = STATE_RESULT

            else:
                if state == STATE_LIVENESS:
                    state            = STATE_WAITING
                    blink_detected   = False
                    eye_was_open     = True
                    eye_closed_since = None
                    blink_min_ear    = 1.0
                    mouth_detected   = False
                    mouth_was_closed = True
                    mouth_open_since = None
                    head_detected    = False
                    turned_left      = False
                    turned_right     = False

            if state == STATE_RESULT:
                cv2.putText(frame, result_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 2)
            cv2.imshow("Authentication (Q = exit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    cap = get_camera()
    if cap is None:
        print("Camera cannot be opened.")
    else:
        try:
            main(cap)
        finally:
            release_camera(cap)