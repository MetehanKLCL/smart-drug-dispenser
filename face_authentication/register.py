# ⚠️  LEGACY SCRIPT — KULLANMA!
# Bu dosya ESKİ kayıt sistemine aittir ve farklı bir veritabanına yazar:
#   DB_PATH = ~/dispenser/faces.db   (kök dizin)
#
# Yeni sistem ~/dispenser/pi_backend/faces.db kullanır.
# Bu script ile kaydedilen yüzler yeni auth sistemi tarafından GÖRILMEZ.
#
# Yüz kayıt için: Flutter uygulamasından /api/face/register kullanın
# veya: python3 -m face_authentication.pi_face_register --first-name ... --last-name ...
#
# Bu dosyayı doğrudan çalıştırmak veritabanı karışıklığına neden olur!

import cv2
import os
import sqlite3
import requests
from dotenv import load_dotenv
import numpy as np
import urllib.request
import mediapipe as mp
import face_recognition
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from camera import get_camera, read_frame, release_camera

load_dotenv()

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faces.db")

# Eye landmark indices for visualization
LEFT_EYE  = {"top": 159, "bot": 145, "left": 33,  "right": 133}
RIGHT_EYE = {"top": 386, "bot": 374, "left": 362, "right": 263}

COLLECT_FRAMES = 30  # Number of frames to collect for average embedding


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Model downloading...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def save_user(conn, patient_id, first_name, last_name, vector):
    """Save face embedding to local_users table."""
    blob = vector.tobytes()
    conn.execute(
        "INSERT INTO local_users (patient_id, first_name, last_name, vector) VALUES (?, ?, ?, ?)",
        (patient_id, first_name, last_name, blob)
    )
    conn.commit()
    print(f"'{first_name} {last_name}' saved to local_users.")

def fetch_patient_id(first_name: str, last_name: str) -> str:
    """
    Ask API by name of patient, return UUID.
    Search by first name first, if not found, search by last name.
    """
    try:
        # Search by first name first
        response = requests.get(
            f"{os.getenv('API_URL')}/patients/search",
            params={"name": first_name}
        )
        if response.status_code == 200:
            patients = response.json()["patients"]
            # If multiple results, filter by last name
            for p in patients:
                if p["last_name"].lower() == last_name.lower():
                    return p["patient_id"]
            # If no last name match, return first result
            if patients:
                return patients[0]["patient_id"]

        print("Patient not found, searching by last name...")
        response = requests.get(
            f"{os.getenv('API_URL')}/patients/search",
            params={"name": last_name}
        )
        if response.status_code == 200:
            patients = response.json()["patients"]
            if patients:
                return patients[0]["patient_id"]

    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Is Uvicorn running?")

    return None

def main(cap):
    ensure_model()
    conn = sqlite3.connect(DB_PATH)

    first_name = input("First name: ").strip()
    if not first_name:
        print("First name cannot be empty.")
        return None

    last_name = input("Last name: ").strip()
    if not last_name:
        print("Last name cannot be empty.")
        return None

    print(f"Searching in AWS Database: {first_name} {last_name}...")
    patient_id = fetch_patient_id(first_name, last_name)

    if not patient_id:
        print("Patient not found in AWS. Please add the patient to the system first.")
        return None

    print(f"Patient found → {patient_id}")

    if cap is None:
        print("Camera cannot be opened.")
        conn.close()
        return

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )

    STATE_WAITING    = "waiting"
    STATE_COLLECTING = "collecting"

    state     = STATE_WAITING
    collected = []  # Collected face_recognition embeddings

    print("Look at the camera. SPACE = start collecting, Q = exit")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = read_frame(cap)
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(img)

            h, w, _ = frame.shape

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Draw all 478 landmarks
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

                # Draw eye points in orange
                for idx in [LEFT_EYE["top"], LEFT_EYE["bot"], LEFT_EYE["left"], LEFT_EYE["right"],
                            RIGHT_EYE["top"], RIGHT_EYE["bot"], RIGHT_EYE["left"], RIGHT_EYE["right"]]:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 165, 255), -1)

                if state == STATE_WAITING:
                    cv2.putText(frame, "Face found - SPACE to start",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                elif state == STATE_COLLECTING:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_frame)
                    if encodings:
                        collected.append(encodings[0])

                    progress = len(collected)
                    cv2.putText(frame, f"Collecting... {progress}/{COLLECT_FRAMES}",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, "Slowly turn your head left and right",
                                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    if len(collected) >= COLLECT_FRAMES:
                        avg_vec = np.mean(collected, axis=0).astype(np.float32)
                        avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-6)
                        save_user(conn, patient_id, first_name, last_name, avg_vec)

                        cv2.putText(frame, "SAVED!", (30, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        cv2.imshow("Record (SPACE = start, Q = exit)", frame)
                        cv2.waitKey(1500)
                        break

            else:
                cv2.putText(frame, "Face not found",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if state == STATE_COLLECTING:
                    collected = []
                    state     = STATE_WAITING

            cv2.imshow("Record (SPACE = start, Q = exit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if result.face_landmarks and state == STATE_WAITING:
                    collected = []
                    state     = STATE_COLLECTING

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