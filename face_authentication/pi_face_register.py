"""
Pi HTTP yüz kaydı: örnek toplama + SQLite (local_users / face_samples).

CLI: python -m face_authentication.pi_face_register --first-name ... --last-name ...
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
import uuid

import cv2
import face_recognition
import numpy as np

from .facade import FaceCamera


def _repo_paths():
    pkg = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(pkg, ".."))
    pi_backend = os.path.join(root, "pi_backend")
    return root, pi_backend


_REPO_ROOT, _PI_BACKEND = _repo_paths()


def faces_db_path() -> str:
    """Her bağlantıda çözümlenir; state_machine.LOCAL_DB ile aynı FACES_DB kuralı."""
    return os.environ.get("FACES_DB", os.path.join(_PI_BACKEND, "faces.db"))


# Geriye dönük uyumluluk (modül yüklendiği andaki yol; bağlantılarda faces_db_path kullanın).
DB_PATH = faces_db_path()


def ensure_local_users_table(conn: sqlite3.Connection) -> None:
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


def _upsample_for_hog() -> int:
    try:
        n = int(os.environ.get("FACE_REGISTER_UPSAMPLE", "2"))
        return max(1, min(3, n))
    except ValueError:
        return 2


def capture_face_encodings(
    max_samples: int = 5,
    timeout_sec: int | None = None,
) -> list[np.ndarray]:
    if timeout_sec is None:
        try:
            timeout_sec = int(os.environ.get("FACE_REGISTER_TIMEOUT_SEC", "90"))
        except ValueError:
            timeout_sec = 90

    cam = FaceCamera()
    if not cam.open():
        raise RuntimeError("Camera could not be opened")

    backend = cam.backend
    print(f"\n[Register] Camera opened (backend: {backend})")
    if backend == "opencv":
        print(
            "[Register] Note: using OpenCV/V4L2 (picamera2 not importable in this venv). "
            "Install with: pip install picamera2 (and system libcamera) for more reliable capture."
        )
    print(f"[Register] Auto-capturing {max_samples} samples...")
    print("[Register] Stand ~50cm in front of the camera, face forward.\n")

    encodings: list[np.ndarray] = []
    started = time.time()
    frame_count = 0
    bad_reads = 0
    upsample = _upsample_for_hog()
    # OpenCV path needs more frames before HOG sees a stable image
    warmup_frames = 40 if backend == "opencv" else 10
    last_status = started

    try:
        while len(encodings) < max_samples and (time.time() - started) < timeout_sec:
            ok, rgb = cam.read_rgb()
            if not ok:
                bad_reads += 1
                if bad_reads == 1 or bad_reads % 250 == 0:
                    print(
                        "[Register] Warning: many camera read failures — "
                        "check light / no other app using the camera."
                    )
                time.sleep(0.05)
                continue

            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            frame_count += 1
            if frame_count <= warmup_frames:
                continue

            now = time.time()
            if now - last_status >= 5.0:
                mean_px = float(rgb.mean())
                print(
                    f"[Register] … still capturing ({now - started:.0f}s): "
                    f"good_frames={frame_count}, mean_brightness≈{mean_px:.0f} "
                    f"(if ~0, image may be blank)"
                )
                last_status = now

            face_locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=upsample,
                model="hog",
            )

            if len(face_locations) == 0:
                if frame_count % 20 == 0:
                    print("[Register] No face detected — move closer or improve light.")
                time.sleep(0.08)
                continue

            if len(face_locations) > 1:
                print(f"[Register] {len(face_locations)} faces — need exactly 1")
                time.sleep(0.3)
                continue

            enc = face_recognition.face_encodings(rgb, known_face_locations=face_locations)
            if not enc:
                print("[Register] Encoding failed, retrying...")
                continue

            encodings.append(enc[0].astype(np.float32))
            print(f"[Register] Sample {len(encodings)}/{max_samples} captured")
            time.sleep(0.5)

    finally:
        cam.release()

    elapsed = time.time() - started
    print(f"[Register] Done: {len(encodings)} samples in {elapsed:.1f}s")
    return encodings


def _ensure_face_samples_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS face_samples (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            vector     BLOB NOT NULL,
            label      TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES local_users(patient_id)
        )
        """
    )
    conn.commit()


def check_face_duplicates(
    new_encoding: np.ndarray,
    threshold: float = 0.6,
    exclude_patient_id: str | None = None,
) -> dict | None:
    conn = sqlite3.connect(faces_db_path())
    try:
        try:
            rows = conn.execute(
                "SELECT patient_id, vector FROM face_samples"
            ).fetchall()
        except sqlite3.OperationalError:
            return None

        if not rows:
            return None

        min_distance = float("inf")
        closest_patient = None

        for patient_id, vector_blob in rows:
            if exclude_patient_id and patient_id == exclude_patient_id:
                continue
            existing_enc = np.frombuffer(vector_blob, dtype=np.float32)
            distance = face_recognition.face_distance([existing_enc], new_encoding)[0]

            if distance < min_distance:
                min_distance = distance
                closest_patient = patient_id

        if min_distance <= threshold and closest_patient:
            user = conn.execute(
                "SELECT first_name, last_name FROM local_users WHERE patient_id = ?",
                (closest_patient,)
            ).fetchone()
            if user:
                return {
                    "patient_id": closest_patient,
                    "first_name": user[0],
                    "last_name": user[1],
                    "distance": float(min_distance),
                }

        return None
    finally:
        conn.close()


def save_user_embedding(
    patient_id: str,
    first_name: str,
    last_name: str,
    avg_encoding: np.ndarray,
    individual_encodings: list[np.ndarray] | None = None,
) -> None:
    from datetime import datetime, timezone

    conn = sqlite3.connect(faces_db_path())
    try:
        ensure_local_users_table(conn)
        _ensure_face_samples_table(conn)

        blob = avg_encoding.astype(np.float32).tobytes()
        conn.execute(
            """
            INSERT INTO local_users (patient_id, first_name, last_name, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                vector=excluded.vector
            """,
            (patient_id, first_name, last_name, blob),
        )

        if individual_encodings:
            conn.execute(
                "DELETE FROM face_samples WHERE patient_id = ?", (patient_id,)
            )
            now = datetime.now(timezone.utc).isoformat()
            for i, enc in enumerate(individual_encodings):
                conn.execute(
                    """
                    INSERT INTO face_samples (patient_id, vector, label, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (patient_id, enc.astype(np.float32).tobytes(), f"sample_{i}", now),
                )

        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Register face to faces.db")
    parser.add_argument("--patient-id", dest="patient_id", default=None)
    parser.add_argument("--first-name", dest="first_name", required=True)
    parser.add_argument("--last-name", dest="last_name", required=True)
    parser.add_argument("--samples", dest="samples", type=int, default=5)
    args = parser.parse_args()

    patient_id = args.patient_id or str(uuid.uuid4())
    first_name = args.first_name.strip()
    last_name = args.last_name.strip()

    if not first_name or not last_name:
        raise ValueError("first_name and last_name must not be empty")

    print(f"[Register] DB path: {faces_db_path()}")
    print(f"[Register] patient_id: {patient_id}")
    print(f"[Register] name: {first_name} {last_name}")

    encodings = capture_face_encodings(max_samples=max(1, args.samples))
    if not encodings:
        print("[Register] No valid face samples captured. Nothing was saved.")
        return

    avg = np.mean(encodings, axis=0).astype(np.float32)
    save_user_embedding(
        patient_id, first_name, last_name, avg,
        individual_encodings=encodings,
    )

    print("\n[Register] Success!")
    print(f"[Register] Saved user: {first_name} {last_name}")
    print(f"[Register] patient_id: {patient_id}")
    print(f"[Register] samples used: {len(encodings)}")


if __name__ == "__main__":
    main()
