"""
Pi kiosk için: face_authentication/auth.py ile aynı MediaPipe Face Landmarker
+ blink / ağız / baş liveness mantığı (tasks API, IMAGE modu).

OpenCV penceresi yok; sadece RGB kare üzerinde state machine.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# auth.py ile aynı sabitler
EAR_THRESHOLD = 0.28  # MediaPipe normalize coords: açık göz ~0.25–0.40, kapalı ~0.05–0.15
BLINK_MIN_EAR = 0.22  # daha düşük = gerçek kapanış gerektirir (fotoğraf geçemez)
MAR_THRESHOLD = 0.15
TIME_LIMIT = float(os.environ.get("KIOSK_AUTH_TIME_LIMIT", "12"))  # blink zorunlu → biraz daha süre
BLINK_MIN_DURATION = 0.05
BLINK_MAX_DURATION = 0.5
MOUTH_MIN_DURATION = 0.1
MOUTH_MAX_DURATION = 0.6

LEFT_EYE = {"top": 159, "bot": 145, "left": 33, "right": 133}
RIGHT_EYE = {"top": 386, "bot": 374, "left": 362, "right": 263}
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
NOSE_TIP = 4

STATE_WAITING = "waiting"
STATE_LIVENESS = "liveness"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "face_landmarker.task")


def _default_model_path() -> str:
    env = os.environ.get("FACE_LANDMARKER_TASK", "").strip()
    if env and os.path.isfile(env):
        return env
    for base in (_REPO_ROOT, os.path.join(_REPO_ROOT, "pi_backend")):
        cand = os.path.join(base, "face_landmarker.task")
        if os.path.isfile(cand):
            return cand
    return DEFAULT_MODEL_PATH


def get_ear(landmarks: list, eye: dict) -> float:
    vertical = abs(landmarks[eye["top"]].y - landmarks[eye["bot"]].y)
    horizontal = abs(landmarks[eye["right"]].x - landmarks[eye["left"]].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def get_mar(landmarks: list) -> float:
    vertical = abs(landmarks[LOWER_LIP].y - landmarks[UPPER_LIP].y)
    horizontal = abs(landmarks[RIGHT_MOUTH].x - landmarks[LEFT_MOUTH].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class LandmarkerLivenessSession:
    """auth.py main() içindeki WAITING/LIVENESS döngüsünün kare kare karşılığı."""

    def __init__(self, model_path: str | None = None):
        path = model_path or _default_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"face_landmarker.task bulunamadı: {path}")

        import mediapipe as mp
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp
        self.reset()

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def reset(self) -> None:
        self.state = STATE_WAITING
        self.start_time: float | None = None
        self.blink_detected = False
        self.eye_was_open = True
        self.eye_closed_since: float | None = None
        self.blink_min_ear = 1.0
        self.mouth_detected = False
        self.mouth_was_closed = True
        self.mouth_open_since: float | None = None
        self.head_detected = False
        self.turned_left = False
        self.turned_right = False

    def step(self, frame_rgb: np.ndarray, now: float) -> dict[str, Any]:
        """
        Dönüş örnekleri:
          { "action": "continue", "has_face": bool, "phase": str, ... }
          { "action": "encode_now" }
          { "action": "liveness_failed" }
        """
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)

        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(img)
        h, w = frame_rgb.shape[:2]

        out: dict[str, Any] = {
            "action": "continue",
            "has_face": bool(result.face_landmarks),
            "phase": self.state,
            "w": w,
            "h": h,
            "blink": self.blink_detected,
            "mouth": self.mouth_detected,
            "head": self.head_detected,
        }
        if self.start_time is not None:
            out["remaining"] = max(0.0, TIME_LIMIT - (now - self.start_time))
        else:
            out["remaining"] = TIME_LIMIT

        if not result.face_landmarks:
            if self.state == STATE_LIVENESS:
                self._reset_liveness_trackers()
                self.state = STATE_WAITING
                self.start_time = None
            return out

        landmarks = result.face_landmarks[0]
        ear = (get_ear(landmarks, LEFT_EYE) + get_ear(landmarks, RIGHT_EYE)) / 2.0
        mar = get_mar(landmarks)
        nose_x = landmarks[NOSE_TIP].x
        eye_closed = ear < EAR_THRESHOLD
        mouth_open = mar > MAR_THRESHOLD

        if self.state == STATE_WAITING:
            self.start_time = now
            self._reset_liveness_trackers()
            self.state = STATE_LIVENESS
            return out

        # STATE_LIVENESS
        if not self.blink_detected:
            if eye_closed and self.eye_was_open:
                self.eye_closed_since = now
                self.blink_min_ear = ear
            if eye_closed and not self.eye_was_open:
                self.blink_min_ear = min(self.blink_min_ear, ear)
            if not eye_closed and not self.eye_was_open:
                if self.eye_closed_since is not None:
                    duration = now - self.eye_closed_since
                    if BLINK_MIN_DURATION <= duration <= BLINK_MAX_DURATION and self.blink_min_ear < BLINK_MIN_EAR:
                        self.blink_detected = True
                        _log.info("Landmarker: blink OK")
                    self.eye_closed_since = None
                    self.blink_min_ear = 1.0
            self.eye_was_open = not eye_closed

        if not self.mouth_detected:
            if mouth_open and self.mouth_was_closed:
                self.mouth_open_since = now
            if not mouth_open and not self.mouth_was_closed:
                if self.mouth_open_since is not None:
                    duration = now - self.mouth_open_since
                    if MOUTH_MIN_DURATION <= duration <= MOUTH_MAX_DURATION:
                        self.mouth_detected = True
                        _log.info("Landmarker: mouth OK")
                    self.mouth_open_since = None
            self.mouth_was_closed = not mouth_open

        if not self.head_detected:
            left_face = landmarks[234].x
            right_face = landmarks[454].x
            face_width = abs(right_face - left_face)
            if face_width > 0:
                nose_norm = (nose_x - left_face) / face_width
                if nose_norm < 0.45:
                    self.turned_right = True
                if nose_norm > 0.55:
                    self.turned_left = True
                if self.turned_left and self.turned_right:
                    self.head_detected = True
                    _log.info("Landmarker: head turn OK")

        # Blink zorunlu (fotoğrafla geçilemez) + ağız VEYA baş çevirme.
        # OR kullanımı fotoğrafı sallayarak head_detected tetiklemeye açıktı.
        liveness_passed = self.blink_detected and (self.mouth_detected or self.head_detected)
        out["blink"] = self.blink_detected
        out["mouth"] = self.mouth_detected
        out["head"] = self.head_detected
        out["remaining"] = max(0.0, TIME_LIMIT - (now - (self.start_time or now)))

        # Encode immediately when liveness is confirmed — don't wait for the full time limit
        if liveness_passed:
            return {"action": "encode_now"}

        if self.start_time is not None and now - self.start_time >= TIME_LIMIT:
            return {"action": "liveness_failed"}

        return out

    def _reset_liveness_trackers(self) -> None:
        self.blink_detected = False
        self.eye_was_open = True
        self.eye_closed_since = None
        self.blink_min_ear = 1.0
        self.mouth_detected = False
        self.mouth_was_closed = True
        self.mouth_open_since = None
        self.head_detected = False
        self.turned_left = False
        self.turned_right = False


def rgb_frame_to_encoding(frame_rgb: np.ndarray) -> np.ndarray | None:
    import face_recognition

    locs = face_recognition.face_locations(frame_rgb, model="hog")
    if not locs:
        return None
    encs = face_recognition.face_encodings(frame_rgb, locs)
    if not encs:
        return None
    return encs[0].astype(np.float32)


def match_expected_patient(
    face_users: list[dict],
    avg_encoding: np.ndarray,
    expected_patient_id: str,
    distance_threshold: float,
) -> dict | None:
    import face_recognition

    for user in face_users:
        if user["patient_id"] != expected_patient_id:
            continue

        # Use all available samples (multi-sample → better accuracy)
        vectors = user.get("encodings") or [user["encoding"]]
        distances = face_recognition.face_distance(vectors, avg_encoding)
        dist = float(np.min(distances))

        _log.info(
            "match_expected_patient: %s dist=%.4f (best of %d samples) threshold=%.2f",
            user["name"], dist, len(vectors), distance_threshold,
        )

        if dist <= distance_threshold:
            return {
                "matched": True,
                "patient_id": user["patient_id"],
                "name": user["name"],
                "score": float(1.0 - dist),
                "distance": float(dist),
                "is_expected": True,
            }
    return None
