"""
Pi / API tarafı için kamera: `face_authentication.camera` (Picamera2, OpenCV).

Aynı arayüz: open(), read(), read_rgb(), release(), backend.
"""

from __future__ import annotations

import cv2

from .camera import get_camera, read_frame, release_camera


def _infer_camera_backend(cam) -> str:
    if cam is None:
        return "none"
    if hasattr(cam, "capture_array"):
        return "picamera2"
    return "opencv"


class FaceCamera:
    """Yüz kaydı ve headless auth için birleşik kamera."""

    def __init__(self, width: int = 640, height: int = 480):
        self._width = width
        self._height = height
        self._cam = None
        self._opened = False
        self._backend_label = "unknown"

    def open(self) -> bool:
        try:
            self._cam = get_camera()
            self._backend_label = _infer_camera_backend(self._cam)
            self._opened = self._cam is not None
            return self._opened
        except Exception as e:
            print(f"[FaceCamera] open failed: {e}", flush=True)
            self._cam = None
            self._opened = False
            self._backend_label = "none"
            return False

    def read(self):
        if not self._opened or self._cam is None:
            return False, None
        return read_frame(self._cam)

    def read_rgb(self):
        ok, bgr = self.read()
        if not ok or bgr is None:
            return False, None
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        if self._cam is not None:
            release_camera(self._cam)
            self._cam = None
        self._opened = False
        self._backend_label = "unknown"

    @property
    def backend(self) -> str:
        return self._backend_label
