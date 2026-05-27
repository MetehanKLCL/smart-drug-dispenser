import time

import cv2

# Pi detection once at startup (not on every frame in read_frame)
_IS_RPI = None

_DEFAULT_W, _DEFAULT_H = 640, 480
# Prove the pipeline delivers frames before we trust OpenCV on Pi
_OPENCV_VERIFY_READS = 80
_OPENCV_GOOD_STREAK = 5


def _opencv_try_open(idx: int, use_v4l2: bool, w: int, h: int):
    """
    Try one OpenCV capture mode. Returns cap if we get a short streak of valid frames.
    Avoid CAP_PROP_BUFFERSIZE=1 — some libcamera V4L2 bridges fail every read() with it.
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if use_v4l2 else cv2.VideoCapture(idx)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            yuyv = cv2.VideoWriter_fourcc(*"YUYV")
            cap.set(cv2.CAP_PROP_FOURCC, yuyv)
        except Exception:
            pass

        good = 0
        for _ in range(_OPENCV_VERIFY_READS):
            ret, frame = cap.read()
            if ret and frame is not None and getattr(frame, "size", 0) > 0:
                good += 1
                if good >= _OPENCV_GOOD_STREAK:
                    _log_cam(
                        f"Opened via OpenCV index={idx} "
                        f"backend={'V4L2' if use_v4l2 else 'default'} ({w}x{h})"
                    )
                    return cap
            else:
                good = 0
            time.sleep(0.01)
    except Exception as e:
        _log_cam(f"OpenCV idx={idx} v4l2={use_v4l2}: {e}")

    try:
        cap.release()
    except Exception:
        pass
    return None


def _opencv_open_pi(w: int, h: int):
    """Try all /dev/videoN devices on the system; return first that yields real frames.

    Pi 5 + libcamera exposes capture devices at high indices (e.g. /dev/video19-35)
    instead of the traditional /dev/video0. We discover the list from the filesystem
    rather than hard-coding a small range.
    """
    import glob
    import re

    # Gather all /dev/videoN paths, sorted numerically.
    video_paths = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.search(r"\d+", p).group()) if re.search(r"\d+", p) else 999,
    )
    indices = []
    for path in video_paths:
        m = re.search(r"\d+", path)
        if m:
            indices.append(int(m.group()))

    # Always include 0-2 first as a fast path on non-Pi systems.
    for idx in [i for i in (0, 1, 2) if i not in indices] + indices:
        for use_v4l2 in (False, True):
            cap = _opencv_try_open(idx, use_v4l2, w, h)
            if cap is not None:
                return cap
    return None


def _log_cam(msg: str) -> None:
    print(f"[face_camera] {msg}", flush=True)


def _detect_raspberry_pi():
    """
    Prefer /sys/firmware/devicetree/base/model (reliable on Pi 4/5 and newer images).
    Fall back to /proc/cpuinfo containing 'Raspberry Pi' for older setups.
    """
    try:
        with open("/sys/firmware/devicetree/base/model", "rb") as f:
            raw = f.read()
        model = raw.decode("utf-8", errors="replace").replace("\x00", "").strip().lower()
        if "raspberry pi" in model:
            return True
    except OSError:
        pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            text = f.read()
        if "Raspberry Pi" in text:
            return True
    except OSError:
        pass

    return False


def is_raspberry_pi():
    """True if running on a Raspberry Pi (device tree and/or /proc/cpuinfo)."""
    global _IS_RPI
    if _IS_RPI is None:
        _IS_RPI = _detect_raspberry_pi()
    return _IS_RPI


def get_camera():
    """
    Opens and returns the correct camera for the device.
    On the Raspberry Pi: picamera2 with explicit configure (required on Bookworm),
    then OpenCV /dev/video0 fallback if libcamera is busy or picamera2 fails.
    On Mac: OpenCV with the default camera (0 = internal camera).
    """
    if not is_raspberry_pi():
        cap = cv2.VideoCapture(0)
        return cap

    w, h = _DEFAULT_W, _DEFAULT_H
    try:
        from picamera2 import Picamera2
        from libcamera import controls as _lc

        picam2 = Picamera2()
        # Video configuration: much higher FPS than still_configuration
        config = picam2.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},
            controls={
                "FrameRate": 30,
                # IMX708 NoIR Wide: no IR cut filter → strong red cast.
                # Daylight ColourGains (R, B) correct this for indoor/outdoor use.
                # R gain ~1.0, B gain ~2.2 pulls reds down and blues up.
                "ColourGains": (1.0, 2.2),
                "AwbEnable": False,
                "AeEnable": True,
                "NoiseReductionMode": _lc.draft.NoiseReductionModeEnum.Fast,
            },
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.3)
        _log_cam(f"Opened via picamera2 ({w}x{h})")
        return picam2
    except Exception as e:
        _log_cam(f"picamera2 open failed: {e}")

    cap = _opencv_open_pi(w, h)
    if cap is not None:
        return cap

    raise RuntimeError(
        "Could not open any camera on Raspberry Pi (picamera2 and OpenCV both failed). "
        "Install: sudo apt install -y python3-libcamera python3-picamera2 && "
        "use venv --system-site-packages or pip install picamera2. "
        "Also: no other process using camera (KVS/gst), user in group 'video'."
    )


def read_frame(camera):
    """
    Reads and returns a frame from the camera.
    For both cameras, returns the same format (success, frame)
    success: True/False (whether the frame was captured)
    frame: numpy array (image data)
    """
    if hasattr(camera, "capture_array"):
        frame = camera.capture_array()
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame
    for _ in range(25):
        ret, frame = camera.read()
        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        time.sleep(0.015)
    return False, None


def release_camera(camera):
    """
    Closes the camera.
    Picamera2: stop() + close(); OpenCV: release().
    """
    try:
        if hasattr(camera, "stop") and hasattr(camera, "close"):
            camera.stop()
            camera.close()
        else:
            camera.release()
    except Exception as e:
        _log_cam(f"release error: {e}")
