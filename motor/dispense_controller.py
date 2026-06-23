"""
Dispense controller for the SmartDrugDispenser.

Manages the 15-minute dispensing window triggered by CMD_TRIGGER_DISPENSE.

Flow:
    1. Mobile sends CMD_TRIGGER_DISPENSE with patient_id
    2. DispenseController.start_window() is called from ble_server
    3. A background thread runs for up to 15 minutes:
       a. Calls authenticate_user() repeatedly (with cooldown between attempts)
       b. If score >= 0.6 → rotate motor to correct slot, open gate, log "dispensed"
       c. If 15 minutes expire without success → log "missed"
    4. Result is posted to API dispensing-logs endpoint
    5. BLE notification sent back to mobile via callback

Motor is passed in from outside (api_server.py) to avoid GPIO busy conflicts.
"""

import os
import time
import uuid
import sqlite3
import threading
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from face_authentication.headless_auth import authenticate_user
    HAS_FACE_AUTH = True
except ImportError:
    HAS_FACE_AUTH = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DB    = os.path.join(_SCRIPT_DIR, "faces.db")

WINDOW_SECONDS        = 15 * 60
AUTH_RETRY_COOLDOWN   = 5
FACE_SCORE_THRESHOLD  = 0.6


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Dispense] {msg}", flush=True)


def _ensure_sync_queue_table():
    try:
        conn = sqlite3.connect(LOCAL_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                log_id            TEXT PRIMARY KEY,
                schedule_id       TEXT,
                patient_id        TEXT NOT NULL,
                status            TEXT NOT NULL,
                face_auth_score   REAL,
                dispensing_at     TEXT,
                taken_at          TEXT,
                device_timestamp  TEXT,
                error_details     TEXT,
                is_synced         INTEGER DEFAULT 0,
                retry_count       INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        _log(f"ensure_sync_queue_table failed: {e}")


def _save_to_sync_queue(patient_id, schedule_id, status, score, error_details=None):
    """Persist a dispensing log locally for later sync if API is unreachable."""
    _ensure_sync_queue_table()
    try:
        conn = sqlite3.connect(LOCAL_DB)
        conn.execute("""
            INSERT OR IGNORE INTO sync_queue
                (log_id, patient_id, schedule_id, status,
                 face_auth_score, dispensing_at, device_timestamp, error_details, is_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            str(uuid.uuid4()),
            patient_id,
            schedule_id,
            status,
            score,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat(),
            error_details,
        ))
        conn.commit()
        conn.close()
        _log(f"Log saved to sync_queue (status={status})")
    except Exception as e:
        _log(f"Failed to save to sync_queue: {e}")


class DispenseController:
    def __init__(self, api_base_url: str, on_success=None, on_failure=None, motor=None):
        """
        Args:
            api_base_url: API base URL
            on_success:   callback(patient_id, name, score)
            on_failure:   callback(reason)
            motor:        MotorController instance passed from outside (avoids GPIO busy)
        """
        self._api_url    = api_base_url.rstrip("/")
        self._on_success = on_success
        self._on_failure = on_failure
        self._motor      = motor  # Motor is passed in, not created here
        self._active     = False
        self._lock       = threading.Lock()

    def start_window(self, patient_id: str, schedule_id: str = None, slot_id: int = None):
        """Start the 15-minute dispensing window in a background thread."""
        with self._lock:
            if self._active:
                _log("Dispense window already active — ignoring duplicate trigger")
                return
            self._active = True

        _log(f"Starting 15-min window: patient={patient_id[:8]}... "
             f"schedule={schedule_id or 'N/A'} slot={slot_id}")

        thread = threading.Thread(
            target=self._window_loop,
            args=(patient_id, schedule_id, slot_id),
            daemon=True,
        )
        thread.start()

    def _window_loop(self, patient_id, schedule_id, slot_id):
        deadline   = time.time() + WINDOW_SECONDS
        attempt    = 0
        success    = False
        auth_name  = None
        auth_score = None

        try:
            while time.time() < deadline:
                attempt += 1
                remaining = int(deadline - time.time())
                _log(f"Auth attempt #{attempt} ({remaining}s remaining)")

                if not HAS_FACE_AUTH:
                    _log("face_auth_headless not available — simulating success for testing")
                    auth_name  = "Test User"
                    auth_score = 0.85
                    success    = True
                    break

                result = authenticate_user()

                if result.get("status") == "success":
                    pid   = result["patient_id"]
                    name  = result.get("name", "Unknown")
                    score = result.get("score", 0.0)
                    _log(f"Auth result: {name} score={score:.2f}")

                    if score >= FACE_SCORE_THRESHOLD and pid == patient_id:
                        auth_name  = name
                        auth_score = score
                        success    = True
                        break
                    elif pid != patient_id:
                        _log(f"Wrong patient: expected {patient_id[:8]}... got {pid[:8]}...")
                    else:
                        _log(f"Score too low: {score:.2f} < {FACE_SCORE_THRESHOLD}")
                else:
                    reason = result.get("reason", "unknown")
                    _log(f"Auth failed: {reason}")

                if time.time() + AUTH_RETRY_COOLDOWN < deadline:
                    time.sleep(AUTH_RETRY_COOLDOWN)
                else:
                    break

            if success:
                self._handle_success(patient_id, schedule_id, slot_id, auth_name, auth_score)
            else:
                self._handle_timeout(patient_id, schedule_id)

        except Exception as e:
            _log(f"Unexpected error in dispense window: {e}")
            self._post_log(patient_id, schedule_id, "error", None, str(e))
            if self._on_failure:
                self._on_failure(f"Error: {e}")
        finally:
            with self._lock:
                self._active = False

    def _handle_success(self, patient_id, schedule_id, slot_id, name, score):
        _log(f"ACCESS GRANTED: {name} (score={score:.2f})")

        if self._motor and slot_id is not None:
            _log(f"Rotating to slot {slot_id}")
            self._motor.rotate_to_slot(slot_id)
            self._motor.open_gate()
            time.sleep(3)
            self._motor.close_gate()
        else:
            _log("[DRY-RUN] Motor action skipped (no motor or no slot_id)")

        self._post_log(patient_id, schedule_id, "dispensed", score)

        if self._on_success:
            self._on_success(patient_id, name, score)

    def _handle_timeout(self, patient_id, schedule_id):
        _log("15-minute window expired — MISSED DOSE")
        self._post_log(patient_id, schedule_id, "missed", None,
                       "15-minute authentication window expired")
        if self._on_failure:
            self._on_failure("Timeout: 15-minute window expired")

    def _post_log(self, patient_id, schedule_id, status, score, error_details=None):
        payload = {
            "patient_id":       patient_id,
            "schedule_id":      schedule_id,
            "status":           status,
            "face_auth_score":  score,
            "device_timestamp": datetime.utcnow().isoformat(),
            "error_details":    error_details,
        }

        if not HAS_REQUESTS:
            _log("requests library not available — saving to sync_queue")
            _save_to_sync_queue(patient_id, schedule_id, status, score, error_details)
            return

        url = f"{self._api_url}/dispensing-logs"
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code in (200, 201):
                _log(f"Dispensing log posted to API (status={status})")
            else:
                _log(f"API returned {resp.status_code}: {resp.text}")
                _save_to_sync_queue(patient_id, schedule_id, status, score, error_details)
        except Exception as e:
            _log(f"API unreachable: {e} — saving to sync_queue")
            _save_to_sync_queue(patient_id, schedule_id, status, score, error_details)

    @property
    def is_active(self) -> bool:
        return self._active