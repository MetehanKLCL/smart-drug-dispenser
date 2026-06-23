"""
SmartDrugDispenser — Dispenser State Machine

IDLE -> ROTATING -> LOADING_MODE -> SLOT_READY ->
WAITING_FOR_PATIENT -> FACE_MATCHED -> DISPENSING -> IDLE
Any state -> ERROR (on failure), ERROR -> IDLE (via reset())

Thread-safe: all public methods are guarded by threading.Lock.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
import uuid
import sqlite3
import threading
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List, Dict, Any


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DB = os.path.join(_SCRIPT_DIR, "faces.db")


TOTAL_SLOTS = 14
WINDOW_SECONDS = 5 * 60          # 5-minute face-auth window
FACE_SCORE_THRESHOLD = 0.6       # 1.0 - euclidean distance
AUTH_RETRY_COOLDOWN = 3           # seconds between face-auth attempts


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [StateMachine] {msg}", flush=True)





class DispenserState(str, Enum):
    """Possible dispenser states. Inherits str for JSON serialization."""
    IDLE = "idle"
    ROTATING = "rotating"
    LOADING_MODE = "loading_mode"
    SLOT_READY = "slot_ready"
    WAITING_FOR_PATIENT = "waiting_for_patient"
    FACE_MATCHED = "face_matched"
    DISPENSING = "dispensing"
    ERROR = "error"





@dataclass
class DispenserContext:
    """Snapshot of the current dispenser state."""

    state: DispenserState = DispenserState.IDLE

    # ── Slot & patient ──
    current_patient_id: Optional[str] = None
    current_patient_name: Optional[str] = None
    selected_slot: Optional[int] = None

    # ── Barcode loading ──
    barcode_count: int = 0
    scanned_barcodes: List[str] = field(default_factory=list)

    # ── Motor / servo ──
    motor_busy: bool = False
    servo_open: bool = False

    # ── Face auth window ──
    window_start: Optional[float] = None
    window_deadline: Optional[float] = None
    auth_attempts: int = 0
    last_auth_score: Optional[float] = None

    # ── Camera ──
    camera_active: bool = False

    # ── Error ──
    last_error: Optional[str] = None
    last_error_time: Optional[str] = None

    # ── Timestamps ──
    state_changed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary for the Flask API."""
        d = {
            "state": self.state.value,
            "current_patient_id": self.current_patient_id,
            "current_patient_name": self.current_patient_name,
            "selected_slot": self.selected_slot,
            "barcode_count": self.barcode_count,
            "scanned_barcodes": self.scanned_barcodes[-10:],   # last 10
            "motor_busy": self.motor_busy,
            "servo_open": self.servo_open,
            "camera_active": self.camera_active,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "state_changed_at": self.state_changed_at,
        }
        # Window info (only meaningful in WAITING_FOR_PATIENT)
        if self.window_deadline is not None:
            remaining = max(0, int(self.window_deadline - time.time()))
            d["window_remaining_sec"] = remaining
            d["auth_attempts"] = self.auth_attempts
            d["last_auth_score"] = self.last_auth_score
        else:
            d["window_remaining_sec"] = None
            d["auth_attempts"] = 0
            d["last_auth_score"] = None

        return d


def _ensure_tables():
    """Create tables required by the state machine if they don't exist."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        # Preserve existing slot_bindings table (used by ble_server.py)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slot_bindings (
                slot_id     INTEGER PRIMARY KEY,
                patient_id  TEXT NOT NULL,
                pill_count  INTEGER DEFAULT 0,
                committed   INTEGER DEFAULT 0,
                updated_at  TEXT
            )
        """)

        # Barcode details per slot
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slot_medications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_id         INTEGER NOT NULL,
                patient_id      TEXT NOT NULL,
                barcode         TEXT,
                scanned_at      TEXT NOT NULL,
                FOREIGN KEY (slot_id) REFERENCES slot_bindings(slot_id)
            )
        """)

        # Face auth attempt log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_auth_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      TEXT,
                matched_patient TEXT,
                score           REAL,
                liveness_ok     INTEGER,
                slot_dispensed  INTEGER,
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()


def _db_bind_slot(slot_id: int, patient_id: str):
    """Assign (or reassign) a slot to a patient."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        conn.execute("""
            INSERT INTO slot_bindings (slot_id, patient_id, pill_count, committed, updated_at)
            VALUES (?, ?, 0, 0, ?)
            ON CONFLICT(slot_id) DO UPDATE SET
                patient_id = excluded.patient_id,
                pill_count = 0,
                committed  = 0,
                updated_at = excluded.updated_at
        """, (slot_id, patient_id, datetime.now(timezone.utc).isoformat()))
        # Clear old barcode records for this slot
        conn.execute(
            "DELETE FROM slot_medications WHERE slot_id = ?", (slot_id,)
        )
        conn.commit()
    finally:
        conn.close()


def _db_add_barcode(slot_id: int, patient_id: str, barcode: str) -> int:
    """Insert a barcode scan record and return the updated pill count."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        conn.execute("""
            INSERT INTO slot_medications (slot_id, patient_id, barcode, scanned_at)
            VALUES (?, ?, ?, ?)
        """, (slot_id, patient_id, barcode, datetime.now(timezone.utc).isoformat()))

        conn.execute("""
            UPDATE slot_bindings
            SET pill_count = pill_count + 1, updated_at = ?
            WHERE slot_id = ?
        """, (datetime.now(timezone.utc).isoformat(), slot_id))

        conn.commit()

        row = conn.execute(
            "SELECT pill_count FROM slot_bindings WHERE slot_id = ?",
            (slot_id,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _db_commit_slot(slot_id: int):
    """Mark slot loading as committed."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        conn.execute("""
            UPDATE slot_bindings SET committed = 1, updated_at = ?
            WHERE slot_id = ?
        """, (datetime.now(timezone.utc).isoformat(), slot_id))
        conn.commit()
    finally:
        conn.close()


def _db_get_slot_for_patient(patient_id: str) -> Optional[int]:
    """Find the committed slot for a patient."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        row = conn.execute(
            "SELECT slot_id FROM slot_bindings WHERE patient_id = ? AND committed = 1",
            (patient_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _db_get_any_committed_slot() -> Optional[tuple]:
    """Return (patient_id, slot_id) for the first committed slot found."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        row = conn.execute(
            "SELECT patient_id, slot_id FROM slot_bindings WHERE committed = 1 LIMIT 1",
        ).fetchone()
        return (row[0], row[1]) if row else None
    finally:
        conn.close()


def _db_log_face_auth(patient_id: Optional[str], matched: Optional[str],
                       score: Optional[float], liveness: bool,
                       slot: Optional[int], status: str):
    """Log a face authentication attempt."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        conn.execute("""
            INSERT INTO face_auth_log
                (patient_id, matched_patient, score, liveness_ok,
                 slot_dispensed, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id, matched, score, int(liveness),
            slot, status, datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()





class DispenserStateMachine:
    """Central state machine managing the full dispenser workflow. Thread-safe."""

    def __init__(
        self,
        motor_controller=None,
        on_state_change: Optional[Callable] = None,
        on_notify: Optional[Callable] = None,
    ):
        _ensure_tables()

        self._ctx = DispenserContext()
        self._lock = threading.Lock()
        self._motor = motor_controller
        self._on_state_change = on_state_change
        self._on_notify = on_notify

        # Background thread refs
        self._auth_thread: Optional[threading.Thread] = None
        self._auth_cancel = threading.Event()

        _log("Initialized (IDLE)")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def state(self) -> DispenserState:
        return self._ctx.state

    @property
    def context(self) -> DispenserContext:
        return self._ctx

    @property
    def snapshot(self) -> DispenserContext:
        """Alias for context (used by tests)."""
        return self._ctx

    def get_state_dict(self) -> Dict[str, Any]:
        """Thread-safe state snapshot for the Flask API."""
        with self._lock:
            return self._ctx.to_dict()

    # ── State transition helper ─────────────────────────────────────────────

    def _set_state(self, new_state: DispenserState, reason: str = ""):
        """Transition state and fire callback."""
        old = self._ctx.state
        self._ctx.state = new_state
        self._ctx.state_changed_at = datetime.now(timezone.utc).isoformat()

        arrow = f"{old.value} → {new_state.value}"
        _log(f"STATE: {arrow}" + (f"  ({reason})" if reason else ""))

        if self._on_state_change:
            try:
                self._on_state_change(old, new_state, self._ctx.to_dict())
            except Exception as e:
                _log(f"on_state_change callback error: {e}")

    # -- Slot Binding & Rotation --

    def bind_slot(self, patient_id: str, slot_id: int,
                  patient_name: str = "") -> Dict[str, Any]:
        """Bind a patient to a slot and start wheel rotation."""
        with self._lock:
            # Validation
            if self._ctx.state not in (
                DispenserState.IDLE,
                DispenserState.SLOT_READY,
                DispenserState.ERROR,
            ):
                return {
                    "ok": False,
                    "message": f"Cannot bind slot in state '{self._ctx.state.value}'. "
                               f"Reset first.",
                    "state": self._ctx.state.value,
                }

            if slot_id < 0 or slot_id >= TOTAL_SLOTS:
                return {
                    "ok": False,
                    "message": f"Invalid slot {slot_id} (valid: 0-{TOTAL_SLOTS - 1})",
                    "state": self._ctx.state.value,
                }

            # Update context
            self._ctx.current_patient_id = patient_id
            self._ctx.current_patient_name = patient_name
            self._ctx.selected_slot = slot_id
            self._ctx.barcode_count = 0
            self._ctx.scanned_barcodes = []
            self._ctx.last_error = None

            # Persist to DB
            try:
                _db_bind_slot(slot_id, patient_id)
            except Exception as e:
                self._set_state(DispenserState.ERROR, f"DB error: {e}")
                self._ctx.last_error = str(e)
                return {
                    "ok": False,
                    "message": f"Database error: {e}",
                    "state": self._ctx.state.value,
                }

            # State → ROTATING
            self._set_state(DispenserState.ROTATING, f"slot={slot_id}")
            self._ctx.motor_busy = True

        # Rotate motor in a separate thread (outside lock)
        def _rotate():
            success = True
            if self._motor:
                try:
                    success = self._motor.rotate_to_slot(slot_id)
                except Exception as e:
                    _log(f"Motor error: {e}")
                    success = False

            with self._lock:
                self._ctx.motor_busy = False
                if success:
                    self._set_state(
                        DispenserState.LOADING_MODE,
                        f"slot {slot_id} ready for loading",
                    )
                else:
                    self._set_state(DispenserState.ERROR, "Motor rotation failed")
                    self._ctx.last_error = "Motor rotation failed"
                    self._ctx.last_error_time = datetime.now(timezone.utc).isoformat()

        thread = threading.Thread(target=_rotate, daemon=True)
        thread.start()

        return {
            "ok": True,
            "message": f"Rotating to slot {slot_id} for patient {patient_name or patient_id[:8]}",
            "state": DispenserState.ROTATING.value,
        }

    # -- Barcode Scanning --

    def increment_barcode(self, barcode_data: str = "") -> Dict[str, Any]:
        """Record a barcode scan from the app camera and increment pill count."""
        with self._lock:
            if self._ctx.state != DispenserState.LOADING_MODE:
                return {
                    "ok": False,
                    "count": self._ctx.barcode_count,
                    "message": f"Not in loading mode (current: {self._ctx.state.value})",
                }

            if self._ctx.selected_slot is None or self._ctx.current_patient_id is None:
                return {
                    "ok": False,
                    "count": self._ctx.barcode_count,
                    "message": "No slot or patient selected",
                }

            # Persist to DB
            try:
                new_count = _db_add_barcode(
                    self._ctx.selected_slot,
                    self._ctx.current_patient_id,
                    barcode_data,
                )
            except Exception as e:
                _log(f"DB barcode error: {e}")
                return {
                    "ok": False,
                    "count": self._ctx.barcode_count,
                    "message": f"Database error: {e}",
                }

            self._ctx.barcode_count = new_count
            if barcode_data:
                self._ctx.scanned_barcodes.append(barcode_data)

            _log(f"BARCODE: slot={self._ctx.selected_slot} "
                 f"count={new_count} barcode='{barcode_data[:20] if barcode_data else 'N/A'}'")

            return {
                "ok": True,
                "count": new_count,
                "barcode": barcode_data,
                "message": f"Pill #{new_count} scanned",
            }

    # -- Commit Slot --

    def commit_slot(self) -> Dict[str, Any]:
        """Finalize slot loading after all barcodes are scanned."""
        with self._lock:
            if self._ctx.state != DispenserState.LOADING_MODE:
                return {
                    "ok": False,
                    "message": f"Not in loading mode (current: {self._ctx.state.value})",
                }

            if self._ctx.barcode_count == 0:
                return {
                    "ok": False,
                    "message": "No pills scanned yet. Scan at least one barcode.",
                }

            slot_id = self._ctx.selected_slot
            count = self._ctx.barcode_count

            try:
                _db_commit_slot(slot_id)
            except Exception as e:
                return {"ok": False, "message": f"Database error: {e}"}

            if self._motor:
                self._motor.open_gate()
                self._ctx.servo_open = True

            self._set_state(
                DispenserState.SLOT_READY,
                f"slot {slot_id} committed with {count} pills",
            )

            _log(f"COMMIT: slot={slot_id} pills={count} "
                 f"patient={self._ctx.current_patient_id[:8]}...")

            return {
                "ok": True,
                "pill_count": count,
                "slot": slot_id,
                "patient_id": self._ctx.current_patient_id,
                "message": f"Slot {slot_id} committed with {count} pills",
            }

    # -- Trigger Dispense Window --

    def trigger_dispense(
        self,
        patient_id: Optional[str] = None,
        schedule_id: Optional[str] = None,
        window_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Open camera and start the face-auth window for dispensing."""
        with self._lock:
            # Reject if already in an active dispense flow
            if self._ctx.state in (
                DispenserState.WAITING_FOR_PATIENT,
                DispenserState.FACE_MATCHED,
                DispenserState.DISPENSING,
            ):
                return {
                    "ok": False,
                    "message": f"Already in {self._ctx.state.value} state",
                }

            # Resolve patient ID
            pid = patient_id or self._ctx.current_patient_id

            # Slot check
            slot = self._ctx.selected_slot
            if slot is None and pid:
                slot = _db_get_slot_for_patient(pid)

            # If still no patient/slot, auto-select from any committed slot
            if not pid or slot is None:
                any_committed = _db_get_any_committed_slot()
                if any_committed:
                    pid, slot = any_committed
                else:
                    return {
                        "ok": False,
                        "message": "No committed slot found. Load pills first.",
                    }

            # Configure window
            duration = window_seconds or WINDOW_SECONDS
            now = time.time()

            self._ctx.current_patient_id = pid
            self._ctx.selected_slot = slot
            self._ctx.window_start = now
            self._ctx.window_deadline = now + duration
            self._ctx.auth_attempts = 0
            self._ctx.last_auth_score = None
            self._ctx.camera_active = True

            # Reset auth cancel flag
            self._auth_cancel.clear()

            self._set_state(
                DispenserState.WAITING_FOR_PATIENT,
                f"patient={pid[:8]}... slot={slot} window={duration}s",
            )

            _log(f"DISPENSE WINDOW: {duration}s for patient {pid[:8]}... slot={slot}")

        # Start timeout watcher (returns to IDLE on expiry)
        def _timeout_watcher():
            self._auth_cancel.wait(timeout=duration)

            with self._lock:
                if self._ctx.state == DispenserState.WAITING_FOR_PATIENT:
                    _log("TIMEOUT: 5-minute window expired — MISSED DOSE")
                    self._ctx.camera_active = False

                    _db_log_face_auth(
                        patient_id=pid,
                        matched=None,
                        score=None,
                        liveness=False,
                        slot=slot,
                        status="timeout_missed",
                    )

                    self._set_state(DispenserState.IDLE, "window expired")

                    # BLE notify: missed dose
                    if self._on_notify:
                        self._on_notify([0xA2])  # EVT_MISSED_DOSE

        self._auth_thread = threading.Thread(
            target=_timeout_watcher, daemon=True,
        )
        self._auth_thread.start()

        # Start face auth loop — runs authenticate_user() repeatedly until
        # match, cancel, or window expiry.
        def _face_auth_worker():
            try:
                from face_authentication.headless_auth import authenticate_user
            except ImportError:
                _log("face_authentication.headless_auth not available — face auth disabled")
                return

            _log("Face auth worker started")

            while not self._auth_cancel.is_set():
                with self._lock:
                    if self._ctx.state != DispenserState.WAITING_FOR_PATIENT:
                        break

                result = authenticate_user()   # blocking ~5-8s

                if self._auth_cancel.is_set():
                    break

                status = result.get("status")

                if status == "success":
                    matched_pid  = result["patient_id"]
                    score        = result["score"]
                    name         = result.get("name", "")

                    match_result = self.on_face_matched(
                        matched_patient_id=matched_pid,
                        score=score,
                        name=name,
                        liveness_ok=True,
                    )
                    if match_result.get("ok"):
                        self.dispense()   # auto-dispense on confirmed match
                    break                 # success or wrong patient — stop loop

                elif status == "failed":
                    reason = result.get("reason", "unknown")
                    _log(f"Face auth attempt: {reason} — retrying")
                    if reason == "camera_unavailable":
                        break             # don't retry if camera is broken
                    # Brief pause before next attempt
                    self._auth_cancel.wait(timeout=AUTH_RETRY_COOLDOWN)

            _log("Face auth worker exited")

        face_thread = threading.Thread(target=_face_auth_worker, daemon=True)
        face_thread.start()

        return {
            "ok": True,
            "window_seconds": duration,
            "patient_id": pid,
            "slot": slot,
            "message": f"Camera active. Waiting for patient face ({duration}s window)",
        }

    # -- Face Authentication Result --

    def on_face_matched(
        self,
        matched_patient_id: str,
        score: float,
        name: str = "",
        liveness_ok: bool = True,
    ) -> Dict[str, Any]:
        """Handle a face recognition result from face_auth_headless."""
        with self._lock:
            if self._ctx.state != DispenserState.WAITING_FOR_PATIENT:
                return {
                    "ok": False,
                    "message": f"Not waiting for patient (current: {self._ctx.state.value})",
                }

            self._ctx.auth_attempts += 1
            self._ctx.last_auth_score = score

            expected_pid = self._ctx.current_patient_id
            slot = self._ctx.selected_slot

            # Liveness check
            if not liveness_ok:
                _db_log_face_auth(expected_pid, matched_patient_id, score,
                                   False, slot, "liveness_failed")
                _log(f"FACE AUTH: Liveness check failed (attempt #{self._ctx.auth_attempts})")
                return {
                    "ok": False,
                    "message": "Liveness check failed. Please try again.",
                    "attempts": self._ctx.auth_attempts,
                }

            # Score check
            if score < FACE_SCORE_THRESHOLD:
                _db_log_face_auth(expected_pid, matched_patient_id, score,
                                   True, slot, "low_score")
                _log(f"FACE AUTH: Score too low ({score:.2f} < {FACE_SCORE_THRESHOLD})")
                return {
                    "ok": False,
                    "message": f"Score too low: {score:.2f}",
                    "attempts": self._ctx.auth_attempts,
                }

            # Patient ID match
            if matched_patient_id != expected_pid:
                _db_log_face_auth(expected_pid, matched_patient_id, score,
                                   True, slot, "wrong_patient")
                _log(f"FACE AUTH: Wrong patient "
                     f"(expected={expected_pid[:8]}... got={matched_patient_id[:8]}...)")
                return {
                    "ok": False,
                    "message": "Face does not match the expected patient",
                    "attempts": self._ctx.auth_attempts,
                }

            # ✅ Authentication successful
            _db_log_face_auth(expected_pid, matched_patient_id, score,
                               True, slot, "success")

            self._ctx.camera_active = False
            self._auth_cancel.set()   # Stop timeout watcher

            self._set_state(
                DispenserState.FACE_MATCHED,
                f"{name or matched_patient_id[:8]}... score={score:.2f}",
            )

            _log(f"╔══════════════════════════════════════════╗")
            _log(f"║  ACCESS GRANTED: {name or 'Patient':<23} ║")
            _log(f"║  Score: {score:.2f}   Slot: {slot:<22}  ║")
            _log(f"╚══════════════════════════════════════════╝")

            return {
                "ok": True,
                "action": "dispense",
                "patient_id": matched_patient_id,
                "score": score,
                "slot": slot,
                "message": f"Access granted for {name or matched_patient_id[:8]}",
            }

    # -- Dispense (Motor + Servo) --

    def dispense(self) -> Dict[str, Any]:
        """Rotate wheel and open gate after face auth succeeds."""
        with self._lock:
            if self._ctx.state != DispenserState.FACE_MATCHED:
                return {
                    "ok": False,
                    "message": f"Cannot dispense in state '{self._ctx.state.value}'",
                }

            slot = self._ctx.selected_slot
            patient_id = self._ctx.current_patient_id

            self._set_state(DispenserState.DISPENSING, f"slot={slot}")
            self._ctx.motor_busy = True

        # Motor + servo control (outside lock)
        def _dispense_worker():
            success = True
            try:
                if self._motor:
                    _log(f"Rotating to slot {slot}")
                    self._motor.rotate_to_slot(slot)

                    _log("Opening gate")
                    self._motor.open_gate()

                    with self._lock:
                        self._ctx.servo_open = True

                    # Wait 5 seconds for patient to take medication
                    time.sleep(5)

                    _log("Closing gate")
                    self._motor.close_gate()

                    with self._lock:
                        self._ctx.servo_open = False
                else:
                    _log("[DRY-RUN] Motor not available — simulating dispense")
                    time.sleep(1)

            except Exception as e:
                _log(f"Dispense motor error: {e}")
                success = False

            with self._lock:
                self._ctx.motor_busy = False

                if success:
                    _log(f"DISPENSED: slot={slot} patient={patient_id[:8]}...")

                    # BLE notify: pill taken
                    if self._on_notify:
                        self._on_notify([0xA1])  # EVT_PILL_TAKEN

                    self._set_state(DispenserState.IDLE, "dispense complete")
                    self._reset_context()
                else:
                    self._set_state(DispenserState.ERROR, "motor error during dispense")
                    self._ctx.last_error = "Motor error during dispense"
                    self._ctx.last_error_time = datetime.now(timezone.utc).isoformat()

                    if self._on_notify:
                        self._on_notify([0xA3, 0x02])  # EVT_HARDWARE_ERROR, ERR_MOTOR

        thread = threading.Thread(target=_dispense_worker, daemon=True)
        thread.start()

        return {
            "ok": True,
            "slot": slot,
            "patient_id": patient_id,
            "message": f"Dispensing from slot {slot}",
        }

    # -- Camera Control (Manual Override) --

    def open_camera_manual(
        self,
        patient_id: Optional[str] = None,
        duration: int = WINDOW_SECONDS,
    ) -> Dict[str, Any]:
        """Manual camera override from the caregiver app."""
        with self._lock:
            if self._ctx.state in (
                DispenserState.WAITING_FOR_PATIENT,
                DispenserState.DISPENSING,
            ):
                return {
                    "ok": False,
                    "message": f"Already active ({self._ctx.state.value})",
                }

        # Delegate to trigger_dispense (will acquire lock)
        return self.trigger_dispense(
            patient_id=patient_id,
            window_seconds=duration,
        )

    # -- Reset & Error Recovery --

    def reset(self) -> Dict[str, Any]:
        """Reset to IDLE from any state. Stops motor, closes camera, clears context."""
        with self._lock:
            # Stop running auth thread
            self._auth_cancel.set()
            if self._auth_thread and self._auth_thread.is_alive():
                self._auth_thread.join(timeout=1)
                self._auth_thread = None

            # Close gate if open
            if self._motor and self._ctx.servo_open:
                try:
                    self._motor.close_gate()
                except Exception:
                    pass

            old_state = self._ctx.state
            self._reset_context()
            self._set_state(DispenserState.IDLE, f"reset from {old_state.value}")

            return {
                "ok": True,
                "message": f"Reset from {old_state.value} to IDLE",
                "state": DispenserState.IDLE.value,
            }

    def _reset_context(self):
        """Clear all context fields. Must be called under lock."""
        self._ctx.current_patient_id = None
        self._ctx.current_patient_name = None
        self._ctx.selected_slot = None
        self._ctx.barcode_count = 0
        self._ctx.scanned_barcodes = []
        self._ctx.motor_busy = False
        self._ctx.servo_open = False
        self._ctx.window_start = None
        self._ctx.window_deadline = None
        self._ctx.auth_attempts = 0
        self._ctx.last_auth_score = None
        self._ctx.camera_active = False
        self._ctx.last_error = None
        self._ctx.last_error_time = None

    # -- Slot Query --

    @staticmethod
    def get_all_slots() -> List[Dict[str, Any]]:
        """Return all slot bindings."""
        conn = sqlite3.connect(LOCAL_DB)
        try:
            rows = conn.execute("""
                SELECT slot_id, patient_id, pill_count, committed, updated_at
                FROM slot_bindings
                ORDER BY slot_id
            """).fetchall()
            return [
                {
                    "slot_id": r[0],
                    "patient_id": r[1],
                    "pill_count": r[2],
                    "committed": bool(r[3]),
                    "updated_at": r[4],
                }
                for r in rows
            ]
        finally:
            conn.close()

    @staticmethod
    def get_slot_medications(slot_id: int) -> List[Dict[str, Any]]:
        """Return barcode records for a given slot."""
        conn = sqlite3.connect(LOCAL_DB)
        try:
            rows = conn.execute("""
                SELECT id, barcode, scanned_at
                FROM slot_medications
                WHERE slot_id = ?
                ORDER BY id
            """, (slot_id,)).fetchall()
            return [
                {
                    "id": r[0],
                    "barcode": r[1],
                    "scanned_at": r[2],
                }
                for r in rows
            ]
        finally:
            conn.close()

    @staticmethod
    def get_face_auth_logs(limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent face auth attempts."""
        conn = sqlite3.connect(LOCAL_DB)
        try:
            rows = conn.execute("""
                SELECT id, patient_id, matched_patient, score,
                       liveness_ok, slot_dispensed, status, created_at
                FROM face_auth_log
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [
                {
                    "id": r[0],
                    "patient_id": r[1],
                    "matched_patient": r[2],
                    "score": r[3],
                    "liveness_ok": bool(r[4]),
                    "slot_dispensed": r[5],
                    "status": r[6],
                    "created_at": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()
