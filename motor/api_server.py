"""Flask REST API for the Smart Drug Dispenser Pi backend."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import uuid
import sqlite3
import threading
import time
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from state_machine import DispenserStateMachine, DispenserState, LOCAL_DB

app = Flask(__name__)

# Singleton state machine — initialized in main()
sm: DispenserStateMachine | None = None


def _ok(data: dict, status: int = 200):
    return jsonify(data), status


def _err(msg: str, status: int = 400):
    return jsonify({"ok": False, "message": msg}), status


# ── State ───────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state():
    if not sm:
        return _err("State machine not initialized", 503)
    data = sm.get_state_dict()
    data["ok"] = True
    return _ok(data)


# ── Slot binding ────────────────────────────────────────────────────────

@app.post("/api/bind-slot")
def bind_slot():
    if not sm:
        return _err("State machine not initialized", 503)
    body = request.get_json(silent=True) or {}
    patient_id = body.get("patient_id")
    slot_id = body.get("slot_id")
    patient_name = body.get("patient_name", "")

    if not patient_id or slot_id is None:
        return _err("patient_id and slot_id are required")

    try:
        slot_id = int(slot_id)
    except (ValueError, TypeError):
        return _err("slot_id must be an integer")

    result = sm.bind_slot(patient_id, slot_id, patient_name)
    status = 200 if result["ok"] else 409
    return _ok(result, status)


# ── Barcode scan ────────────────────────────────────────────────────────

@app.post("/api/barcode")
def barcode_scan():
    if not sm:
        return _err("State machine not initialized", 503)
    body = request.get_json(silent=True) or {}
    barcode = body.get("barcode", "")
    result = sm.increment_barcode(barcode)
    status = 200 if result["ok"] else 409
    return _ok(result, status)


# ── Commit slot ─────────────────────────────────────────────────────────

@app.post("/api/commit-slot")
def commit_slot():
    if not sm:
        return _err("State machine not initialized", 503)
    result = sm.commit_slot()
    status = 200 if result["ok"] else 409
    return _ok(result, status)


# ── Trigger dispense ────────────────────────────────────────────────────

@app.post("/api/trigger-dispense")
def trigger_dispense():
    if not sm:
        return _err("State machine not initialized", 503)
    body = request.get_json(silent=True) or {}
    patient_id = body.get("patient_id")
    schedule_id = body.get("schedule_id")
    window_sec = body.get("window_seconds")

    result = sm.trigger_dispense(
        patient_id=patient_id,
        schedule_id=schedule_id,
        window_seconds=int(window_sec) if window_sec and str(window_sec).strip() else None,
    )
    status = 200 if result["ok"] else 409
    return _ok(result, status)


# ── Manual camera override ──────────────────────────────────────────────

@app.post("/api/camera/open")
def camera_open():
    if not sm:
        return _err("State machine not initialized", 503)
    data = request.get_json(silent=True) or {}
    patient_id = data.get("patient_id") or None
    result = sm.open_camera_manual(patient_id=patient_id)
    status = 200 if result["ok"] else 409
    return _ok(result, status)


# ── Reset ───────────────────────────────────────────────────────────────

@app.post("/api/reset")
def reset():
    if not sm:
        return _err("State machine not initialized", 503)
    result = sm.reset()
    return _ok(result)


# ── Query endpoints ─────────────────────────────────────────────────────

@app.get("/api/slots")
def list_slots():
    slots = DispenserStateMachine.get_all_slots()
    return _ok({"ok": True, "slots": slots})


@app.get("/api/slots/<int:slot_id>/medications")
def slot_medications(slot_id: int):
    meds = DispenserStateMachine.get_slot_medications(slot_id)
    return _ok({"ok": True, "slot_id": slot_id, "medications": meds})


@app.delete("/api/slots/<int:slot_id>")
def delete_slot(slot_id: int):
    """Remove a slot binding and its barcode records."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        cur = conn.execute(
            "DELETE FROM slot_bindings WHERE slot_id = ?", (slot_id,)
        )
        conn.execute(
            "DELETE FROM slot_medications WHERE slot_id = ?", (slot_id,)
        )
        conn.commit()
    finally:
        conn.close()
    if cur.rowcount == 0:
        return _err(f"Slot {slot_id} not found", 404)
    return _ok({"ok": True, "deleted_slot": slot_id})


@app.get("/api/face-auth-logs")
def face_auth_logs():
    limit = request.args.get("limit", 20, type=int)
    logs = DispenserStateMachine.get_face_auth_logs(limit)
    return _ok({"ok": True, "logs": logs})


@app.delete("/api/face-auth-logs")
def clear_face_auth_logs():
    """Delete all face auth log entries."""
    conn = sqlite3.connect(LOCAL_DB)
    try:
        conn.execute("DELETE FROM face_auth_log")
        conn.commit()
    finally:
        conn.close()
    return _ok({"ok": True, "message": "Face auth logs cleared"})


# ── Face registration ───────────────────────────────────────────────────

@app.post("/api/face/register")
def face_register():
    """Trigger face registration via Pi camera. Captures N samples."""
    body = request.get_json(silent=True) or {}
    patient_id = body.get("patient_id")
    first_name = body.get("first_name", "")
    last_name = body.get("last_name", "")
    samples = body.get("samples", 5)
    allow_duplicate = body.get("allow_duplicate", False)

    if not patient_id or not first_name or not last_name:
        return _err("patient_id, first_name, and last_name are required")

    try:
        from face_authentication.pi_face_register import (
            capture_face_encodings,
            save_user_embedding,
            check_face_duplicates,
        )
        import numpy as np

        encodings = capture_face_encodings(max_samples=min(samples, 10))
        if not encodings:
            return _err("No face samples captured", 422)

        avg = np.mean(encodings, axis=0).astype(np.float32)

        # Check for duplicate face before registration.
        # Exclude the current patient so re-registration doesn't self-block.
        if not allow_duplicate:
            duplicate = check_face_duplicates(avg, threshold=0.6,
                                              exclude_patient_id=patient_id)
            if duplicate:
                # Return conflict status with duplicate details
                return _ok({
                    "ok": False,
                    "duplicate_found": True,
                    "existing_patient_id": duplicate["patient_id"],
                    "existing_name": f"{duplicate['first_name']} {duplicate['last_name']}",
                    "similarity_distance": duplicate["distance"],
                    "message": f"This face matches an existing patient: {duplicate['first_name']} {duplicate['last_name']}. "
                              f"Set allow_duplicate=true to override.",
                }, 409)

        save_user_embedding(
            patient_id, first_name, last_name, avg,
            individual_encodings=encodings,
        )

        return _ok({
            "ok": True,
            "patient_id": patient_id,
            "samples_captured": len(encodings),
            "message": f"Registered {first_name} {last_name} with {len(encodings)} samples",
        })
    except Exception as e:
        return _err(f"Registration failed: {e}", 500)


@app.get("/api/face/users")
def face_users():
    """List registered face users (without vectors)."""
    import sqlite3
    from state_machine import LOCAL_DB
    conn = sqlite3.connect(LOCAL_DB)
    try:
        try:
            rows = conn.execute(
                "SELECT patient_id, first_name, last_name FROM local_users"
            ).fetchall()
        except sqlite3.OperationalError:
            return _ok({"ok": True, "users": []})

        # Count samples per user
        users = []
        for pid, fn, ln in rows:
            sample_count = 0
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM face_samples WHERE patient_id = ?", (pid,)
                ).fetchone()
                sample_count = row[0] if row else 0
            except sqlite3.OperationalError:
                pass
            users.append({
                "patient_id": pid,
                "first_name": fn,
                "last_name": ln,
                "sample_count": sample_count,
            })

        return _ok({"ok": True, "users": users})
    finally:
        conn.close()


# ── Patients ────────────────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(LOCAL_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables():
    """Create core tables if missing (idempotent)."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id     TEXT PRIMARY KEY,
                first_name     TEXT NOT NULL,
                last_name      TEXT NOT NULL,
                date_of_birth  TEXT,
                created_at     TEXT NOT NULL,
                cloud_synced_at TEXT,
                deleted_at     TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                medication_id       TEXT PRIMARY KEY,
                patient_id          TEXT NOT NULL,
                medication_name     TEXT NOT NULL,
                pill_barcode        TEXT,
                pill_color_shape    TEXT,
                remaining_count     INTEGER DEFAULT 0,
                low_stock_threshold INTEGER DEFAULT 5,
                expiry_date         TEXT,
                created_at          TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS local_schedules (
                schedule_id     TEXT PRIMARY KEY,
                patient_id      TEXT NOT NULL,
                medication_id   TEXT,
                planned_time    TEXT,
                dosage_quantity INTEGER DEFAULT 1,
                slot_id         INTEGER,
                is_active       INTEGER DEFAULT 1,
                start_date      TEXT,
                end_date        TEXT,
                synced_at       TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                log_id          TEXT PRIMARY KEY,
                patient_id      TEXT NOT NULL,
                schedule_id     TEXT,
                status          TEXT NOT NULL,
                face_auth_score REAL,
                dispensing_at   TEXT,
                taken_at        TEXT,
                device_timestamp TEXT,
                error_details   TEXT,
                is_synced       INTEGER DEFAULT 0,
                retry_count     INTEGER DEFAULT 0
            )""")
        conn.commit()

    # Migrate existing tables — add columns that may not exist yet
    _run_migrations()


def _run_migrations():
    """Add missing columns to existing tables (safe to run on every startup)."""
    migrations = [
        ("patients", "cloud_synced_at", "TEXT"),
        ("patients", "deleted_at",      "TEXT"),
    ]
    with _db() as conn:
        for table, column, col_type in migrations:
            existing = [
                row[1] for row in conn.execute(f"PRAGMA table_info({table})")
            ]
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                print(f"[Migration] Added {table}.{column}", flush=True)
        conn.commit()


_ensure_tables()


def _ensure_default_accounts() -> None:
    """Create default caregiver/patient accounts from .env if they don't exist."""
    from auth import create_user
    import dotenv as _dotenv
    _env: dict = {}
    _env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_file):
        for line in open(_env_file).read().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                _env[k.strip()] = v.strip()

    email    = _env.get("DEFAULT_CAREGIVER_EMAIL", "caregiver@medidispense.local")
    password = _env.get("DEFAULT_CAREGIVER_PASSWORD", "MediPass2024!")

    result = create_user(email, password, role="caregiver")
    if result.get("ok"):
        print(f"[Auth] Default caregiver created: {email}", flush=True)
    # "Email already registered" is fine — account already exists


_ensure_default_accounts()


@app.get("/api/patients")
def list_patients():
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE deleted_at IS NULL ORDER BY last_name, first_name"
        ).fetchall()
    patients = [dict(r) for r in rows]
    return _ok({"ok": True, "patients": patients})


@app.post("/api/patients")
def create_patient():
    body = request.get_json(silent=True) or {}
    first = body.get("first_name", "").strip()
    last  = body.get("last_name", "").strip()
    if not first or not last:
        return _err("first_name and last_name are required")

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        conn.execute(
            "INSERT INTO patients (patient_id, first_name, last_name, date_of_birth, created_at, cloud_synced_at) "
            "VALUES (?,?,?,?,?,?)",
            (pid, first, last, body.get("date_of_birth"), now, None),
        )
        conn.commit()
    return _ok({"ok": True, "patient_id": pid, "first_name": first, "last_name": last}, 201)


@app.get("/api/patients/<patient_id>")
def get_patient(patient_id: str):
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id=?", (patient_id,)
        ).fetchone()
    if not row:
        return _err("Patient not found", 404)
    return _ok({"ok": True, **dict(row)})


@app.put("/api/patients/<patient_id>")
def update_patient(patient_id: str):
    body = request.get_json(silent=True) or {}
    first = body.get("first_name", "").strip()
    last  = body.get("last_name", "").strip()
    if not first or not last:
        return _err("first_name and last_name are required")

    with _db() as conn:
        cur = conn.execute(
            "UPDATE patients SET first_name=?, last_name=?, date_of_birth=? WHERE patient_id=?",
            (first, last, body.get("date_of_birth"), patient_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        return _err("Patient not found", 404)
    return _ok({"ok": True, "patient_id": patient_id, "first_name": first, "last_name": last})


@app.delete("/api/patients/<patient_id>")
def delete_patient(patient_id: str):
    """
    Full patient removal:
    - Soft-delete in patients table (for AWS sync)
    - Hard-delete face data (face_samples, local_users)
    - Hard-delete local schedules, slot bindings, medications
    """
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        row = conn.execute(
            "SELECT patient_id FROM patients WHERE patient_id=? AND deleted_at IS NULL",
            (patient_id,)
        ).fetchone()
        if not row:
            return _err("Patient not found", 404)

        # Soft delete in patients table (needed for sync to push deletion to AWS)
        conn.execute(
            "UPDATE patients SET deleted_at=? WHERE patient_id=?",
            (now, patient_id)
        )

        # Hard-delete face data when patient is deleted
        conn.execute("DELETE FROM face_samples WHERE patient_id=?", (patient_id,))
        try:
            conn.execute("DELETE FROM local_users WHERE patient_id=?", (patient_id,))
        except Exception:
            pass

        # Hard delete local schedules and slot bindings
        conn.execute("DELETE FROM local_schedules WHERE patient_id=?", (patient_id,))
        conn.execute("DELETE FROM slot_bindings WHERE patient_id=?", (patient_id,))

        # Hard delete medications
        conn.execute("DELETE FROM medications WHERE patient_id=?", (patient_id,))

        conn.commit()

    return _ok({"ok": True, "deleted": patient_id})


# ── Medications ──────────────────────────────────────────────────────────

@app.get("/api/medications/<patient_id>")
def list_medications(patient_id: str):
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM medications WHERE patient_id=? ORDER BY medication_name",
            (patient_id,),
        ).fetchall()
    return _ok({"ok": True, "medications": [dict(r) for r in rows]})


@app.post("/api/medications")
def create_medication():
    body = request.get_json(silent=True) or {}
    pid  = body.get("patient_id", "").strip()
    name = body.get("medication_name", "").strip()
    if not pid or not name:
        return _err("patient_id and medication_name are required")

    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _db() as conn:
        conn.execute(
            """INSERT INTO medications
               (medication_id, patient_id, medication_name, pill_barcode,
                pill_color_shape, remaining_count, low_stock_threshold, expiry_date, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                mid, pid, name,
                body.get("pill_barcode"),
                body.get("pill_color_shape"),
                int(body.get("remaining_count", 0)),
                int(body.get("low_stock_threshold", 5)),
                body.get("expiry_date"),
                now,
            ),
        )
        conn.commit()
    return _ok({"ok": True, "medication_id": mid, "medication_name": name}, 201)


# ── Schedules ────────────────────────────────────────────────────────────

@app.get("/api/schedules/<patient_id>")
def list_schedules(patient_id: str):
    with _db() as conn:
        rows = conn.execute("""
            SELECT ls.schedule_id, ls.patient_id, ls.medication_id,
                   m.medication_name, ls.planned_time, ls.dosage_quantity,
                   ls.slot_id, ls.is_active, ls.start_date, ls.end_date,
                   m.remaining_count
            FROM local_schedules ls
            LEFT JOIN medications m ON ls.medication_id = m.medication_id
            WHERE ls.patient_id = ?
            ORDER BY ls.planned_time
        """, (patient_id,)).fetchall()
    return _ok({"ok": True, "schedules": [dict(r) for r in rows]})


@app.post("/api/schedules")
def create_schedule():
    body = request.get_json(silent=True) or {}
    med_id       = body.get("medication_id", "").strip()
    planned_time = body.get("planned_time", "08:00").strip()
    if not med_id:
        return _err("medication_id is required")

    with _db() as conn:
        med_row = conn.execute(
            "SELECT patient_id, remaining_count FROM medications WHERE medication_id=?",
            (med_id,)
        ).fetchone()
    if not med_row:
        return _err("Medication not found", 404)

    # Dosage cannot exceed loaded pill count
    dosage_quantity = int(body.get("dosage_quantity", 1))
    remaining = med_row["remaining_count"] or 0
    if remaining > 0 and dosage_quantity > remaining:
        return _err(
            f"Dosage ({dosage_quantity}) exceeds loaded pill count ({remaining})", 400
        )
    if dosage_quantity < 1:
        return _err("Dosage must be at least 1", 400)

    sid        = str(uuid.uuid4())
    patient_id = med_row["patient_id"]
    start_date = body.get("start_date") or datetime.now(timezone.utc).date().isoformat()

    # Auto-look up which slot this patient's medication is assigned to
    with _db() as conn:
        slot_row = conn.execute(
            "SELECT slot_id FROM slot_bindings WHERE patient_id=? LIMIT 1",
            (patient_id,)
        ).fetchone()
    slot_id = slot_row["slot_id"] if slot_row else None

    with _db() as conn:
        conn.execute("""
            INSERT INTO local_schedules
                (schedule_id, patient_id, medication_id, planned_time,
                 dosage_quantity, slot_id, is_active, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            sid, patient_id, med_id, planned_time,
            dosage_quantity,
            slot_id,
            start_date,
            body.get("end_date"),
        ))
        conn.commit()

    # Push to AWS immediately; retry on next background sync if it fails
    try:
        from sync_service import SyncService
        svc = SyncService()
        if not svc._config_error():
            aws = svc._rds_connect(svc.env)
            try:
                svc._push_schedules(aws)
            finally:
                aws.close()
    except Exception as e:
        print(f"[Schedule] AWS push skipped (will retry on sync): {e}", flush=True)

    return _ok({"ok": True, "schedule_id": sid}, 201)


@app.delete("/api/schedules/<schedule_id>")
def delete_schedule(schedule_id: str):
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM local_schedules WHERE schedule_id=?", (schedule_id,)
        )
        conn.commit()
    if cur.rowcount == 0:
        return _err("Schedule not found", 404)
    return _ok({"ok": True, "deleted": schedule_id})


# ── Auth ────────────────────────────────────────────────────────────────

@app.post("/api/auth/signup")
@app.post("/api/auth/signup")
def auth_signup():
    """
    Register a new user.
    Body: { email, password, model_id? (optional), patient_id? (for patients) }
    If model_id matches DISPENSER_MODEL_ID → role = caregiver, else role = patient.
    """
    from auth import create_user, DISPENSER_MODEL_ID
    body = request.get_json(silent=True) or {}
    email      = body.get("email", "").strip()
    password   = body.get("password", "")
    model_id   = body.get("model_id", "").strip()
    patient_id = body.get("patient_id")

    if not email or not password:
        return _err("email and password are required")

    # Determine role from model_id
    if model_id:
        if model_id != DISPENSER_MODEL_ID:
            return _err("Invalid Model ID", 403)
        role = "caregiver"
        patient_id = None
    else:
        role = "patient"

    result = create_user(email, password, role=role, patient_id=patient_id)
    status = 200 if result["ok"] else 409

    # If local signup succeeded, also register in AWS RDS roles table
    if result["ok"]:
        try:
            import urllib.request
            import json as _json
            aws_url = os.environ.get(
                "AWS_API_URL",
                "https://s766ccq1c7.execute-api.eu-north-1.amazonaws.com/default"
            )
            payload = _json.dumps({
                "email":      email,
                "role_type":  role,
                "first_name": body.get("first_name", ""),
                "last_name":  body.get("last_name", ""),
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{aws_url}/auth/register-role",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            print(f"[Auth] AWS role registered: {email}", flush=True)
        except Exception as e:
            print(f"[Auth] AWS role registration failed (non-critical): {e}", flush=True)

    return _ok(result, status)


@app.post("/api/auth/login")
def auth_login():
    """
    Authenticate a user.
    Body: { email, password }
    Returns: { ok, role, patient_id, email }
    """
    from auth import authenticate_user
    body = request.get_json(silent=True) or {}
    email    = body.get("email", "").strip()
    password = body.get("password", "")

    if not email or not password:
        return _err("email and password are required")

    result = authenticate_user(email, password)
    status = 200 if result["ok"] else 401
    return _ok(result, status)


@app.get("/api/auth/model-id-hint")
def model_id_hint():
    """
    Returns the first 4 chars of Model ID as a hint (not the full value).
    Helps users verify they have the right ID without exposing it.
    """
    from auth import DISPENSER_MODEL_ID
    hint = DISPENSER_MODEL_ID[:4] + "*" * (len(DISPENSER_MODEL_ID) - 4)
    return _ok({"ok": True, "hint": hint})


# ── Cloud sync ──────────────────────────────────────────────────────────

@app.get("/api/sync/status")
def sync_status():
    """Return sync configuration and pending log count."""
    from sync_service import SyncService
    return _ok(SyncService().get_status())


@app.post("/api/sync")
def sync_full():
    """Trigger a full bidirectional sync (Pi ↔ AWS). Runs in the request thread (~5-10s)."""
    from sync_service import SyncService
    result = SyncService().full_sync()
    status = 200 if result.get("ok") else 500
    return _ok(result, status)


@app.post("/api/sync/push")
def sync_push():
    """Push only: patients + medications + dispensing logs → AWS."""
    from sync_service import SyncService
    svc = SyncService()
    err = svc._config_error()
    if err:
        return _err(err, 503)
    try:
        import psycopg2, psycopg2.extras
        aws = svc._rds_connect(svc.env)  # type: ignore[attr-defined]
    except Exception as e:
        return _err(f"RDS connection failed: {e}", 503)
    try:
        results = {
            "push_patients":        svc._push_patients(aws),
            "push_medications":     svc._push_medications(aws),
            "push_dispensing_logs": svc._push_dispensing_logs(aws),
        }
    finally:
        aws.close()
    return _ok({"ok": True, "results": results})


@app.post("/api/sync/pull")
def sync_pull():
    """Pull only: patients + medications + schedules ← AWS (cloud wins)."""
    from sync_service import SyncService
    svc = SyncService()
    err = svc._config_error()
    if err:
        return _err(err, 503)
    try:
        aws = svc._rds_connect(svc.env)  # type: ignore[attr-defined]
    except Exception as e:
        return _err(f"RDS connection failed: {e}", 503)
    try:
        results = {
            "pull_patients":   svc._pull_patients(aws),
            "pull_medications": svc._pull_medications(aws),
            "pull_schedules":  svc._pull_schedules(aws),
        }
    finally:
        aws.close()
    return _ok({"ok": True, "results": results})


# ── Health check ────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return _ok({
        "ok": True,
        "service": "MediDispense Pi Backend",
        "state": sm.state.value if sm else "not_initialized",
    })


# ── Error handlers ──────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return _err("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(_):
    return _err("Internal server error", 500)


# ── Entry point ─────────────────────────────────────────────────────────

def create_app(motor_controller=None) -> Flask:
    """Factory for external usage (e.g. gunicorn or testing)."""
    global sm
    sm = DispenserStateMachine(motor_controller=motor_controller)
    return app


def _start_background_sync(interval_minutes: int = 60) -> None:
    """Run full_sync() in a daemon thread, then reschedule itself."""
    import threading

    def _run():
        print("[Sync] Background sync started", flush=True)
        try:
            from sync_service import SyncService
            result = SyncService().full_sync()
            ok = result.get("ok", False)
            synced_at = result.get("synced_at", "?")
            print(f"[Sync] {'OK' if ok else 'PARTIAL'} at {synced_at}", flush=True)
            if result.get("errors"):
                for e in result["errors"]:
                    print(f"[Sync]   error: {e}", flush=True)
        except Exception as e:
            print(f"[Sync] Background sync error: {e}", flush=True)
        finally:
            # Reschedule next run
            t = threading.Timer(interval_minutes * 60, _run)
            t.daemon = True
            t.start()

    # First run after 10s (let server fully start), then every interval_minutes
    t = threading.Timer(10, _run)
    t.daemon = True
    t.start()
    print(f"[Sync] Background sync scheduled every {interval_minutes} min", flush=True)


# ── Kiosk / Scheduler endpoints ────────────────────────────────────────

# Global reference to kiosk scheduler (set if kiosk_app is running)
_kiosk_scheduler = None


def set_kiosk_scheduler(scheduler):
    """Called by kiosk_app to allow API-triggered dispensing."""
    global _kiosk_scheduler
    _kiosk_scheduler = scheduler


@app.post("/api/dispense/trigger")
def api_trigger_dispense_schedule():
    """Manually trigger a specific schedule for dispensing (for testing).

    Body: {"schedule_id": "..."}
    """
    body = request.get_json(silent=True) or {}
    schedule_id = body.get("schedule_id")
    if not schedule_id:
        return _err("schedule_id is required")

    if not _kiosk_scheduler:
        return _err("Kiosk scheduler not running", 503)

    result = _kiosk_scheduler.trigger_now(schedule_id)
    if result:
        return _ok({
            "ok": True,
            "message": f"Triggered: {result.medication_name} for {result.patient_name}",
            "schedule": {
                "schedule_id": result.schedule_id,
                "patient_name": result.patient_name,
                "medication_name": result.medication_name,
                "slot_id": result.slot_id,
            },
        })
    return _err("Schedule not found", 404)


@app.get("/api/dispense/next")
def api_next_schedule():
    """Get the next upcoming scheduled medication."""
    if not _kiosk_scheduler:
        return _err("Kiosk scheduler not running", 503)

    nxt = _kiosk_scheduler.get_next_schedule()
    if nxt:
        return _ok({
            "ok": True,
            "next_schedule": {
                "schedule_id": nxt.schedule_id,
                "patient_name": nxt.patient_name,
                "medication_name": nxt.medication_name,
                "planned_time": nxt.planned_time,
                "slot_id": nxt.slot_id,
                "dosage_quantity": nxt.dosage_quantity,
            },
        })
    return _ok({"ok": True, "next_schedule": None,
                "message": "No upcoming schedules"})


@app.post("/api/servo/test")
def api_servo_test():
    """Test the servo motor (open/close cycle). For hardware testing."""
    body = request.get_json(silent=True) or {}
    action = body.get("action", "cycle")  # cycle, open, close

    try:
        from servo_control import ServoController
        servo = ServoController()
        if action == "open":
            ok = servo.open_gate()
            return _ok({"ok": ok, "message": "Gate opened" if ok else "Open failed"})
        elif action == "close":
            ok = servo.close_gate()
            return _ok({"ok": ok, "message": "Gate closed" if ok else "Close failed"})
        else:
            ok = servo.dispense_cycle()
            return _ok({"ok": ok, "message": "Cycle complete" if ok else "Cycle failed"})
    except Exception as e:
        return _err(f"Servo error: {e}", 500)


# ── Main ───────────────────────────────────────────────────────────────

AUTO_SYNC_INTERVAL = 300  # seconds (5 minutes)

def _auto_sync_loop():
    """Background thread: sync with AWS every AUTO_SYNC_INTERVAL seconds."""
    # Wait a bit after startup before first sync
    time.sleep(30)
    while True:
        try:
            from sync_service import SyncService
            result = SyncService().run_sync()
            if result.get("ok"):
                print(f"[AutoSync] OK — patients pushed: {result.get('results', {}).get('push_patients', {})}", flush=True)
            else:
                print(f"[AutoSync] Failed: {result.get('error')}", flush=True)
        except Exception as e:
            print(f"[AutoSync] Error: {e}", flush=True)
        time.sleep(AUTO_SYNC_INTERVAL)


def main():
    global sm

    # Optional: import motor controller if available
    motor = None
    try:
        from motor_controller import MotorController
        motor = MotorController()
        print("[API] Motor controller loaded")
    except Exception:
        print("[API] Motor controller not available — dry run mode")

    sm = DispenserStateMachine(motor_controller=motor)

    # Ensure tables exist
    _ensure_tables()
    _ensure_default_accounts()

    # Start background cloud sync
    sync_interval = int(os.environ.get("SYNC_INTERVAL_MINUTES", "60"))
    _start_background_sync(interval_minutes=sync_interval)

    host  = os.environ.get("API_HOST",  "0.0.0.0")
    port  = int(os.environ.get("API_PORT",  "5000"))
    debug = os.environ.get("API_DEBUG", "0") == "1"

    # Start background auto-sync thread
    sync_thread = threading.Thread(target=_auto_sync_loop, name="auto-sync", daemon=True)
    sync_thread.start()
    print(f"[API] Auto-sync enabled every {AUTO_SYNC_INTERVAL}s", flush=True)

    print(f"[API] Starting on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
