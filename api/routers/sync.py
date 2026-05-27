from fastapi import APIRouter, Depends, HTTPException
from api.database import get_aws, get_local

router = APIRouter()


@router.post("/pull/{patient_id}")
def pull_schedules(patient_id: str, aws=Depends(get_aws), local=Depends(get_local)):
    """
    Gets active schedules from AWS database and save to local SQLite database.
    Pi will call this every 60 minutes.
    """
    # 1. Get schedules from AWS database
    cur = aws.cursor()
    cur.execute("""
        SELECT
            ms.schedule_id,
            m.patient_id,
            ms.planned_time,
            ms.dosage_quantity,
            ms.is_active,
            ms.start_date,
            ms.end_date
        FROM medication_schedules ms
        JOIN medications m ON ms.medication_id = m.medication_id
        WHERE m.patient_id = %s
          AND ms.is_active = TRUE
    """, (patient_id,))
    schedules = cur.fetchall()

    # 2. Save to local SQLite database
    local_cur = local.cursor()
    inserted = 0
    updated  = 0

    for s in schedules:
        # Check if already exists
        existing = local_cur.execute(
            "SELECT schedule_id FROM local_schedules WHERE schedule_id = ?",
            (str(s["schedule_id"]),)
        ).fetchone()

        if existing:
            # Update if exists
            local_cur.execute("""
                UPDATE local_schedules SET
                    planned_time     = ?,
                    dosage_quantity  = ?,
                    is_active        = ?,
                    start_date       = ?,
                    end_date         = ?
                WHERE schedule_id = ?
            """, (
                str(s["planned_time"]),
                s["dosage_quantity"],
                1 if s["is_active"] else 0,
                str(s["start_date"]),
                str(s["end_date"]) if s["end_date"] else None,
                str(s["schedule_id"])
            ))
            updated += 1
        else:
            # Insert if not exists
            local_cur.execute("""
                INSERT INTO local_schedules
                    (schedule_id, patient_id, planned_time, dosage_quantity, is_active, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(s["schedule_id"]),
                str(s["patient_id"]),
                str(s["planned_time"]),
                s["dosage_quantity"],
                1 if s["is_active"] else 0,
                str(s["start_date"]),
                str(s["end_date"]) if s["end_date"] else None
            ))
            inserted += 1

    local.commit()

    return {
        "message":  "Schedules synced",
        "inserted": inserted,
        "updated":  updated,
        "total":    len(schedules)
    }


@router.post("/push/{patient_id}")
def push_logs(patient_id: str, aws=Depends(get_aws), local=Depends(get_local)):
    """
    Sends pending logs to AWS database.
    Pi will call this when internet is available.
    """
    # 1. Get pending logs
    local_cur = local.cursor()
    pending = local_cur.execute("""
        SELECT * FROM sync_queue
        WHERE patient_id = ? AND is_synced = 0
    """, (patient_id,)).fetchall()

    if not pending:
        return {"message": "No logs to send", "pushed": 0, "failed": 0, "total": 0}

    # 2. Send to AWS database
    aws_cur = aws.cursor()
    pushed  = 0
    failed  = 0

    for log in pending:
        try:
            aws_cur.execute("""
                INSERT INTO dispensing_logs
                    (log_id, patient_id, schedule_id, status,
                     face_auth_score, dispensing_at, taken_at,
                     device_timestamp, error_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (log_id) DO NOTHING
            """, (
                log["log_id"],
                log["patient_id"],
                log["schedule_id"],
                log["status"],
                log["face_auth_score"],
                log["dispensing_at"],
                log["taken_at"],
                log["device_timestamp"],
                log["error_details"],
            ))
            # If successful, set is_synced = 1
            local_cur.execute("""
                UPDATE sync_queue SET is_synced = 1, retry_count = retry_count + 1
                WHERE log_id = ?
            """, (log["log_id"],))
            pushed += 1
        except Exception as e:
            # If failed, increment retry_count
            local_cur.execute("""
                UPDATE sync_queue SET retry_count = retry_count + 1
                WHERE log_id = ?
            """, (log["log_id"],))
            failed += 1

    aws.commit()
    local.commit()

    return {
        "message": "Sync completed",
        "pushed":  pushed,
        "failed":  failed,
        "total":   len(pending)
    }