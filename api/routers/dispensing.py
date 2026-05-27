from fastapi import APIRouter, Depends, HTTPException
from api.database import get_aws
from pydantic import BaseModel
from typing import Optional
import boto3
import json
import os

router = APIRouter()


class DispensingLogCreate(BaseModel):
    """
    Log data received from Pi after face authentication.
    Pi will send this information in a POST request.
    """
    patient_id:       str
    schedule_id:      Optional[str] = None
    status:           str            # 'dispensed', 'taken', 'missed', 'error'
    face_auth_score:  Optional[float] = None
    device_timestamp: Optional[str] = None
    error_details:    Optional[str] = None


def _put_to_kinesis(data: dict):
    """Writes the dispensing event to Kinesis Data Streams."""
    kinesis = boto3.client(
        "kinesis",
        region_name=os.getenv("APP_REGION", "eu-north-1"),
    )
    kinesis.put_record(
        StreamName=os.getenv("KINESIS_STREAM_NAME", "drug-dispenser-logs"),
        Data=json.dumps(data, default=str),
        PartitionKey=data["patient_id"],
    )


@router.post("")
@router.post("/")
def create_dispensing_log(log: DispensingLogCreate, aws=Depends(get_aws)):
    """
    Saves the dispensing event to the AWS database.
    Called after ACCESS GRANTED in the Pi face authentication flow.
    After saving, notifies linked caregivers via FCM push notification.
    """
    cur = aws.cursor()

    # 1. Insert dispensing log
    cur.execute("""
        INSERT INTO dispensing_logs (
            patient_id,
            schedule_id,
            status,
            face_auth_score,
            device_timestamp,
            error_details
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING log_id, dispensing_at
    """, (
        log.patient_id,
        log.schedule_id,
        log.status,
        log.face_auth_score,
        log.device_timestamp,
        log.error_details,
    ))
    aws.commit()
    result       = cur.fetchone()
    log_id       = str(result["log_id"])
    dispensing_at = str(result["dispensing_at"])

    # 2. Write to Kinesis Data Streams
    try:
        print("[KINESIS] Writing event...")
        _put_to_kinesis({
            "log_id":           log_id,
            "patient_id":       log.patient_id,
            "schedule_id":      log.schedule_id,
            "status":           log.status,
            "face_auth_score":  log.face_auth_score,
            "device_timestamp": log.device_timestamp,
            "error_details":    log.error_details,
            "dispensing_at":    dispensing_at,
        })
        print("[KINESIS] Success.")
    except Exception as e:
        print(f"[KINESIS WARNING] {e}")

    # 3. Notify caregivers — fetch patient name first, then push
    try:
        cur.execute(
            "SELECT first_name, last_name FROM patients WHERE patient_id::text = %s",
            (log.patient_id,)
        )
        patient_row  = cur.fetchone()
        patient_name = f"{patient_row['first_name']} {patient_row['last_name']}" if patient_row else "Hasta"

        # Get planned_time from schedule if available
        planned_time = None
        if log.schedule_id:
            cur.execute(
                "SELECT planned_time FROM medication_schedules WHERE schedule_id::text = %s",
                (log.schedule_id,)
            )
            sched_row = cur.fetchone()
            if sched_row:
                planned_time = str(sched_row["planned_time"])[:5]  # HH:MM

        from api.routers.notifications import notify_caregivers_of_dispensing
        notify_caregivers_of_dispensing(
            aws=aws,
            patient_id=log.patient_id,
            patient_name=patient_name,
            status=log.status,
            planned_time=planned_time,
        )
    except Exception as e:
        print(f"[NOTIFY] Caregiver notification failed (non-critical): {e}")

    return {
        "log_id":        log_id,
        "dispensing_at": dispensing_at,
        "status":        log.status,
    }


@router.get("/{patient_id}")
def get_dispensing_logs(patient_id: str, aws=Depends(get_aws)):
    """Returns the dispensing history for a patient."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            log_id,
            patient_id,
            schedule_id,
            status,
            face_auth_score,
            dispensing_at,
            taken_at,
            device_timestamp,
            error_details
        FROM dispensing_logs
        WHERE patient_id = %s
        ORDER BY dispensing_at DESC
    """, (patient_id,))
    logs = cur.fetchall()

    if not logs:
        raise HTTPException(status_code=404, detail="No logs found for this patient")

    return {"logs": [dict(l) for l in logs]}


@router.get("/{patient_id}/analytics")
def get_patient_analytics(
    patient_id: str,
    start_date: str,
    end_date: str,
    aws=Depends(get_aws)
):
    """
    Returns dispensing analytics for a patient between start_date and end_date.
    start_date and end_date format: YYYY-MM-DD
    """
    cur = aws.cursor()

    # Overall summary counts
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'dispensed') AS total_dispensed,
            COUNT(*) FILTER (WHERE status = 'taken')     AS total_taken,
            COUNT(*) FILTER (WHERE status = 'missed')    AS total_missed
        FROM dispensing_logs
        WHERE patient_id = %s
          AND dispensing_at::date BETWEEN %s::date AND %s::date
    """, (patient_id, start_date, end_date))

    summary         = cur.fetchone()
    total_dispensed = summary["total_dispensed"] or 0
    total_taken     = summary["total_taken"] or 0
    total_missed    = summary["total_missed"] or 0
    adherence_rate  = round(
        (total_taken / (total_taken + total_missed) * 100), 1
    ) if (total_taken + total_missed) > 0 else 0.0

    # Daily breakdown
    cur.execute("""
        SELECT
            dispensing_at::date AS date,
            COUNT(*) FILTER (WHERE status = 'dispensed') AS dispensed,
            COUNT(*) FILTER (WHERE status = 'taken')     AS taken,
            COUNT(*) FILTER (WHERE status = 'missed')    AS missed
        FROM dispensing_logs
        WHERE patient_id = %s
          AND dispensing_at::date BETWEEN %s::date AND %s::date
        GROUP BY dispensing_at::date
        ORDER BY dispensing_at::date
    """, (patient_id, start_date, end_date))

    daily = cur.fetchall()

    daily_stats = []
    for row in daily:
        day_date = row["date"]

        cur.execute("""
            SELECT DISTINCT
                dl.status,
                sm.medication_name
            FROM dispensing_logs dl
            JOIN medication_schedules ms
              ON ms.schedule_id::text = dl.schedule_id::text
            JOIN slot_medications sm
              ON sm.slot_id = ms.slot_id
             AND sm.patient_id::text = dl.patient_id::text
            WHERE dl.patient_id::text = %s
              AND dl.dispensing_at::date = %s::date
              AND dl.status IN ('dispensed', 'taken', 'missed')
              AND sm.medication_name IS NOT NULL
              AND sm.medication_name <> ''
        """, (patient_id, str(day_date)))
        med_rows = cur.fetchall()

        dispensed_meds = []
        taken_meds = []
        missed_meds = []

        for med_row in med_rows:
            status = med_row["status"]
            med_name = med_row["medication_name"]
            if status == "dispensed":
                dispensed_meds.append(med_name)
            elif status == "taken":
                taken_meds.append(med_name)
            elif status == "missed":
                missed_meds.append(med_name)

        daily_stats.append({
            "date": str(day_date),
            "dispensed": row["dispensed"],
            "taken": row["taken"],
            "missed": row["missed"],
            "dispensed_medications": sorted(set(dispensed_meds)),
            "taken_medications": sorted(set(taken_meds)),
            "missed_medications": sorted(set(missed_meds)),
        })

    return {
        "patient_id":      patient_id,
        "start_date":      start_date,
        "end_date":        end_date,
        "total_dispensed": total_dispensed,
        "total_taken":     total_taken,
        "total_missed":    total_missed,
        "adherence_rate":  adherence_rate,
        "daily_stats": daily_stats,
    }