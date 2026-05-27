from fastapi import APIRouter, Depends, HTTPException
from api.database import get_aws
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()


class SlotMedicationItem(BaseModel):
    medication_id:   str
    medication_name: str
    target_count:    int = 0
    loaded_count:    int = 0


class ScheduleCreate(BaseModel):
    patient_id:     str
    slot_id:        int
    planned_time:   str
    frequency_type: str = "daily"
    week_days:      Optional[str] = ""
    start_date:     str
    end_date:       Optional[str] = None
    window_seconds: Optional[int] = 300
    group_id:       Optional[str] = None
    medications:    List[SlotMedicationItem] = []


class ScheduleUpdate(BaseModel):
    planned_time:   Optional[str] = None
    frequency_type: Optional[str] = None
    week_days:      Optional[str] = None
    start_date:     Optional[str] = None
    end_date:       Optional[str] = None
    window_seconds: Optional[int] = None
    medications:    Optional[List[SlotMedicationItem]] = None


# ── GET ──────────────────────────────────────────────────────────────────────

@router.get("/{patient_id}")
def get_patient_schedules(patient_id: str, aws=Depends(get_aws)):
    """Returns all active schedules for a patient with their medications."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            ms.schedule_id,
            ms.slot_id,
            ms.patient_id,
            ms.planned_time,
            ms.frequency_type,
            ms.week_days,
            ms.is_active,
            ms.start_date,
            ms.end_date,
            ms.window_seconds,
            ms.group_id
        FROM medication_schedules ms
        WHERE ms.patient_id = %s
          AND ms.is_active = TRUE
          AND ms.start_date <= CURRENT_DATE
          AND (ms.end_date IS NULL OR ms.end_date >= CURRENT_DATE)
        ORDER BY ms.planned_time
    """, (patient_id,))
    schedules = cur.fetchall()

    result = []
    for s in schedules:
        cur.execute("""
            SELECT medication_id, medication_name, target_count, loaded_count
            FROM slot_medications
            WHERE slot_id = %s AND patient_id = %s
        """, (s["slot_id"], patient_id))
        meds = cur.fetchall()
        row = dict(s)
        row["planned_time"] = str(s["planned_time"])
        row["start_date"]   = str(s["start_date"]) if s["start_date"] else None
        row["end_date"]     = str(s["end_date"])   if s["end_date"]   else None
        row["medications"]  = [dict(m) for m in meds]
        result.append(row)

    return {"schedules": result}


# ── POST ─────────────────────────────────────────────────────────────────────

@router.post("")
@router.post("/")
def create_schedule(schedule: ScheduleCreate, aws=Depends(get_aws)):
    """Creates a schedule and binds slot to patient."""
    cur = aws.cursor()

    cur.execute("""
        INSERT INTO slot_bindings (slot_id, patient_id, status, updated_at)
        VALUES (%s, %s, 'active', NOW())
        ON CONFLICT (slot_id) DO UPDATE
        SET patient_id = EXCLUDED.patient_id,
            status     = 'active',
            updated_at = NOW()
    """, (schedule.slot_id, schedule.patient_id))

    if schedule.medications:
        cur.execute("DELETE FROM slot_medications WHERE slot_id = %s AND patient_id = %s",
                    (schedule.slot_id, schedule.patient_id))
        for med in schedule.medications:
            cur.execute("""
                INSERT INTO slot_medications
                    (slot_id, patient_id, medication_id, medication_name,
                     target_count, loaded_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                schedule.slot_id,
                schedule.patient_id,
                med.medication_id,
                med.medication_name,
                med.target_count,
                med.loaded_count,
            ))

    medication_id = schedule.medications[0].medication_id if schedule.medications else None

    cur.execute("""
        INSERT INTO medication_schedules
            (medication_id, patient_id, slot_id, planned_time, frequency_type,
             week_days, is_active, start_date, end_date, window_seconds, group_id)
        VALUES (%s, %s, %s, %s::time, %s, %s, TRUE, %s::date, %s::date, %s, %s)
        RETURNING schedule_id, planned_time
    """, (
        medication_id,
        schedule.patient_id,
        schedule.slot_id,
        schedule.planned_time,
        schedule.frequency_type,
        schedule.week_days or "",
        schedule.start_date,
        schedule.end_date,
        schedule.window_seconds or 300,
        schedule.group_id,
    ))
    aws.commit()
    result = cur.fetchone()
    return {
        "message":      "Schedule created",
        "schedule_id":  str(result["schedule_id"]),
        "planned_time": str(result["planned_time"]),
    }


# ── PUT ──────────────────────────────────────────────────────────────────────

@router.put("/group/{group_id}")
def update_schedule_group(group_id: str, schedule: ScheduleUpdate, aws=Depends(get_aws)):
    """Updates all schedules in a group."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE medication_schedules SET
            planned_time   = COALESCE(%s::time, planned_time),
            frequency_type = COALESCE(%s, frequency_type),
            week_days      = COALESCE(%s, week_days),
            start_date     = COALESCE(%s::date, start_date),
            end_date       = COALESCE(%s::date, end_date),
            window_seconds = COALESCE(%s, window_seconds)
        WHERE group_id = %s
        RETURNING schedule_id
    """, (
        schedule.planned_time,
        schedule.frequency_type,
        schedule.week_days,
        schedule.start_date,
        schedule.end_date,
        schedule.window_seconds,
        group_id,
    ))
    aws.commit()
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Group not found")
    return {"message": "Group updated", "group_id": group_id, "updated": len(rows)}


@router.put("/{schedule_id}")
def update_schedule(schedule_id: str, schedule: ScheduleUpdate, aws=Depends(get_aws)):
    """Updates a single schedule."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE medication_schedules SET
            planned_time   = COALESCE(%s::time, planned_time),
            frequency_type = COALESCE(%s, frequency_type),
            week_days      = COALESCE(%s, week_days),
            start_date     = COALESCE(%s::date, start_date),
            end_date       = COALESCE(%s::date, end_date),
            window_seconds = COALESCE(%s, window_seconds)
        WHERE schedule_id = %s
        RETURNING schedule_id
    """, (
        schedule.planned_time,
        schedule.frequency_type,
        schedule.week_days,
        schedule.start_date,
        schedule.end_date,
        schedule.window_seconds,
        schedule_id,
    ))
    aws.commit()
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"message": "Schedule updated", "schedule_id": str(result["schedule_id"])}


# ── PATCH ────────────────────────────────────────────────────────────────────

@router.patch("/group/{group_id}/active")
def toggle_schedule_group(group_id: str, aws=Depends(get_aws)):
    """Toggles is_active for all schedules in a group."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE medication_schedules
        SET is_active = NOT is_active
        WHERE group_id = %s
        RETURNING is_active
    """, (group_id,))
    aws.commit()
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Group not found")
    return {"message": "Group toggled", "is_active": rows[0]["is_active"]}


@router.patch("/{schedule_id}/active")
def toggle_schedule(schedule_id: str, aws=Depends(get_aws)):
    """Toggles is_active for a single schedule."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE medication_schedules
        SET is_active = NOT is_active
        WHERE schedule_id = %s
        RETURNING is_active
    """, (schedule_id,))
    aws.commit()
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"message": "Schedule toggled", "is_active": result["is_active"]}


# ── DELETE ───────────────────────────────────────────────────────────────────

@router.delete("/group/{group_id}")
def delete_schedule_group(group_id: str, aws=Depends(get_aws)):
    """Deletes all schedules in a group."""
    cur = aws.cursor()
    cur.execute("""
        DELETE FROM medication_schedules WHERE group_id = %s
        RETURNING schedule_id
    """, (group_id,))
    aws.commit()
    rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Group not found")
    return {"message": "Group deleted", "deleted": len(rows)}


@router.delete("/{schedule_id}")
def delete_schedule(schedule_id: str, aws=Depends(get_aws)):
    """Deletes a single schedule."""
    cur = aws.cursor()
    cur.execute("""
        DELETE FROM medication_schedules WHERE schedule_id = %s
        RETURNING schedule_id
    """, (schedule_id,))
    aws.commit()
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"message": "Schedule deleted", "schedule_id": schedule_id}
