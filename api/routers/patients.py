from fastapi import APIRouter, Depends, HTTPException
from api.database import get_aws
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class PatientCreate(BaseModel):
    first_name:           str
    last_name:            str
    date_of_birth:        Optional[str] = None
    timezone:             Optional[str] = "Europe/Istanbul"
    device_serial_number: Optional[str] = None


class PatientUpdate(BaseModel):
    first_name:           Optional[str] = None
    last_name:            Optional[str] = None
    date_of_birth:        Optional[str] = None
    timezone:             Optional[str] = None
    device_serial_number: Optional[str] = None


@router.get("")
@router.get("/")
def get_all_patients(aws=Depends(get_aws)):
    """Gets all patients information from AWS database."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            patient_id,
            first_name,
            last_name,
            date_of_birth,
            device_serial_number,
            battery_level,
            is_online,
            last_seen_at,
            COALESCE(face_registered, FALSE) AS face_registered
        FROM patients
        WHERE deleted_at IS NULL
        ORDER BY first_name
    """)
    patients = cur.fetchall()
    return {"patients": [dict(p) for p in patients]}


@router.get("/search")
def search_patient(name: str, aws=Depends(get_aws)):
    """Searches for a patient by name from AWS database."""
    cur = aws.cursor()
    cur.execute("""
        SELECT 
            patient_id,
            first_name,
            last_name,
            date_of_birth
        FROM patients
        WHERE (LOWER(first_name) = LOWER(%s)
           OR LOWER(last_name)  = LOWER(%s))
          AND deleted_at IS NULL
        ORDER BY first_name
    """, (name, name))
    results = cur.fetchall()

    if not results:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {"patients": [dict(p) for p in results]}


@router.get("/caregivers")
def get_all_caregivers(aws=Depends(get_aws)):
    """Gets all caregivers from caregivers table."""
    cur = aws.cursor()
    cur.execute("""
        SELECT role_id, first_name, last_name, email, role_type
        FROM caregivers
        WHERE role_type = 'caregiver'
        ORDER BY first_name
    """)
    caregivers = cur.fetchall()
    return {"caregivers": [dict(c) for c in caregivers]}


@router.get("/my-patients/{caregiver_id}")
def get_caregiver_patients(caregiver_id: str, aws=Depends(get_aws)):
    """Gets only patients linked to this caregiver."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            p.patient_id,
            p.first_name,
            p.last_name,
            p.date_of_birth,
            p.device_serial_number,
            p.battery_level,
            p.is_online,
            p.last_seen_at,
            COALESCE(p.face_registered, FALSE) AS face_registered
        FROM patients p
        JOIN patient_caregivers pc ON pc.patient_id = p.patient_id
        WHERE pc.caregiver_id = %s
          AND p.deleted_at IS NULL
        ORDER BY p.first_name
    """, (caregiver_id,))
    patients = cur.fetchall()
    return {"patients": [dict(p) for p in patients]}


@router.get("/{patient_id}")
def get_patient(patient_id: str, aws=Depends(get_aws)):
    """Gets a single patient's information from AWS database."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            patient_id,
            first_name,
            last_name,
            date_of_birth,
            device_serial_number,
            battery_level,
            is_online,
            last_seen_at,
            COALESCE(face_registered, FALSE) AS face_registered
        FROM patients
        WHERE patient_id = %s
          AND deleted_at IS NULL
    """, (patient_id,))
    patient = cur.fetchone()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return dict(patient)


@router.post("")
@router.post("/")
def create_patient(patient: PatientCreate, aws=Depends(get_aws)):
    """Creates a new patient in the AWS database."""
    cur = aws.cursor()
    cur.execute("""
        INSERT INTO patients (first_name, last_name, date_of_birth, timezone, device_serial_number)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING patient_id, first_name, last_name
    """, (
        patient.first_name,
        patient.last_name,
        patient.date_of_birth,
        patient.timezone,
        patient.device_serial_number,
    ))
    aws.commit()
    result = cur.fetchone()
    return {
        "message":    "Patient created",
        "patient_id": str(result["patient_id"]),
        "first_name": result["first_name"],
        "last_name":  result["last_name"],
    }


@router.put("/{patient_id}")
def update_patient(patient_id: str, patient: PatientUpdate, aws=Depends(get_aws)):
    """Updates the existing patient's information."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE patients SET
            first_name           = COALESCE(%s, first_name),
            last_name            = COALESCE(%s, last_name),
            date_of_birth        = COALESCE(%s::date, date_of_birth),
            timezone             = COALESCE(%s, timezone),
            device_serial_number = COALESCE(%s, device_serial_number)
        WHERE patient_id = %s
        RETURNING patient_id, first_name, last_name
    """, (
        patient.first_name,
        patient.last_name,
        patient.date_of_birth,
        patient.timezone,
        patient.device_serial_number,
        patient_id,
    ))
    aws.commit()
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": "Patient updated", "patient_id": str(result["patient_id"])}


@router.delete("/{patient_id}")
def delete_patient(patient_id: str, aws=Depends(get_aws)):
    """Deletes the patient and related medications and dispensing_logs (CASCADE)."""
    cur = aws.cursor()
    cur.execute("DELETE FROM patients WHERE patient_id = %s RETURNING patient_id", (patient_id,))
    aws.commit()
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": "Patient deleted", "patient_id": patient_id}