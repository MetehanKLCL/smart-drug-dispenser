import os
import json
import hashlib
import secrets
import boto3
from fastapi import APIRouter, Depends, HTTPException
from api.database import get_aws
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

CAREGIVER_MODEL_ID = "MEDI-FENS402-2026"


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"pbkdf2:{salt}:{key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        _, salt, key_hex = stored_hash.split(":", 2)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return secrets.compare_digest(key.hex(), key_hex)
    except Exception:
        return False


class CaregiverSignup(BaseModel):
    first_name:   str
    last_name:    str
    email:        str
    password:     str
    model_id:     str
    phone_number: Optional[str] = None


class CaregiverLogin(BaseModel):
    email:    str
    password: str


class PatientLogin(BaseModel):
    email:    str
    password: str


class PatientAccountCreate(BaseModel):
    first_name:    str
    last_name:     str
    email:         str
    password:      str
    caregiver_id:  str
    date_of_birth: Optional[str] = None


class PatientAccountUpdate(BaseModel):
    patient_id: str
    email:      Optional[str] = None
    password:   Optional[str] = None


@router.post("/caregiver/signup")
def caregiver_signup(request: CaregiverSignup, aws=Depends(get_aws)):
    """
    Caregiver signup. Requires valid Model ID (MEDI-FENS402-2026).
    Creates caregiver account in caregivers table.
    """
    if request.model_id != CAREGIVER_MODEL_ID:
        raise HTTPException(status_code=403, detail="Invalid Model ID")

    cur = aws.cursor()
    cur.execute("SELECT caregiver_id FROM caregivers WHERE email = %s", (request.email,))
    if cur.fetchone():
        raise HTTPException(status_code=409, detail="Email already registered")

    password_hash = _hash_password(request.password)

    cur.execute("""
        INSERT INTO caregivers (role_type, first_name, last_name, email, password, phone_number, model_id)
        VALUES ('caregiver', %s, %s, %s, %s, %s, %s)
        RETURNING caregiver_id, email, first_name, last_name
    """, (
        request.first_name,
        request.last_name,
        request.email,
        password_hash,
        request.phone_number,
        request.model_id,
    ))
    aws.commit()
    result = cur.fetchone()

    try:
        sns_client = boto3.client("sns", region_name="eu-north-1")
        sns_client.subscribe(
            TopicArn=os.getenv("SNS_TOPIC_ARN"),
            Protocol="email",
            Endpoint=request.email,
            Attributes={
                "FilterPolicy": json.dumps({"email": [request.email]})
            }
        )
        print(f"[SNS] Subscribed {request.email} to topic")
    except Exception as e:
        print(f"[SNS] Subscription failed (non-critical): {e}")

    return {
        "ok":           True,
        "message":      "Caregiver account created",
        "caregiver_id": str(result["caregiver_id"]),
        "email":        result["email"],
        "first_name":   result["first_name"],
        "last_name":    result["last_name"],
        "role":         "caregiver",
    }


@router.post("/caregiver/login")
def caregiver_login(request: CaregiverLogin, aws=Depends(get_aws)):
    """Caregiver login. Returns caregiver_id and profile info."""
    cur = aws.cursor()
    cur.execute("""
        SELECT caregiver_id, first_name, last_name, email, password
        FROM caregivers WHERE email = %s
    """, (request.email,))
    caregiver = cur.fetchone()

    if not caregiver:
        raise HTTPException(status_code=401, detail="Email not found")
    if not _verify_password(request.password, caregiver["password"]):
        raise HTTPException(status_code=401, detail="Incorrect password")

    return {
        "ok":           True,
        "caregiver_id": str(caregiver["caregiver_id"]),
        "email":        caregiver["email"],
        "first_name":   caregiver["first_name"],
        "last_name":    caregiver["last_name"],
        "role":         "caregiver",
    }


@router.post("/patient/create")
def create_patient_account(request: PatientAccountCreate, aws=Depends(get_aws)):
    """Caregiver creates a patient account from dashboard."""
    cur = aws.cursor()

    cur.execute("SELECT caregiver_id FROM caregivers WHERE caregiver_id = %s", (request.caregiver_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Caregiver not found")

    cur.execute("SELECT patient_id FROM patients WHERE email = %s", (request.email,))
    if cur.fetchone():
        raise HTTPException(status_code=409, detail="Email already registered")

    password_hash = _hash_password(request.password)

    cur.execute("""
        INSERT INTO patients (first_name, last_name, email, password, date_of_birth)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING patient_id, first_name, last_name, email
    """, (
        request.first_name,
        request.last_name,
        request.email,
        password_hash,
        request.date_of_birth,
    ))
    aws.commit()
    result     = cur.fetchone()
    patient_id = str(result["patient_id"])

    cur.execute("""
        INSERT INTO patient_caregivers (patient_id, caregiver_id, relationship_type)
        VALUES (%s, %s, 'caregiver')
    """, (patient_id, request.caregiver_id))
    aws.commit()

    return {
        "ok":         True,
        "message":    "Patient account created and linked to caregiver",
        "patient_id": patient_id,
        "email":      result["email"],
        "first_name": result["first_name"],
        "last_name":  result["last_name"],
        "role":       "patient",
    }


@router.post("/patient/login")
def patient_login(request: PatientLogin, aws=Depends(get_aws)):
    """Patient login."""
    cur = aws.cursor()
    cur.execute("""
        SELECT patient_id, first_name, last_name, email, password
        FROM patients WHERE email = %s AND deleted_at IS NULL
    """, (request.email,))
    patient = cur.fetchone()

    if not patient:
        raise HTTPException(status_code=401, detail="Email not found")
    if not _verify_password(request.password, patient["password"]):
        raise HTTPException(status_code=401, detail="Incorrect password")

    return {
        "ok":         True,
        "patient_id": str(patient["patient_id"]),
        "email":      patient["email"],
        "first_name": patient["first_name"],
        "last_name":  patient["last_name"],
        "role":       "patient",
    }


@router.patch("/patient-account")
def update_patient_account(request: PatientAccountUpdate, aws=Depends(get_aws)):
    """Caregiver updates a patient's email and/or password."""
    cur = aws.cursor()
    cur.execute(
        "SELECT patient_id FROM patients WHERE patient_id = %s AND deleted_at IS NULL",
        (request.patient_id,),
    )
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Patient not found")

    updates = []
    values  = []

    if request.email:
        cur.execute(
            "SELECT patient_id FROM patients WHERE email = %s AND patient_id != %s",
            (request.email, request.patient_id),
        )
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="Email already in use")
        updates.append("email = %s")
        values.append(request.email)

    if request.password:
        updates.append("password = %s")
        values.append(_hash_password(request.password))

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update")

    values.append(request.patient_id)
    cur.execute(
        f"UPDATE patients SET {', '.join(updates)} WHERE patient_id = %s",
        values,
    )
    aws.commit()
    return {"ok": True, "message": "Account updated"}


@router.post("/register-role")
def register_role():
    """Legacy endpoint — kept for backward compatibility."""
    return {"message": "Use /auth/caregiver/signup instead"}