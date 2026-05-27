import boto3
import uuid
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from api.database import get_aws

router = APIRouter()


class MedicationCreate(BaseModel):
    """
    Medication creation data.
    Will be used to create a new medication in the AWS database.
    """
    patient_id:         str
    medication_name:    str
    pill_image_url:     Optional[str] = None
    pill_barcode:       Optional[str] = None
    pill_color_shape:   Optional[str] = None
    remaining_count:    Optional[int] = 0
    low_stock_threshold: Optional[int] = 5
    expiry_date:        Optional[str] = None


class MedicationUpdate(BaseModel):
    """
    Medication update data.
    Will be used to update an existing medication in the AWS database.
    """
    medication_name:     Optional[str] = None
    remaining_count:     Optional[int] = None
    low_stock_threshold: Optional[int] = None
    expiry_date:         Optional[str] = None
    pill_color_shape:    Optional[str] = None
    pill_barcode:        Optional[str] = None


@router.get("")
@router.get("/")
def get_all_medications(aws=Depends(get_aws)):
    """Lists all medications across all patients."""
    cur = aws.cursor()
    cur.execute("""
        SELECT
            medication_id,
            patient_id,
            medication_name,
            pill_image_url,
            pill_barcode,
            pill_color_shape,
            remaining_count,
            low_stock_threshold,
            expiry_date
        FROM medications
        ORDER BY medication_name
    """)
    medications = cur.fetchall()

    if not medications:
        raise HTTPException(status_code=404, detail="No medications found")

    return {"medications": [dict(m) for m in medications]}


@router.get("/{patient_id}")
def get_patient_medications(patient_id: str, aws=Depends(get_aws)):
    """Lists all medications for an existing patient in the AWS database."""
    cur = aws.cursor()
    cur.execute("""
        SELECT medication_id, medication_name, remaining_count,
               low_stock_threshold, expiry_date
        FROM medications
        WHERE patient_id = %s
        ORDER BY medication_name
    """, (patient_id,))
    medications = cur.fetchall()

    if not medications:
        raise HTTPException(status_code=404, detail="No medications found for this patient")

    return {"medications": [dict(m) for m in medications]}


@router.post("")
@router.post("/")
def create_medication(medication: MedicationCreate, aws=Depends(get_aws)):
    """Creates a new medication for an existing patient in the AWS database."""
    cur = aws.cursor()
    cur.execute("""
        INSERT INTO medications 
            (patient_id, medication_name, pill_image_url, pill_barcode,
             pill_color_shape, remaining_count, low_stock_threshold, expiry_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING medication_id, medication_name
    """, (
        medication.patient_id,
        medication.medication_name,
        medication.pill_image_url,
        medication.pill_barcode,
        medication.pill_color_shape,
        medication.remaining_count,
        medication.low_stock_threshold,
        medication.expiry_date,
    ))
    aws.commit()
    result = cur.fetchone()

    return {
        "message":       "Medication created",
        "medication_id": str(result["medication_id"]),
        "medication_name": result["medication_name"],
    }


@router.post("/upload-image/{medication_id}")
def upload_medication_image(
    medication_id: str,
    file: UploadFile = File(...),
    aws=Depends(get_aws)
):
    """Uploads a medication image to S3 and saves the URL to the medications table."""

    # 1. Gets the file extension (.jpg, .png, etc.)
    extension = file.filename.split(".")[-1]

    # 2. Creates a unique file name
    s3_key = f"medications/{medication_id}/{uuid.uuid4()}.{extension}"

    # 3. Uploads the file to S3 bucket
    s3_client = boto3.client(
        "s3",
        region_name=os.getenv("APP_REGION", "eu-north-1")
    )

    try:
        s3_client.upload_fileobj(
            file.file,
            os.getenv("S3_BUCKET_NAME"),
            s3_key,
            ExtraArgs={"ContentType": file.content_type}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {e}")

    # 4. Creates the S3 URL
    image_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('APP_REGION')}.amazonaws.com/{s3_key}"

    # 5. Saves the URL to the medications table
    cur = aws.cursor()
    cur.execute("""
        UPDATE medications
        SET pill_image_url = %s
        WHERE medication_id = %s
        RETURNING medication_id, medication_name, pill_image_url
    """, (image_url, medication_id))
    aws.commit()
    result = cur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Medication not found")

    return {
        "message": "Image uploaded",
        "medication_id": str(result["medication_id"]),
        "medication_name": result["medication_name"],
        "image_url": result["pill_image_url"]
    }


@router.put("/{medication_id}")
def update_medication(medication_id: str, medication: MedicationUpdate, aws=Depends(get_aws)):
    """Updates the existing medication's information."""
    cur = aws.cursor()
    cur.execute("""
        UPDATE medications SET
            medication_name     = COALESCE(%s, medication_name),
            remaining_count     = COALESCE(%s, remaining_count),
            low_stock_threshold = COALESCE(%s, low_stock_threshold),
            expiry_date         = COALESCE(%s::date, expiry_date),
            pill_color_shape    = COALESCE(%s, pill_color_shape),
            pill_barcode        = COALESCE(%s, pill_barcode)
        WHERE medication_id = %s
        RETURNING medication_id, medication_name
    """, (
        medication.medication_name,
        medication.remaining_count,
        medication.low_stock_threshold,
        medication.expiry_date,
        medication.pill_color_shape,
        medication.pill_barcode,
        medication_id,
    ))
    aws.commit()
    result = cur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Medication not found")
    return {"message": "Medication updated", "medication_id": str(result["medication_id"])}    


@router.delete("/{medication_id}")
def delete_medication(medication_id: str, aws=Depends(get_aws)):
    """Deletes the medication and related schedules (CASCADE)."""
    cur = aws.cursor()
    cur.execute("DELETE FROM medications WHERE medication_id = %s RETURNING medication_id", (medication_id,))
    aws.commit()
    result = cur.fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="Medication not found")
    return {"message": "Medication deleted", "medication_id": medication_id}