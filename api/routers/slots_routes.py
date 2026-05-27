from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from api.database import get_aws

router = APIRouter()


def _normalize_barcode(value: Optional[str]) -> str:
    raw = (value or "").strip()
    compact = "".join(raw.split())
    if compact.startswith("]") and len(compact) >= 3:
        compact = compact[3:]
    return compact


class SlotMedicationItem(BaseModel):
    medication_id:   str
    medication_name: str
    barcode:         Optional[str] = None
    target_count:    int = 0
    loaded_count:    int = 0


class SetSlotMedicationsRequest(BaseModel):
    patient_id:  Optional[str] = None
    medications: List[SlotMedicationItem] = []


class BindSlotRequest(BaseModel):
    patient_id: str
    status:     str = "active"


class BarcodeIncrementRequest(BaseModel):
    barcode: str
    slot_id: int



@router.get("")
@router.get("/")
def get_all_slots(aws=Depends(get_aws)):
    """Returns all slots with their bound patient and loaded medications."""
    cur = aws.cursor()
    cur.execute("""
        SELECT sb.slot_id, sb.patient_id, sb.status, sb.updated_at,
               p.first_name, p.last_name
        FROM slot_bindings sb
        LEFT JOIN patients p ON p.patient_id::text = sb.patient_id
        ORDER BY sb.slot_id
    """)
    rows = cur.fetchall()

    slots = []
    for r in rows:
        cur.execute("""
            SELECT id, medication_id, medication_name, barcode,
                   target_count, loaded_count
            FROM slot_medications
            WHERE slot_id = %s
            ORDER BY id
        """, (r["slot_id"],))
        meds = cur.fetchall()
        slots.append({
            "slot_id":      r["slot_id"],
            "patient_id":   r["patient_id"],
            "patient_name": f"{r['first_name'] or ''} {r['last_name'] or ''}".strip(),
            "status":       r["status"],
            "updated_at":   str(r["updated_at"]),
            "medications":  [dict(m) for m in meds],
        })

    return {"slots": slots}


@router.get("/available")
def get_available_slots(aws=Depends(get_aws)):
    """Returns slot IDs that are empty and available for binding."""
    cur = aws.cursor()
    cur.execute("""
        SELECT slot_id FROM slot_bindings
        WHERE patient_id IS NULL OR status = 'empty'
        ORDER BY slot_id
    """)
    rows = cur.fetchall()
    return {"available_slots": [r["slot_id"] for r in rows]}


@router.get("/{slot_id}/medications")
def get_slot_medications(slot_id: int, aws=Depends(get_aws)):
    """Returns the list of medications loaded into a specific slot."""
    cur = aws.cursor()
    cur.execute("""
        SELECT id, medication_id, medication_name, barcode,
               target_count, loaded_count
        FROM slot_medications
        WHERE slot_id = %s
        ORDER BY id
    """, (slot_id,))
    rows = cur.fetchall()
    return {"medications": [dict(r) for r in rows]}


@router.post("/{slot_id}/barcode")
def barcode_scanned(slot_id: int, body: BarcodeIncrementRequest, aws=Depends(get_aws)):
    """Called when Pi scans a barcode. Increments loaded_count for matching medication."""
    cur = aws.cursor()

    scanned = _normalize_barcode(body.barcode)

    # Fast path: exact normalized barcode match
    cur.execute("""
        UPDATE slot_medications sm
        SET loaded_count = loaded_count + 1
        WHERE sm.slot_id = %s AND (sm.barcode = %s OR sm.barcode IS NULL)
        RETURNING loaded_count, target_count
    """, (slot_id, scanned))
    row = cur.fetchone()

    # Backward compatibility for old rows that may contain spaces/newlines/prefixes.
    if not row:
        cur.execute("""
            SELECT id, barcode
            FROM slot_medications
            WHERE slot_id = %s
            ORDER BY id
        """, (slot_id,))
        candidates = cur.fetchall()
        match_id = None
        for candidate in candidates:
            if _normalize_barcode(candidate["barcode"]) == scanned and scanned:
                match_id = candidate["id"]
                break
        if match_id is not None:
            cur.execute("""
                UPDATE slot_medications
                SET barcode = %s, loaded_count = loaded_count + 1
                WHERE id = %s
                RETURNING loaded_count, target_count
            """, (scanned, match_id))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Medication not found in slot")

    loaded  = row["loaded_count"]
    target  = row["target_count"]

    # Hedef doluysa slot'u loaded yap
    if loaded >= target:
        cur.execute("""
            UPDATE slot_bindings SET status = 'loaded', updated_at = NOW()
            WHERE slot_id = %s
        """, (slot_id,))

    aws.commit()
    return {
        "slot_id":      slot_id,
        "loaded_count": loaded,
        "target_count": target,
        "slot_status":  "loaded" if loaded >= target else "loading",
    }


@router.post("/{slot_id}/medications")
def set_slot_medications(
    slot_id: int,
    body: SetSlotMedicationsRequest,
    aws=Depends(get_aws)
):
    """Replaces all medications for a slot. Deletes existing entries then inserts new ones."""
    cur = aws.cursor()

    cur.execute("DELETE FROM slot_medications WHERE slot_id = %s", (slot_id,))

    for med in body.medications:
        normalized_barcode = _normalize_barcode(med.barcode)
        cur.execute("""
            INSERT INTO slot_medications
                (slot_id, patient_id, medication_id, medication_name,
                 barcode, target_count, loaded_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            slot_id,
            body.patient_id,
            med.medication_id,
            med.medication_name,
            normalized_barcode or None,
            med.target_count,
            med.loaded_count,
        ))

    aws.commit()
    return {"message": "Slot medications updated successfully", "slot_id": slot_id}


@router.post("/{slot_id}/bind")
def bind_slot(slot_id: int, body: BindSlotRequest, aws=Depends(get_aws)):
    """Binds a patient to a physical slot. Updates existing binding if slot already occupied."""
    cur = aws.cursor()
    cur.execute("""
        INSERT INTO slot_bindings (slot_id, patient_id, status, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (slot_id) DO UPDATE
        SET patient_id = EXCLUDED.patient_id,
            status     = EXCLUDED.status,
            updated_at = NOW()
    """, (slot_id, body.patient_id, body.status))
    aws.commit()
    return {"message": "Slot bound successfully", "slot_id": slot_id, "patient_id": body.patient_id}


@router.delete("/{slot_id}")
def delete_slot(slot_id: int, aws=Depends(get_aws)):
    """Clears a slot by removing its medications and resetting the patient binding."""
    cur = aws.cursor()

    cur.execute("DELETE FROM slot_medications WHERE slot_id = %s", (slot_id,))

    cur.execute("""
        UPDATE slot_bindings
        SET patient_id = NULL, status = 'empty', updated_at = NOW()
        WHERE slot_id = %s
    """, (slot_id,))

    aws.commit()
    return {"message": "Slot cleared successfully", "slot_id": slot_id}