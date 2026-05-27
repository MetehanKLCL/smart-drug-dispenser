import boto3
import json
import os
import urllib.request
from fastapi import APIRouter, Depends
from api.database import get_aws
from pydantic import BaseModel

router = APIRouter()

sns = boto3.client("sns", region_name=os.getenv("APP_REGION", "eu-north-1"))
s3  = boto3.client("s3",  region_name=os.getenv("APP_REGION", "eu-north-1"))

SNS_TOPIC_ARN      = os.getenv("SNS_TOPIC_ARN")
FIREBASE_SA_BUCKET = "drug-dispenser-analytics-datalake"
FIREBASE_SA_KEY    = "config/firebase-service-account.json"


def _ensure_notification_log_table(aws):
    """Create reminder dedupe table if it doesn't exist."""
    cur = aws.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notification_log (
            id            BIGSERIAL PRIMARY KEY,
            schedule_id   TEXT NOT NULL,
            sent_date     DATE NOT NULL,
            patient_id    TEXT NOT NULL,
            reminder_type TEXT NOT NULL,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (schedule_id, sent_date, patient_id, reminder_type)
        )
        """
    )
    aws.commit()


def _get_firebase_access_token() -> str:
    """Gets OAuth2 access token from Firebase service account stored in S3."""
    import google.oauth2.service_account
    import google.auth.transport.requests
    obj = s3.get_object(Bucket=FIREBASE_SA_BUCKET, Key=FIREBASE_SA_KEY)
    sa_info = json.loads(obj["Body"].read().decode("utf-8"))
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/firebase.messaging"],
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def _send_fcm_push(fcm_token: str, title: str, body: str):
    """Sends push notification to a single device via FCM v1 API."""
    project_id   = "drug-dispenser-7f379"
    access_token = _get_firebase_access_token()
    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    payload = json.dumps({
        "message": {
            "token": fcm_token,
            "notification": {"title": title, "body": body},
            "android": {"priority": "HIGH"},
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def notify_caregivers_of_dispensing(aws, patient_id: str, patient_name: str, status: str, planned_time: str = None):
    """
    Sends a push notification to all caregivers linked to a patient
    when a dispensing event (taken, missed, error) occurs.
    Called from dispensing.py after a log is created.
    'dispensed' status is skipped — patient hasn't taken the pill yet.
    """
    if status == "dispensed":
        return

    cur = aws.cursor()
    cur.execute("""
        SELECT c.fcm_token
        FROM caregivers c
        JOIN patient_caregivers pc ON pc.caregiver_id = c.caregiver_id
        WHERE pc.patient_id::text = %s AND c.fcm_token IS NOT NULL
    """, (patient_id,))
    caregivers = cur.fetchall()

    if not caregivers:
        return

    time_str = f" ({planned_time})" if planned_time else ""

    if status == "taken":
        title = "✅ İlaç Alındı"
        body  = f"{patient_name}{time_str} saatindeki ilacını aldı."
    elif status == "missed":
        title = "⚠️ İlaç Kaçırıldı"
        body  = f"{patient_name}{time_str} saatindeki ilacını kaçırdı!"
    elif status == "error":
        title = "❌ Cihaz Hatası"
        body  = f"{patient_name} için cihaz hatası oluştu."
    else:
        return

    for cg in caregivers:
        try:
            _send_fcm_push(cg["fcm_token"], title, body)
            print(f"[NOTIFY] {status} → caregiver notified for {patient_name}")
        except Exception as e:
            print(f"[NOTIFY] Caregiver push failed: {e}")


class FcmTokenRequest(BaseModel):
    fcm_token: str
    role_id:   str | None = None
    email:     str | None = None
    role:      str | None = None


@router.post("/register-token")
def register_fcm_token(request: FcmTokenRequest, aws=Depends(get_aws)):
    """Registers FCM token for caregiver or patient."""
    cur = aws.cursor()
    if request.role == "patient" and request.email:
        cur.execute("UPDATE patients SET fcm_token = %s WHERE email = %s", (request.fcm_token, request.email))
    elif request.email:
        cur.execute("UPDATE caregivers SET fcm_token = %s WHERE email = %s", (request.fcm_token, request.email))
    else:
        return {"message": "FCM token registered (no user linked yet)"}
    aws.commit()
    return {"message": "FCM token registered successfully"}


@router.post("/send-push")
def send_push_to_all(aws=Depends(get_aws)):
    """Sends push notification to all caregivers with registered FCM tokens."""
    cur = aws.cursor()
    cur.execute("SELECT fcm_token FROM caregivers WHERE fcm_token IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        return {"message": "No registered devices found", "sent": 0}
    sent = failed = 0
    for row in rows:
        try:
            _send_fcm_push(fcm_token=row["fcm_token"], title="💊 Drug Dispenser Alert", body="A patient has been flagged as high risk. Please check the app.")
            sent += 1
        except Exception as e:
            print(f"[FCM] Push failed: {e}")
            failed += 1
    return {"message": f"Push sent to {sent} devices", "sent": sent, "failed": failed}


@router.post("/test-push")
def test_push_notification(aws=Depends(get_aws)):
    """Sends a test push notification to all registered devices (caregivers + patients)."""
    cur = aws.cursor()
    tokens = []
    cur.execute("SELECT fcm_token FROM caregivers WHERE fcm_token IS NOT NULL")
    tokens += [r["fcm_token"] for r in cur.fetchall()]
    cur.execute("SELECT fcm_token FROM patients WHERE fcm_token IS NOT NULL")
    tokens += [r["fcm_token"] for r in cur.fetchall()]
    if not tokens:
        return {"message": "No registered devices.", "sent": 0}
    sent = 0
    for token in tokens:
        try:
            _send_fcm_push(fcm_token=token, title="🧪 Test Notification", body="Drug Dispenser push notification is working correctly!")
            sent += 1
        except Exception as e:
            print(f"[FCM] Test push failed: {e}")
    return {"message": f"Test push sent to {sent} devices", "sent": sent}


@router.post("/send-reminders")
def send_medication_reminders(aws=Depends(get_aws)):
    """
    Checks medication_schedules and sends push reminders to patients.
    Sends two notifications: 5 minutes before and at exact scheduled time.
    Called automatically by EventBridge every minute.
    """
    from datetime import datetime, timezone, timedelta

    now   = datetime.now(timezone.utc) + timedelta(hours=3)
    today = now.date()
    sent = skipped = 0

    print(f"[REMINDER] now={now.strftime('%H:%M')}, today={today}")
    try:
        _ensure_notification_log_table(aws)
    except Exception as e:
        print(f"[REMINDER] notification_log ensure failed: {e}")
        aws.rollback()

    for reminder_type, minute_offset, title, body_template in [
        ("early", 5, "⏰ Yakında İlaç Zamanı!", "5 dakika sonra saat {time} — ilacınızı almayı unutmayın!"),
        ("exact", 0, "💊 İlaç Zamanı!",         "Saat {time} — ilacınızı almanın tam zamanı!"),
    ]:
        try:
            cur = aws.cursor()
            check_time   = now + timedelta(minutes=minute_offset)
            window_start = (check_time - timedelta(minutes=2)).strftime("%H:%M")
            window_end   = check_time.strftime("%H:%M")

            print(f"[REMINDER] {reminder_type} window: {window_start} - {window_end}")

            cur.execute("""
                SELECT ms.schedule_id, ms.planned_time,
                       p.patient_id, p.first_name, p.last_name, p.fcm_token
                FROM medication_schedules ms
                JOIN patients p ON p.patient_id::text = ms.patient_id::text
                WHERE ms.is_active = TRUE
                  AND ms.start_date <= %s
                  AND (ms.end_date IS NULL OR ms.end_date >= %s)
                  AND to_char(ms.planned_time, 'HH24:MI') BETWEEN %s AND %s
                  AND p.fcm_token IS NOT NULL
                  AND p.deleted_at IS NULL
            """, (today, today, window_start, window_end))

            schedules = cur.fetchall()
            print(f"[REMINDER] {reminder_type} found {len(schedules)} schedules")

            for s in schedules:
                try:
                    cur.execute("""
                        INSERT INTO notification_log (schedule_id, sent_date, patient_id, reminder_type)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (schedule_id, sent_date, patient_id, reminder_type)
                        DO NOTHING
                    """, (str(s["schedule_id"]), today, str(s["patient_id"]), reminder_type))
                    aws.commit()
                except Exception as ie:
                    # Reminder delivery must not depend on notification_log insert.
                    print(f"[REMINDER] log insert warning ({reminder_type}): {ie}")
                    aws.rollback()

                try:
                    time_str = str(s["planned_time"])[:5]
                    _send_fcm_push(fcm_token=s["fcm_token"], title=title, body=body_template.format(time=time_str))
                    sent += 1
                    print(f"[REMINDER] {reminder_type} → {s['first_name']} {s['last_name']} @ {time_str}")
                except Exception as e:
                    print(f"[REMINDER] Push failed: {e}")

        except Exception as e:
            print(f"[REMINDER] Loop error ({reminder_type}): {e}")
            aws.rollback()

    return {"sent": sent, "skipped": skipped}