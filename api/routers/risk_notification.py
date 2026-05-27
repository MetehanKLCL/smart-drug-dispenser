import boto3
import json
import os
import math
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends
from api.database import get_aws

router = APIRouter()

s3  = boto3.client("s3", region_name=os.getenv("APP_REGION", "eu-north-1"))
sns = boto3.client("sns", region_name=os.getenv("APP_REGION", "eu-north-1"))

RISK_THRESHOLD = 70
SNS_TOPIC_ARN  = os.getenv("SNS_TOPIC_ARN")
MODEL_BUCKET   = "drug-dispenser-frankfurt"
MODEL_KEY      = "sklearn/model.json"
FEATURE_COLS   = [
    "total_doses", "missed_count", "dispensed_count", "taken_count",
    "error_count", "avg_face_score", "min_face_score", "miss_rate",
    "miss_rate_7d", "avg_face_7d", "error_count_7d"
]


def _load_model():
    """Downloads logistic regression model from S3."""
    s3_frankfurt = boto3.client("s3", region_name="eu-central-1")
    obj = s3_frankfurt.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _predict(model: dict, features: list) -> float:
    w    = model["w"]
    b    = model["b"]
    mean = model["mean"]
    std  = model["std"]
    x_norm = [(features[i] - mean[i]) / std[i] for i in range(len(features))]
    z = sum(x_norm[i] * w[i] for i in range(len(w))) + b
    return 1 / (1 + math.exp(-z))


def _get_patient_features(aws) -> list[dict]:
    """Calculates daily features for each patient from dispensing_logs."""
    cur    = aws.cursor()
    today  = datetime.now(timezone.utc).date()
    day_30 = today - timedelta(days=30)
    day_7  = today - timedelta(days=7)

    cur.execute("""
        SELECT
            p.patient_id,
            p.first_name,
            p.last_name,
            COUNT(dl.log_id)                                          AS total_doses,
            SUM(CASE WHEN dl.status='missed'     THEN 1 ELSE 0 END)  AS missed_count,
            SUM(CASE WHEN dl.status='dispensed'  THEN 1 ELSE 0 END)  AS dispensed_count,
            SUM(CASE WHEN dl.status='taken'      THEN 1 ELSE 0 END)  AS taken_count,
            SUM(CASE WHEN dl.status='error'      THEN 1 ELSE 0 END)  AS error_count,
            AVG(dl.face_auth_score)                                   AS avg_face_score,
            MIN(dl.face_auth_score)                                   AS min_face_score,
            CASE WHEN COUNT(dl.log_id) > 0
                 THEN SUM(CASE WHEN dl.status='missed' THEN 1 ELSE 0 END)::float / COUNT(dl.log_id)
                 ELSE 0 END                                           AS miss_rate,
            SUM(CASE WHEN dl.dispensing_at >= %s AND dl.status='missed'  THEN 1 ELSE 0 END)::float /
            NULLIF(SUM(CASE WHEN dl.dispensing_at >= %s THEN 1 ELSE 0 END), 0) AS miss_rate_7d,
            AVG(CASE WHEN dl.dispensing_at >= %s THEN dl.face_auth_score END)  AS avg_face_7d,
            SUM(CASE WHEN dl.dispensing_at >= %s AND dl.status='error' THEN 1 ELSE 0 END) AS error_count_7d
        FROM patients p
        LEFT JOIN dispensing_logs dl
            ON dl.patient_id = p.patient_id
            AND dl.dispensing_at >= %s
        WHERE p.deleted_at IS NULL
        GROUP BY p.patient_id, p.first_name, p.last_name
    """, (day_7, day_7, day_7, day_7, day_30))

    return [dict(r) for r in cur.fetchall()]


def _send_sns_to_caregiver(caregiver_email: str, subject: str, message: str):
    """Sends SNS email only to specific caregiver via filter policy."""
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject,
        Message=message,
        MessageAttributes={
            "email": {
                "DataType": "String",
                "StringValue": caregiver_email,
            }
        }
    )


def _send_fcm_push(fcm_token: str, title: str, body: str):
    """Sends push notification via FCM v1 API."""
    import urllib.request
    s3_frankfurt = boto3.client("s3", region_name="eu-central-1")
    obj = s3_frankfurt.get_object(
        Bucket="drug-dispenser-analytics-datalake",
        Key="config/firebase-service-account.json"
    )
    sa_info = json.loads(obj["Body"].read().decode("utf-8"))

    import google.oauth2.service_account
    import google.auth.transport.requests
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    credentials.refresh(google.auth.transport.requests.Request())

    url     = "https://fcm.googleapis.com/v1/projects/drug-dispenser-7f379/messages:send"
    payload = json.dumps({
        "message": {
            "token": fcm_token,
            "notification": {"title": title, "body": body},
            "android": {"priority": "HIGH"},
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


@router.get("/risk-scores")
def get_risk_scores(aws=Depends(get_aws)):
    """Returns latest risk scores for all patients."""
    try:
        model    = _load_model()
        patients = _get_patient_features(aws)

        if not patients:
            return {"message": "No patients found.", "scores": []}

        results = []
        for p in patients:
            features   = [p.get(col) or 0 for col in FEATURE_COLS]
            prob       = _predict(model, features)
            risk_score = round(prob * 100, 1)
            results.append({
                "patient_id": str(p["patient_id"]),
                "first_name": p["first_name"],
                "last_name":  p["last_name"],
                "risk_score": risk_score,
                "risk_level": "HIGH" if risk_score >= 85 else "MEDIUM" if risk_score >= RISK_THRESHOLD else "LOW",
            })

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return {
            "total_patients":  len(results),
            "high_risk_count": sum(1 for r in results if r["risk_score"] >= RISK_THRESHOLD),
            "scores":          results,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/send-notifications")
def send_risk_notifications(aws=Depends(get_aws)):
    """
    Calculates risk scores and sends notifications to high-risk patients' caregivers.
    Called by EventBridge every night.
    """
    try:
        model    = _load_model()
        patients = _get_patient_features(aws)

        if not patients:
            return {"message": "No patients found.", "notified": 0}

        cur       = aws.cursor()
        notified  = 0
        high_risk = []

        for p in patients:
            features   = [p.get(col) or 0 for col in FEATURE_COLS]
            prob       = _predict(model, features)
            risk_score = round(prob * 100, 1)

            if risk_score >= RISK_THRESHOLD:
                patient_name = f"{p['first_name']} {p['last_name']}"
                risk_level   = "HIGH" if risk_score >= 85 else "MEDIUM"

                high_risk.append({"patient": patient_name, "risk_score": risk_score, "risk_level": risk_level})

                # Get caregivers for this patient
                cur.execute("""
                    SELECT c.fcm_token, c.email
                    FROM caregivers c
                    JOIN patient_caregivers pc ON pc.caregiver_id = c.caregiver_id
                    WHERE pc.patient_id = %s
                """, (str(p["patient_id"]),))
                caregivers = cur.fetchall()

                for cg in caregivers:
                    # Send email via SNS filter to this specific caregiver
                    try:
                        _send_sns_to_caregiver(
                            caregiver_email=cg["email"],
                            subject=f"⚠️ High Medication Risk Alert — {patient_name}",
                            message=(
                                f"ALERT: {patient_name} has been flagged as high risk.\n"
                                f"Risk Score: {risk_score}/100\n"
                                f"Risk Level: {risk_level}\n"
                                f"Please check the Drug Dispenser app immediately."
                            )
                        )
                    except Exception as e:
                        print(f"[SNS] Email failed: {e}")

                    # Send FCM push
                    if cg["fcm_token"]:
                        try:
                            _send_fcm_push(
                                fcm_token=cg["fcm_token"],
                                title=f"⚠️ {patient_name} — High Risk",
                                body=f"Risk Score: {risk_score}/100. Please check the app.",
                            )
                        except Exception as e:
                            print(f"[FCM] Push failed: {e}")

                # Send notification to patient's own FCM token
                cur.execute(
                    "SELECT fcm_token FROM patients WHERE patient_id = %s AND fcm_token IS NOT NULL",
                    (str(p["patient_id"]),)
                )
                patient_row = cur.fetchone()
                if patient_row:
                    try:
                        _send_fcm_push(
                            fcm_token=patient_row["fcm_token"],
                            title="💊 İlaç Uyarısı",
                            body=f"Risk skorunuz yüksek: {risk_score}/100. Lütfen ilacınızı almayı unutmayın.",
                        )
                    except Exception as e:
                        print(f"[FCM] Patient push failed: {e}")

                notified += 1
                print(f"[RISK] Notified for {patient_name}: {risk_score}")

        return {
            "message":   f"{notified} high-risk patients notified.",
            "notified":  notified,
            "high_risk": high_risk,
            "threshold": RISK_THRESHOLD,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/test-notification")
def test_notification():
    """Sends a test SNS notification."""
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="🧪 Drug Dispenser - Test Notification",
        Message=(
            "This is a test notification from the Drug Dispenser Early Warning System.\n"
            "If you received this, SNS is working correctly!\n\n"
            "System: Erken Uyarı Sistemi\n"
            "Status: Active"
        )
    )
    return {"message": "Test notification sent. Check your email/SMS."}