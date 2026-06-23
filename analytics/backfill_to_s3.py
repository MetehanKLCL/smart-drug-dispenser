# analytics/backfill_to_s3.py
import boto3
import json
import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv

# Load the .env file from the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# RDS connection
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode=os.getenv("DB_SSLMODE", "require"),
)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# S3 client setup
s3 = boto3.client(
    "s3",
    region_name=os.getenv("APP_REGION", "eu-north-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

BUCKET = "drug-dispenser-analytics-datalake"

# Fetch all dispensing logs
cur.execute("""
    SELECT
        log_id::text,
        patient_id::text,
        schedule_id::text,
        status,
        face_auth_score,
        dispensing_at,
        taken_at,
        device_timestamp,
        error_details
    FROM dispensing_logs
    ORDER BY dispensing_at
""")

logs = cur.fetchall()
print(f"{len(logs)} records found, writing to S3...")

saved = 0
for log in logs:
    data = dict(log)
    
    for key in ["dispensing_at", "taken_at", "device_timestamp"]:
        if data[key] is not None:
            data[key] = str(data[key])

    try:
        dt = datetime.fromisoformat(data["dispensing_at"])
    except:
        dt = datetime.utcnow()

    s3_key = (
        f"raw/"
        f"year={dt.year}/"
        f"month={dt.month:02d}/"
        f"day={dt.day:02d}/"
        f"patient_id={data['patient_id']}/"
        f"{data['log_id']}.json"
    )

    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=json.dumps(data, default=str),
        ContentType="application/json"
    )
    saved += 1
    if saved % 100 == 0:
        print(f"  {saved}/{len(logs)} written...")

print(f"\nDone: {saved} records written to S3.")
cur.close()
conn.close()