# convert_to_csv.py
import boto3
import pandas as pd
import pyarrow.parquet as pq
import io
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

BUCKET = "drug-dispenser-analytics-datalake"
PREFIX = "curated/"

# List all parquet files
response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
parquet_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".parquet")]

print(f"{len(parquet_files)} parquet files found")

dfs = []
for key in parquet_files:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Total {len(df)} rows, columns: {df.columns.tolist()}")

# will_miss_tomorrow ilk kolona al (SageMaker hedef kolonu ilk olmalı)
cols = ["will_miss_tomorrow"] + [c for c in df.columns if c != "will_miss_tomorrow"]
df = df[cols].dropna()

# Upload CSV to Frankfurt bucket
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False, header=False)

s3_frankfurt = boto3.client(
    "s3",
    region_name="eu-central-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

s3_frankfurt.put_object(
    Bucket="drug-dispenser-frankfurt",
    Key="train/train.csv",
    Body=csv_buffer.getvalue().encode("utf-8"),
)

print("CSV uploaded to Frankfurt: s3://drug-dispenser-frankfurt/train/train.csv")
print(f"Total rows: {len(df)}")