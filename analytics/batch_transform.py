import boto3
import sagemaker
from sagemaker.transformer import Transformer
import os
import json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

REGION  = "eu-central-1"
BUCKET  = "drug-dispenser-frankfurt"
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")

boto_session = boto3.Session(
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
sm_session = sagemaker.Session(boto_session=boto_session)
sm_client  = boto_session.client("sagemaker")

# Find the latest successful training job's model name
response = sm_client.list_training_jobs(
    StatusEquals="Completed",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

if not response["TrainingJobSummaries"]:
    print("No completed training jobs found. Run sagemaker_launch.py first.")
    exit(1)

training_job_name = response["TrainingJobSummaries"][0]["TrainingJobName"]
print(f"Using training job: {training_job_name}")

# Create model from training job
model_name = f"drug-dispenser-xgboost-model"
try:
    # Delete if it exists
    sm_client.delete_model(ModelName=model_name)
except:
    pass

training_job_info = sm_client.describe_training_job(TrainingJobName=training_job_name)
model_artifact    = training_job_info["ModelArtifacts"]["S3ModelArtifacts"]
container         = training_job_info["AlgorithmSpecification"]["TrainingImage"]

sm_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image":       container,
        "ModelDataUrl": model_artifact,
    },
    ExecutionRoleArn=SAGEMAKER_ROLE,
)
print(f"Model created: {model_name}")

# Batch Transform — read daily features for all patients, generate scores
transformer = Transformer(
    model_name=model_name,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{BUCKET}/sagemaker/scores/",
    sagemaker_session=sm_session,
    strategy="MultiRecord",
    accept="text/csv",
    assemble_with="Line",
)


transformer.transform(
    data=f"s3://{BUCKET}/train/",
    content_type="text/csv",
    split_type="Line",
    wait=True,
)


print(f"Batch Transform complete. Scores saved to s3://{BUCKET}/sagemaker/scores/")