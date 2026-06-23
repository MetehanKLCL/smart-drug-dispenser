import boto3
import sagemaker
from sagemaker.estimator import Estimator
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

REGION         = "eu-central-1"
BUCKET         = "drug-dispenser-frankfurt"
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")

boto_session = boto3.Session(
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
sm_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=BUCKET,
)

container = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
)

s3_input  = "s3://drug-dispenser-frankfurt/train/"
s3_output = "s3://drug-dispenser-frankfurt/sagemaker/output/"

xgb = Estimator(
    image_uri=container,
    role=SAGEMAKER_ROLE,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=s3_output,
    sagemaker_session=sm_session,
    hyperparameters={
        "max_depth":        5,
        "eta":              0.2,
        "num_round":        100,
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
    }
)

print("Training starting...")
xgb.fit({"train": sagemaker.inputs.TrainingInput(
    s3_input,
    content_type="text/csv"
)})
print(f"Training complete. Model saved to: {s3_output}")
