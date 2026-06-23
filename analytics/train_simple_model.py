import boto3
import pandas as pd
import io
import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

s3 = boto3.client(
    "s3",
    region_name="eu-central-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

obj = s3.get_object(Bucket="drug-dispenser-frankfurt", Key="train/train.csv")
df  = pd.read_csv(io.BytesIO(obj["Body"].read()), header=None)

y = df.iloc[:, 0].values.astype(float)
X = df.select_dtypes(include="number").iloc[:, 1:].values.astype(float)

# Normalize
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_norm = (X - mean) / std

# Logistic regression with gradient descent
np.random.seed(42)
w  = np.zeros(X_norm.shape[1])
b  = 0.0
lr = 0.01

for _ in range(1000):
    z    = X_norm @ w + b
    pred = 1 / (1 + np.exp(-z))
    err  = pred - y
    w   -= lr * (X_norm.T @ err) / len(y)
    b   -= lr * err.mean()

# Save weights as JSON
model_data = {
    "w":    w.tolist(),
    "b":    float(b),
    "mean": mean.tolist(),
    "std":  std.tolist(),
}
model_json = json.dumps(model_data)
print(f"Model boyutu: {len(model_json)/1024:.1f} KB")

s3.put_object(
    Bucket="drug-dispenser-frankfurt",
    Key="sklearn/model.json",
    Body=model_json.encode(),
)
print("Model yuklendi: s3://drug-dispenser-frankfurt/sklearn/model.json")

z     = X_norm @ w + b
preds = (1 / (1 + np.exp(-z))) > 0.5
acc   = (preds == y).mean()
print(f"Accuracy: {acc:.3f}")