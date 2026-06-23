import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth",    type=int,   default=5)
    parser.add_argument("--eta",          type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int,   default=100)
    parser.add_argument("--objective",    type=str,   default="binary:logistic")
    parser.add_argument("--eval_metric",  type=str,   default="auc")
    parser.add_argument("--model-dir",    type=str,   default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train",        type=str,   default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    args = parser.parse_args()

    # Read all parquet files from curated zone
    dfs = []
    for root, dirs, files in os.walk(args.train):
        for f in files:
            if f.endswith(".parquet"):
                dfs.append(pd.read_parquet(os.path.join(root, f)))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows")

    features = [
        "total_doses",
        "missed_count",
        "dispensed_count",
        "taken_count",
        "error_count",
        "avg_face_score",
        "min_face_score",
        "miss_rate",
        "miss_rate_7d",
        "avg_face_7d",
        "error_count_7d",
    ]
    target = "will_miss_tomorrow"

    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.eta,
        n_estimators=args.n_estimators,
        objective=args.objective,
        eval_metric=args.eval_metric,
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.4f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print(f"Model saved to {args.model_dir}")