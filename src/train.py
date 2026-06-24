"""
Train the "production" model for the DVC pipeline (random forest -- picked
after comparing baseline/RF/XGBoost runs in MLflow, see experiment-tracking
branch / docs/architecture.md).

Usage: python src/train.py
Reads:  data/processed/train.csv, params.yaml (train:)
Writes: models/model.pkl
"""
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        class_weight="balanced",  # Enrolled is ~35% the size of Graduate
        n_jobs=-1,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] fit RandomForestClassifier on {len(X)} rows, saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
