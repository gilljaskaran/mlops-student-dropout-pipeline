"""
Evaluate the trained model on the held-out test set and write DVC-tracked
metrics.

Usage: python src/evaluate.py
Reads:  models/model.pkl, data/processed/test.csv, params.yaml (evaluate:)
Writes: metrics.json, docs/confusion_matrix.png
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

MODEL_PATH = Path("models/model.pkl")
TEST_PATH = Path("data/processed/test.csv")
LABEL_MAP_PATH = Path("data/processed/label_map.json")

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]  # index order matches LABEL_MAP


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["Target"])
    y_test = test_df["Target"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1, 2]
    )
    macro_f1 = f1.mean()  # equivalent to average="macro" but we want per-class too

    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(3)
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix plot, mostly useful for the README/report
    Path("docs").mkdir(exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion matrix -- test set")
    plt.tight_layout()
    plt.savefig("docs/confusion_matrix.png", dpi=150)

    print(f"[evaluate] accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")


if __name__ == "__main__":
    main()
