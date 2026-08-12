"""
FastAPI inference service for the student dropout / academic success model.

Model-as-service pattern (Week 8): the model is packaged and served as its
own process, independent of the DVC training pipeline. Loads
`models/model.pkl` (the `train` stage's output) and exposes REST endpoints
for prediction, health checks, and request logging that feeds the
monitoring / drift-detection pipeline (Week 9) and the retraining trigger
(Week 10) -- see src/monitor_drift.py and src/check_retrain_trigger.py.

The request schema is NOT hand-typed. RandomForestClassifier (like any
scikit-learn estimator fit on a pandas DataFrame) records the training
column names on `model.feature_names_in_`. We build the Pydantic schema
from that at startup, so the API can never silently drift out of sync with
whatever columns `src/prepare.py` actually produced.

Run locally:
    uvicorn app.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import csv
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOG_DIR / "predictions.csv"

# Index order is fixed by the LABEL_MAP in src/prepare.py -- do not reorder.
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]

logger = logging.getLogger("uvicorn.error")

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run `dvc repro` (or `dvc pull`) "
        "to produce models/model.pkl before starting the API -- see README.md."
    )

model = joblib.load(MODEL_PATH)

if not hasattr(model, "feature_names_in_"):
    raise RuntimeError(
        "Loaded model has no feature_names_in_, meaning it wasn't trained on "
        "a pandas DataFrame with column names. Retrain via `dvc repro` with "
        "the current src/train.py before serving it."
    )

FEATURE_NAMES: List[str] = list(model.feature_names_in_)


def _model_version() -> str:
    if os.getenv("MODEL_VERSION"):
        return os.getenv("MODEL_VERSION")
    digest = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest()[:8]
    return f"rf-{digest}"


MODEL_VERSION = _model_version()

# Build the request schema from the model's actual training columns. All
# features in this dataset are integer-coded categoricals or numeric grades
# (see docs/dataset.md) -- float is a safe, permissive supertype for both.
_fields = {
    name: (float, Field(..., description=f"Value for feature '{name}'")) for name in FEATURE_NAMES
}
StudentFeatures = create_model("StudentFeatures", **_fields)


def _as_dict(pydantic_obj) -> dict:
    # pydantic v1 vs v2 compatibility
    return pydantic_obj.model_dump() if hasattr(pydantic_obj, "model_dump") else pydantic_obj.dict()


class PredictionResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: Dict[str, float]
    model_version: str


class BatchRequest(BaseModel):
    records: List[StudentFeatures]


app = FastAPI(
    title="Student Dropout Prediction API",
    description=(
        "Serves the DVC/MLflow-trained RandomForestClassifier from the "
        "mlops-student-dropout-pipeline project (MAI201 Phase 2)."
    ),
    version="1.0.0",
)


def _log_prediction(features: dict, prediction: str, probabilities: dict) -> None:
    """Append one row per request to logs/predictions.csv.

    This is the raw material for src/monitor_drift.py: EvidentlyAI compares
    this "current" data against data/processed/train.csv ("reference") to
    check for data drift. The prediction distribution itself is one of the
    four monitored artifacts covered in Week 9 (accuracy, predictions,
    features, raw inputs).
    """
    is_new = not PREDICTIONS_LOG.exists()
    fieldnames = [
        "timestamp",
        *FEATURE_NAMES,
        "prediction",
        *[f"proba_{c}" for c in CLASS_NAMES],
    ]
    with open(PREDICTIONS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **features,
            "prediction": prediction,
        }
        row.update({f"proba_{c}": probabilities[c] for c in CLASS_NAMES})
        writer.writerow(row)


@app.get("/")
def root():
    return {
        "service": "Student Dropout Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "model_version": MODEL_VERSION,
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
    }


@app.get("/monitoring/recent-predictions")
def recent_predictions(limit: int = 500):
    """Last N logged requests as JSON records, so a monitoring/retraining
    job (see .github/workflows/retrain.yml, src/monitor_drift.py) can pull
    "current" production data from a live deployment without needing
    direct filesystem/database access to wherever this is hosted."""
    if not PREDICTIONS_LOG.exists():
        return {"count": 0, "records": []}
    df = pd.read_csv(PREDICTIONS_LOG)
    tail = df.tail(limit)
    return {"count": len(tail), "records": tail.to_dict(orient="records")}


def _predict_one(features) -> PredictionResponse:
    row = _as_dict(features)
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    probabilities = {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(len(CLASS_NAMES))}
    prediction = CLASS_NAMES[pred_idx]
    _log_prediction(row, prediction, probabilities)
    return PredictionResponse(
        prediction=prediction,
        prediction_index=pred_idx,
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    try:
        return _predict_one(features)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(batch: BatchRequest):
    try:
        return [_predict_one(record) for record in batch.records]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
