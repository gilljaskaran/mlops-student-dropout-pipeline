"""
Shared pytest fixtures.

CI never has access to the real (DVC-tracked) data.csv or model.pkl -- see
docs/dataset.md and the note in docs/deployment.md about the DVC remote.
So tests build a tiny synthetic model with the same *shape* of contract the
real one has (a DataFrame-fit sklearn classifier with feature_names_in_),
rather than depending on the actual trained artifact. This keeps CI fast
and independent of data access, per the Week 6 "test on sample data" guidance.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

SYNTHETIC_FEATURES = [
    "Marital status",
    "Application mode",
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
]


@pytest.fixture(scope="session")
def synthetic_model_path(tmp_path_factory):
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "Marital status": rng.randint(1, 6, n),
            "Application mode": rng.randint(1, 18, n),
            "Age at enrollment": rng.randint(17, 50, n),
            "Curricular units 1st sem (approved)": rng.randint(0, 10, n),
            "Curricular units 2nd sem (approved)": rng.randint(0, 10, n),
        }
    )
    y = rng.randint(0, 3, n)  # 0=Dropout, 1=Enrolled, 2=Graduate

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


@pytest.fixture()
def client(synthetic_model_path, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(synthetic_model_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MODEL_VERSION", "test-fixture")

    # app.main reads env vars and loads the model at import time, so make
    # sure we get a fresh import for every test run (module caching would
    # otherwise reuse whatever model the first test loaded).
    import sys

    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient

    from app import main as app_main

    return TestClient(app_main.app), app_main
