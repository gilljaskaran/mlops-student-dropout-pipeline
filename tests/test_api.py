"""
Integration tests for the FastAPI inference service (app/main.py).

Uses the synthetic model fixture from conftest.py so these run in CI without
needing the real DVC-tracked data/model artifacts.
"""


def test_health(client):
    api, _ = client
    resp = api.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == "test-fixture"


def test_model_info(client):
    api, app_main = client
    resp = api.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classes"] == ["Dropout", "Enrolled", "Graduate"]
    assert set(body["features"]) == set(app_main.FEATURE_NAMES)
    assert body["n_features"] == len(app_main.FEATURE_NAMES)


def _sample_payload(feature_names):
    return {name: 1.0 for name in feature_names}


def test_predict_returns_valid_class(client):
    api, app_main = client
    payload = _sample_payload(app_main.FEATURE_NAMES)
    resp = api.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ["Dropout", "Enrolled", "Graduate"]
    assert body["prediction_index"] in [0, 1, 2]
    assert set(body["probabilities"].keys()) == {"Dropout", "Enrolled", "Graduate"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_missing_field_returns_422(client):
    api, app_main = client
    payload = _sample_payload(app_main.FEATURE_NAMES)
    payload.pop(app_main.FEATURE_NAMES[0])
    resp = api.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch(client):
    api, app_main = client
    payload = {"records": [_sample_payload(app_main.FEATURE_NAMES) for _ in range(3)]}
    resp = api.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    for item in body:
        assert item["prediction"] in ["Dropout", "Enrolled", "Graduate"]


def test_predictions_are_logged(client, tmp_path):
    api, app_main = client
    payload = _sample_payload(app_main.FEATURE_NAMES)
    api.post("/predict", json=payload)
    log_path = app_main.PREDICTIONS_LOG
    assert log_path.exists()
    content = log_path.read_text()
    assert "prediction" in content  # header row present


def test_recent_predictions_endpoint(client):
    api, app_main = client
    payload = _sample_payload(app_main.FEATURE_NAMES)
    api.post("/predict", json=payload)
    resp = api.get("/monitoring/recent-predictions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert "prediction" in body["records"][0]
