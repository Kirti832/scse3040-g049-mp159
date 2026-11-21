import joblib
from fastapi.testclient import TestClient

from app.main import app


class _DummyRegressor:
    def predict(self, X):
        return [142.5 for _ in range(len(X))]


def test_predict_success_and_json_log(tmp_path, monkeypatch, capsys):
    model_file = tmp_path / "model.joblib"
    joblib.dump(_DummyRegressor(), model_file)
    monkeypatch.setenv("MP159_MODEL_PATH", str(model_file))

    client = TestClient(app)
    payload = {
        "pm25_today": 120.0,
        "wind_kmph": 8.5,
        "humidity_pct": 55.0,
        "temp_c": 29.0,
        "traffic_index": 60.0,
        "stubble_fires": 45.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["pm25_next_day_pred"] == 142.5

    out = capsys.readouterr().out
    assert '"message": "prediction_completed"' in out


def test_predict_rejects_high_stubble_fires_and_low_pm25_422():
    client = TestClient(app)
    base_payload = {
        "pm25_today": 120.0,
        "wind_kmph": 8.5,
        "humidity_pct": 55.0,
        "temp_c": 29.0,
        "traffic_index": 60.0,
        "stubble_fires": 45.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }

    high_fires = {**base_payload, "stubble_fires": 701.0}
    assert client.post("/predict", json=high_fires).status_code == 422

    low_pm25 = {**base_payload, "pm25_today": 5.0}
    assert client.post("/predict", json=low_pm25).status_code == 422
