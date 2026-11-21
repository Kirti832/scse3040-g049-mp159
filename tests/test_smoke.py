"""Starter tests including FastAPI TestClient verification."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd

from mp159.data_audit import find_duplicate_rows

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def test_find_duplicate_rows_detects_extra_rows() -> None:
    df = pd.DataFrame(
        {
            "station_zone": ["Central", "Central", "Residential"],
            "pm25_today": [120.0, 120.0, 85.0],
        }
    )
    result = find_duplicate_rows(df)
    assert result["present"] is True
    assert result["count"] == 1


def test_health_endpoint_with_test_client() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
