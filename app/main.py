"""FastAPI prediction service for MP159 Jaipur Air Quality."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp159.pipeline import add_engineered_features  # noqa: E402


class JsonStdoutFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload.update(record.extra_data)
        return json.dumps(payload)


class DynamicStdoutHandler(logging.StreamHandler):
    """Always write to the current sys.stdout so pytest capsys captures output."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


logger = logging.getLogger("mp159_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
_handler = DynamicStdoutHandler()
_handler.setFormatter(JsonStdoutFormatter())
logger.addHandler(_handler)


class PredictRequest(BaseModel):
    """Exact input payload required by POST /predict."""

    pm25_today: float = Field(..., ge=7.5)
    wind_kmph: float
    humidity_pct: float
    temp_c: float
    traffic_index: float
    stubble_fires: float = Field(..., le=700)
    day_type: str
    station_zone: str


class PredictResponse(BaseModel):
    """Prediction response schema."""

    pm25_next_day_pred: float


app = FastAPI(title="MP159 Jaipur Air Quality API", version="1.0.0")
_MODEL_CACHE: dict[str, Any] = {"model": None, "path": None}


def _load_findings() -> dict[str, Any]:
    findings_path = Path(os.getenv("MP159_FINDINGS_PATH", ROOT_DIR / "reports" / "findings.yaml"))
    if findings_path.exists():
        with open(findings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    return {}


def get_model() -> Any:
    """Load and cache the trained model artifact."""
    model_path = Path(os.getenv("MP159_MODEL_PATH", ROOT_DIR / "models" / "model.joblib"))
    if _MODEL_CACHE["model"] is None or _MODEL_CACHE["path"] != str(model_path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _MODEL_CACHE["model"] = joblib.load(model_path)
        _MODEL_CACHE["path"] = str(model_path)
    return _MODEL_CACHE["model"]


def build_feature_frame(payload: PredictRequest) -> pd.DataFrame:
    """Convert request payload into feature DataFrame matching pipeline features."""
    findings = _load_findings()
    stuck_info = findings.get("data_issues", {}).get("stuck_sensor", {})
    stuck_col = stuck_info.get("column", "humidity_pct")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row: dict[str, Any] = {
        "date": today_str,
        "pm25_today": float(payload.pm25_today),
        "wind_kmph": float(payload.wind_kmph),
        "humidity_pct": float(payload.humidity_pct),
        "temp_c": float(payload.temp_c),
        "traffic_index": float(payload.traffic_index),
        "stubble_fires": float(payload.stubble_fires),
        "day_type": payload.day_type.strip().lower(),
        "station_zone": payload.station_zone.strip(),
    }
    if stuck_col == "humidity_pct":
        row["humidity_stuck"] = 0

    df = pd.DataFrame([row])
    return add_engineered_features(df, date_col="date")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict next-day PM2.5 concentration."""
    try:
        model = get_model()
        features_df = build_feature_frame(payload)
        pred_val = float(model.predict(features_df)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    logger.info(
        "prediction_completed",
        extra={
            "extra_data": {
                "station_zone": payload.station_zone,
                "day_type": payload.day_type,
                "pm25_today": payload.pm25_today,
                "stubble_fires": payload.stubble_fires,
                "prediction": pred_val,
            }
        },
    )
    return PredictResponse(pm25_next_day_pred=pred_val)
