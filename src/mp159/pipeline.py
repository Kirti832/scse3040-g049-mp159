"""Data cleaning, feature engineering, model training, and MLflow tracking for MP159."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DAY_TYPE_MAP = {
    "weekday": "weekday",
    "weekend": "weekend",
    "festival": "festival",
    "wee.": "weekend",
    "fes.": "festival",
}

SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file from disk."""
    return pd.read_csv(Path(path))


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file from disk."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_day_type(series: pd.Series) -> pd.Series:
    """Normalize casing, whitespace, and abbreviations in day_type."""
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.replace(DAY_TYPE_MAP)


def fix_unit_mixup(df: pd.DataFrame, findings: dict) -> pd.DataFrame:
    """Fix unit scaling issue discovered in findings.yaml."""
    out = df.copy()
    unit_info = findings.get("data_issues", {}).get("unit_mixup", findings.get("unit_mixup", {}))
    if unit_info.get("present"):
        col = unit_info.get("column", "wind_kmph")
        zone = unit_info.get("group_value", unit_info.get("zone", "Residential"))
        factor = float(unit_info.get("factor", 1000))
        if col in out.columns and "station_zone" in out.columns and factor != 0:
            mask = out["station_zone"] == zone
            out.loc[mask, col] = out.loc[mask, col].astype(float) / factor
    return out


def fix_stuck_sensor(df: pd.DataFrame, findings: dict, date_col: str = "date") -> pd.DataFrame:
    """Flag and interpolate stuck sensor readings identified in findings.yaml."""
    out = df.copy()
    stuck_info = findings.get("data_issues", {}).get("stuck_sensor", findings.get("stuck_sensor", {}))
    col = stuck_info.get("column", "humidity_pct")
    start_date = stuck_info.get("start_date")
    end_date = stuck_info.get("end_date")

    out["humidity_stuck"] = 0
    if stuck_info.get("present") and col in out.columns and date_col in out.columns:
        out = out.sort_values(date_col).reset_index(drop=True)
        in_window = pd.Series(True, index=out.index)
        if start_date:
            in_window &= out[date_col].astype(str) >= str(start_date)
        if end_date:
            in_window &= out[date_col].astype(str) <= str(end_date)

        window_vals = out.loc[in_window, col]
        if len(window_vals) > 0:
            stuck_val = window_vals.mode().iloc[0]
            stuck_mask = in_window & (out[col] == stuck_val)
            out.loc[stuck_mask, "humidity_stuck"] = 1
            out.loc[stuck_mask, col] = np.nan
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            out[col] = out[col].fillna(float(stuck_val))
    return out


def clean_dataset(
    df: pd.DataFrame,
    findings: dict,
    date_col: str = "date",
    drop_dupes: bool = True,
) -> pd.DataFrame:
    """Apply all confirmed data cleaning fixes."""
    out = df.copy()
    issues = findings.get("data_issues", findings)
    if drop_dupes and issues.get("duplicate_rows", {}).get("present"):
        out = out.drop_duplicates().reset_index(drop=True)

    out = fix_unit_mixup(out, findings)

    if issues.get("spelling_variants", {}).get("present"):
        sp_col = issues["spelling_variants"].get("column", "day_type")
        if sp_col in out.columns:
            out[sp_col] = clean_day_type(out[sp_col])

    out = fix_stuck_sensor(out, findings, date_col=date_col)
    return out


def add_engineered_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Create calendar and air-quality interaction features."""
    out = df.copy()
    if date_col in out.columns:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out["month"] = dt.dt.month.fillna(1).astype(int)
        out["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    if "pm25_today" in out.columns and "wind_kmph" in out.columns:
        out["stagnation_ratio"] = out["pm25_today"] / (out["wind_kmph"].clip(lower=0.0) + 1.0)

    if "stubble_fires" in out.columns and "temp_c" in out.columns:
        out["fire_heat_index"] = out["stubble_fires"] * out["temp_c"] / 100.0

    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def _register_and_alias_champion(
    model_name: str,
    run_id: str,
    alias: str,
    tracking_uri: str,
) -> str:
    """Register a model from a run and assign the champion alias."""
    client = MlflowClient(tracking_uri=tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version_str = str(mv.version)
    client.set_registered_model_alias(model_name, alias, version_str)
    return version_str


def run_step4(params_path: str = "params.yaml") -> dict:
    """Run cleaning, feature engineering, 3+ MLflow runs with data_sha256, and v1 champion registration."""
    params = load_yaml(params_path)
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    model_cfg = params["model"]

    findings = load_yaml(data_cfg["findings_path"])
    target_col = data_cfg["target_col"]
    date_col = data_cfg["date_col"]
    base_features = list(data_cfg["feature_cols"])

    raw_train_path = data_cfg["raw_train_path"]
    data_sha256 = compute_file_sha256(raw_train_path)

    raw_train = load_csv(raw_train_path)
    holdout_df = load_csv(data_cfg["holdout_features_path"])

    cleaned_train = clean_dataset(raw_train, findings, date_col=date_col, drop_dupes=True)
    feat_train = add_engineered_features(cleaned_train, date_col=date_col)

    cleaned_holdout = clean_dataset(holdout_df, findings, date_col=date_col, drop_dupes=False)
    feat_holdout = add_engineered_features(cleaned_holdout, date_col=date_col)

    random_state = int(model_cfg["random_state"])

    # 1. Baseline model on uncleaned features (deduplicated for split)
    raw_no_dupes = raw_train.drop_duplicates().copy()
    X_base = raw_no_dupes[base_features].copy()
    y_base = raw_no_dupes[target_col].astype(float)

    X_b_tr, X_b_val, y_b_tr, y_b_val = train_test_split(
        X_base, y_base, test_size=0.2, random_state=random_state
    )
    baseline_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_b_tr)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    baseline_pipe.fit(X_b_tr, y_b_tr)
    base_preds = baseline_pipe.predict(X_b_val)
    base_mae = float(mean_absolute_error(y_b_val, base_preds))
    base_rmse = float(root_mean_squared_error(y_b_val, base_preds))

    # 2. Improved RandomForest on cleaned + engineered features
    improved_features = [c for c in feat_train.columns if c not in {target_col, data_cfg["id_col"], date_col}]
    X_imp = feat_train[improved_features].copy()
    y_imp = feat_train[target_col].astype(float)

    X_i_tr, X_i_val, y_i_tr, y_i_val = train_test_split(
        X_imp, y_imp, test_size=0.2, random_state=random_state
    )
    rf_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_i_tr)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=int(model_cfg["n_estimators"]),
                    max_depth=int(model_cfg["max_depth"]),
                    min_samples_split=int(model_cfg["min_samples_split"]),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_i_tr, y_i_tr)
    rf_preds = rf_pipe.predict(X_i_val)
    rf_mae = float(mean_absolute_error(y_i_val, rf_preds))
    rf_rmse = float(root_mean_squared_error(y_i_val, rf_preds))

    # 3. Third experiment: GradientBoostingRegressor on cleaned + engineered features
    gbr_pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_i_tr)),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )
    gbr_pipe.fit(X_i_tr, y_i_tr)
    gbr_preds = gbr_pipe.predict(X_i_val)
    gbr_mae = float(mean_absolute_error(y_i_val, gbr_preds))
    gbr_rmse = float(root_mean_squared_error(y_i_val, gbr_preds))

    # Select best improved model by validation MAE
    candidates = [
        ("improved_rf", rf_pipe, rf_mae, rf_rmse),
        ("improved_gbr", gbr_pipe, gbr_mae, gbr_rmse),
    ]
    best_name, best_pipe, best_mae, best_rmse = min(candidates, key=lambda item: item[2])

    tracking_uri = mlflow_cfg["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name="baseline_ridge"):
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "model_type": "Ridge",
                "alpha": 1.0,
                "cleaned_data": False,
            }
        )
        mlflow.log_metrics({"val_mae": base_mae, "val_rmse": base_rmse})

    rf_run_id = ""
    with mlflow.start_run(run_name="improved_rf") as rf_run:
        rf_run_id = rf_run.info.run_id
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "model_type": "RandomForestRegressor",
                "n_estimators": int(model_cfg["n_estimators"]),
                "max_depth": int(model_cfg["max_depth"]),
                "min_samples_split": int(model_cfg["min_samples_split"]),
                "cleaned_data": True,
                "num_features": len(improved_features),
            }
        )
        mlflow.log_metrics(
            {
                "val_mae": rf_mae,
                "val_rmse": rf_rmse,
                "mae_improvement": float(base_mae - rf_mae),
            }
        )
        mlflow.sklearn.log_model(
            rf_pipe,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    gbr_run_id = ""
    with mlflow.start_run(run_name="improved_gbr") as gbr_run:
        gbr_run_id = gbr_run.info.run_id
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "cleaned_data": True,
                "num_features": len(improved_features),
            }
        )
        mlflow.log_metrics(
            {
                "val_mae": gbr_mae,
                "val_rmse": gbr_rmse,
                "mae_improvement": float(base_mae - gbr_mae),
            }
        )
        mlflow.sklearn.log_model(
            gbr_pipe,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    best_run_id = rf_run_id if best_name == "improved_rf" else gbr_run_id
    champion_version = _register_and_alias_champion(
        model_name=mlflow_cfg["registered_model_name"],
        run_id=best_run_id,
        alias=mlflow_cfg["champion_alias"],
        tracking_uri=tracking_uri,
    )

    # Save best model artifact & holdout predictions
    model_path = Path(mlflow_cfg["model_artifact_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, model_path)

    holdout_preds = best_pipe.predict(feat_holdout[improved_features])
    pred_path = Path(data_cfg["predictions_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            data_cfg["id_col"]: feat_holdout[data_cfg["id_col"]],
            "prediction": holdout_preds,
        }
    )
    pred_df.to_csv(pred_path, index=False)

    metrics = {
        "data_sha256": data_sha256,
        "cleaned_train_rows": int(len(feat_train)),
        "baseline_val_mae": base_mae,
        "baseline_val_rmse": base_rmse,
        "improved_rf_val_mae": rf_mae,
        "improved_rf_val_rmse": rf_rmse,
        "improved_gbr_val_mae": gbr_mae,
        "improved_gbr_val_rmse": gbr_rmse,
        "best_model": best_name,
        "best_val_mae": best_mae,
        "best_val_rmse": best_rmse,
        "mae_improvement": float(base_mae - best_mae),
        "best_run_id": best_run_id,
        "registered_model_name": mlflow_cfg["registered_model_name"],
        "champion_version": champion_version,
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    results = run_step4()
    print(json.dumps(results, indent=2))