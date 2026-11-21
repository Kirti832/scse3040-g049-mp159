"""Stage 4 drift monitoring, per-zone MAE evaluation, v2 retraining, and champion promotion."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_SRC_DIR = _Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from mp159.pipeline import (
    SKOPS_TRUSTED_TYPES,
    add_engineered_features,
    build_preprocessor,
    clean_dataset,
    compute_file_sha256,
    load_csv,
    load_yaml,
)


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric series."""
    exp_arr = pd.to_numeric(expected, errors="coerce").dropna().to_numpy(dtype=float)
    act_arr = pd.to_numeric(actual, errors="coerce").dropna().to_numpy(dtype=float)
    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(exp_arr, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_arr, bins=edges)
    act_counts, _ = np.histogram(act_arr, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(len(exp_arr), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(act_arr), 1), eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def _save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_stage4(params_path: str = "params.yaml") -> dict[str, Any]:
    """Execute PSI drift check, v1 zone MAE, v2 retraining, promotion rule, and holdout scoring."""
    params = load_yaml(params_path)
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]
    model_cfg = params["model"]

    findings_path = Path(data_cfg["findings_path"])
    findings = load_yaml(findings_path)

    target_col = data_cfg["target_col"]
    date_col = data_cfg["date_col"]
    id_col = data_cfg["id_col"]
    base_features = list(data_cfg["feature_cols"])

    raw_train_path = data_cfg["raw_train_path"]
    later_path = data_cfg.get("later_path", "data/raw/later.csv")
    holdout_path = data_cfg["holdout_features_path"]

    data_sha256 = compute_file_sha256(raw_train_path)

    raw_train = load_csv(raw_train_path)
    raw_later = load_csv(later_path)
    raw_holdout = load_csv(holdout_path)

    clean_train = add_engineered_features(
        clean_dataset(raw_train, findings, date_col=date_col, drop_dupes=True),
        date_col=date_col,
    )
    clean_later = add_engineered_features(
        clean_dataset(raw_later, findings, date_col=date_col, drop_dupes=True),
        date_col=date_col,
    )
    clean_holdout = add_engineered_features(
        clean_dataset(raw_holdout, findings, date_col=date_col, drop_dupes=False),
        date_col=date_col,
    )

    if date_col in clean_later.columns:
        clean_later = clean_later.sort_values(date_col).reset_index(drop=True)

    # 1. PSI per numeric column across 10 bins
    psi_bins = int(params.get("monitoring", {}).get("psi_bins", 10))
    numeric_cols = [
        c
        for c in base_features
        if c in clean_train.columns
        and c in clean_later.columns
        and pd.api.types.is_numeric_dtype(clean_train[c])
    ]
    psi_by_feature: dict[str, float] = {
        col: compute_psi(clean_train[col], clean_later[col], bins=psi_bins)
        for col in numeric_cols
    }
    most_drifted_feature = (
        max(psi_by_feature.items(), key=lambda item: item[1])[0]
        if psi_by_feature
        else None
    )

    # Load v1 model (current model.joblib)
    v1_model_path = Path(mlflow_cfg["model_artifact_path"])
    v1_model = joblib.load(v1_model_path)

    feature_cols = [c for c in clean_train.columns if c not in {target_col, id_col, date_col}]

    # 2. v1 MAE per station_zone on later.csv
    mae_by_group: dict[str, float] = {}
    if "station_zone" in clean_later.columns and target_col in clean_later.columns:
        for zone, grp in clean_later.groupby("station_zone", dropna=False):
            if len(grp) > 0:
                preds_grp = v1_model.predict(grp[feature_cols])
                mae_by_group[str(zone)] = float(
                    mean_absolute_error(grp[target_col].astype(float), preds_grp)
                )
    worst_served_group = (
        max(mae_by_group.items(), key=lambda item: item[1])[0]
        if mae_by_group
        else None
    )

    # 3. Retrain v2 on cleaned train.csv + first 70% of sorted later.csv; evaluate on last 30%
    split_ratio = float(params.get("monitoring", {}).get("retrain_later_split", 0.70))
    split_idx = max(1, min(len(clean_later) - 1, int(len(clean_later) * split_ratio)))
    later_train = clean_later.iloc[:split_idx].copy()
    later_check = clean_later.iloc[split_idx:].copy()

    combined_train = pd.concat([clean_train, later_train], ignore_index=True)
    X_v2_tr = combined_train[feature_cols].copy()
    y_v2_tr = combined_train[target_col].astype(float)

    v2_model = Pipeline(
        steps=[
            ("prep", build_preprocessor(X_v2_tr)),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=int(model_cfg["random_state"]),
                ),
            ),
        ]
    )
    v2_model.fit(X_v2_tr, y_v2_tr)

    mae_v1_on_check = float(
        mean_absolute_error(
            later_check[target_col].astype(float),
            v1_model.predict(later_check[feature_cols]),
        )
    )
    mae_v2_on_check = float(
        mean_absolute_error(
            later_check[target_col].astype(float),
            v2_model.predict(later_check[feature_cols]),
        )
    )

    # 4. Promotion rule: v2 MAE must be at least 12% lower than v1 MAE
    improvement_threshold = float(
        params.get("monitoring", {}).get("promotion_min_improvement", 0.12)
    )
    relative_improvement = (
        (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
        if mae_v1_on_check > 0
        else 0.0
    )
    promote_v2 = relative_improvement >= improvement_threshold
    decision = "promote" if promote_v2 else "keep"

    tracking_uri = mlflow_cfg["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    reg_model_name = mlflow_cfg["registered_model_name"]
    champion_alias = mlflow_cfg["champion_alias"]
    client = MlflowClient(tracking_uri=tracking_uri)

    v2_run_id = ""
    with mlflow.start_run(run_name="retrained_v2_gbr") as run:
        v2_run_id = run.info.run_id
        mlflow.log_params(
            {
                "data_sha256": data_sha256,
                "model_type": "GradientBoostingRegressor",
                "retrain_later_split": split_ratio,
                "decision": decision,
            }
        )
        mlflow.log_metrics(
            {
                "mae_v1_on_check": mae_v1_on_check,
                "mae_v2_on_check": mae_v2_on_check,
                "relative_mae_improvement": float(relative_improvement),
            }
        )
        mlflow.sklearn.log_model(
            v2_model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    mv_v2 = mlflow.register_model(
        model_uri=f"runs:/{v2_run_id}/model",
        name=reg_model_name,
    )
    v2_version_str = str(mv_v2.version)

    if promote_v2:
        client.set_registered_model_alias(reg_model_name, champion_alias, v2_version_str)
        joblib.dump(v2_model, v1_model_path)
        champion_model = v2_model
        active_champion_version = v2_version_str
    else:
        champion_model = v1_model
        active_champion_version = str(findings.get("tracking", {}).get("champion_version", "1"))

    # 5. Predict on holdout_features.csv using @champion model
    holdout_preds = champion_model.predict(clean_holdout[feature_cols])
    pred_path = Path(data_cfg["predictions_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            id_col: clean_holdout[id_col],
            "prediction": holdout_preds,
        }
    ).to_csv(pred_path, index=False)

    # 6. Update reports/findings.yaml cleanly
    findings.setdefault("tracking", {})
    findings["tracking"]["registered_model_name"] = reg_model_name
    findings["tracking"]["champion_version"] = active_champion_version
    findings["tracking"]["v2_run_id"] = v2_run_id
    findings["tracking"]["v2_version"] = v2_version_str

    findings.setdefault("monitoring", {})
    findings["monitoring"]["psi_by_feature"] = psi_by_feature
    findings["monitoring"]["most_drifted_feature"] = most_drifted_feature
    findings["monitoring"]["mae_by_group"] = mae_by_group
    findings["monitoring"]["worst_served_group"] = worst_served_group
    findings["monitoring"]["mae_v1_on_check"] = mae_v1_on_check
    findings["monitoring"]["mae_v2_on_check"] = mae_v2_on_check
    findings["monitoring"]["relative_mae_improvement"] = float(relative_improvement)
    findings["monitoring"]["decision"] = decision

    _save_yaml(findings_path, findings)

    summary = {
        "most_drifted_feature": most_drifted_feature,
        "psi_by_feature": psi_by_feature,
        "worst_served_group": worst_served_group,
        "mae_by_group": mae_by_group,
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "relative_mae_improvement": float(relative_improvement),
        "decision": decision,
        "champion_version": active_champion_version,
        "predictions_file": str(pred_path),
    }
    Path("reports/stage4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage4(), indent=2))

