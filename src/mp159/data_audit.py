"""Audit utilities for diagnosing the 7 data problem types in MP159 train.csv."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    """Load pipeline configuration from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_leaking_columns(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "pm25_next_day",
    id_col: str = "record_id",
) -> dict:
    """Find non-target, non-ID columns present in train_df but missing from holdout_df."""
    ignore_cols = {target_col, id_col}
    leaking = [col for col in train_df.columns if col not in holdout_df.columns and col not in ignore_cols]
    return {
        "present": len(leaking) > 0,
        "columns": leaking,
        "train_only_all": [col for col in train_df.columns if col not in holdout_df.columns],
    }


def summarize_by_zone(train_df: pd.DataFrame, group_col: str = "station_zone") -> dict:
    """Compute min, mean, max, and median per station_zone for numeric columns."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = {}
    if group_col in train_df.columns:
        for zone, grp in train_df.groupby(group_col, dropna=False):
            zone_stats = {}
            for col in num_cols:
                s = grp[col].dropna()
                zone_stats[col] = {
                    "min": float(s.min()) if len(s) else None,
                    "mean": float(s.mean()) if len(s) else None,
                    "median": float(s.median()) if len(s) else None,
                    "max": float(s.max()) if len(s) else None,
                }
            summary[str(zone)] = zone_stats
    return summary


def find_sentinel_candidates(
    train_df: pd.DataFrame,
    sentinels: tuple[float, ...] = (-1, -99, -999, -9999, 999, 9999),
) -> dict:
    """Check numeric columns for common sentinel placeholder values."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    findings = {}
    for col in num_cols:
        col_hits = {}
        for val in sentinels:
            cnt = int((train_df[col] == val).sum())
            if cnt > 0:
                col_hits[str(val)] = cnt
        if col_hits:
            findings[col] = col_hits
    return findings


def find_duplicate_rows(train_df: pd.DataFrame) -> dict:
    """Count exact duplicate rows in train_df."""
    extra_dupes = int(train_df.duplicated().sum())
    return {
        "present": extra_dupes > 0,
        "count": extra_dupes,
    }


def find_stuck_sensor_runs(
    train_df: pd.DataFrame,
    date_col: str = "date",
    min_run_length: int = 5,
) -> list[dict]:
    """Find consecutive runs of identical values after sorting by date."""
    if date_col not in train_df.columns:
        return []

    sorted_df = train_df.sort_values(date_col).reset_index(drop=True)
    num_cols = [c for c in sorted_df.select_dtypes(include=[np.number]).columns if c != "pm25_next_day"]
    runs = []

    for col in num_cols:
        series = sorted_df[col]
        dates = sorted_df[date_col].astype(str)
        run_start = 0

        for i in range(1, len(series) + 1):
            is_same = False
            if i < len(series):
                v_prev, v_curr = series.iloc[i - 1], series.iloc[i]
                if pd.notna(v_prev) and pd.notna(v_curr) and v_prev == v_curr:
                    is_same = True
            if not is_same:
                run_len = i - run_start
                if run_len >= min_run_length and pd.notna(series.iloc[run_start]):
                    runs.append(
                        {
                            "column": col,
                            "stuck_value": float(series.iloc[run_start]),
                            "length": int(run_len),
                            "start_date": str(dates.iloc[run_start]),
                            "end_date": str(dates.iloc[i - 1]),
                        }
                    )
                run_start = i

    runs.sort(key=lambda x: x["length"], reverse=True)
    return runs


def inspect_target_distribution(
    train_df: pd.DataFrame,
    target_col: str = "pm25_next_day",
) -> dict:
    """Inspect top target values to spot ~10x typo outliers and gaps."""
    if target_col not in train_df.columns:
        return {}
    s = pd.to_numeric(train_df[target_col], errors="coerce").dropna().sort_values(ascending=False)
    top_20 = [float(x) for x in s.head(20).tolist()]
    q99 = float(np.percentile(s, 99))
    median_val = float(s.median())
    high_candidates = s[s > max(q99 * 1.5, median_val * 4)]
    return {
        "min": float(s.min()),
        "median": median_val,
        "p95": float(np.percentile(s, 95)),
        "p99": q99,
        "max": float(s.max()),
        "top_20_values": top_20,
        "high_candidate_count": int(len(high_candidates)),
        "high_candidate_values": [float(x) for x in high_candidates.tolist()],
    }


def inspect_text_columns(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame | None = None,
) -> dict:
    """Inspect object/string columns in train.csv (and compare with later.csv)."""
    obj_cols = train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    report = {}
    for col in obj_cols:
        train_vc = train_df[col].value_counts(dropna=False).to_dict()
        entry = {
            "train_value_counts": {str(k): int(v) for k, v in train_vc.items()},
        }
        if later_df is not None and col in later_df.columns:
            later_vc = later_df[col].value_counts(dropna=False).to_dict()
            entry["later_value_counts"] = {str(k): int(v) for k, v in later_vc.items()}
        report[col] = entry
    return report


def run_full_audit(params_path: str = "params.yaml") -> dict:
    """Run all audit checks and return a structured report dictionary."""
    params = load_params(params_path)
    train_df = pd.read_csv(params["data"]["raw_train_path"])
    later_df = pd.read_csv(params["data"]["raw_later_path"])
    holdout_df = pd.read_csv(params["data"]["holdout_features_path"])

    return {
        "shapes": {
            "train": list(train_df.shape),
            "later": list(later_df.shape),
            "holdout": list(holdout_df.shape),
        },
        "leaking_column": detect_leaking_columns(
            train_df,
            holdout_df,
            target_col=params["data"]["target_col"],
            id_col=params["data"]["id_col"],
        ),
        "duplicate_rows": find_duplicate_rows(train_df),
        "sentinel_candidates": find_sentinel_candidates(train_df),
        "stuck_sensor_runs": find_stuck_sensor_runs(
            train_df,
            date_col=params["data"]["date_col"],
        ),
        "target_distribution": inspect_target_distribution(
            train_df,
            target_col=params["data"]["target_col"],
        ),
        "zone_numeric_stats": summarize_by_zone(
            train_df,
            group_col=params["data"]["group_col"],
        ),
        "text_columns": inspect_text_columns(train_df, later_df),
    }


def print_compact_summary(report: dict) -> None:
    """Print a concise summary of the 7 data checks to the terminal."""
    print("SHAPES:", report["shapes"])
    print("1. LEAKING COLUMNS:", report["leaking_column"])
    print("2. DUPLICATE ROWS:", report["duplicate_rows"])
    print("3. SENTINEL CANDIDATES:", report["sentinel_candidates"])
    print("4. STUCK SENSOR RUNS (top 5):", report["stuck_sensor_runs"][:5])
    td = report["target_distribution"]
    print(
        "5. TARGET DIST:",
        {
            "min": td.get("min"),
            "median": td.get("median"),
            "p99": td.get("p99"),
            "max": td.get("max"),
            "top_10": td.get("top_20_values", [])[:10],
        },
    )
    print("6. TEXT COLUMNS (keys & counts):", {
        col: {
            "train": info.get("train_value_counts"),
            "later": info.get("later_value_counts"),
        }
        for col, info in report["text_columns"].items()
    })
    print("7. ZONE STATS AVAILABLE FOR ZONES:", list(report["zone_numeric_stats"].keys()))


if __name__ == "__main__":
    audit_report = run_full_audit()
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print_compact_summary(audit_report)