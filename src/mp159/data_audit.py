"""Data inspection utilities for diagnosing data issues in train.csv."""

from pathlib import Path
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    """Load pipeline configuration from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_duplicates(df: pd.DataFrame) -> dict:
    """Count extra duplicate rows in a DataFrame."""
    extra_dupes = int(df.duplicated().sum())
    return {"present": extra_dupes > 0, "count": extra_dupes if extra_dupes > 0 else None}


def summarize_dataset(csv_path: str) -> dict:
    """Return basic row/column metadata for a CSV file."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_by_col": {k: int(v) for k, v in df.isna().sum().items()},
    }
