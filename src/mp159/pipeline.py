"""Placeholder training, monitoring, and prediction module for MP159."""

from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file from disk."""
    return pd.read_csv(Path(path))
