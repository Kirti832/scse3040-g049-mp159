import pandas as pd
from src.mp159.pipeline import (
    add_engineered_features,
    clean_dataset,
    clean_day_type,
)


def _sample_findings() -> dict:
    return {
        "data_issues": {
            "duplicate_rows": {"present": True, "count": 1},
            "unit_mixup": {
                "present": True,
                "column": "wind_kmph",
                "group_value": "Residential",
                "factor": 1000,
            },
            "spelling_variants": {
                "present": True,
                "column": "day_type",
                "true_levels": 3,
            },
            "stuck_sensor": {
                "present": True,
                "column": "humidity_pct",
                "start_date": "2026-04-25",
                "end_date": "2026-05-10",
            },
        }
    }


def test_clean_day_type_normalizes_variants():
    s = pd.Series(["WEEKDAY", " wee. ", "fes.", "Festival"])
    out = clean_day_type(s)
    assert list(out) == ["weekday", "weekend", "festival", "festival"]


def test_clean_dataset_fixes_dupes_units_and_stuck_humidity():
    df = pd.DataFrame(
        {
            "record_id": ["R1", "R1", "R2", "R3", "R4"],
            "date": [
                "2026-04-24",
                "2026-04-24",
                "2026-04-25",
                "2026-04-26",
                "2026-04-27",
            ],
            "station_zone": ["Residential", "Residential", "Residential", "Central", "Industrial"],
            "wind_kmph": [12000.0, 12000.0, 14000.0, 10.0, 8.0],
            "humidity_pct": [60.0, 60.0, 49.0, 49.0, 70.0],
            "day_type": ["weekday", "weekday", "wee.", "fes.", "weekend"],
            "pm25_today": [100.0, 100.0, 120.0, 140.0, 90.0],
            "temp_c": [30.0, 30.0, 32.0, 35.0, 28.0],
            "traffic_index": [50.0, 50.0, 55.0, 65.0, 45.0],
            "stubble_fires": [10.0, 10.0, 12.0, 20.0, 5.0],
            "pm25_next_day": [110.0, 110.0, 130.0, 150.0, 95.0],
        }
    )
    cleaned = clean_dataset(df, _sample_findings(), date_col="date", drop_dupes=True)
    assert len(cleaned) == 4
    res_wind = cleaned.loc[cleaned["record_id"] == "R1", "wind_kmph"].iloc[0]
    assert res_wind == 12.0
    assert set(cleaned["day_type"]).issubset({"weekday", "weekend", "festival"})
    assert cleaned["humidity_stuck"].sum() >= 1


def test_add_engineered_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": ["2026-01-10", "2026-01-11"],
            "pm25_today": [150.0, 200.0],
            "wind_kmph": [4.0, 9.0],
            "stubble_fires": [50.0, 100.0],
            "temp_c": [20.0, 25.0],
        }
    )
    feat = add_engineered_features(df, date_col="date")
    for col in ["month", "day_of_week", "is_weekend", "stagnation_ratio", "fire_heat_index"]:
        assert col in feat.columns
