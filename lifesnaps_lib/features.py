import numpy as np
import pandas as pd

# 19 features selected from notebook 01 via XGBoost cumulative-95% importance.
# These are the best-performing reduced feature set (notebook 02).
FEATURE_COLS = [
    # Activity (raw)
    "steps", "distance", "very_active_minutes", "moderately_active_minutes", "lightly_active_minutes",
    # Activity (engineered)
    "TotalActiveMinutes", "ActiveRatio", "StepsPerActiveMin",
    # Heart rate
    "bpm", "resting_hr", "hr_zone_cardio",
    # HRV / advanced Fitbit
    "nremhr", "rmssd", "filteredDemographicVO2Max",
    # Skin temperature
    "nightly_temperature", "daily_temperature_variation",
    # Demographics
    "bmi",
    # Mood (SEMA app, ~66% missing — imputed per-user)
    "ALERT", "TIRED",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by the pipeline.

    Operates on a copy so the original DataFrame is not mutated.
    Expects a 'date' column (datetime) and the raw Fitbit/SEMA columns.
    """
    df = df.copy()

    # Time
    df["DayOfWeek"] = df["date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Activity composites
    df["TotalActiveMinutes"] = (
        df["very_active_minutes"]
        + df["moderately_active_minutes"]
        + df["lightly_active_minutes"]
    )
    df["ActiveRatio"] = df["TotalActiveMinutes"] / (
        df["TotalActiveMinutes"] + df["sedentary_minutes"] + 1e-9
    )
    df["StepsPerActiveMin"] = np.where(
        df["TotalActiveMinutes"] > 0,
        df["steps"] / df["TotalActiveMinutes"],
        0,
    )

    # HR zone alias used in feature list
    df["hr_zone_cardio"] = df["minutes_in_default_zone_2"]

    # Cast all feature columns to numeric
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
