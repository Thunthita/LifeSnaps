"""
predict.py — Load saved model and predict calories
Usage: python predict.py
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# ── Available models ──────────────────────────────────────────────────────────
MODEL_FILES = {
    'Gradient Boosting': 'gradient_boosting_gkf_tuned.joblib',
    'Random Forest':     'random_forest_gkf_tuned.joblib',
    'XGBoost':           'xgboost_gkf_tuned.joblib',
    'LightGBM':          'lightgbm_gkf_tuned.joblib',
    'Ridge':             'ridge_gkf_tuned.joblib',
    'Lasso':             'lasso_gkf_tuned.joblib',
    'ElasticNet':        'elasticnet_gkf_tuned.joblib',
    'Linear Regression': 'linear_regression_gkf_tuned.joblib',
}


def compute_engineered_features(row: dict) -> dict:
    """Compute the 3 engineered features from raw inputs."""
    total_active = (row['very_active_minutes']
                    + row['moderately_active_minutes']
                    + row['lightly_active_minutes'])
    sedentary = row.get('sedentary_minutes', 0)
    row['TotalActiveMinutes']  = total_active
    row['ActiveRatio']         = total_active / (total_active + sedentary + 1e-9)
    row['StepsPerActiveMin']   = row['steps'] / total_active if total_active > 0 else 0
    return row


def predict(raw_inputs: dict, model_name: str = 'Gradient Boosting') -> float:
    """
    Predict calories for one day's data.

    Parameters
    ----------
    raw_inputs : dict
        Must include:
          steps, distance, very_active_minutes, moderately_active_minutes,
          lightly_active_minutes, bpm, resting_hr, hr_zone_cardio,
          filteredDemographicVO2Max, nightly_temperature,
          daily_temperature_variation, bmi
        Optional:
          sedentary_minutes (used for ActiveRatio, default 0)
          id (user identifier, default 'new_user')
    model_name : str
        One of the keys in MODEL_FILES. Default: 'Gradient Boosting'

    Returns
    -------
    float : predicted calories (kcal)
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES)}")

    model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    pipe = joblib.load(model_path)

    row = dict(raw_inputs)
    row.setdefault('id', 'new_user')
    row.setdefault('sedentary_minutes', 0)
    row = compute_engineered_features(row)

    feature_cols = [
        'steps', 'distance', 'very_active_minutes', 'moderately_active_minutes',
        'lightly_active_minutes', 'TotalActiveMinutes', 'ActiveRatio',
        'StepsPerActiveMin', 'bpm', 'resting_hr', 'hr_zone_cardio',
        'filteredDemographicVO2Max', 'nightly_temperature',
        'daily_temperature_variation', 'bmi',
    ]

    df = pd.DataFrame([row])[['id'] + feature_cols]
    return float(pipe.predict(df)[0])


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    example = {
        'id':                        'user_001',
        'steps':                     8000,
        'distance':                  6.1,
        'very_active_minutes':       30,
        'moderately_active_minutes': 20,
        'lightly_active_minutes':    60,
        'sedentary_minutes':         800,
        'bpm':                       72,
        'resting_hr':                58,
        'hr_zone_cardio':            15,
        'filteredDemographicVO2Max': 42.0,
        'nightly_temperature':       0.1,
        'daily_temperature_variation': 0.3,
        'bmi':                       22.5,
    }

    print('=== Calorie Prediction Example ===')
    print(f'Input: {example["steps"]} steps, {example["very_active_minutes"]} very active min, BMI {example["bmi"]}')
    print()

    for model_name in MODEL_FILES:
        cal = predict(example, model_name=model_name)
        print(f'  {model_name:<22} → {cal:.0f} kcal')
