import os
import joblib
import pandas as pd
import numpy as np

from .features import FEATURE_COLS, engineer_features


def load_model(model_path: str):
    """Load a saved pipeline from a .joblib file.

    Parameters
    ----------
    model_path : str
        Path to a .joblib file saved by train_best_model() or the notebook.

    Returns
    -------
    sklearn Pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_calories(
    pipe,
    df: pd.DataFrame,
    feature_cols: list = None,
    apply_engineering: bool = True,
) -> np.ndarray:
    """Predict daily calorie burn for new data.

    Parameters
    ----------
    pipe : sklearn Pipeline
        A fitted pipeline (from load_model or train_best_model).
    df : pd.DataFrame
        Raw daily data. Must contain 'id', 'date', and the raw Fitbit columns
        needed by engineer_features if apply_engineering=True.
    feature_cols : list, optional
        Feature columns used during training. Defaults to FEATURE_COLS (19 features).
    apply_engineering : bool
        Whether to run engineer_features() on df first. Set False if already done.

    Returns
    -------
    np.ndarray of predicted calories (kcal).

    Example
    -------
    >>> pipe = load_model("saved_models/xgboost_best.joblib")
    >>> preds = predict_calories(pipe, new_df)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if apply_engineering:
        df = engineer_features(df)

    X = df[["id"] + feature_cols]
    return pipe.predict(X)
