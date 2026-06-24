import os
import json
import warnings

import numpy as np
import optuna
import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
import lightgbm as lgb

from .features import FEATURE_COLS, engineer_features
from .pipeline import build_pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "rais_anonymized", "csv_rais_anonymized",
    "daily_fitbit_sema_df_unprocessed.csv",
)

# ── Optuna search spaces ──────────────────────────────────────────────────────

def _xgb_objective(trial, X, y, groups, feature_cols, n_splits=3):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 500),
        max_depth        = trial.suggest_int("max_depth", 2, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 1.0, 10.0),
    )
    pipe = build_pipeline(
        model=XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=2),
        feature_cols=feature_cols,
    )
    gkf = GroupKFold(n_splits=n_splits)
    return cross_val_score(
        pipe, X, y, cv=gkf, groups=groups,
        scoring="neg_mean_absolute_error", n_jobs=2,
    ).mean()


def _lgbm_objective(trial, X, y, groups, feature_cols, n_splits=3):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 50, 500),
        max_depth         = trial.suggest_int("max_depth", 2, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 1.0, 10.0),
    )
    pipe = build_pipeline(
        model=lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=2),
        feature_cols=feature_cols,
    )
    gkf = GroupKFold(n_splits=n_splits)
    return cross_val_score(
        pipe, X, y, cv=gkf, groups=groups,
        scoring="neg_mean_absolute_error", n_jobs=2,
    ).mean()


def _ridge_objective(trial, X, y, groups, feature_cols, n_splits=3):
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    pipe = build_pipeline(model=Ridge(alpha=alpha), feature_cols=feature_cols)
    gkf = GroupKFold(n_splits=n_splits)
    return cross_val_score(
        pipe, X, y, cv=gkf, groups=groups,
        scoring="neg_mean_absolute_error", n_jobs=2,
    ).mean()


def _lasso_objective(trial, X, y, groups, feature_cols, n_splits=3):
    alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    pipe = build_pipeline(
        model=Lasso(alpha=alpha, max_iter=5000), feature_cols=feature_cols
    )
    gkf = GroupKFold(n_splits=n_splits)
    return cross_val_score(
        pipe, X, y, cv=gkf, groups=groups,
        scoring="neg_mean_absolute_error", n_jobs=2,
    ).mean()


_OBJECTIVES = {
    "xgboost": _xgb_objective,
    "lightgbm": _lgbm_objective,
    "ridge": _ridge_objective,
    "lasso": _lasso_objective,
}


def _build_best_model(name: str, best_params: dict):
    if name == "xgboost":
        return XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=2)
    if name == "lightgbm":
        return lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=2)
    if name == "ridge":
        return Ridge(**best_params)
    if name == "lasso":
        return Lasso(**best_params, max_iter=5000)
    raise ValueError(f"Unknown model name: {name}")


# ── Public API ────────────────────────────────────────────────────────────────

def train_best_model(
    model_name: str = "xgboost",
    feature_cols: list = None,
    n_trials: int = 30,
    n_splits: int = 3,
    data_path: str = None,
    save_dir: str = None,
) -> tuple:
    """Tune and train the best pipeline, then save it to disk.

    Parameters
    ----------
    model_name : str
        One of 'xgboost', 'lightgbm', 'ridge', 'lasso'. Default: 'xgboost'.
    feature_cols : list, optional
        Feature columns to use. Defaults to the 19-feature set.
    n_trials : int
        Number of Optuna trials. Default: 30.
    n_splits : int
        GroupKFold splits. Default: 3.
    data_path : str, optional
        Path to the daily CSV. Defaults to the project data path.
    save_dir : str, optional
        Directory to save model + metadata. Defaults to '<project_root>/saved_models'.

    Returns
    -------
    (pipeline, metadata) : tuple
        Fitted sklearn Pipeline and a dict with run info.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if data_path is None:
        data_path = DATA_PATH
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # ── Load & prep data ──────────────────────────────────────────────────────
    print(f"Loading data from {data_path} ...")
    df_raw = pd.read_csv(data_path, parse_dates=["date"])
    df = df_raw.dropna(subset=["calories"]).copy()
    df = df[df["calories"] >= 500].copy()
    df = engineer_features(df)

    X = df[["id"] + feature_cols]
    y = df["calories"]
    groups = df["id"]
    print(f"Dataset: {len(y)} rows | {df['id'].nunique()} users | {len(feature_cols)} features")

    # ── Optuna tuning ─────────────────────────────────────────────────────────
    if model_name not in _OBJECTIVES:
        raise ValueError(f"model_name must be one of {list(_OBJECTIVES.keys())}")

    print(f"Tuning {model_name} with Optuna ({n_trials} trials, GroupKFold n_splits={n_splits}) ...")
    obj_fn = _OBJECTIVES[model_name]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: obj_fn(trial, X, y, groups, feature_cols, n_splits),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )
    best_mae = -study.best_value
    best_params = study.best_params
    print(f"Best GroupKFold MAE: {best_mae:.1f} kcal  |  params: {best_params}")

    # ── Fit on all data ───────────────────────────────────────────────────────
    model = _build_best_model(model_name, best_params)
    pipe = build_pipeline(model=model, feature_cols=feature_cols)
    pipe.fit(X, y)
    print("Fitted pipeline on full dataset.")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, f"{model_name}_best.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "gkf_mae": round(best_mae, 2),
        "n_trials": n_trials,
        "n_splits": n_splits,
        "n_samples": int(len(y)),
        "n_users": int(df["id"].nunique()),
        "model_path": model_path,
    }
    meta_path = os.path.join(save_dir, f"{model_name}_best_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model → {model_path}")
    print(f"Saved metadata → {meta_path}")
    return pipe, meta
