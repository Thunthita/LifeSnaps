from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .preprocessing import PerUserMedianImputer
from .features import FEATURE_COLS


def build_pipeline(model=None, feature_cols=None) -> Pipeline:
    """Return a full sklearn Pipeline: PerUserMedianImputer → StandardScaler → model.

    Parameters
    ----------
    model : sklearn estimator, optional
        Defaults to XGBRegressor (best model from notebook 02).
    feature_cols : list of str, optional
        Defaults to the 19-feature set from FEATURE_COLS.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if model is None:
        model = XGBRegressor(random_state=42, verbosity=0, n_jobs=2)

    return Pipeline([
        ("imputer", PerUserMedianImputer(feature_cols=feature_cols)),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])
