import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerUserMedianImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using per-user medians computed only from training data.

    Expects 'id' as the first column; drops it before returning the array.
    Falls back to global median for users unseen during fit.
    """

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        cols = ["id"] + list(self.feature_names_in_)
        return pd.DataFrame(X, columns=cols)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [c for c in X.columns if c != "id"]
            X_df = X
        else:
            if self.feature_cols is not None:
                self.feature_names_in_ = list(self.feature_cols)
            cols = ["id"] + list(self.feature_names_in_)
            X_df = pd.DataFrame(X, columns=cols)

        self.impute_cols_ = [c for c in self.feature_names_in_ if c in X_df.columns]
        self.user_medians_ = {}
        self.global_medians_ = {}
        for col in self.impute_cols_:
            self.user_medians_[col] = X_df.groupby("id")[col].median().to_dict()
            gm = X_df[col].median()
            self.global_medians_[col] = 0.0 if pd.isna(gm) else gm
        return self

    def transform(self, X):
        X_df = self._to_df(X)
        for col in self.impute_cols_:
            mask = X_df[col].isna()
            if mask.any():
                X_df.loc[mask, col] = (
                    X_df.loc[mask, "id"]
                    .map(self.user_medians_[col])
                    .fillna(self.global_medians_[col])
                )
        X_df[self.impute_cols_] = X_df[self.impute_cols_].fillna(0)
        return X_df.drop(columns=["id"]).values
