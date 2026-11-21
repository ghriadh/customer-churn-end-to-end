"""Custom scikit-learn transformers.

Rule followed by every class: `fit` learns from the training data only,
`transform` applies what was learned, and the input is never modified in place.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill missing values of `column` with its median within groups (e.g. Contract x OnlineSecurity).

    Medians are learned in `fit` (on the training folds), then reused in `transform`.
    Groups never seen during `fit` fall back to the global median.
    """

    def __init__(self, column="TotalCharges", groupby=("Contract", "OnlineSecurity")):
        self.column = column
        self.groupby = groupby

    def fit(self, X, y=None):
        keys = list(self.groupby)
        self.medians_ = X.groupby(keys)[self.column].median()
        self.global_median_ = X[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        missing = X[self.column].isna()
        if missing.any():
            keys = list(self.groupby)
            groups = list(X.loc[missing, keys].itertuples(index=False, name=None))
            fill = self.medians_.reindex(groups).to_numpy(dtype=float)
            X.loc[missing, self.column] = np.where(np.isnan(fill), self.global_median_, fill)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
