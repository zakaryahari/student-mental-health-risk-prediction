import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - 1.5 * iqr
        self.upper_bounds_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.lower_bounds_, self.upper_bounds_, axis=1).to_numpy()
