"""Predict churn for new customers with the saved model (preparation pipeline + model + threshold)."""
import joblib
import pandas as pd

from churn.config import MODEL_PATH
from churn.data import prepare_features


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(customers, model=None):
    """`customers`: DataFrame (or list of dicts) with the raw dataset columns.
    Returns the churn probability and the predicted label (using the tuned threshold)."""
    model = model or load_model()
    X = prepare_features(pd.DataFrame(customers))
    return pd.DataFrame({
        "churn_probability": model.predict_proba(X)[:, 1],
        "churn": model.predict(X).astype(bool),
    }, index=X.index)
