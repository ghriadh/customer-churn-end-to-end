"""Predict churn for new customers with a saved model (preparation pipeline + model + tuned threshold)."""
import json

import joblib
import pandas as pd

from churn.config import METRICS_PATH, model_path
from churn.data import prepare_features


def selected_model_name():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)["selected_model"]


def load_model(name=None):
    return joblib.load(model_path(name or selected_model_name()))


def predict(customers, model=None):
    """`customers`: DataFrame (or list of dicts) with the raw dataset columns.
    Returns the churn probability and the predicted label (using the tuned threshold)."""
    model = model or load_model()
    X = prepare_features(pd.DataFrame(customers))
    return pd.DataFrame({
        "churn_probability": model.predict_proba(X)[:, 1],
        "churn": model.predict(X).astype(bool),
    }, index=X.index)
