"""Metrics, cross-validated comparison of configurations, and final test evaluation."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score, make_scorer,
    precision_score, recall_score, roc_auc_score, balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn.config import SEED
from churn.pipeline import build_pipeline

# Metrics computed for every configuration. "average_precision" is the PR-AUC.
SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}
# Criteria available to tune the decision threshold (they trade precision against recall)
THRESHOLD_SCORERS = {
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),  # weighs recall twice as much as precision
    "balanced_accuracy": "balanced_accuracy",
}


def make_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cross_validate_config(estimator, X, y, cv):
    scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORERS, n_jobs=-1)
    row = {}
    for name in SCORERS:
        row[name] = scores[f"test_{name}"].mean()
        row[f"{name}_std"] = scores[f"test_{name}"].std()
    row["fit_time"] = scores["fit_time"].mean()
    return row


def compare_configs(configs, X, y, cv, verbose=True):
    """Cross-validate each configuration on the training set only. Returns one row per config."""
    rows = []
    for i, config in enumerate(configs, 1):
        row = {**config, **cross_validate_config(build_pipeline(**config), X, y, cv)}
        rows.append(row)
        if verbose:
            print(f"[{i}/{len(configs)}] {config} -> PR-AUC {row['average_precision']:.3f}  F1 {row['f1']:.3f}")
    return pd.DataFrame(rows)


def baseline_scores(X, y, cv):
    """Reference point: always predict the majority class (no churn)."""
    return cross_validate_config(DummyClassifier(strategy="most_frequent"), X, y, cv)


def test_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }
