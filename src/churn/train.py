"""End-to-end training: split -> compare methods (CV) -> tune best -> tune threshold -> test once -> save."""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV

from churn.config import ARTIFACTS_DIR, CONFUSION_MATRIX_PATH, METRICS_PATH, MODEL_PATH, RESULTS_PATH, SEED
from churn.data import clean_data, load_data, split_data
from churn.evaluation import (
    SCORERS, THRESHOLD_SCORERS, baseline_scores, compare_configs, make_cv, test_metrics,
)
from churn.pipeline import PARAM_GRIDS, build_pipeline, experiment_configs

CONFIG_KEYS = ["model", "imputer", "transform", "scaler", "imbalance"]


def train(metric="average_precision", threshold_metric="f1", drop_duplicates=False, configs=None, verbose=True):
    if metric not in SCORERS:
        raise ValueError(f"metric must be one of {list(SCORERS)}")
    if threshold_metric not in THRESHOLD_SCORERS:
        raise ValueError(f"threshold_metric must be one of {list(THRESHOLD_SCORERS)}")
    log = print if verbose else (lambda *a, **k: None)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Clean (stateless) and split FIRST: the test set is not touched until step 6
    X, y = clean_data(load_data(), drop_duplicates=drop_duplicates)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = make_cv()
    log(f"train: {len(X_train)} rows, test: {len(X_test)} rows, churn rate: {y.mean():.1%}")

    # 2. Baseline and comparison of all preparation/model combinations (CV on train only)
    baseline = baseline_scores(X_train, y_train, cv)
    results = compare_configs(configs or experiment_configs(), X_train, y_train, cv, verbose=verbose)
    results = results.sort_values(metric, ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    # 3. Tune the hyperparameters of the best configuration (CV on train only)
    best_config = results.loc[0, CONFIG_KEYS].to_dict()
    log(f"best configuration by {metric}: {best_config}")
    search = GridSearchCV(build_pipeline(**best_config), PARAM_GRIDS[best_config["model"]],
                          scoring=SCORERS[metric], cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    log(f"best hyperparameters: {search.best_params_} ({metric} {search.best_score_:.3f})")

    # 4. Tune the decision threshold (default 0.5 is rarely optimal with imbalanced classes)
    model = TunedThresholdClassifierCV(search.best_estimator_, scoring=THRESHOLD_SCORERS[threshold_metric],
                                       cv=cv, random_state=SEED)
    model.fit(X_train, y_train)
    log(f"decision threshold tuned for {threshold_metric}: {model.best_threshold_:.3f}")

    # 5. Final evaluation, ONCE, on the untouched test set
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    pred_default = (proba >= 0.5).astype(int)
    metrics = {
        "selection_metric": metric,
        "threshold_metric": threshold_metric,
        "drop_duplicates": drop_duplicates,
        "best_config": best_config,
        "best_params": search.best_params_,
        "threshold": float(model.best_threshold_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_cv": {k: baseline[k] for k in SCORERS},
        "test": test_metrics(y_test, pred, proba),
        "test_threshold_0.5": test_metrics(y_test, pred_default, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    log("test metrics:", {k: round(v, 3) for k, v in metrics["test"].items()})

    # 6. Save artifacts
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No churn", "Churn"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Test set - {best_config['model']} (threshold {model.best_threshold_:.2f})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close()
    return model, results, metrics
