"""End-to-end training.

split -> compare all configurations (CV on train) -> for each algorithm, tune its best
configuration and its decision threshold (CV on train) -> evaluate each final model once
on the untouched test set -> save models, results and interpretation data.

The "selected" model is the algorithm with the best cross-validated score for the chosen
metric; the test set is never used to choose between models.
"""
import json
import os

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV

from churn.config import (
    ARTIFACTS_DIR, COEFFICIENTS_PATH, CONFUSION_DIR, IMPORTANCE_PATH, METRICS_PATH, MODELS_DIR, RESULTS_PATH, SEED,
    TEST_PREDICTIONS_PATH, model_path,
)
from churn.data import FEATURES, clean_data, load_data, split_data
from churn.evaluation import (
    SCORERS, THRESHOLD_SCORERS, baseline_scores, compare_configs, make_cv, test_metrics,
)
from churn.pipeline import PARAM_GRIDS, build_pipeline, experiment_configs

CONFIG_KEYS = ["model", "imputer", "transform", "scaler", "imbalance"]


def _logreg_coefficients(model):
    """Signed coefficients of a fitted logistic regression (inside the tuned-threshold wrapper)."""
    pipe = model.estimator_
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefs}).sort_values("coefficient", ascending=False)


def _save_confusion_png(cm, title, path):
    disp = ConfusionMatrixDisplay(np.asarray(cm, dtype=int), display_labels=["No churn", "Churn"])
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def train(metric="average_precision", threshold_metric="f1", drop_duplicates=False, configs=None,
          models=None, verbose=True):
    if metric not in SCORERS:
        raise ValueError(f"metric must be one of {list(SCORERS)}")
    if threshold_metric not in THRESHOLD_SCORERS:
        raise ValueError(f"threshold_metric must be one of {list(THRESHOLD_SCORERS)}")
    log = print if verbose else (lambda *a, **k: None)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CONFUSION_DIR, exist_ok=True)

    # 1. Stateless cleaning, then split: the test set is not used until step 4
    X, y = clean_data(load_data(), drop_duplicates=drop_duplicates)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = make_cv()
    log(f"train: {len(X_train)} rows, test: {len(X_test)} rows, churn rate: {y.mean():.1%}")

    # 2. Baseline and comparison of every configuration (CV on train)
    baseline = baseline_scores(X_train, y_train, cv)
    results = compare_configs(configs or experiment_configs(models=models), X_train, y_train, cv, verbose=verbose)
    results = results.sort_values(metric, ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)

    # 3. For each algorithm: tune hyperparameters and threshold of its best configuration (CV on train)
    summary, predictions, importances = {}, {"y_true": y_test.to_numpy()}, []
    for name in results["model"].unique():
        best_config = results[results["model"] == name].iloc[0][CONFIG_KEYS].to_dict()
        search = GridSearchCV(build_pipeline(**best_config), PARAM_GRIDS[name],
                              scoring=SCORERS[metric], cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = TunedThresholdClassifierCV(search.best_estimator_, scoring=THRESHOLD_SCORERS[threshold_metric],
                                           cv=cv, random_state=SEED)
        model.fit(X_train, y_train)

        # 4. Evaluation on the test set (for information: selection is based on CV only)
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        summary[name] = {
            "config": best_config,
            "params": search.best_params_,
            "cv_score": float(search.best_score_),
            "threshold": float(model.best_threshold_),
            "test": test_metrics(y_test, pred, proba),
            "test_threshold_0.5": test_metrics(y_test, (proba >= 0.5).astype(int), proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        predictions[name] = proba
        joblib.dump(model, model_path(name), compress=3)
        row = results[results["model"] == name].iloc[0]
        _save_confusion_png([[row.tn, row.fp], [row.fn, row.tp]], f"{name.replace('_', ' ').title()} - cross-validation (threshold 0.5)",
                            os.path.join(CONFUSION_DIR, f"cv_{name}.png"))
        _save_confusion_png(summary[name]["confusion_matrix"],
                            f"{name.replace('_', ' ').title()} - test set (threshold {model.best_threshold_:.2f})",
                            os.path.join(CONFUSION_DIR, f"test_{name}.png"))

        # Model-agnostic importance of each original feature (drop in PR-AUC when it is shuffled)
        imp = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                     n_repeats=10, random_state=SEED, n_jobs=-1)
        importances.append(pd.DataFrame({"model": name, "feature": FEATURES,
                                         "importance": imp.importances_mean, "std": imp.importances_std}))
        if name == "logistic_regression":
            _logreg_coefficients(model).to_csv(COEFFICIENTS_PATH, index=False)
        log(f"{name}: {metric} (CV) {search.best_score_:.3f} | threshold {model.best_threshold_:.2f} | "
            f"test recall {summary[name]['test']['recall']:.3f} f1 {summary[name]['test']['f1']:.3f}")

    selected = max(summary, key=lambda n: summary[n]["cv_score"])
    log(f"selected model (best {metric} in CV): {selected}")

    # 5. Save results
    pd.DataFrame(predictions).to_csv(TEST_PREDICTIONS_PATH, index=False)
    pd.concat(importances).to_csv(IMPORTANCE_PATH, index=False)
    metrics = {
        "selection_metric": metric,
        "threshold_metric": threshold_metric,
        "drop_duplicates": drop_duplicates,
        "selected_model": selected,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate": float(y.mean()),
        "baseline_cv": {k: baseline[k] for k in SCORERS},
        "models": summary,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics
