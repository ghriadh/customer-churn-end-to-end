"""Build a full, leak-free pipeline from a set of choices.

Each preparation choice (imputation, numeric transformation, scaler, class imbalance
strategy, model) is a parameter, so different methods can be compared fairly with the
same cross-validation. The pipeline is fitted on training data only; inside a
cross-validation it is refitted on each training fold, and resampling (over/under/SMOTE)
is applied to training folds only, never to validation or test data.
"""
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.preprocessing import (
    FunctionTransformer, MinMaxScaler, OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler,
)

from churn.config import SEED
from churn.data import CAT_COLS, NUM_COLS
from churn.transformers import GroupMedianImputer

IMPUTERS = ["group_median", "median", "zero"]
TRANSFORMS = ["none", "log", "sqrt", "yeo-johnson"]
SCALERS = ["standard", "minmax", "robust"]
IMBALANCE = ["none", "class_weight", "over", "under", "smote"]

# name -> (factory taking class_weight, is_linear, supports class_weight)
MODELS = {
    "logistic_regression": (lambda cw: LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED), True, True),
    # RidgeClassifier has no predict_proba: calibration turns its scores into probabilities
    "ridge": (lambda cw: CalibratedClassifierCV(RidgeClassifier(class_weight=cw), cv=3), True, True),
    # log_loss = logistic regression trained by stochastic gradient descent (gives probabilities)
    "sgd": (lambda cw: SGDClassifier(loss="log_loss", class_weight=cw, random_state=SEED), True, True),
    "random_forest": (lambda cw: RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=SEED), False, True),
    "hist_gradient_boosting": (lambda cw: HistGradientBoostingClassifier(class_weight=cw, random_state=SEED), False, True),
    "adaboost": (lambda cw: AdaBoostClassifier(random_state=SEED), False, False),
}

# Small hyperparameter grids, used to tune the best configuration found
PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 50]},
    "ridge": {"model__estimator__alpha": [0.1, 1, 10, 100]},
    "sgd": {"model__alpha": [1e-4, 1e-3, 1e-2], "model__penalty": ["l2", "l1"]},
    "random_forest": {"model__max_depth": [None, 8, 12], "model__min_samples_leaf": [1, 5, 10]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [None, 3, 6], "model__l2_regularization": [0, 1]},
    "adaboost": {"model__learning_rate": [0.1, 0.5, 1.0], "model__n_estimators": [50, 200]},
}


def _numeric_transform(name):
    if name == "none":
        return "passthrough"
    if name == "log":
        # log1p rather than log: tenure can be 0 and log(0) = -inf
        return FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    if name == "sqrt":
        return FunctionTransformer(np.sqrt, feature_names_out="one-to-one")
    if name == "yeo-johnson":
        # Box-Cox family that accepts zeros; parameters are learned in fit
        return PowerTransformer(method="yeo-johnson", standardize=False)
    raise ValueError(f"unknown transform: {name}")


def _scaler(name):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[name]


def _sampler(name):
    if name in ("none", "class_weight"):
        return "passthrough"
    if name == "over":
        return RandomOverSampler(random_state=SEED)
    if name == "under":
        return RandomUnderSampler(random_state=SEED)
    if name == "smote":
        return SMOTE(random_state=SEED)
    raise ValueError(f"unknown imbalance strategy: {name}")


def build_pipeline(model="logistic_regression", imputer="group_median", transform="log",
                   scaler="standard", imbalance="none"):
    factory, _, supports_cw = MODELS[model]
    if imbalance == "class_weight" and not supports_cw:
        raise ValueError(f"{model} does not support class_weight")

    if imputer == "group_median":
        group_imputer, num_imputer = GroupMedianImputer(), "passthrough"
    elif imputer == "median":
        group_imputer, num_imputer = "passthrough", SimpleImputer(strategy="median")
    elif imputer == "zero":
        # The 11 missing TotalCharges all have tenure = 0: new customers who have paid nothing yet
        group_imputer, num_imputer = "passthrough", SimpleImputer(strategy="constant", fill_value=0)
    else:
        raise ValueError(f"unknown imputer: {imputer}")

    numeric = Pipeline([
        ("impute", num_imputer),
        ("transform", _numeric_transform(transform)),
        ("scale", _scaler(scaler)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CAT_COLS),
    ], verbose_feature_names_out=False)

    return Pipeline([
        ("group_impute", group_imputer),
        ("preprocess", preprocess),
        ("resample", _sampler(imbalance)),
        ("model", factory("balanced" if imbalance == "class_weight" else None)),
    ])


def experiment_configs(models=None, imputers=("group_median",), transforms=("none", "log", "yeo-johnson"),
                       scalers=("standard", "robust"), imbalance=("none", "class_weight", "over", "smote")):
    """All combinations to compare. Tree models are insensitive to monotonic transforms and
    scaling, so for them only the imputation and imbalance strategies are varied."""
    configs = []
    for model in models or MODELS:
        _, is_linear, supports_cw = MODELS[model]
        for imp in imputers:
            for tr in (transforms if is_linear else ("none",)):
                for sc in (scalers if is_linear else ("standard",)):
                    for imb in imbalance:
                        if imb == "class_weight" and not supports_cw:
                            continue
                        configs.append(dict(model=model, imputer=imp, transform=tr, scaler=sc, imbalance=imb))
    return configs
