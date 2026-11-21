"""Loading, stateless cleaning and train/test split.

Everything here is applied *before* the split on purpose: these steps do not learn
anything from the data (no statistics are computed), so they cannot leak information
from the test set. Every step that learns something lives in the pipeline (see pipeline.py).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DATA_PATH, SEED

TARGET = "Churn"
ID_COL = "customerID"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = CAT_COLS + NUM_COLS


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def prepare_features(df):
    """Return the feature columns with clean types. Used for training *and* prediction."""
    X = df[FEATURES].copy()
    # TotalCharges is read as text because 11 rows contain " " (customers with tenure = 0)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    # SeniorCitizen is coded 0/1 but is a categorical variable
    X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)
    return X


def clean_data(df, drop_duplicates=False):
    """Raw dataframe -> (X, y). y is 1 for churned customers, 0 otherwise."""
    df = df.drop(columns=ID_COL)
    if drop_duplicates:
        # Rows identical on every feature (different customers, same profile)
        df = df.drop_duplicates().reset_index(drop=True)
    X = prepare_features(df)
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def split_data(X, y, test_size=0.2, seed=SEED):
    """Stratified split, done once, before any step that learns from the data."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
