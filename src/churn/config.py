"""Project paths and constants, resolved from the repository root so code works from any directory."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(ROOT_DIR, "data", "customer_churn.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "cv_results.csv")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TEST_PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "test_predictions.csv")
IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")
COEFFICIENTS_PATH = os.path.join(ARTIFACTS_DIR, "logreg_coefficients.csv")


def model_path(name):
    return os.path.join(MODELS_DIR, f"{name}.pkl")

SEED = 1234
