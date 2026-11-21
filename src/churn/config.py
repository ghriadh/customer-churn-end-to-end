"""Project paths and constants, resolved from the repository root so code works from any directory."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(ROOT_DIR, "data", "customer_churn.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
APP_CSS_PATH = os.path.join(ROOT_DIR, "app", "style.css")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "cv_results.csv")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")

SEED = 1234
