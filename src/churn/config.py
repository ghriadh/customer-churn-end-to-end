"""Project paths, resolved from the repository root so code works from any working directory."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(ROOT_DIR, "data", "customer_churn.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
APP_CSS_PATH = os.path.join(ROOT_DIR, "app", "style.css")
