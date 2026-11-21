"""Train the churn model from the command line.

Examples:
    python run_pipeline.py
    python run_pipeline.py --metric f1 --threshold-metric f2
    python run_pipeline.py --quick
"""
import argparse

from churn.evaluation import SCORERS, THRESHOLD_SCORERS
from churn.pipeline import experiment_configs
from churn.prediction import predict
from churn.train import train


def main():
    parser = argparse.ArgumentParser(description="Customer churn training pipeline")
    parser.add_argument("--metric", default="average_precision", choices=list(SCORERS),
                        help="metric used to choose the best configuration (default: PR-AUC)")
    parser.add_argument("--threshold-metric", default="f1", choices=list(THRESHOLD_SCORERS),
                        help="metric maximized when tuning the decision threshold")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="drop rows identical on every feature (22 rows)")
    parser.add_argument("--quick", action="store_true", help="small grid, for a fast check")
    args = parser.parse_args()

    configs = None
    if args.quick:
        configs = experiment_configs(models=["logistic_regression", "hist_gradient_boosting"],
                                     transforms=("log",), scalers=("standard",),
                                     imbalance=("none", "class_weight"))
    train(metric=args.metric, threshold_metric=args.threshold_metric,
          drop_duplicates=args.drop_duplicates, configs=configs)

    example = {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 4,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0, "TotalCharges": 300.0,
    }
    print("example prediction:", predict([example]).iloc[0].to_dict())


if __name__ == "__main__":
    main()
