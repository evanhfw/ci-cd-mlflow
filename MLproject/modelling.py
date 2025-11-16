import pandas as pd
import numpy as np
import argparse
import dagshub
import mlflow
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Optional
from pathlib import Path

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(Path("preprocessed/X_train.csv"))
    X_test = pd.read_csv(Path("preprocessed/X_test.csv"))
    X_train = _ensure_float_features(X_train)
    X_test = _ensure_float_features(X_test)
    y_train = pd.read_csv(Path("preprocessed/y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(Path("preprocessed/y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def _ensure_float_features(features: pd.DataFrame) -> pd.DataFrame:
    integer_columns = features.select_dtypes(include=[np.integer]).columns
    if not integer_columns.empty:
        features[integer_columns] = features[integer_columns].astype("float64")
    return features

def setup_mlflow() -> None:
    dagshub.init(repo_owner='evanhfw', repo_name='MSML_Evan', mlflow=True)
    mlflow.autolog()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest model with hyperparameters")
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest (default: 100)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of the tree (default: None, unlimited)"
    )
    return parser.parse_args()

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None
) -> None:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    input_example = X_train[0:5]

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

def main() -> None:
    args = parse_args()
    # setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()
    train_model(X_train, y_train, args.n_estimators, args.max_depth)

if __name__ == "__main__":
    main()