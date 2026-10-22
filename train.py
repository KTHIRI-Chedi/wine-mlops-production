"""
Training script for Wine Quality model using ElasticNet.
Implements prepare_features_train(df) as a pure function.
"""
from typing import Tuple
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"


def prepare_features_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pure feature transformation function for training data.
    Separates the 11 feature inputs from the target quality score.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def build_pipeline(alpha: float = 0.5, l1_ratio: float = 0.5) -> Pipeline:
    """Builds the standardized ElasticNet regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)),
        ]
    )


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/winequality-red.csv"
    print(f"Reading dataset from {data_path}...")
    df = pd.read_csv(data_path, sep=";")
    X, y = prepare_features_train(df)
    pipe = build_pipeline()
    pipe.fit(X, y)
    print(f"Trained baseline ElasticNet model successfully on {len(X)} rows.")
