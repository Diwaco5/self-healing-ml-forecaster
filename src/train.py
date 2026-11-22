import argparse
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import settings
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features


def train_model(
    df: pd.DataFrame,
    alias: str = "champion",
    n_estimators: int = 150,
    learning_rate: float = 0.05,
) -> tuple[str, float]:
    """Train LightGBM on chronological split, register in MLflow, and assign alias."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    featured_df = build_features(df, is_training=True)

    # Strict chronological 80/20 split (never shuffle time series)
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    val_df = featured_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]

    with mlflow.start_run(run_name=f"lgbm-{alias}") as run:
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "random_state": settings.RANDOM_SEED,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        rmse = float(root_mean_squared_error(y_val, preds))

        mlflow.log_params(params)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)

        model_info = mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=settings.MODEL_NAME,
            input_example=X_train.iloc[:3],
        )

        # Assign alias (@champion or @challenger) to the newly registered version
        version = model_info.registered_model_version
        client.set_registered_model_alias(
            name=settings.MODEL_NAME,
            alias=alias,
            version=str(version),
        )
        print(f"Registered {settings.MODEL_NAME} v{version} as @{alias} | Val MAE: {mae:.3f}")
        return run.info.run_id, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", type=str, default="champion")
    args = parser.parse_args()

    ref_df = pd.read_parquet(settings.REFERENCE_DATA_PATH)
    train_model(ref_df, alias=args.alias)
