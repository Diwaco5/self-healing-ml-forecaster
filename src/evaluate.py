import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from src.config import settings
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features


def evaluate_and_promote(eval_df: pd.DataFrame) -> bool:
    """Compare @challenger vs @champion on holdout data; promote only if MAE improves."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    featured = build_features(eval_df, is_training=True)
    X_eval, y_eval = featured[FEATURE_COLUMNS], featured[TARGET_COLUMN]

    champion_uri = f"models:/{settings.MODEL_NAME}@champion"
    challenger_uri = f"models:/{settings.MODEL_NAME}@challenger"

    champion_model = mlflow.pyfunc.load_model(champion_uri)
    challenger_model = mlflow.pyfunc.load_model(challenger_uri)

    champ_mae = float(mean_absolute_error(y_eval, champion_model.predict(X_eval)))
    chall_mae = float(mean_absolute_error(y_eval, challenger_model.predict(X_eval)))

    improvement = (champ_mae - chall_mae) / champ_mae
    print(f"Champion MAE: {champ_mae:.3f} | Challenger MAE: {chall_mae:.3f} | Improvement: {improvement:.2%}")

    if improvement >= settings.MIN_IMPROVEMENT_PCT:
        challenger_meta = client.get_model_version_by_alias(settings.MODEL_NAME, "challenger")
        client.set_registered_model_alias(
            name=settings.MODEL_NAME,
            alias="champion",
            version=challenger_meta.version,
        )
        print(f"PROMOTED: Version {challenger_meta.version} is now @champion!")
        return True

    print("REJECTED: Challenger did not meet improvement threshold. Retaining current @champion.")
    return False
