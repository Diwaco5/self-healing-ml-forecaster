import pandas as pd
from prefect import flow, task
from sqlalchemy import create_engine

from src.config import settings
from src.evaluate import evaluate_and_promote
from src.monitor import run_drift_check
from src.train import train_model


@task(name="Check Production Drift")
def detect_drift_task(window_size: int = 500) -> dict:
    report = run_drift_check(window_size=window_size)
    print(
        f"[Monitor] Drift Share: {report['drift_share']:.1%} "
        f"(Threshold: {settings.DRIFT_SHARE_THRESHOLD:.1%}) -> Detected: {report['drift_detected']}"
    )
    return report


@task(name="Fetch Labeled Production Window")
def fetch_retraining_data_task(window_size: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = create_engine(settings.DB_URI)
    query = f"""
        SELECT timestamp, ambient_temp, industrial_activity, grid_price_index,
               is_weekend, hour, actual_demand_mwh AS demand_mwh
        FROM prediction_logs
        WHERE actual_demand_mwh IS NOT NULL
        ORDER BY id DESC
        LIMIT {window_size}
    """
    recent_df = pd.read_sql(query, engine).sort_values("timestamp").reset_index(drop=True)

    # First 75% of recent shifted data for retraining; last 25% for holdout validation gate
    split_idx = int(len(recent_df) * 0.75)
    retrain_df = recent_df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = recent_df.iloc[split_idx:].reset_index(drop=True)
    return retrain_df, holdout_df


@task(name="Train Challenger Model")
def train_challenger_task(retrain_df: pd.DataFrame) -> str:
    run_id, val_mae = train_model(retrain_df, alias="challenger")
    return run_id


@task(name="Evaluate & Promote Gatekeeper")
def gatekeeper_task(holdout_df: pd.DataFrame) -> bool:
    return evaluate_and_promote(holdout_df)


@flow(name="Self-Healing Demand Forecaster Pipeline")
def self_healing_pipeline(window_size: int = 600) -> dict:
    drift_report = detect_drift_task(window_size=window_size)

    if not drift_report.get("drift_detected", False):
        print("[Flow] System healthy. No retraining required.")
        return {"action": "none", "promoted": False}

    print("[Flow] Drift threshold breached! Triggering automated retraining...")
    retrain_df, holdout_df = fetch_retraining_data_task(window_size=window_size)
    train_challenger_task(retrain_df)
    promoted = gatekeeper_task(holdout_df)

    return {"action": "retrained", "promoted": promoted}


if __name__ == "__main__":
    self_healing_pipeline()
