import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error

from flows.retraining_flow import self_healing_pipeline
from src.api.main import app, load_champion_model
from src.config import settings
from src.features import build_features


def replay_regime(client: TestClient, df: pd.DataFrame, label: str) -> float:
    """Send rows through FastAPI /predict and compute live MAE."""
    actuals = []
    preds = []

    for _, row in df.iterrows():
        payload = {
            "timestamp": str(row["timestamp"]),
            "ambient_temp": float(row["ambient_temp"]),
            "industrial_activity": float(row["industrial_activity"]),
            "grid_price_index": float(row["grid_price_index"]),
            "is_weekend": int(row["is_weekend"]),
            "hour": int(row["hour"]),
            "demand_lag_1h": float(row["demand_lag_1h"]),
            "demand_lag_24h": float(row["demand_lag_24h"]),
            "demand_rolling_mean_24h": float(row["demand_rolling_mean_24h"]),
            "actual_demand_mwh": float(row["demand_mwh"]),
        }
        res = client.post("/predict", json=payload)
        res.raise_for_status()
        body = res.json()

        actuals.append(float(row["demand_mwh"]))
        preds.append(float(body["predicted_demand_mwh"]))

    mae = float(mean_absolute_error(actuals, preds))
    print(f"[{label}] Rows: {len(df)} | Active Model: {body['model_version']} | Live MAE: {mae:.2f} MWh")
    return mae


def run_simulation() -> None:
    load_champion_model()
    client = TestClient(app)

    stream_raw = pd.read_parquet(settings.STREAM_DATA_PATH)
    stream_df = build_features(stream_raw, is_training=True)

    m7 = stream_df[stream_df["regime"] == "month_7_normal"]
    m8 = stream_df[stream_df["regime"] == "month_8_drift"]
    m9 = stream_df[stream_df["regime"] == "month_9_post_drift"]

    print("\n--- PHASE 1: MONTH 7 (NORMAL PRODUCTION TRAFFIC) ---")
    replay_regime(client, m7, "Month 7 - Normal")
    self_healing_pipeline(window_size=500)

    print("\n--- PHASE 2: MONTH 8 (HEATWAVE + CONCEPT DRIFT INJECTED) ---")
    replay_regime(client, m8, "Month 8 - Drifted (v1)")
    result = self_healing_pipeline(window_size=600)

    if result.get("promoted"):
        client.post("/reload-model")

    print("\n--- PHASE 3: MONTH 9 (POST-HEALING RECOVERY) ---")
    replay_regime(client, m9, "Month 9 - Recovered (v2)")


if __name__ == "__main__":
    run_simulation()
