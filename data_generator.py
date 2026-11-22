import numpy as np
import pandas as pd
from src.config import settings


def generate_dataset(seed: int = settings.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 9 months of hourly demand data split into Reference (6m) and Stream (3m)."""
    rng = np.random.default_rng(seed)

    total_hours = 24 * 30 * 9
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=total_hours, freq="h")

    hours = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    month_idx = np.arange(total_hours) // (24 * 30)

    daily_temp_cycle = 10.0 * np.sin(2 * np.pi * (hours - 8) / 24)
    seasonal_temp = 65.0 + 15.0 * np.sin(2 * np.pi * np.arange(total_hours) / (24 * 365))
    ambient_temp = seasonal_temp + daily_temp_cycle + rng.normal(0, 2.5, total_hours)

    industrial_activity = np.where(is_weekend == 1, 55.0, 85.0) + rng.normal(0, 5.0, total_hours)
    grid_price_index = 40.0 + 0.3 * industrial_activity + rng.normal(0, 3.0, total_hours)

    drift_mask = month_idx >= 7
    ambient_temp[drift_mask] += 18.0
    grid_price_index[drift_mask] *= 1.35

    base_demand = (
        120.0
        + 1.8 * industrial_activity
        + 0.9 * np.maximum(ambient_temp - 68.0, 0) ** 1.2
        - 0.4 * grid_price_index
        + rng.normal(0, 6.0, total_hours)
    )

    concept_shift = np.where(
        drift_mask,
        1.4 * np.maximum(ambient_temp - 72.0, 0) ** 1.35 - 15.0,
        0.0,
    )
    demand_mwh = np.round(base_demand + concept_shift, 2)

    regime_labels = np.select(
        [month_idx < 6, month_idx == 6, month_idx == 7, month_idx >= 8],
        ["reference", "month_7_normal", "month_8_drift", "month_9_post_drift"],
        default="reference",
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ambient_temp": np.round(ambient_temp, 2),
            "industrial_activity": np.round(industrial_activity, 2),
            "grid_price_index": np.round(grid_price_index, 2),
            "is_weekend": is_weekend,
            "hour": hours,
            "regime": regime_labels,
            "demand_mwh": demand_mwh,
        }
    )

    ref_df = df[df["regime"] == "reference"].reset_index(drop=True)
    stream_df = df[df["regime"] != "reference"].reset_index(drop=True)
    return ref_df, stream_df


def save_datasets() -> None:
    ref_df, stream_df = generate_dataset()
    ref_df.to_parquet(settings.REFERENCE_DATA_PATH, index=False)
    stream_df.to_parquet(settings.STREAM_DATA_PATH, index=False)
    print(f"Saved reference data: {ref_df.shape} -> {settings.REFERENCE_DATA_PATH}")
    print(f"Saved stream data:    {stream_df.shape} -> {settings.STREAM_DATA_PATH}")


if __name__ == "__main__":
    save_datasets()
