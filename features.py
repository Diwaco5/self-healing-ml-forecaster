import pandas as pd

FEATURE_COLUMNS = [
    "ambient_temp",
    "industrial_activity",
    "grid_price_index",
    "is_weekend",
    "hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_rolling_mean_24h",
]
TARGET_COLUMN = "demand_mwh"


def build_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Create leak-free temporal features sorted strictly by timestamp."""
    data = df.sort_values("timestamp").copy()

    data["demand_lag_1h"] = data["demand_mwh"].shift(1)
    data["demand_lag_24h"] = data["demand_mwh"].shift(24)
    data["demand_rolling_mean_24h"] = data["demand_mwh"].shift(1).rolling(window=24).mean()

    if is_training:
        data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    else:
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # Enforce float64 across all features for strict MLflow schema compatibility
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].astype("float64")
    return data
