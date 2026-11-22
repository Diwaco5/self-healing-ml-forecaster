import pandas as pd
from src.data_generator import generate_dataset
from src.features import FEATURE_COLUMNS, build_features


def test_no_future_data_leakage() -> None:
    """Modifying target values at t >= 100 must not alter features at t < 100."""
    ref_df, _ = generate_dataset(seed=42)
    sample = ref_df.iloc[:150].copy()

    features_clean = build_features(sample, is_training=True)

    corrupted = sample.copy()
    corrupted.loc[100:, "demand_mwh"] = 999999.0
    features_corrupted = build_features(corrupted, is_training=True)

    cutoff_ts = sample.loc[99, "timestamp"]
    clean_past = features_clean[features_clean["timestamp"] <= cutoff_ts][FEATURE_COLUMNS]
    corrupted_past = features_corrupted[features_corrupted["timestamp"] <= cutoff_ts][FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(clean_past, corrupted_past)
