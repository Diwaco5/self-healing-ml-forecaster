import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import create_engine

from src.config import settings
from src.features import FEATURE_COLUMNS, build_features


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and production arrays."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Clip zeros to avoid division by zero / log(0)
    expected_pct = np.clip(expected_counts, 1e-4, 1.0)
    actual_pct = np.clip(actual_counts, 1e-4, 1.0)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(psi, 4)


def run_drift_check(window_size: int = 500) -> dict:
    """Compare recent production logs against reference training distribution."""
    ref_raw = pd.read_parquet(settings.REFERENCE_DATA_PATH)
    ref_df = build_features(ref_raw, is_training=True)

    engine = create_engine(settings.DB_URI)
    query = f"SELECT * FROM prediction_logs ORDER BY id DESC LIMIT {window_size}"
    prod_df = pd.read_sql(query, engine)

    if len(prod_df) < 50:
        return {"drift_detected": False, "reason": "Insufficient production samples", "drift_share": 0.0}

    feature_reports = {}
    drifted_count = 0

    for col in FEATURE_COLUMNS:
        ref_vals = ref_df[col].to_numpy(dtype=float)
        prod_vals = prod_df[col].to_numpy(dtype=float)

        ks_stat, p_value = ks_2samp(ref_vals, prod_vals)
        psi_val = calculate_psi(ref_vals, prod_vals)

        # PSI >= 0.20 indicates significant distribution shift
        is_drifted = bool(psi_val >= 0.20 and p_value < 0.01)
        if is_drifted:
            drifted_count += 1

        feature_reports[col] = {
            "psi": psi_val,
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drifted": is_drifted,
        }

    drift_share = round(drifted_count / len(FEATURE_COLUMNS), 4)
    drift_detected = drift_share >= settings.DRIFT_SHARE_THRESHOLD

    report = {
        "samples_evaluated": len(prod_df),
        "drifted_features": drifted_count,
        "total_features": len(FEATURE_COLUMNS),
        "drift_share": drift_share,
        "threshold": settings.DRIFT_SHARE_THRESHOLD,
        "drift_detected": drift_detected,
        "features": feature_reports,
    }

    report_path = Path(settings.DATA_DIR) / "latest_drift_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    summary = run_drift_check()
    print(json.dumps(summary, indent=2))
