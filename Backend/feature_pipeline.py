import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_SCHEMA = [
    "open", "high", "low", "close", "volume",
    "ma20", "ma50", "ema20",
    "rsi", "macd",
    "volatility", "atr",
    "daily_return_1d",
    "lagged_return_t1", "lagged_return_t3", "lagged_return_t5",
    "dist_from_ma50",
]

# Compatibility aliases for historical artifacts that used mixed naming styles.
_COLUMN_ALIASES = {
    "MA20": "ma20",
    "MA50": "ma50",
    "EMA20": "ema20",
    "RSI": "rsi",
    "MACD": "macd",
    "Rolling_Vol_20d_std": "volatility",
    "rolling_vol_20d_std": "volatility",
    "ATR": "atr",
    "Daily_Return_1d": "daily_return_1d",
    "daily_return": "daily_return_1d",
    "Lagged_Return_t1": "lagged_return_t1",
    "Lagged_Return_t3": "lagged_return_t3",
    "Lagged_Return_t5": "lagged_return_t5",
    "Dist_from_MA50": "dist_from_ma50",
}


def _to_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for source_col, target_col in _COLUMN_ALIASES.items():
        if source_col in df.columns and target_col not in df.columns:
            rename_map[source_col] = target_col
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_feature_schema(feature_names: Sequence[str]) -> None:
    actual = list(feature_names)
    if actual == FEATURE_SCHEMA:
        return

    missing = [c for c in FEATURE_SCHEMA if c not in actual]
    extra = [c for c in actual if c not in FEATURE_SCHEMA]
    raise ValueError(
        "Feature schema mismatch. "
        f"Expected exact order={FEATURE_SCHEMA}; got={actual}. "
        f"Missing={missing}; Extra={extra}"
    )


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise KeyError("Input dataframe must contain 'time'.")

    work_df = _to_canonical_columns(df.copy())
    work_df["time"] = pd.to_datetime(work_df["time"], utc=True, errors="coerce")
    if work_df["time"].isnull().any():
        raise ValueError("Time column contains invalid timestamps.")

    work_df = work_df.sort_values("time").reset_index(drop=True)

    for col in FEATURE_SCHEMA:
        if col not in work_df.columns:
            work_df[col] = np.nan

    work_df = work_df[["time"] + FEATURE_SCHEMA]
    work_df = work_df.ffill().bfill()
    work_df[FEATURE_SCHEMA] = work_df[FEATURE_SCHEMA].astype(np.float32)
    return work_df


def assert_sequence_integrity(df: pd.DataFrame, seq_len: int) -> None:
    if len(df) < int(seq_len):
        raise ValueError(f"Sequence too short: need >= {seq_len} rows, got {len(df)}")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Sequence integrity failed: time must be monotonic increasing.")

    nan_cols = df[FEATURE_SCHEMA].columns[df[FEATURE_SCHEMA].isnull().any()].tolist()
    if nan_cols:
        raise ValueError(f"Sequence integrity failed: NaN values remain in columns {nan_cols}")


def summarize_input_stats(feature_frame: pd.DataFrame) -> dict:
    matrix = feature_frame[FEATURE_SCHEMA].to_numpy(dtype=np.float32)
    return {
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
    }


def log_input_stats(feature_frame: pd.DataFrame, prefix: str = "Input stats") -> dict:
    stats = summarize_input_stats(feature_frame)
    logger.info("%s:", prefix)
    logger.info("mean: %.6f", stats["mean"])
    logger.info("std: %.6f", stats["std"])
    logger.info("min: %.6f", stats["min"])
    logger.info("max: %.6f", stats["max"])
    return stats


def to_model_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Align inference features with the training contract.

    During training, OHLC channels are transformed to one-step returns before
    scaler_X is fit. Inference must apply the same transform before scaling.
    """
    work_df = feature_frame.copy()

    missing = [c for c in FEATURE_SCHEMA if c not in work_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns required by model contract: {missing}")

    price_cols = ["open", "high", "low", "close"]
    work_df[price_cols] = (
        work_df[price_cols]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    work_df[FEATURE_SCHEMA] = (
        work_df[FEATURE_SCHEMA]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
        .astype(np.float32)
    )
    return work_df


def enforce_scaled_anomaly_guard(
    scaled_matrix: np.ndarray,
    mean_tolerance: float = 5.0,
    std_low: float = 0.01,
    std_high: float = 10.0,
    abs_max_tolerance: float = 25.0,
) -> dict:
    mean_val = float(np.mean(scaled_matrix))
    std_val = float(np.std(scaled_matrix))
    min_val = float(np.min(scaled_matrix))
    max_val = float(np.max(scaled_matrix))

    anomaly_reasons = []
    if abs(mean_val) > mean_tolerance:
        anomaly_reasons.append(f"|mean|={abs(mean_val):.6f} > {mean_tolerance}")
    if std_val < std_low or std_val > std_high:
        anomaly_reasons.append(f"std={std_val:.6f} outside [{std_low}, {std_high}]")
    if max(abs(min_val), abs(max_val)) > abs_max_tolerance:
        anomaly_reasons.append(
            f"max_abs={max(abs(min_val), abs(max_val)):.6f} > {abs_max_tolerance}"
        )

    stats = {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "anomaly": bool(anomaly_reasons),
        "reasons": anomaly_reasons,
    }

    if stats["anomaly"]:
        raise ValueError("Input distribution anomaly detected: " + "; ".join(anomaly_reasons))

    return stats
