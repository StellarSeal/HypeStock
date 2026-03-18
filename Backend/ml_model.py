import os
import json
import logging
import torch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import text

from feature_pipeline import (
    FEATURE_SCHEMA,
    normalize_features,
    validate_feature_schema,
    assert_sequence_integrity,
    log_input_stats,
    to_model_feature_frame,
    enforce_scaled_anomaly_guard,
)
from train import predict, MultiMetricPredictor
from database import sync_engine

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Global Config & Inference State
# ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'output_model')

# Fixed Forecast Range according to specification parity limits
PYTORCH_FORECAST_DAYS = 7
LOOKBACK_WINDOW = 120

_metadata_cache = {
    'scaler_X': None,
    'scaler_Y': None,
    'features': None,
    'symbol_mapping': None
}
_model_instances = {}

def load_metadata():
    if _metadata_cache['scaler_X'] is None:
        try:
            _metadata_cache['scaler_X'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_X.pkl'))
            _metadata_cache['scaler_Y'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_Y.pkl'))
            with open(os.path.join(MODELS_DIR, 'features.json'), 'r') as f:
                loaded_features = json.load(f)
            validate_feature_schema(loaded_features)
            _metadata_cache['features'] = loaded_features
            with open(os.path.join(MODELS_DIR, 'symbol_mapping.json'), 'r') as f:
                _metadata_cache['symbol_mapping'] = json.load(f)

            scaler_features = int(getattr(_metadata_cache['scaler_X'], 'n_features_in_', -1))
            if scaler_features != len(FEATURE_SCHEMA):
                raise ValueError(
                    f"scaler_X feature dimension mismatch: expected {len(FEATURE_SCHEMA)}, got {scaler_features}"
                )
            logger.info("✅ ML Metadata (Scalers/Features) Loaded Successfully.")
        except Exception as e:
            _metadata_cache['scaler_X'] = None
            _metadata_cache['scaler_Y'] = None
            _metadata_cache['features'] = None
            _metadata_cache['symbol_mapping'] = None
            logger.error(f"⚠️ Failed to load ML metadata from {MODELS_DIR}: {e}")
    return _metadata_cache

def get_model(num_symbols: int, num_features: int, num_targets: int = 4):
    days = PYTORCH_FORECAST_DAYS
    if days not in _model_instances:
        model = MultiMetricPredictor(
            num_symbols=num_symbols, 
            num_features=num_features, 
            lookback=LOOKBACK_WINDOW, 
            forecast_horizon=days,
            num_target_metrics=num_targets
        )
        candidate_paths = [
            os.path.join(MODELS_DIR, f'{days}d', 'best_model.pth'),
            os.path.join(MODELS_DIR, f'{days}d', f'{days}d.pth'),
            os.path.join(MODELS_DIR, f'{days}d.pth'),
        ]
        model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if model_path is None:
            raise FileNotFoundError(f"Checkpoint not found for {days}D horizon in {MODELS_DIR}")

        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info(f"✅ Loaded {days}D predictor successfully on {device}")
        except Exception as e:
            raise RuntimeError(f"Could not decode checkpoint bindings for {model_path}: {e}") from e
            
        model.to(device)
        model.eval()
        _model_instances[days] = model
        
    return _model_instances[days]

# ────────────────────────────────────────────────────────────
# Data Acquisition
# ────────────────────────────────────────────────────────────
def fetch_stock_data(symbol: str, limit: int, _fallback_data: list) -> pd.DataFrame:
    """Fetch full canonical feature set from DB using exact timestamp alignment."""
    try:
        query = text("""
            SELECT *
            FROM (
                SELECT p."time", p.open, p.high, p.low, p.close, p.volume,
                       m.ma20, m.ma50, m.ema20,
                       m.rsi, m.macd,
                       m.rolling_vol_20d_std AS volatility,
                       m.atr,
                       m.daily_return_1d,
                       m.lagged_return_t1,
                       m.lagged_return_t3,
                       m.lagged_return_t5,
                       m.dist_from_ma50
                FROM stock_prices p
                INNER JOIN metrics m ON p.symbol = m.symbol AND p."time" = m."time"
                WHERE p.symbol = :sym
                ORDER BY p."time" DESC
                LIMIT :limit
            ) latest
            ORDER BY latest."time" ASC
        """)
        with sync_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sym": symbol, "limit": limit})

        if df.empty:
            raise ValueError(f"No rows returned for symbol={symbol}")

        if len(df) < int(limit):
            logger.warning(
                "Strict DB parity load for %s returned %d/%d rows. "
                "This can indicate timestamp mismatch between stock_prices and metrics.",
                symbol,
                len(df),
                int(limit),
            )

        norm_df = normalize_features(df)
        assert_sequence_integrity(norm_df, seq_len=1)
        return norm_df

    except Exception as e:
        raise RuntimeError(f"Database query failed for strict inference pipeline: {e}") from e

# ────────────────────────────────────────────────────────────
# Prediction Triggers
# ────────────────────────────────────────────────────────────
def predict_future_prices(symbol: str, historical_data: list):
    meta = load_metadata()
    if not meta['features'] or not meta['scaler_X'] or not meta['scaler_Y']:
        return {"available": False, "message": "ML Pipeline missing trained scalers/metadata in the output_model folder."}

    features = meta['features']
    try:
        validate_feature_schema(features)
    except Exception as e:
        return {"available": False, "message": f"Feature schema validation failed: {e}"}

    # Extends limit bounds slightly to allow localized rolling features to populate.
    try:
        df = fetch_stock_data(symbol, LOOKBACK_WINDOW + 50, historical_data)
    except Exception as e:
        return {"available": False, "message": f"Strict DB load failed: {e}"}

    try:
        assert_sequence_integrity(df, LOOKBACK_WINDOW)
    except Exception as e:
        return {"available": False, "message": f"Sequence integrity validation failed: {e}"}

    if symbol not in meta['symbol_mapping']:
        return {"available": False, "message": f"Symbol mapping missing for '{symbol}'. Retrain or refresh symbol_mapping.json."}
    sym_id = meta['symbol_mapping'][symbol]

    window_df = df.tail(LOOKBACK_WINDOW).copy()
    anomaly_message = None
    log_input_stats(window_df, prefix="Input stats (raw)")
    try:
        model_window_df = to_model_feature_frame(window_df)
        log_input_stats(model_window_df, prefix="Input stats (model contract)")

        x_window = model_window_df[FEATURE_SCHEMA].to_numpy(dtype=np.float32)
        x_scaled = meta['scaler_X'].transform(x_window)
        scaled_stats = {
            'mean': float(np.mean(x_scaled)),
            'std': float(np.std(x_scaled)),
            'min': float(np.min(x_scaled)),
            'max': float(np.max(x_scaled)),
        }
        try:
            enforce_scaled_anomaly_guard(x_scaled)
        except ValueError as e:
            anomaly_message = str(e)
            logger.warning(
                "Scaled anomaly guard triggered: %s. Proceeding with inference (degraded confidence advisory).",
                anomaly_message,
            )

        logger.info(
            "Scaled input stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
            scaled_stats['mean'],
            scaled_stats['std'],
            scaled_stats['min'],
            scaled_stats['max'],
        )
    except Exception as e:
        logger.error("Input preflight failed: %s", e)
        return {"available": False, "message": f"Input preflight failed: {e}"}

    recent_vol = float(window_df['volatility'].tail(20).mean())
    if not np.isfinite(recent_vol):
        recent_vol = 0.0
    regime_id = 1 
    if recent_vol > 0.03: regime_id = 2
    elif recent_vol < 0.015: regime_id = 0

    # 2. Diagnostic Logging
    logger.info(f"[DEBUG] Executing Inference using external train.predict()")
    logger.info(f"[DEBUG] Binding configuration: Symbol ID {sym_id} / Regime ID {regime_id}")

    # 3. Load Model
    num_targets = meta['scaler_Y'].scale_.shape[0]
    try:
        model = get_model(len(meta['symbol_mapping']), len(features), num_targets)
    except Exception as e:
        return {"available": False, "message": f"Model load failed: {e}"}
    
    # 4. Execute predict from train.py
    try:
        pred_df = predict(
            df, 
            symbol_id=sym_id, 
            regime_id=regime_id,
            model=model, 
            scaler_X=meta['scaler_X'], 
            scaler_Y=meta['scaler_Y'],
            feature_names=FEATURE_SCHEMA,
            device=device
        )
    except Exception as e:
        logger.error(f"Prediction execution failed: {e}")
        return {"available": False, "message": f"Prediction execution failed: {str(e)}"}
        
    last_date_val = df['time'].iloc[-1]
    if isinstance(last_date_val, str):
        try:
            curr_date = datetime.fromisoformat(last_date_val.replace('Z', '+00:00')).replace(tzinfo=None)
        except ValueError:
            curr_date = datetime.now()
    else:
        curr_date = last_date_val if pd.notnull(last_date_val) else datetime.now()
        
    predictions = []
    d = curr_date
    prev_close = float(df['close'].iloc[-1])
    for _, row in pred_df.iterrows():
        d += timedelta(days=1)
        if d.weekday() >= 5: # Skip weekends
            d += timedelta(days=2)

        close_v = float(row.get('close', prev_close))
        if not np.isfinite(close_v) or close_v <= 0:
            close_v = max(0.01, prev_close)

        open_v = float(row.get('open', prev_close))
        if not np.isfinite(open_v) or open_v <= 0:
            open_v = prev_close

        high_v = float(row.get('high', max(open_v, close_v)))
        low_v = float(row.get('low', min(open_v, close_v)))

        if not np.isfinite(high_v) or high_v <= 0:
            high_v = max(open_v, close_v)
        if not np.isfinite(low_v) or low_v <= 0:
            low_v = min(open_v, close_v)

        high_v = max(high_v, open_v, close_v)
        low_v = max(0.01, min(low_v, open_v, close_v))
            
        predictions.append({
            "date": d.isoformat() + 'Z',
            "open": float(open_v),
            "high": float(high_v),
            "low": float(low_v),
            "close": float(close_v)
        })
        prev_close = close_v
        
    response = {
        "available": True,
        "model_used": f"MultiMetric Seq2Seq ({PYTORCH_FORECAST_DAYS}D)",
        "predictions": predictions
    }
    if anomaly_message:
        response["message"] = anomaly_message

    return response


# ────────────────────────────────────────────────────────────
# Prediction Aggregator (PyTorch 7d)
# ────────────────────────────────────────────────────────────
def predict_ensemble(symbol: str, historical_data: list) -> dict:
    if not historical_data or len(historical_data) < LOOKBACK_WINDOW:
        return {"available": False, "message": f"Dataset constraint: model requires {LOOKBACK_WINDOW} days of localized data."}

    try:
        torch_result = predict_future_prices(symbol, historical_data)
    except Exception as e:
        logger.warning(f"PyTorch {PYTORCH_FORECAST_DAYS}D failed for {symbol}: {e}")
        torch_result = {"available": False, "message": str(e)}

    if not torch_result.get("available"):
        return {
            "available": False,
            "message": torch_result.get("message") or "No prediction model available. Validation failed on dataset components."
        }

    last_close = float(historical_data[-1].get("close", 0))
    trend = "neutral"
    if torch_result.get("predictions"):
        end_7d = torch_result["predictions"][-1]["close"]
        pct_7d = (end_7d - last_close) / max(last_close, 1e-9)
        if pct_7d > 0.01:
            trend = "bullish"
        elif pct_7d < -0.01:
            trend = "bearish"

    msg_parts = []
    if torch_result.get("predictions"):
        end_price = torch_result["predictions"][-1]["close"]
        pct = ((end_price - last_close) / max(last_close, 1e-9)) * 100
        msg_parts.append(f"{PYTORCH_FORECAST_DAYS}D: {end_price:.2f} ({pct:+.1f}%)")
    if torch_result.get("message"):
        msg_parts.append(torch_result["message"])
    msg = " ".join(part for part in msg_parts if part)

    return {
        "available": True,
        "model_used": torch_result.get("model_used"),
        "message": msg or None,
        "trend": trend,
        "confidence": None,
        "predictions": torch_result.get("predictions"),
    }