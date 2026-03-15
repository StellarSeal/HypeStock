import os
import json
import math
import logging
from functools import lru_cache
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from features import build_features

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding (Cached)
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def _build_sinusoidal_cpu(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe.unsqueeze(0)

def sinusoidal_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    return _build_sinusoidal_cpu(seq_len, d_model).to(device, non_blocking=True)

# ────────────────────────────────────────────────────────────
# PyTorch Multi-Metric Predictor (Seq2Seq GRU)
# ────────────────────────────────────────────────────────────
class MultiMetricPredictor(nn.Module):
    def __init__(self, num_symbols, num_features, lookback=120,
                 forecast_horizon=90, num_target_metrics=5,
                 symbol_dim=16, regime_dim=8, model_dim=128,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.forecast_horizon   = forecast_horizon
        self.num_target_metrics = num_target_metrics
        self.model_dim          = model_dim

        self.symbol_embedding = nn.Embedding(num_symbols, symbol_dim)
        self.regime_embedding = nn.Embedding(3, regime_dim)

        self.input_proj = nn.Linear(num_features + symbol_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool   = nn.Linear(model_dim, 1)

        ctx_dim = model_dim + symbol_dim + regime_dim
        self.ctx_proj = nn.Linear(ctx_dim, model_dim)

        self.decoder_gru_parallel = nn.GRU(
            input_size=num_target_metrics + model_dim,
            hidden_size=model_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_gru_cell = nn.GRUCell(num_target_metrics + model_dim, model_dim)
        self.mu_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_target_metrics),
        )
        self.vol_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_target_metrics),
        )

    def _share_gru_weights(self):
        gru = self.decoder_gru_parallel
        cell = self.decoder_gru_cell
        cell.weight_ih = gru.weight_ih_l0
        cell.weight_hh = gru.weight_hh_l0
        cell.bias_ih   = gru.bias_ih_l0
        cell.bias_hh   = gru.bias_hh_l0

    def encode(self, x, sym_id, regime_id):
        B, T, _ = x.shape
        sym_embed = self.symbol_embedding(sym_id)
        sym_seq   = sym_embed.unsqueeze(1).expand(-1, T, -1)

        features  = self.input_proj(torch.cat([x, sym_seq], dim=-1))
        features  = features + sinusoidal_encoding(T, self.model_dim, x.device)
        encoded   = self.transformer(features)

        attn_w  = torch.softmax(self.attn_pool(encoded), dim=1)
        pooled  = (attn_w * encoded).sum(dim=1)

        regime_embed = self.regime_embedding(regime_id)
        ctx = self.ctx_proj(torch.cat([pooled, sym_embed, regime_embed], dim=-1))
        return ctx

    def forward(self, x, sym_id, regime_id, teacher_targets=None):
        B = x.size(0)
        ctx = self.encode(x, sym_id, regime_id)

        if teacher_targets is not None:
            zero_step = torch.zeros(B, 1, self.num_target_metrics, device=x.device)
            decoder_in_tgt = torch.cat([zero_step, teacher_targets[:, :-1, :]], dim=1)
            ctx_expanded   = ctx.unsqueeze(1).expand(-1, self.forecast_horizon, -1)
            decoder_in     = torch.cat([decoder_in_tgt, ctx_expanded], dim=-1)

            h0  = ctx.unsqueeze(0)
            out, _ = self.decoder_gru_parallel(decoder_in, h0)
            mu = torch.tanh(self.mu_head(out))
            sigma = nn.functional.softplus(self.vol_head(out))
            past_returns = x[:, :, 0:1]
            realized_vol = torch.std(past_returns, dim=1, keepdim=True)
            sigma = sigma * (1.0 + realized_vol)
            return mu * sigma
        else:
            self._share_gru_weights()
            h    = ctx
            step = torch.zeros(B, self.num_target_metrics, device=x.device)
            past_returns = x[:, :, 0:1]
            realized_vol = torch.std(past_returns, dim=1, keepdim=True).squeeze(1)
            preds = []
            for _ in range(self.forecast_horizon):
                gru_in = torch.cat([step, ctx], dim=-1)
                h      = self.decoder_gru_cell(gru_in, h)
                mu = torch.tanh(self.mu_head(h))
                sigma = nn.functional.softplus(self.vol_head(h))
                sigma = sigma * (1.0 + realized_vol)
                step = mu * sigma
                preds.append(step.unsqueeze(1))
            return torch.cat(preds, dim=1)

# ────────────────────────────────────────────────────────────
# Global Config & Inference State
# ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

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
_persona_models = {}

def load_metadata():
    if _metadata_cache['scaler_X'] is None:
        try:
            _metadata_cache['scaler_X'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_X.pkl'))
            _metadata_cache['scaler_Y'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_Y.pkl'))
            with open(os.path.join(MODELS_DIR, 'features.json'), 'r') as f:
                _metadata_cache['features'] = json.load(f)
            with open(os.path.join(MODELS_DIR, 'symbol_mapping.json'), 'r') as f:
                _metadata_cache['symbol_mapping'] = json.load(f)
            logger.info("✅ ML Metadata (Scalers/Features) Loaded Successfully.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load ML metadata from {MODELS_DIR}: {e}")
    return _metadata_cache

def load_persona_models():
    global _persona_models
    if not _persona_models:
        try:
            index_path = os.path.join(BASE_DIR, 'personA', 'deploy_index.json')
            with open(index_path, 'r') as f:
                idx = json.load(f)
            for item in idx:
                path = os.path.join(BASE_DIR, 'personA', item['path'])
                _persona_models[item['task']] = joblib.load(path)
            logger.info("✅ Person A Walkforward Ensemble Loaded Successfully.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load Person A models: {e}")
    return _persona_models

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
            os.path.join(MODELS_DIR, f'{days}d', f'{days}d.pth'),
            os.path.join(MODELS_DIR, f'{days}d', 'best_model.pth'),
            os.path.join(MODELS_DIR, f'{days}d.pth'),
        ]
        model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        
        if model_path:
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                logger.info(f"✅ Loaded {days}D predictor successfully on {device}")
            except Exception as e:
                logger.error(f"⚠️ Warning: Could not decode bindings for {model_path}: {e}")
        else:
            logger.warning(f"⚠️ Checkpoint not found for {days}D horizon. Yielding fallback.")
            
        model.to(device)
        model.eval()
        _model_instances[days] = model
        
    return _model_instances[days]

# ────────────────────────────────────────────────────────────
# Prediction Triggers
# ────────────────────────────────────────────────────────────
def predict_future_prices(symbol: str, historical_data: list):
    meta = load_metadata()
    if not meta['features'] or not meta['scaler_X'] or not meta['scaler_Y']:
        return {"available": False, "message": "ML Pipeline missing trained scalers/metadata in the models folder."}

    df = pd.DataFrame(historical_data)
    
    # 1. Lookback Validation
    if len(df) < LOOKBACK_WINDOW:
        return {"available": False, "message": f"Dataset failure: Require at least {LOOKBACK_WINDOW} temporal rows, obtained {len(df)}."}

    # 2. Extract shared features
    try:
        df = build_features(df, is_training=False)
    except ValueError as e:
        return {"available": False, "message": str(e)}

    features = meta['features']
    
    # 3. Explicit Missing Column Guardrail (No Zero Filling)
    missing = [f for f in features if f not in df.columns]
    if missing:
        return {"available": False, "message": f"Feature Parity Mismatch. Missing localized components: {missing}"}

    # 4. NaN validation guardrail
    if df[features].isna().any().any():
        return {"available": False, "message": "Corrupted state: NaN values observed resolving features matrix."}

    window = df[features].values[-LOOKBACK_WINDOW:].astype(np.float32)
    window_scaled = meta['scaler_X'].transform(window)
    x = torch.tensor(window_scaled).unsqueeze(0).to(device)
    
    sym_id = meta['symbol_mapping'].get(symbol, 0)
    sym_t = torch.tensor([sym_id], dtype=torch.long, device=device)

    recent_vol = df['close'].pct_change().tail(20).std()
    regime_id = 1 
    if recent_vol > 0.03: regime_id = 2
    elif recent_vol < 0.015: regime_id = 0
    reg_t = torch.tensor([regime_id], dtype=torch.long, device=device)

    # 5. Diagnostic Logging
    logger.info(f"[DEBUG] Feature columns verified: {len(features)}")
    logger.info(f"[DEBUG] Executing Inference tensor shape: {x.shape}")
    logger.info(f"[DEBUG] Binding configuration: Symbol ID {sym_id} / Regime ID {regime_id}")
    logger.info(f"[DEBUG] Last timestep feature localized vector fragment: {x[0, -1, :5].tolist()}...")

    num_targets = meta['scaler_Y'].scale_.shape[0]
    model = get_model(len(meta['symbol_mapping']), len(features), num_targets)
    with torch.no_grad():
        pred_ret_scaled = model(x, sym_t, reg_t, teacher_targets=None).squeeze(0).cpu().numpy()

    pred_ret = meta['scaler_Y'].inverse_transform(pred_ret_scaled)
    forecast_steps = min(PYTORCH_FORECAST_DAYS, pred_ret.shape[0])

    last_close = float(df['close'].iloc[-1])
    last_date_val = df['time'].iloc[-1]
    
    if isinstance(last_date_val, str):
        try:
            curr_date = datetime.fromisoformat(last_date_val.replace('Z', '+00:00')).replace(tzinfo=None)
        except:
            curr_date = datetime.now()
    else:
        curr_date = last_date_val if last_date_val else datetime.now()

    close_idx = pred_ret.shape[1] - 1
    predictions = []
    p = np.array([last_close] * pred_ret.shape[1])
    for t in range(forecast_steps):
        curr_date += timedelta(days=1)
        if curr_date.weekday() >= 5:
            curr_date += timedelta(days=2) 
            
        p = p * (1.0 + pred_ret[t])
        pred_close_val = float(p[close_idx])
        
        predictions.append({
            "date": curr_date.isoformat() + 'Z',
            "close": pred_close_val
        })
        
    return {
        "available": True,
        "model_used": f"MultiMetric Seq2Seq ({PYTORCH_FORECAST_DAYS}D)",
        "predictions": predictions
    }


def predict_regression_prices(symbol: str, historical_data: list):
    """
    Executes Person A's Ensemble Walkforward Models over localized trend vectors.
    """
    models = load_persona_models()
    if not models:
        return {"available": False, "message": "Person A Walkforward models are not available."}

    df = pd.DataFrame(historical_data)
    if len(df) < LOOKBACK_WINDOW:
        return {"available": False, "message": f"Dataset failure: Require at least {LOOKBACK_WINDOW} temporal rows."}

    try:
        df = build_features(df, is_training=False)
    except ValueError as e:
        return {"available": False, "message": str(e)}

    row = df.iloc[-1:]

    results = {}
    for task, model in models.items():
        if hasattr(model, 'feature_names_in_'):
            X = row[model.feature_names_in_]
        else:
            X = row.drop(columns=['time', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'], errors='ignore')

        if task.endswith('_clf'):
            try:
                prob = model.predict_proba(X)[0][1]
            except Exception:
                prob = model.predict(X)[0] 
            results[task] = float(prob)
        else:
            pred = model.predict(X)[0]
            results[task] = float(pred)

    last_close = float(df['close'].iloc[-1])
    last_date_val = df['time'].iloc[-1]
    
    if isinstance(last_date_val, str):
        try:
            curr_date = datetime.fromisoformat(last_date_val.replace('Z', '+00:00')).replace(tzinfo=None)
        except:
            curr_date = datetime.now()
    else:
        curr_date = last_date_val if last_date_val else datetime.now()

    r_1d = results.get('next_day_return_reg', 0)
    close_1d = last_close * (1 + r_1d)

    mean_5d_rel = results.get('next_week_mean_close_5d_reg', 0)
    mean_5d = last_close * (1 + mean_5d_rel)

    tte_raw = results.get('trend_duration_tte_reg', 0)
    tte = np.expm1(tte_raw)

    predictions = []
    d = curr_date
    for i in range(5):
        d += timedelta(days=1)
        if d.weekday() >= 5:
            d += timedelta(days=2)

        # Gradual non-linear trend curve (power curve) to avoid both strictly linear and "teeth" patterns
        progress = i / 4.0 if i > 0 else 0.0
        # Apply convex curvature for uptrends and concave for downtrends to simulate momentum
        curve_factor = progress ** 1.35 if mean_5d > close_1d else progress ** 0.85
        
        pred_close = close_1d + (mean_5d - close_1d) * curve_factor

        predictions.append({
            "date": d.isoformat() + 'Z',
            "close": max(0.01, float(pred_close))
        })

    prob_up = results.get('price_direction_clf', 0)
    prob_res = results.get('break_resistance_1_5d_clf', 0)
    prob_sup = results.get('break_support_1_5d_clf', 0)

    msg = (
        f"<div class='space-y-3 mt-4 text-slate-300'>"
        f"<p><strong class='text-sky-400'>1-Day Outlook:</strong> {(prob_up*100):.1f}% probability of upward movement (Target Ret: {(r_1d*100):.2f}%).</p>"
        f"<p><strong class='text-sky-400'>5-Day Outlook:</strong> Projected Trajectory Mean is {mean_5d:.2f}.</p>"
        f"<p><strong class='text-purple-400'>Trend Analytics:</strong> Est. Trend Duration of {tte:.1f} days. "
        f"Resistance Break Prob: {(prob_res*100):.1f}%, Support Break Prob: {(prob_sup*100):.1f}%.</p>"
        f"</div>"
    )

    return {
        "available": True,
        "model_used": "Person A Walkforward Ensemble",
        "message": msg,
        "predictions": predictions,
        "raw_scores": {
            "prob_up": float(prob_up),
            "prob_resistance_break": float(prob_res),
            "prob_support_break": float(prob_sup),
            "next_day_return": float(r_1d),
            "trend_duration": float(tte),
        },
    }

# ────────────────────────────────────────────────────────────
# Ensemble Prediction (PyTorch 7d + PersonA)
# ────────────────────────────────────────────────────────────
def predict_ensemble(symbol: str, historical_data: list) -> dict:
    if not historical_data or len(historical_data) < LOOKBACK_WINDOW:
        return {"available": False, "message": f"Dataset constraint: Ensemble requires {LOOKBACK_WINDOW} days of localized data."}

    try:
        torch_result = predict_future_prices(symbol, historical_data)
    except Exception as e:
        logger.warning(f"PyTorch {PYTORCH_FORECAST_DAYS}D failed for {symbol}: {e}")
        torch_result = {"available": False}

    try:
        persona_result = predict_regression_prices(symbol, historical_data)
    except Exception as e:
        logger.warning(f"PersonA failed for {symbol}: {e}")
        persona_result = {"available": False}

    if not torch_result.get("available") and persona_result.get("available"):
        return {
            "available": True,
            "message": persona_result.get("message", ""),
            "predictions": persona_result.get("predictions"),
        }
    if not torch_result.get("available"):
        return {"available": False, "message": "No prediction models available. Validation failed on dataset components."}

    last_close = float(historical_data[-1].get("close", 0))
    trend = "neutral"
    if torch_result.get("predictions"):
        end_7d = torch_result["predictions"][-1]["close"]
        pct_7d = (end_7d - last_close) / max(last_close, 1e-9)
        if pct_7d > 0.01:
            trend = "bullish"
        elif pct_7d < -0.01:
            trend = "bearish"

    confidence = None
    if persona_result.get("available") and persona_result.get("raw_scores"):
        prob_up = persona_result["raw_scores"].get("prob_up", 0.5)
        confidence = round(max(prob_up, 1.0 - prob_up), 3)

    msg_parts = []
    if torch_result.get("predictions"):
        end_price = torch_result["predictions"][-1]["close"]
        pct = ((end_price - last_close) / max(last_close, 1e-9)) * 100
        msg_parts.append(f"{PYTORCH_FORECAST_DAYS}D: {end_price:.2f} ({pct:+.1f}%)")
    if persona_result.get("available") and persona_result.get("message"):
        msg_parts.append(persona_result["message"])
    msg = " ".join(msg_parts)

    return {
        "available": True,
        "message": msg or None,
        "trend": trend,
        "confidence": confidence,
        "predictions": torch_result.get("predictions"),
    }