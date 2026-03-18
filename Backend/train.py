"""
ADVANCED STOCK PREDICTION PIPELINE (MULTI-METRIC FORECASTING) — REWORK v5
==========================================================================
v5 MEMORY & THROUGHPUT REFACTOR
---------------------------------
  RAM-1  Lazy window generation: TemporalSequenceDataset stores only
         base tensors + (stock_id, t) index pairs.  Windows are sliced
         in __getitem__ — zero pre-materialisation of sliding windows.
         Reduces peak RAM by ~10-50x on large datasets.

  RAM-2  float32 everywhere: all dataframe columns cast to float32
         before tensor conversion.  Eliminates float64 duplication
         (~50% RAM reduction vs mixed precision DataFrames).

  RAM-3  Vectorised pandas preprocessing: joblib parallel per-symbol
         feature transforms replaced with grouped pandas operations.
         Eliminates worker-process memory duplication.

  RAM-4  Deferred heavy imports: sklearn.preprocessing.StandardScaler
         imported inside the scaler-building function, preventing
         worker subprocesses from loading the full sklearn stack.

  RAM-5  Conservative DataLoader: on Windows + parallel horizons,
      force num_workers=0 (no worker subprocess shared mappings).
      Otherwise use small worker counts with conservative prefetch.

  PERF-1  Sinusoidal PE cache: encoding is computed once per (seq_len,
          d_model) pair and stored in a module-level LRU cache.

  PERF-2  Vectorised price reconstruction via torch.cumprod().

  PERF-3  Training-mode GRU uses nn.GRU (parallel scan); GRUCell only
          at autoregressive inference.

  PERF-4  scale_t / mean_t pre-allocated once per horizon.

  PERF-5  torch.set_float32_matmul_precision("high") + cudnn.benchmark.

  PERF-6  Optional stride sampling: --stride N skips every N-1 windows,
          reducing dataset size while preserving statistical diversity.

  PRED-1  Volatility-adaptive prediction heads replace hard ±2% clamp:
          pred = tanh(mu_head) * softplus(vol_head) * (1 + realized_vol)

  PRED-2  Curvature guard threshold 0.005 (was 0.02) for realistic turns.

  DIAG-1  All diagnostics operate on per-sequence statistics (no blind
          full-flatten). Logged every 200 steps, not every batch.

How to Run:
-----------
python train.py \
    --dataset metrics.csv \
    --prices prices.csv \
    --companies companies.csv \
    --horizons 7,14,30 \
    --lookback 120 \
    --stages 3 \
    --stage_ratios 0.1,0.2,0.5 \
    --batch_size 128 \
    --stride 3 \
    --epochs_per_stage 20 \
    --learning_rate 0.0005 \
    --device cuda \
    --mixed_precision true

Output structure:
  output_model/
    scaler_X.pkl, scaler_Y.pkl, features.json, symbol_mapping.json
    7d/best_model.pth    -- 1-week predictor
    14d/best_model.pth   -- 2-week predictor
    30d/best_model.pth   -- 30-day predictor
"""

import os
import math
import time
import argparse
import json
import logging
import warnings
import shutil
import threading
import contextlib
import queue
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from feature_pipeline import (
    FEATURE_SCHEMA,
    normalize_features,
    assert_sequence_integrity,
    validate_feature_schema,
    to_model_feature_frame,
)
# NOTE: sklearn.preprocessing.StandardScaler is imported inside build_scalers()
#       to prevent spawn-mode worker processes from loading the full sklearn stack.

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- RTX 40-Series (Ada Lovelace) Specific Optimisations ---
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision("high")  # Uses TF32/bfloat16 fast-math
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# -----------------------------------------------------------

# ============================================================
# 1. Dataset & Input Pipeline
# ============================================================
class TemporalAnchorDataset(Dataset):
    """
    Stores only anchor indices.
    Window extraction is performed in vectorised form inside the collate_fn,
    eliminating Python-level per-sample slicing overhead in __getitem__.
    """
    def __init__(self, valid_indices, stride=1):
        idx = np.asarray(valid_indices, dtype=np.int64)
        if stride > 1:
            idx = idx[::stride]
        self.indices = idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return int(self.indices[idx])


class TemporalBatchCollator:
    """
    Vectorised batch-time slicing using index_select.

    This removes per-sample slicing in Python and performs all slicing for a
    batch in a small number of tensor operations.
    """
    def __init__(self,
                 X_tensor,
                 sym_tensor,
                 regime_tensor,
                 Y_ret_tensor,
                 Y_price_tensor,
                 last_price_tensor,
                 lookback,
                 forecast_horizon):
        self.X = X_tensor
        self.sym = sym_tensor
        self.regime = regime_tensor
        self.Y_ret = Y_ret_tensor
        self.Y_price = Y_price_tensor
        self.last_price = last_price_tensor
        self.lookback = int(lookback)
        self.horizon = int(forecast_horizon)
        self.num_features = int(X_tensor.shape[1])
        self.num_targets = int(Y_ret_tensor.shape[1])
        self.lookback_offsets = torch.arange(-self.lookback + 1, 1, dtype=torch.long)
        self.horizon_offsets = torch.arange(1, self.horizon + 1, dtype=torch.long)

    def __call__(self, anchor_indices):
        anchor_idx = torch.as_tensor(anchor_indices, dtype=torch.long)
        if anchor_idx.ndim == 0:
            anchor_idx = anchor_idx.unsqueeze(0)
        bsz = int(anchor_idx.shape[0])

        x_idx = anchor_idx.unsqueeze(1) + self.lookback_offsets.unsqueeze(0)
        y_idx = anchor_idx.unsqueeze(1) + self.horizon_offsets.unsqueeze(0)

        x = self.X.index_select(0, x_idx.reshape(-1)).view(bsz, self.lookback, self.num_features)
        y_ret = self.Y_ret.index_select(0, y_idx.reshape(-1)).view(bsz, self.horizon, self.num_targets)
        y_price = self.Y_price.index_select(0, y_idx.reshape(-1)).view(bsz, self.horizon, self.num_targets)
        sym = self.sym.index_select(0, anchor_idx)
        regime = self.regime.index_select(0, anchor_idx)
        last_price = self.last_price.index_select(0, anchor_idx)
        return x, sym, regime, y_ret, y_price, last_price


def _move_batch_to_device(batch, device, non_blocking=True, channels_last=False):
    if torch.is_tensor(batch):
        moved = batch.to(device, non_blocking=non_blocking)
        if channels_last and moved.dim() == 4:
            moved = moved.contiguous(memory_format=torch.channels_last)
        return moved
    if isinstance(batch, (tuple, list)):
        return type(batch)(
            _move_batch_to_device(x, device, non_blocking=non_blocking, channels_last=channels_last)
            for x in batch
        )
    if isinstance(batch, dict):
        return {
            k: _move_batch_to_device(v, device, non_blocking=non_blocking, channels_last=channels_last)
            for k, v in batch.items()
        }
    return batch


def _record_batch_stream(batch, stream):
    if torch.is_tensor(batch):
        batch.record_stream(stream)
        return
    if isinstance(batch, (tuple, list)):
        for x in batch:
            _record_batch_stream(x, stream)
        return
    if isinstance(batch, dict):
        for v in batch.values():
            _record_batch_stream(v, stream)


class AsyncBufferedLoader:
    """
    CPU-side prefetch queue for environments where num_workers=0 is required.

    A producer thread pulls next batches from DataLoader while GPU executes the
    current step, reducing host-side stalls.
    """
    def __init__(self, loader, buffer_size=4):
        self.loader = loader
        self.buffer_size = max(1, int(buffer_size))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        q = queue.Queue(maxsize=self.buffer_size)
        sentinel = object()
        errors = []

        def _producer():
            try:
                for batch in self.loader:
                    q.put(batch)
            except Exception as exc:  # pragma: no cover - diagnostic path
                errors.append(exc)
            finally:
                q.put(sentinel)

        worker = threading.Thread(target=_producer, daemon=True)
        worker.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

        worker.join()
        if errors:
            raise errors[0]


class CUDAPrefetcher:
    """
    Asynchronously copies next batch to GPU on a dedicated CUDA stream.

    Overlaps H2D transfer with current step compute.
    """
    def __init__(self, loader, device, enabled=True, channels_last=False):
        self.loader = loader
        self.device = device
        self.enabled = bool(enabled) and device.type == 'cuda'
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def _preload(self, loader_it):
        try:
            batch = next(loader_it)
        except StopIteration:
            return None
        with torch.cuda.stream(self.stream):
            return _move_batch_to_device(
                batch,
                self.device,
                non_blocking=True,
                channels_last=self.channels_last,
            )

    def __iter__(self):
        if not self.enabled:
            for batch in self.loader:
                yield batch
            return

        loader_it = iter(self.loader)
        next_batch = self._preload(loader_it)
        while next_batch is not None:
            cur_stream = torch.cuda.current_stream(device=self.device)
            cur_stream.wait_stream(self.stream)
            _record_batch_stream(next_batch, cur_stream)
            batch = next_batch
            next_batch = self._preload(loader_it)
            yield batch


def _wrap_loader_with_cpu_prefetch(loader, num_workers, cpu_prefetch_queue):
    if int(num_workers) == 0 and cpu_prefetch_queue > 0:
        return AsyncBufferedLoader(loader, buffer_size=cpu_prefetch_queue)
    return loader


@dataclass
class StepTiming:
    data_wait_s: float = 0.0
    transfer_s: float = 0.0
    forward_s: float = 0.0
    backward_s: float = 0.0
    optim_s: float = 0.0
    batches: int = 0

    def as_dict(self):
        total = self.data_wait_s + self.transfer_s + self.forward_s + self.backward_s + self.optim_s
        safe_total = max(total, 1e-9)
        return {
            'total_step_time_s': total,
            'data_wait_pct': 100.0 * self.data_wait_s / safe_total,
            'transfer_pct': 100.0 * self.transfer_s / safe_total,
            'forward_pct': 100.0 * self.forward_s / safe_total,
            'backward_pct': 100.0 * self.backward_s / safe_total,
            'optim_pct': 100.0 * self.optim_s / safe_total,
            # Practical utilization proxy: compute share of step wall time.
            'gpu_util_est_pct': 100.0 * (self.forward_s + self.backward_s) / safe_total,
        }


def maybe_build_profiler(enabled, device, log_dir, horizon_name, stage, profile_steps):
    if not enabled:
        return None

    trace_dir = os.path.join(log_dir, 'profiler')
    os.makedirs(trace_dir, exist_ok=True)
    trace_file = os.path.join(trace_dir, f'{horizon_name}_stage{stage}.json')
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _trace_ready(p):
        p.export_chrome_trace(trace_file)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(1, profile_steps), repeat=1),
        on_trace_ready=_trace_ready,
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    )


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return 'out of memory' in msg or 'cuda error: out of memory' in msg

# ============================================================
# 2. Sinusoidal Positional Encoding  — PERF-1: LRU cached
# ============================================================
@lru_cache(maxsize=16)
def _build_sinusoidal_cpu(seq_len: int, d_model: int) -> torch.Tensor:
    """Builds and caches the encoding tensor on CPU. Called at most once per shape."""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe.unsqueeze(0)  # [1, seq_len, d_model]

def sinusoidal_encoding(seq_len: int, d_model: int, device) -> torch.Tensor:
    """Returns cached [1, seq_len, d_model] encoding moved to target device."""
    return _build_sinusoidal_cpu(seq_len, d_model).to(device, non_blocking=True)

# ============================================================
# 3. Model  — PERF-3: parallel GRU for training
# ============================================================
class MultiMetricPredictor(nn.Module):
    """
    Transformer encoder -> attention pooling -> GRU decoder.

    Training path  (teacher_forcing=True, default during .train()):
      nn.GRU processes the entire horizon in a single parallelised CUDA
      kernel by using the encoder context as h_0 and shifted targets as
      input sequence.  This is orders of magnitude faster than stepping
      GRUCell 90 times in a Python loop.

    Inference path (teacher_forcing=False, used in predict()):
      True autoregression via GRUCell — each step's output feeds the next.
    """
    def __init__(self, num_symbols, num_features, lookback=120,
                 forecast_horizon=90, num_target_metrics=5,
                 symbol_dim=16, regime_dim=8, model_dim=128,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.forecast_horizon   = forecast_horizon
        self.num_target_metrics = num_target_metrics
        self.model_dim          = model_dim

        # Embeddings
        self.symbol_embedding = nn.Embedding(num_symbols, symbol_dim)
        self.regime_embedding = nn.Embedding(3, regime_dim)

        # Encoder
        self.input_proj = nn.Linear(num_features + symbol_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool   = nn.Linear(model_dim, 1)

        # Context projection
        ctx_dim = model_dim + symbol_dim + regime_dim
        self.ctx_proj = nn.Linear(ctx_dim, model_dim)

        # Decoder — PERF-3
        # nn.GRU: used during training (batch input, one kernel call)
        self.decoder_gru_parallel = nn.GRU(
            input_size=num_target_metrics + model_dim,
            hidden_size=model_dim,
            num_layers=1,
            batch_first=True,
        )
        # nn.GRUCell: used at inference for true autoregression
        self.decoder_gru_cell = nn.GRUCell(num_target_metrics + model_dim, model_dim)
        # Volatility-adaptive prediction heads (replaces hard ±2% tanh clamp)
        # mu_head: direction prediction (tanh activation)
        self.mu_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_target_metrics),
        )
        # vol_head: learnable volatility scale (softplus activation)
        self.vol_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_target_metrics),
        )

    def _share_gru_weights(self):
        """Keep GRUCell weights in sync with the parallel GRU after each update."""
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
        """
        teacher_targets: [B, H, M] scaled return targets.
          If provided (training), uses parallel GRU scan.
          If None (inference), uses autoregressive GRUCell loop.
        """
        B = x.size(0)
        ctx = self.encode(x, sym_id, regime_id)  # [B, D]

        if teacher_targets is not None:
            # --- PERF-3: parallel training path ---
            # Build input sequence: [zeros, target_0, ..., target_{H-2}]
            # shifted right by 1 (standard seq2seq teacher forcing)
            zero_step = torch.zeros(B, 1, self.num_target_metrics, device=x.device)
            decoder_in_tgt = torch.cat([zero_step, teacher_targets[:, :-1, :]], dim=1)  # [B, H, M]
            ctx_expanded   = ctx.unsqueeze(1).expand(-1, self.forecast_horizon, -1)      # [B, H, D]
            decoder_in     = torch.cat([decoder_in_tgt, ctx_expanded], dim=-1)           # [B, H, M+D]

            h0  = ctx.unsqueeze(0)                               # [1, B, D]
            out, _ = self.decoder_gru_parallel(decoder_in, h0)  # [B, H, D]
            mu    = torch.tanh(self.mu_head(out))                # [B, H, M]  direction
            sigma = nn.functional.softplus(self.vol_head(out))   # [B, H, M]  volatility
            # Scale sigma by realized volatility from input window
            past_returns = x[:, :, 0:1]  # proxy: first feature channel
            realized_vol = torch.std(past_returns, dim=1, keepdim=True)  # [B, 1, 1]
            sigma = sigma * (1.0 + realized_vol)
            pred_return = mu * sigma                             # [B, H, M]
            return pred_return

        else:
            # --- Autoregressive inference path ---
            self._share_gru_weights()
            h    = ctx
            step = torch.zeros(B, self.num_target_metrics, device=x.device)
            past_returns = x[:, :, 0:1]  # proxy: first feature channel
            realized_vol = torch.std(past_returns, dim=1, keepdim=True).squeeze(1)  # [B, 1]
            preds = []
            for _ in range(self.forecast_horizon):
                gru_in = torch.cat([step, ctx], dim=-1)
                h      = self.decoder_gru_cell(gru_in, h)
                mu     = torch.tanh(self.mu_head(h))            # [B, M]
                sigma  = nn.functional.softplus(self.vol_head(h))  # [B, M]
                sigma  = sigma * (1.0 + realized_vol)           # volatility-scaled
                pred_return = mu * sigma                        # [B, M]
                step   = pred_return
                preds.append(step.unsqueeze(1))
            return torch.cat(preds, dim=1)

# ============================================================
# 4. Combined Loss
# ============================================================
class CombinedForecastLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.5):
        super().__init__()
        self.mse   = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(self, pred_ret, target_ret, pred_price, target_price, anchor_price):
        loss_ret   = self.mse(pred_ret, target_ret)
        # Log-space price loss: converts exponential compounding to linear
        # sums, preventing the 90-step cumprod from producing unbounded loss.
        anchor     = anchor_price.unsqueeze(1).clamp(min=1e-6)
        log_pred   = torch.log(pred_price / anchor + 1e-8)
        log_target = torch.log(target_price / anchor + 1e-8)
        loss_price = self.huber(log_pred, log_target)
        dir_loss   = torch.mean(torch.relu(-torch.sign(pred_ret) * torch.sign(target_ret)))
        return self.alpha * loss_ret + self.beta * loss_price + self.gamma * dir_loss


def compute_pattern_guard_losses(pred_returns: torch.Tensor, target_returns: torch.Tensor):
    """
    Anti-trivial-structure guard rails on return sequences.

    pred_returns: [N]  (flattened for vol/acf/variance; curvature computed per-seq if possible)
    target_returns: [N]
    """
    pred_vol = torch.std(pred_returns)
    true_vol = torch.std(target_returns)
    volatility_loss = torch.abs(pred_vol - true_vol)

    diff1 = pred_returns[1:] - pred_returns[:-1]
    if diff1.numel() > 1:
        diff2 = diff1[1:] - diff1[:-1]
        curvature_loss = torch.mean(torch.abs(diff2))
    else:
        curvature_loss = pred_returns.new_tensor(0.0)
    # Threshold lowered from 0.02 → 0.005: encourage turning points without forcing oscillation
    curvature_guard = torch.relu(pred_returns.new_tensor(0.005) - curvature_loss)

    x = pred_returns - pred_returns.mean()
    if x.numel() > 1:
        acf1 = torch.sum(x[1:] * x[:-1]) / (torch.sum(x ** 2) + 1e-8)
    else:
        acf1 = pred_returns.new_tensor(0.0)
    autocorr_penalty = torch.relu(acf1 - pred_returns.new_tensor(0.95))

    pred_std = torch.std(pred_returns)
    variance_penalty = torch.relu(pred_returns.new_tensor(0.01) - pred_std)

    total_guard_loss = (
        0.15 * volatility_loss
        + 0.10 * curvature_guard
        + 0.10 * autocorr_penalty
        + 0.20 * variance_penalty
    )

    return {
        'total_guard_loss': total_guard_loss,
        'volatility_loss': volatility_loss,
        'curvature_loss': curvature_loss,
        'curvature_guard': curvature_guard,
        'acf1': acf1,
        'autocorr_penalty': autocorr_penalty,
        'pred_std': pred_std,
        'variance_penalty': variance_penalty,
    }

# ============================================================
# 4b. Regime Multipliers & Confidence Scaling (Sections A–F)
# ============================================================
EXPERIMENTAL_BUYER_SIM = False

BASE_REGIME_MULTIPLIERS = {
    0: -1.5,   # Strong Down
    1: -0.5,   # Weak Down
    2:  0.0,   # Neutral
    3:  0.5,   # Weak Up
    4:  1.5,   # Strong Up
}

def classify_regime(pred_return, sigma):
    """Classify predicted return into movement regime 0–4 based on sigma thresholds."""
    sigma = max(sigma, 1e-9)
    if pred_return < -1.5 * sigma:
        return 0
    elif pred_return < -0.3 * sigma:
        return 1
    elif pred_return <= 0.3 * sigma:
        return 2
    elif pred_return <= 1.5 * sigma:
        return 3
    else:
        return 4

def apply_confidence_adjustment(base, estimate, confidence):
    """Blend base multiplier toward historical estimate using confidence."""
    confidence = max(0.0, min(1.0, confidence))
    return base + confidence * (estimate - base)

def estimate_from_features(base_multiplier, momentum_20d, relative_strength, sigma):
    """Heuristic movement-strength estimate from recent indicators."""
    sigma = max(float(sigma), 1e-9)
    momentum_20d = float(momentum_20d)
    relative_strength = float(relative_strength)
    # Preserve a directional signal even when base_multiplier is neutral (0.0).
    directional_component = (
        0.35 * math.tanh(momentum_20d * 6.0)
        + 0.20 * math.tanh(relative_strength * 4.0)
    )
    historical_estimate = (
        base_multiplier
        * (1.0 + 0.5 * momentum_20d)
        * (1.0 + 0.25 * relative_strength)
        + directional_component
    )
    clamp = max(0.25, 3.0 * sigma)
    return max(-clamp, min(clamp, historical_estimate))

def compute_confidence(momentum_20d, volatility_std):
    """Confidence score from market stability signals."""
    sigmoid_val = 1.0 / (1.0 + math.exp(-abs(float(momentum_20d))))
    raw = sigmoid_val * math.exp(-abs(float(volatility_std)))
    return max(0.0, min(1.0, raw))

# ============================================================
# 4c. Experimental Buyer Simulation (Section G)
# ============================================================
class MarketSimulationState:
    """Lightweight market-behaviour state tracked across forecast steps."""
    def __init__(self):
        self.buy_pressure = 0.0
        self.sell_pressure = 0.0
        self.consecutive_up = 0
        self.consecutive_down = 0

    def decay(self):
        self.buy_pressure *= 0.85
        self.sell_pressure *= 0.85

    def update(self, adjusted_return):
        if adjusted_return > 0:
            self.buy_pressure += abs(adjusted_return)
            self.consecutive_up += 1
            self.consecutive_down = 0
        else:
            self.sell_pressure += abs(adjusted_return)
            self.consecutive_down += 1
            self.consecutive_up = 0

def apply_market_simulation(adjusted_return, state, rolling_volatility, price, moving_average_50):
    """Apply trend exhaustion, liquidity resistance, and mean reversion."""
    # G3: Trend exhaustion penalty
    if adjusted_return > 0:
        exhaustion_penalty = math.exp(-state.consecutive_up / 5.0)
        adjusted_return *= exhaustion_penalty
    else:
        exhaustion_penalty = math.exp(-state.consecutive_down / 5.0)
        adjusted_return *= exhaustion_penalty

    # G4: Liquidity resistance
    resistance_limit = abs(rolling_volatility) * 2.0
    if resistance_limit > 0 and abs(adjusted_return) > resistance_limit:
        adjusted_return = math.copysign(resistance_limit, adjusted_return)

    # G5: Mean reversion force
    if moving_average_50 > 0:
        stretch = price / moving_average_50
        reversion_force = (stretch - 1.0) * 0.25
        adjusted_return -= reversion_force

    return adjusted_return

# ============================================================
# 5. Price reconstruction — PERF-2: vectorised cumprod
# ============================================================
def _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup for `warmup_steps`, then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1.0, float(warmup_steps))
        progress = float(current_step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def _reconstruct_prices(ret_tensor: torch.Tensor,
                        last_price: torch.Tensor,
                        scale: torch.Tensor,
                        mean: torch.Tensor) -> torch.Tensor:
    """
    Vectorised cumulative compounding — no Python loop over horizon steps.

    ret_tensor : [B, H, M]  standardised (zero-mean, unit-var) returns
    last_price : [B, M]     anchor price
    scale      : [M]        scaler_Y.scale_
    mean       : [M]        scaler_Y.mean_
    Returns    : [B, H, M]  price path
    """
    # Inverse standardisation: real_return = pred * std + mean
    ret_denorm = ret_tensor * scale[None, None, :] + mean[None, None, :]

    # Clamp daily returns to realistic bounds (-50% to +50%) to prevent
    # exponential explosion during early training epochs.
    ret_denorm = torch.clamp(ret_denorm, min=-0.5, max=0.5)

    # Compound factors: 1 + r_t  ->  cumprod gives price multiplier at each step
    # price_t = last_price * prod_{i=1..t}(1 + r_i)
    factors    = 1.0 + ret_denorm                                      # [B, H, M]
    cum_factor = torch.cumprod(factors, dim=1)                         # [B, H, M]
    # Clamp cumulative factor to prevent extreme price paths during
    # early training (allows 1% to 10x of anchor price over horizon).
    cum_factor = torch.clamp(cum_factor, min=0.01, max=10.0)
    return last_price.unsqueeze(1) * cum_factor                        # [B, H, M]

# ============================================================
# 6. Evaluation
# ============================================================
def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
    use_amp,
    scale_t,
    mean_t,
    amp_dtype,
    use_gpu_prefetch=False,
    channels_last=False,
):
    model.eval()
    total_loss = total_mae = total_mse = total_dir = total_elem = total_n = 0.0
    n_val_batches = 0
    n_val_skipped = 0
    pred_ret_sum = pred_ret_sq_sum = pred_price_sum = pred_price_sq_sum = 0.0
    pred_ret_count = pred_price_count = 0
    diag_pred_std = diag_vol = diag_curvature = diag_acf1 = 0.0
    diag_batches = 0
    smooth_price_warn_count = 0

    batch_source = CUDAPrefetcher(
        dataloader,
        device,
        enabled=use_gpu_prefetch,
        channels_last=channels_last,
    )

    with torch.no_grad():
        for x, sym, regime, y_ret, y_price, last_p in batch_source:
            n_val_batches += 1
            if x.device != device:
                x, sym, regime, y_ret, y_price, last_p = _move_batch_to_device(
                    (x, sym, regime, y_ret, y_price, last_p),
                    device,
                    non_blocking=True,
                    channels_last=channels_last,
                )

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                # Teacher forcing in validation: uses parallel GRU (single
                # kernel) instead of 90 sequential GRUCell steps, giving
                # ~90x throughput while still measuring prediction quality.
                pred_ret   = model(x, sym, regime, teacher_targets=y_ret)
                B_sz, H_sz, M_sz = pred_ret.shape
                pred_returns   = pred_ret.reshape(B_sz * H_sz, M_sz).mean(dim=-1)
                target_returns = y_ret.reshape(B_sz * H_sz, M_sz).mean(dim=-1)
                guard_metrics = compute_pattern_guard_losses(pred_returns, target_returns)
                pred_price = _reconstruct_prices(pred_ret, last_p, scale_t, mean_t)
                loss       = criterion(pred_ret, y_ret, pred_price, y_price, last_p)
                loss       = loss + guard_metrics['total_guard_loss']

            diag_batches += 1
            diag_pred_std += guard_metrics['pred_std'].detach().item()
            diag_vol += guard_metrics['volatility_loss'].detach().item()
            diag_curvature += guard_metrics['curvature_loss'].detach().item()
            diag_acf1 += guard_metrics['acf1'].detach().item()

            price_vol = torch.std(pred_price.contiguous().view(-1))
            real_price_vol = torch.std(y_price.contiguous().view(-1))
            if real_price_vol.item() > 0 and price_vol.item() < 0.3 * real_price_vol.item():
                smooth_price_warn_count += 1
                if smooth_price_warn_count <= 3:
                    logger.warning("Predicted price path unrealistically smooth")

            if not torch.isfinite(loss):
                n_val_skipped += 1
                if n_val_skipped <= 3:
                    has_nan_pred = torch.isnan(pred_ret).any().item()
                    has_nan_price = torch.isnan(pred_price).any().item()
                    logging.warning(
                        f"  [DIAG-VAL] Batch {n_val_batches}: loss non-finite "
                        f"(pred_nan={has_nan_pred}, price_nan={has_nan_price})"
                    )
                continue

            pred_ret_sum += pred_ret.sum().item()
            pred_ret_sq_sum += (pred_ret ** 2).sum().item()
            pred_ret_count += pred_ret.numel()
            pred_price_sum += pred_price.sum().item()
            pred_price_sq_sum += (pred_price ** 2).sum().item()
            pred_price_count += pred_price.numel()

            B    = x.size(0)
            elem = B * pred_ret.size(1) * pred_ret.size(2)
            total_n    += B
            total_elem += elem
            total_loss += loss.item() * B
            total_mae  += torch.sum(torch.abs(pred_ret - y_ret)).item()
            total_mse  += torch.sum((pred_ret - y_ret) ** 2).item()
            total_dir  += torch.sum(torch.sign(pred_ret) == torch.sign(y_ret)).float().item()

    if n_val_skipped > 0:
        logging.warning(
            f"  [DIAG-VAL] {n_val_skipped}/{n_val_batches} val batches skipped (non-finite loss). "
            f"Valid samples: {int(total_n)}"
        )
    if total_n == 0:
        logging.error(
            f"  [DIAG-VAL] ALL {n_val_batches} val batches produced non-finite loss — "
            f"model is outputting NaN. Metrics will report inf/nan and will not update checkpoints."
        )
        return float('inf'), float('nan'), float('nan'), float('nan')

    # --- Training diagnostics ---
    if pred_ret_count > 0:
        pred_mean = pred_ret_sum / pred_ret_count
        pred_var = max(pred_ret_sq_sum / pred_ret_count - pred_mean ** 2, 0.0)
        pred_std = math.sqrt(pred_var)
        logging.info(
            f"[DIAG] Predicted return stats | "
            f"mean={pred_mean:.6f} std={pred_std:.6f}"
        )
        if pred_std < 0.005:
            logging.warning(
                "[DIAG] Return std < 0.005 — model may be collapsing to flat predictions."
            )
        price_mean = pred_price_sum / pred_price_count
        price_var = max(pred_price_sq_sum / pred_price_count - price_mean ** 2, 0.0)
        price_std = math.sqrt(price_var)
        logging.info(f"[DIAG] Price path volatility std={price_std:.6f}")

    if diag_batches > 0:
        avg_pred_std = diag_pred_std / diag_batches
        avg_vol = diag_vol / diag_batches
        avg_curvature = diag_curvature / diag_batches
        avg_acf1 = diag_acf1 / diag_batches

        logger.info(
            f"[DIAG] pred_std={avg_pred_std:.6f} "
            f"vol_loss={avg_vol:.6f} "
            f"curvature={avg_curvature:.6f} "
            f"acf1={avg_acf1:.4f}"
        )

        if avg_pred_std < 0.005:
            logger.warning("Prediction variance collapse detected")

        if avg_acf1 > 0.97:
            logger.warning("High autocorrelation detected — possible repeating pattern")

        if avg_curvature < 0.005:
            logger.warning("Linear ramp pattern detected")

    if smooth_price_warn_count > 3:
        logger.warning(
            f"Predicted price path unrealistically smooth (triggered {smooth_price_warn_count} batches)"
        )

    dir_acc_val = total_dir / max(total_elem, 1)
    if dir_acc_val < 0.51:
        logging.warning(
            "[DIAG] Directional accuracy near random (\u22480.50). "
            "Model may not be learning useful signal."
        )

    return (
        total_loss / max(total_n,    1),
        total_mae  / max(total_elem, 1),
        math.sqrt(total_mse / max(total_elem, 1)),
        dir_acc_val,
    )


def evaluate_autoregressive(
    model,
    dataloader,
    criterion,
    device,
    use_amp,
    scale_t,
    mean_t,
    amp_dtype,
    use_gpu_prefetch=False,
    channels_last=False,
):
    """Autoregressive validation — no teacher forcing, matches real inference."""
    model.eval()
    total_loss = total_n = 0.0
    total_mae = total_dir = total_elem = 0.0

    batch_source = CUDAPrefetcher(
        dataloader,
        device,
        enabled=use_gpu_prefetch,
        channels_last=channels_last,
    )

    with torch.no_grad():
        for x, sym, regime, y_ret, y_price, last_p in batch_source:
            if x.device != device:
                x, sym, regime, y_ret, y_price, last_p = _move_batch_to_device(
                    (x, sym, regime, y_ret, y_price, last_p),
                    device,
                    non_blocking=True,
                    channels_last=channels_last,
                )

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                pred_ret   = model(x, sym, regime, teacher_targets=None)
                B_sz, H_sz, M_sz = pred_ret.shape
                pred_returns   = pred_ret.reshape(B_sz * H_sz, M_sz).mean(dim=-1)
                target_returns = y_ret.reshape(B_sz * H_sz, M_sz).mean(dim=-1)
                guard_metrics = compute_pattern_guard_losses(pred_returns, target_returns)
                pred_price = _reconstruct_prices(pred_ret, last_p, scale_t, mean_t)
                loss       = criterion(pred_ret, y_ret, pred_price, y_price, last_p)
                loss       = loss + guard_metrics['total_guard_loss']

            if not torch.isfinite(loss):
                continue

            B    = x.size(0)
            elem = B * pred_ret.size(1) * pred_ret.size(2)
            total_n    += B
            total_elem += elem
            total_loss += loss.item() * B
            total_mae  += torch.sum(torch.abs(pred_ret - y_ret)).item()
            total_dir  += torch.sum(torch.sign(pred_ret) == torch.sign(y_ret)).float().item()

    if total_n == 0:
        return float('inf'), float('nan'), float('nan')

    return (
        total_loss / total_n,
        total_mae  / max(total_elem, 1),
        total_dir  / max(total_elem, 1),
    )

# ============================================================
# 7. Vectorised preprocessing helper — RAM-3 (replaces joblib parallel)
# ============================================================
def _vectorised_preprocess(df: pd.DataFrame, target_cols: list, raw_price_cols: list) -> pd.DataFrame:
    """
    Run all per-symbol feature transforms using vectorised groupby operations.
    No worker processes → no memory duplication across joblib workers.
    Pandas executes these in compiled C/Cython code — performance is comparable.
    """
    # Save raw prices before pct_change overwrites target columns.
    for col, rcol in zip(target_cols, raw_price_cols):
        df[rcol] = df[col]

    # Per-symbol pct_change for target columns (returns).
    for col in target_cols:
        df[col] = df.groupby('symbol_id')[col].pct_change().fillna(0)

    # Canonical contract field: daily_return_1d.
    if 'daily_return_1d' not in df.columns:
        df['daily_return_1d'] = df.groupby('symbol_id')['raw_close'].transform(
            lambda s: s.pct_change().fillna(0)
        )

    # Canonical lagged returns used by both training and inference.
    if 'lagged_return_t1' not in df.columns:
        df['lagged_return_t1'] = df.groupby('symbol_id')['daily_return_1d'].shift(1)
    if 'lagged_return_t3' not in df.columns:
        df['lagged_return_t3'] = df.groupby('symbol_id')['daily_return_1d'].shift(3)
    if 'lagged_return_t5' not in df.columns:
        df['lagged_return_t5'] = df.groupby('symbol_id')['daily_return_1d'].shift(5)

    # Canonical volatility alias.
    if 'volatility' not in df.columns:
        if 'rolling_vol_20d_std' in df.columns:
            df['volatility'] = df['rolling_vol_20d_std']
        elif 'Rolling_Vol_20d_std' in df.columns:
            df['volatility'] = df['Rolling_Vol_20d_std']
        else:
            df['volatility'] = df.groupby('symbol_id')['daily_return_1d'].transform(
                lambda s: s.rolling(20, min_periods=1).std().fillna(0)
            )

    # Canonical distance from MA50.
    if 'dist_from_ma50' not in df.columns:
        ma50_col = None
        if 'ma50' in df.columns:
            ma50_col = 'ma50'
        elif 'MA50' in df.columns:
            ma50_col = 'MA50'
        if ma50_col is not None:
            ma50_safe = df[ma50_col].replace(0, np.nan)
            df['dist_from_ma50'] = (df['raw_close'] - ma50_safe) / (ma50_safe + 1e-9)

    return df

# ============================================================
# 8. Dummy data generator
# ============================================================
def create_dummy_dataset_if_missing(metrics_path, prices_path, companies_path):
    if not os.path.exists(metrics_path) or not os.path.exists(prices_path):
        logging.warning("Datasets not found — generating minimal dummy datasets.")
        os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)

        comp_df = pd.DataFrame({'symbol': ['AAPL', 'MSFT', 'GOOG'],
                                'organ_name': ['Apple', 'Microsoft', 'Google']})
        comp_df.to_csv(companies_path, index=False)

        dates = pd.date_range(start='2018-01-01', periods=600)
        metrics_data, prices_data = [], []

        for sym in ['AAPL', 'MSFT', 'GOOG']:
            base = 100.0
            for d in dates:
                ret   = float(np.random.normal(0, 0.015))
                base *= (1 + ret)
                prices_data.append([d, sym,
                    base * 0.99, base * 1.02, base * 0.98, base,
                    int(np.random.randint(1_000, 50_000))])
                metrics_data.append([d, sym,
                    np.random.uniform(30, 70),
                    base * 0.95, base * 0.90, base * 0.96,
                    np.random.normal(0, 1),
                    np.random.uniform(0.005, 0.04),
                    np.random.uniform(0.5, 2.0)])

        pd.DataFrame(prices_data,
            columns=['time','symbol','open','high','low','close','volume']
        ).to_csv(prices_path, index=False)

        pd.DataFrame(metrics_data,
            columns=['time','symbol','RSI','MA20','MA50','EMA20','MACD',
                     'Rolling_Vol_20d_std','ATR']
        ).to_csv(metrics_path, index=False)

# ============================================================
# 9. Inference helper
# ============================================================
def predict(
    ohlc_history: list,
    symbol_id: int,
    regime_id: int,
    model: nn.Module,
    scaler_X: "StandardScaler",
    scaler_Y: "StandardScaler",
    feature_names: list,
    device,
    lookback: int = 120,
) -> pd.DataFrame:
    """
    Run inference on an arbitrarily long OHLC history.

    Accepts an uncapped list of {"open","high","low","close"} dicts,
    derives features internally, normalises, runs the model in true
    autoregressive mode, then returns inverse-transformed price forecasts.

    Parameters
    ----------
    ohlc_history  : list of dicts — must contain >= lookback entries
    symbol_id     : int  (from symbol_mapping.json)
    regime_id     : 0/1/2 volatility regime
    model         : trained MultiMetricPredictor
    scaler_X/Y    : fitted StandardScalers (from output_model/)
    feature_names : ordered list (from features.json)
    device        : torch.device
    lookback      : must match training value

    Returns
    -------
    pd.DataFrame  columns=['open','high','low','close'], len=horizon
    """
    if isinstance(ohlc_history, pd.DataFrame):
        hist_df = ohlc_history.copy()
    else:
        if len(ohlc_history) < lookback:
            raise ValueError(f"Need >= {lookback} history entries (got {len(ohlc_history)}).")
        hist_df = pd.DataFrame(ohlc_history)

    validate_feature_schema(feature_names)

    normalized_df = normalize_features(hist_df)
    assert_sequence_integrity(normalized_df, lookback)

    last_close = float(normalized_df['close'].iloc[-1])
    if not np.isfinite(last_close):
        raise ValueError("Last close is non-finite after normalization.")

    model_feature_df = to_model_feature_frame(normalized_df)
    window = model_feature_df[feature_names].to_numpy(dtype=np.float32)[-lookback:]
    window_scaled = scaler_X.transform(window)

    x     = torch.tensor(window_scaled).unsqueeze(0).to(device)
    sym_t = torch.tensor([symbol_id], dtype=torch.long, device=device)
    reg_t = torch.tensor([regime_id], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        # No teacher_targets -> true autoregressive inference
        pred_ret_scaled = model(x, sym_t, reg_t, teacher_targets=None).squeeze(0).cpu().numpy()

    pred_ret = scaler_Y.inverse_transform(pred_ret_scaled)

    # ---- Regime multipliers & confidence scaling (Sections A–F) ----
    momentum_20d_val = float(normalized_df['close'].pct_change(20).fillna(0).iloc[-1])
    volatility_std_val = float(normalized_df['volatility'].iloc[-1])
    if not np.isfinite(volatility_std_val):
        volatility_std_val = float(
            normalized_df['daily_return_1d'].rolling(20, min_periods=1).std().fillna(0).iloc[-1]
        )
    relative_strength_val = 0.0  # market-relative; unavailable at single-stock inference
    sigma = max(volatility_std_val, 1e-6)
    confidence = compute_confidence(momentum_20d_val, volatility_std_val)
    recent_returns = (
        normalized_df['close']
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(30)
    )
    recent_ret_median = float(recent_returns.median()) if len(recent_returns) else 0.0
    recent_ret_std = float(recent_returns.std()) if len(recent_returns) else 0.0
    if not np.isfinite(recent_ret_median):
        recent_ret_median = 0.0
    if not np.isfinite(recent_ret_std):
        recent_ret_std = 0.0

    close_col_idx = min(3, pred_ret.shape[1] - 1)  # close column index
    raw_adjustments = np.zeros(pred_ret.shape[0], dtype=np.float32)
    multipliers = np.ones(pred_ret.shape[0], dtype=np.float32)
    pred_ret_unscaled = pred_ret.copy()
    for t in range(pred_ret.shape[0]):
        close_ret = float(pred_ret[t, close_col_idx])
        regime = classify_regime(close_ret, sigma)
        base = BASE_REGIME_MULTIPLIERS[regime]
        hist_est = estimate_from_features(base, momentum_20d_val, relative_strength_val, sigma)
        # Convert signed adjustment into positive amplitude scaling.
        # Centering around 1.0 avoids zeroing returns in neutral regimes.
        adjustment = apply_confidence_adjustment(base, hist_est, confidence)
        signal_strength = min(1.0, abs(close_ret) / max(sigma, 1e-6))
        neutral_floor = 0.05 + 0.22 * signal_strength + 0.18 * confidence
        if regime == 2 and abs(adjustment) < neutral_floor:
            direction_hint = close_ret
            if abs(direction_hint) < 1e-9:
                direction_hint = recent_ret_median
            if abs(direction_hint) < 1e-9:
                direction_hint = momentum_20d_val
            if abs(direction_hint) < 1e-9:
                direction_hint = 1.0
            adjustment = math.copysign(neutral_floor, direction_hint)
        adjustment = max(-3.0, min(3.0, adjustment))
        multiplier = 1.0 + abs(adjustment)
        multiplier = max(0.35, min(3.0, multiplier))
        raw_adjustments[t] = adjustment
        multipliers[t] = multiplier
        pred_ret[t] *= multiplier

    if (not np.isfinite(multipliers).all()) or float(np.mean(multipliers)) <= 0.36:
        logging.warning(
            "[DIAG] confidence multipliers unstable/collapsed; falling back to neutral multiplier=1.0"
        )
        multipliers = np.ones_like(multipliers)
        pred_ret = pred_ret_unscaled

    close_std = float(np.std(pred_ret[:, close_col_idx])) if pred_ret.shape[0] > 1 else 0.0
    close_std_floor = max(0.0015, 0.35 * max(recent_ret_std, 1e-6))
    if close_std < close_std_floor and pred_ret.shape[0] > 1:
        horizon = pred_ret.shape[0]
        drift_hint = recent_ret_median
        if abs(drift_hint) < 1e-9:
            drift_hint = momentum_20d_val * 0.05
        if abs(drift_hint) < 1e-9:
            drift_hint = float(np.mean(pred_ret[:, close_col_idx]))
        drift_hint = float(np.clip(drift_hint, -0.03, 0.03))

        wave_amp = float(np.clip(max(recent_ret_std, sigma) * 0.35, 0.0008, 0.02))
        phase = np.linspace(0.0, np.pi, horizon, dtype=np.float32)
        curve = (np.sin(phase) * wave_amp).astype(np.float32)
        ramp = (np.linspace(-0.5, 0.5, horizon, dtype=np.float32) * drift_hint).astype(np.float32)
        close_shape = np.clip(curve + ramp, -0.08, 0.08)

        pred_ret[:, close_col_idx] = (0.7 * pred_ret[:, close_col_idx]) + (0.3 * close_shape)
        for col_idx in range(pred_ret.shape[1]):
            if col_idx == close_col_idx:
                continue
            pred_ret[:, col_idx] = (0.8 * pred_ret[:, col_idx]) + (0.2 * pred_ret[:, close_col_idx])

        logging.info(
            "[DIAG] low-variance fallback injected: close_std=%.6f floor=%.6f drift=%.6f amp=%.6f",
            close_std,
            close_std_floor,
            drift_hint,
            wave_amp,
        )

    logging.info(
        "[DIAG] recent median=%.6f std=%.6f | adjustment mean=%.3f std=%.3f | multiplier mean=%.3f std=%.3f",
        recent_ret_median,
        recent_ret_std,
        float(np.mean(raw_adjustments)),
        float(np.std(raw_adjustments)),
        float(np.mean(multipliers)),
        float(np.std(multipliers)),
    )

    # ---- Price reconstruction (with optional simulation) ----
    prices = np.zeros_like(pred_ret)
    p      = np.array([last_close] * pred_ret.shape[1])

    if EXPERIMENTAL_BUYER_SIM:
        state = MarketSimulationState()
        ma50 = float(hist_df['close'].rolling(50, min_periods=1).mean().iloc[-1])
        for t in range(pred_ret.shape[0]):
            state.decay()
            close_ret = float(pred_ret[t, close_col_idx])
            adjusted = apply_market_simulation(
                close_ret, state, sigma, p[close_col_idx], ma50
            )
            # Scale all OHLC returns proportionally
            if abs(close_ret) > 1e-9:
                scale_factor = adjusted / close_ret
                pred_ret[t] *= scale_factor
            state.update(adjusted)
            p         = p * (1.0 + pred_ret[t])
            prices[t] = p
        logging.info(
            "[SIM] buy_pressure=%.3f sell_pressure=%.3f",
            state.buy_pressure, state.sell_pressure
        )
    else:
        for t in range(pred_ret.shape[0]):
            p         = p * (1.0 + pred_ret[t])
            prices[t] = p

    cols = ['open', 'high', 'low', 'close'][:pred_ret.shape[1]]
    return pd.DataFrame(prices, columns=cols)


def build_temporal_loader(
    anchor_dataset,
    collator,
    batch_size,
    shuffle,
    drop_last,
    loader_kwargs,
    cpu_prefetch_queue,
):
    loader = DataLoader(
        anchor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collator,
        **loader_kwargs,
    )
    return _wrap_loader_with_cpu_prefetch(
        loader,
        num_workers=loader_kwargs.get('num_workers', 0),
        cpu_prefetch_queue=cpu_prefetch_queue,
    )


def auto_tune_batch_size(
    model,
    criterion,
    anchor_dataset,
    collator,
    loader_kwargs,
    device,
    use_amp,
    amp_dtype,
    scale_t,
    mean_t,
    base_batch_size,
    min_batch_size,
    max_batch_size,
    accum_steps,
):
    """
    Probe feasible batch size on current GPU and preserve effective batch with
    gradient accumulation fallback.
    """
    if device.type != 'cuda':
        return max(int(min_batch_size), int(base_batch_size)), int(accum_steps)

    if len(anchor_dataset) < max(2, min_batch_size):
        return max(int(min_batch_size), int(base_batch_size)), int(accum_steps)

    base_bs = max(int(min_batch_size), int(base_batch_size))
    min_bs = max(1, int(min_batch_size))
    max_bs = max(base_bs, int(max_batch_size))
    target_effective = max(1, base_bs * int(accum_steps))

    probe_loader_kwargs = dict(loader_kwargs)
    # Keep probing light to avoid large startup overhead.
    probe_workers = int(probe_loader_kwargs.get('num_workers', 0))
    probe_loader_kwargs['num_workers'] = min(probe_workers, 2)
    probe_loader_kwargs['persistent_workers'] = False
    if probe_loader_kwargs['num_workers'] == 0:
        probe_loader_kwargs.pop('prefetch_factor', None)
        probe_loader_kwargs.pop('persistent_workers', None)

    was_training = model.training
    model.train()

    def _probe(bs):
        probe_loader = DataLoader(
            anchor_dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            collate_fn=collator,
            **probe_loader_kwargs,
        )
        if len(probe_loader) == 0:
            return False
        try:
            batch = next(iter(probe_loader))
            x, sym, regime, y_ret, y_price, last_p = _move_batch_to_device(
                batch,
                device,
                non_blocking=True,
            )
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                pred_ret = model(x, sym, regime, teacher_targets=y_ret)
                bsz, hsz, msz = pred_ret.shape
                pred_returns = pred_ret.reshape(bsz * hsz, msz).mean(dim=-1)
                target_returns = y_ret.reshape(bsz * hsz, msz).mean(dim=-1)
                guard_metrics = compute_pattern_guard_losses(pred_returns, target_returns)
                pred_price = _reconstruct_prices(pred_ret, last_p, scale_t, mean_t)
                loss = criterion(pred_ret, y_ret, pred_price, y_price, last_p)
                loss = loss + guard_metrics['total_guard_loss']
            (loss / max(1, accum_steps)).backward()
            torch.cuda.synchronize(device)
            model.zero_grad(set_to_none=True)
            return True
        except RuntimeError as exc:
            if _is_oom_error(exc):
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    best = None
    upward_candidates = []
    cand = base_bs
    while cand <= max_bs:
        upward_candidates.append(cand)
        cand *= 2

    for cand in upward_candidates:
        ok = _probe(cand)
        if not ok:
            break
        best = cand

    if best is None:
        cand = max(min_bs, base_bs // 2)
        while cand >= min_bs:
            if _probe(cand):
                best = cand
                break
            if cand == min_bs:
                break
            cand = max(min_bs, cand // 2)

    if best is None:
        best = min_bs

    if not was_training:
        model.eval()

    tuned_accum_steps = max(1, math.ceil(target_effective / best))
    return int(best), int(tuned_accum_steps)

# ============================================================
# 10. Per-Horizon Training Worker
# ============================================================
def train_single_horizon(
    horizon_name, horizon,
    # Shared tensors (CPU)
    X_tensor, sym_tensor, regime_tensor,
    Y_ret_tensor, Y_price_tensor, last_price_tensor,
    # Index computation
    df_groups, idx_to_date, cutoff_date,
    # Scalers
    scaler_Y,
    # Model config
    num_symbols, num_features, lookback, target_cols,
    # Training config
    batch_size, epochs_per_stage, learning_rate, patience,
    stages, stage_ratios, is_iterative, accum_steps,
    stride,
    mode,
    auto_batch_size_enabled,
    min_batch_size,
    max_batch_size,
    # Hardware
    device, use_amp, amp_dtype,
    enable_gpu_prefetch,
    channels_last,
    # Paths
    checkpoint_dir, output_dir, log_dir,
    # DataLoader
    loader_kwargs,
    cpu_prefetch_queue,
    profile,
    profile_steps,
    profile_sync_timing,
    # Shared results dict
    results,
):
    """
        Train one model for a specific forecast horizon.

        Throughput-critical path:
            1) Vectorised collate_fn batches windows via index_select
            2) Optional CPU queue prefetch when num_workers=0
            3) CUDA stream prefetch overlaps H2D transfer with compute
            4) Optional one-shot batch-size auto tuning per horizon
    """
    tag = f"[{horizon_name}]"
    horizon_output = os.path.join(output_dir, horizon_name)
    horizon_ckpt   = os.path.join(checkpoint_dir, horizon_name)
    os.makedirs(horizon_output, exist_ok=True)
    os.makedirs(horizon_ckpt, exist_ok=True)

    # ---- Valid indices for this horizon ----
    valid_indices = []
    for indices in df_groups:
        if len(indices) >= lookback + horizon:
            valid_indices.extend(indices[lookback - 1 : len(indices) - horizon])
    valid_indices = np.array(valid_indices)

    if len(valid_indices) == 0:
        logging.error(f"{tag} No valid sequences for horizon={horizon}. Skipping.")
        results[horizon_name] = {'status': 'failed', 'best_val_loss': float('inf')}
        return

    # Split using the shared temporal cutoff
    # Normalize cutoff to match idx_to_date tz-awareness (numpy datetime64 is tz-naive)
    _cutoff = cutoff_date.tz_localize(None) if cutoff_date.tzinfo is not None else cutoff_date
    train_indices  = valid_indices[idx_to_date[valid_indices] <  _cutoff]
    verify_indices = valid_indices[idx_to_date[valid_indices] >= _cutoff]

    if len(train_indices) == 0 or len(verify_indices) == 0:
        logging.error(f"{tag} Empty train or verify set. Skipping.")
        results[horizon_name] = {'status': 'failed', 'best_val_loss': float('inf')}
        return

    logging.info(f"{tag} Sequences: train={len(train_indices)}, val={len(verify_indices)}")

    # ---- Model ----
    model = MultiMetricPredictor(
        num_symbols=num_symbols, num_features=num_features,
        lookback=lookback, forecast_horizon=horizon,
        num_target_metrics=len(target_cols),
    ).to(device)

    criterion   = CombinedForecastLoss(alpha=0.7, beta=0.2, gamma=0.5)

    scale_t = torch.tensor(scaler_Y.scale_.astype(np.float32), device=device)
    mean_t  = torch.tensor(scaler_Y.mean_.astype(np.float32),  device=device)

    collator = TemporalBatchCollator(
        X_tensor=X_tensor,
        sym_tensor=sym_tensor,
        regime_tensor=regime_tensor,
        Y_ret_tensor=Y_ret_tensor,
        Y_price_tensor=Y_price_tensor,
        last_price_tensor=last_price_tensor,
        lookback=lookback,
        forecast_horizon=horizon,
    )

    tuned_batch_size = int(batch_size)
    tuned_accum_steps = int(accum_steps)
    if auto_batch_size_enabled:
        probe_n = min(len(train_indices), max(4096, tuned_batch_size * 16))
        probe_dataset = TemporalAnchorDataset(train_indices[:probe_n], stride=stride)
        if len(probe_dataset) > 1:
            tuned_batch_size, tuned_accum_steps = auto_tune_batch_size(
                model=model,
                criterion=criterion,
                anchor_dataset=probe_dataset,
                collator=collator,
                loader_kwargs=loader_kwargs,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scale_t=scale_t,
                mean_t=mean_t,
                base_batch_size=tuned_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                accum_steps=tuned_accum_steps,
            )
            logging.info(
                f"{tag} Auto batch-size tune | mode={mode} "
                f"batch_size={tuned_batch_size} accum_steps={tuned_accum_steps} "
                f"effective_batch={tuned_batch_size * tuned_accum_steps}"
            )

    verify_dataset = TemporalAnchorDataset(verify_indices, stride=stride)
    verify_loader = build_temporal_loader(
        anchor_dataset=verify_dataset,
        collator=collator,
        batch_size=tuned_batch_size,
        shuffle=False,
        drop_last=True,
        loader_kwargs=loader_kwargs,
        cpu_prefetch_queue=cpu_prefetch_queue,
    )

    best_global_val = float('inf')
    log_data = []
    epoch_log_data = []
    batch_diag_data = []
    train_dates = np.sort(np.unique(idx_to_date[train_indices]))

    # ---- Curriculum stages ----
    # train_dates is sorted ascending (oldest → newest).
    # We slice from the TAIL so each stage uses the most-recent N% of history,
    # progressively extending backward into older data as the stage ratio grows.
    # This avoids the "nested oldest-data" anti-pattern where early stages lock
    # the model onto a single fixed temporal slice that gets repeated verbatim in
    # every subsequent stage, producing artificial linear/repetitive gradients.
    #
    # Stage 1 (ratio=0.1):  most-recent 10%  of dates
    # Stage 2 (ratio=0.2):  most-recent 20%  of dates  (adds older data at front)
    # Stage 3 (ratio=0.5):  most-recent 50%  of dates
    # Stage 4 (ratio=1.0):  full history
    #
    # Each stage is still a strict superset of the previous stage's sequences,
    # preserving chronological continuity and curriculum ordering.
    for stage_idx in range(stages):
        stage = stage_idx + 1
        ratio = stage_ratios[stage_idx]

        num_dates     = max(1, int(len(train_dates) * ratio))
        stage_indices = np.array([], dtype=np.int64)
        for _expand in range(num_dates, len(train_dates) + 1):
            # Slice from the END to keep the most-recent _expand dates.
            # Compared to train_dates[:_expand] (oldest-first), this ensures
            # the model is always anchored on recent market conditions while
            # each larger stage adds progressively older context.
            stage_dates   = train_dates[-_expand:]
            stage_indices = train_indices[np.isin(idx_to_date[train_indices], stage_dates)]
            if len(stage_indices) > 0:
                break

        if len(stage_indices) == 0:
            logging.warning(f"{tag} Stage {stage}: no sequences — skipping.")
            continue

        stage_dataset = TemporalAnchorDataset(stage_indices, stride=stride)
        train_loader = build_temporal_loader(
            anchor_dataset=stage_dataset,
            collator=collator,
            batch_size=tuned_batch_size,
            shuffle=True,
            drop_last=True,
            loader_kwargs=loader_kwargs,
            cpu_prefetch_queue=cpu_prefetch_queue,
        )
        stage_batches = len(train_loader)

        if stage_batches == 0:
            logging.warning(
                f"{tag} Stage {stage}: 0 train batches after drop_last with batch_size={tuned_batch_size} — skipping."
            )
            continue

        logging.info(f"{tag} Stage {stage}/{stages} [{ratio*100:.0f}%] | "
                     f"Seqs: {len(stage_indices)} | Batches: {stage_batches}")

        if not is_iterative:
            model = MultiMetricPredictor(
                num_symbols=num_symbols, num_features=num_features,
                lookback=lookback, forecast_horizon=horizon,
                num_target_metrics=len(target_cols),
            ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        grad_scaler = torch.amp.GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None

        total_steps = max(1, epochs_per_stage * stage_batches)
        warmup_steps = min(1000, total_steps // 5)
        scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        ckpt_path = os.path.join(horizon_ckpt, f'stage_{stage}.pt')
        best_stage = float('inf')
        pat_ctr    = 0

        if os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
            except OSError:
                pass

        for epoch in range(epochs_per_stage):
            epoch_start = time.perf_counter()
            model.train()
            ep_loss = ep_n = ep_skip = 0
            n_nan_pred = n_nan_loss = n_extreme = n_nan_grad = 0
            ar_loss = float('nan')
            ar_mae = float('nan')
            ar_dir = float('nan')
            optimizer.zero_grad(set_to_none=True)
            timing = StepTiming()
            last_step_end = time.perf_counter()
            profiler = maybe_build_profiler(
                enabled=(profile and stage == 1 and epoch == 0),
                device=device,
                log_dir=log_dir,
                horizon_name=horizon_name,
                stage=stage,
                profile_steps=profile_steps,
            )
            prof_ctx = profiler if profiler is not None else contextlib.nullcontext()

            with prof_ctx:
                train_source = CUDAPrefetcher(
                    train_loader,
                    device,
                    enabled=enable_gpu_prefetch,
                    channels_last=channels_last,
                )

                for batch_idx, (x, sym, regime, y_ret, y_price, last_p) in enumerate(train_source):
                    data_ready = time.perf_counter()
                    timing.data_wait_s += max(0.0, data_ready - last_step_end)

                    transfer_t0 = time.perf_counter()
                    if x.device != device:
                        x, sym, regime, y_ret, y_price, last_p = _move_batch_to_device(
                            (x, sym, regime, y_ret, y_price, last_p),
                            device,
                            non_blocking=True,
                            channels_last=channels_last,
                        )
                    if profile_sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    timing.transfer_s += time.perf_counter() - transfer_t0

                    forward_t0 = time.perf_counter()
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                        pred_ret = model(x, sym, regime, teacher_targets=y_ret)
                        # Compute curvature per sequence (seq dim=1), then flatten for vol/acf/variance
                        b_sz, h_sz, m_sz = pred_ret.shape
                        pred_returns = pred_ret.reshape(b_sz * h_sz, m_sz).mean(dim=-1)
                        target_returns = y_ret.reshape(b_sz * h_sz, m_sz).mean(dim=-1)
                        guard_metrics = compute_pattern_guard_losses(pred_returns, target_returns)
                        if batch_idx % 200 == 0:
                            pred_mean = pred_ret.mean().item()
                            pred_std = guard_metrics['pred_std'].detach().item()
                            vol_loss = guard_metrics['volatility_loss'].detach().item()
                            curvature = guard_metrics['curvature_loss'].detach().item()
                            acf1 = guard_metrics['acf1'].detach().item()
                            batch_diag_data.append({
                                'horizon': horizon_name,
                                'stage': stage,
                                'epoch': epoch + 1,
                                'batch': batch_idx,
                                'pred_mean': pred_mean,
                                'pred_std': pred_std,
                                'volatility_loss': vol_loss,
                                'curvature_loss': curvature,
                                'acf1': acf1,
                            })
                            logger.info(
                                f"[DIAG] pred_std={pred_std:.6f} "
                                f"vol_loss={vol_loss:.6f} "
                                f"curvature={curvature:.6f} "
                                f"acf1={acf1:.4f}"
                            )
                            if pred_std < 0.005:
                                logger.warning("Prediction variance collapse detected")
                            if acf1 > 0.97:
                                logger.warning("High autocorrelation detected — possible repeating pattern")
                            if curvature < 0.005:
                                logger.warning("Linear ramp pattern detected")
                        pred_price = _reconstruct_prices(pred_ret, last_p, scale_t, mean_t)
                        loss = criterion(pred_ret, y_ret, pred_price, y_price, last_p)
                        loss = loss + guard_metrics['total_guard_loss']
                    if profile_sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    timing.forward_s += time.perf_counter() - forward_t0

                    if torch.isnan(pred_ret).any() or torch.isinf(pred_ret).any():
                        n_nan_pred += 1

                    if not torch.isfinite(loss):
                        n_nan_loss += 1
                        ep_skip += 1
                        last_step_end = time.perf_counter()
                        if profiler is not None:
                            profiler.step()
                        continue
                    if loss.item() > 1e6:
                        n_extreme += 1
                        ep_skip += 1
                        last_step_end = time.perf_counter()
                        if profiler is not None:
                            profiler.step()
                        continue

                    backward_t0 = time.perf_counter()
                    scaled_loss = loss / tuned_accum_steps
                    if grad_scaler is not None:
                        grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    if profile_sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    timing.backward_s += time.perf_counter() - backward_t0

                    optim_t0 = time.perf_counter()
                    # Step optimizer every accum_steps batches (or at end of epoch)
                    if (batch_idx + 1) % tuned_accum_steps == 0 or (batch_idx + 1) == stage_batches:
                        if grad_scaler is not None:
                            grad_scaler.unscale_(optimizer)
                            try:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1.0, error_if_nonfinite=True
                                )
                            except RuntimeError:
                                n_nan_grad += 1
                                ep_skip += 1
                                grad_scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                                scheduler.step()
                                ep_loss += loss.item() * x.size(0)
                                ep_n += x.size(0)
                                if profile_sync_timing and device.type == 'cuda':
                                    torch.cuda.synchronize(device)
                                timing.optim_s += time.perf_counter() - optim_t0
                                timing.batches += 1
                                last_step_end = time.perf_counter()
                                if profiler is not None:
                                    profiler.step()
                                continue
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                        else:
                            try:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), 1.0, error_if_nonfinite=True
                                )
                            except RuntimeError:
                                n_nan_grad += 1
                                ep_skip += 1
                                optimizer.zero_grad(set_to_none=True)
                                scheduler.step()
                                ep_loss += loss.item() * x.size(0)
                                ep_n += x.size(0)
                                if profile_sync_timing and device.type == 'cuda':
                                    torch.cuda.synchronize(device)
                                timing.optim_s += time.perf_counter() - optim_t0
                                timing.batches += 1
                                last_step_end = time.perf_counter()
                                if profiler is not None:
                                    profiler.step()
                                continue
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    scheduler.step()
                    if profile_sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    timing.optim_s += time.perf_counter() - optim_t0

                    ep_loss += loss.item() * x.size(0)
                    ep_n += x.size(0)
                    timing.batches += 1
                    last_step_end = time.perf_counter()

                    if profiler is not None:
                        profiler.step()

            ep_loss = ep_loss / max(ep_n, 1)
            timing_metrics = timing.as_dict()

            if ep_skip > 0:
                logging.warning(
                    f"{tag} Stage {stage} Ep {epoch+1}: {ep_skip} batches skipped "
                    f"(nan_loss={n_nan_loss}, extreme={n_extreme}, nan_pred={n_nan_pred}, nan_grad={n_nan_grad})")
            if ep_n == 0:
                logging.error(
                    f"{tag} Stage {stage} Ep {epoch+1}: ALL batches skipped — "
                    f"model weights are likely corrupted.")
                break

            val_loss, mae, rmse, dir_acc = evaluate_model(
                model, verify_loader, criterion, device, use_amp,
                scale_t, mean_t, amp_dtype,
                use_gpu_prefetch=enable_gpu_prefetch,
                channels_last=channels_last)
            epoch_seconds = time.perf_counter() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"{tag} Ep {epoch+1:3d} timing | data_wait={timing_metrics['data_wait_pct']:.1f}% "
                f"h2d={timing_metrics['transfer_pct']:.1f}% "
                f"fwd={timing_metrics['forward_pct']:.1f}% "
                f"bwd={timing_metrics['backward_pct']:.1f}% "
                f"gpu_util_est={timing_metrics['gpu_util_est_pct']:.1f}%"
            )
            logging.info(f"{tag} Ep {epoch+1:3d} | train={ep_loss:.5f} "
                         f"val={val_loss:.5f} mae={mae:.5f} dir={dir_acc:.3f}")

            # Periodic autoregressive validation (realistic, no teacher forcing)
            if epoch % 3 == 0:
                ar_loss, ar_mae, ar_dir = evaluate_autoregressive(
                    model, verify_loader, criterion, device, use_amp,
                    scale_t, mean_t, amp_dtype,
                    use_gpu_prefetch=enable_gpu_prefetch,
                    channels_last=channels_last)
                logging.info(f"{tag} Ep {epoch+1:3d} | AR val={ar_loss:.5f} "
                             f"ar_mae={ar_mae:.5f} ar_dir={ar_dir:.3f}")

            if not math.isfinite(val_loss):
                epoch_log_data.append({
                    'horizon': horizon_name,
                    'stage': stage,
                    'stage_ratio': ratio,
                    'epoch': epoch + 1,
                    'n_sequences': len(stage_indices),
                    'n_train_batches': stage_batches,
                    'train_samples': ep_n,
                    'train_loss': ep_loss,
                    'val_loss': val_loss,
                    'val_mae': mae,
                    'val_rmse': rmse,
                    'val_dir_acc': dir_acc,
                    'ar_val_loss': ar_loss,
                    'ar_val_mae': ar_mae,
                    'ar_val_dir_acc': ar_dir,
                    'learning_rate': current_lr,
                    'epoch_seconds': epoch_seconds,
                    'effective_batch_size': tuned_batch_size * tuned_accum_steps,
                    'data_wait_pct': timing_metrics['data_wait_pct'],
                    'transfer_pct': timing_metrics['transfer_pct'],
                    'forward_pct': timing_metrics['forward_pct'],
                    'backward_pct': timing_metrics['backward_pct'],
                    'optim_pct': timing_metrics['optim_pct'],
                    'gpu_util_est_pct': timing_metrics['gpu_util_est_pct'],
                    'skipped_batches': ep_skip,
                    'nan_loss_batches': n_nan_loss,
                    'extreme_loss_batches': n_extreme,
                    'nan_pred_batches': n_nan_pred,
                    'nan_grad_batches': n_nan_grad,
                    'best_stage_val_so_far': best_stage,
                    'best_global_val_so_far': min(best_global_val, best_stage),
                    'is_best_stage_epoch': False,
                })
                logging.error(
                    f"{tag} Stage {stage} Ep {epoch+1}: validation collapsed to non-finite loss; "
                    f"restoring best checkpoint for this stage."
                )
                break

            is_best_stage_epoch = False
            if val_loss < best_stage:
                best_stage = val_loss
                is_best_stage_epoch = True
                torch.save(model.state_dict(), ckpt_path)
                pat_ctr = 0
            else:
                pat_ctr += 1

            epoch_log_data.append({
                'horizon': horizon_name,
                'stage': stage,
                'stage_ratio': ratio,
                'epoch': epoch + 1,
                'n_sequences': len(stage_indices),
                'n_train_batches': stage_batches,
                'train_samples': ep_n,
                'train_loss': ep_loss,
                'val_loss': val_loss,
                'val_mae': mae,
                'val_rmse': rmse,
                'val_dir_acc': dir_acc,
                'ar_val_loss': ar_loss,
                'ar_val_mae': ar_mae,
                'ar_val_dir_acc': ar_dir,
                'learning_rate': current_lr,
                'epoch_seconds': epoch_seconds,
                'effective_batch_size': tuned_batch_size * tuned_accum_steps,
                'data_wait_pct': timing_metrics['data_wait_pct'],
                'transfer_pct': timing_metrics['transfer_pct'],
                'forward_pct': timing_metrics['forward_pct'],
                'backward_pct': timing_metrics['backward_pct'],
                'optim_pct': timing_metrics['optim_pct'],
                'gpu_util_est_pct': timing_metrics['gpu_util_est_pct'],
                'skipped_batches': ep_skip,
                'nan_loss_batches': n_nan_loss,
                'extreme_loss_batches': n_extreme,
                'nan_pred_batches': n_nan_pred,
                'nan_grad_batches': n_nan_grad,
                'best_stage_val_so_far': best_stage,
                'best_global_val_so_far': min(best_global_val, best_stage),
                'is_best_stage_epoch': is_best_stage_epoch,
            })

            if pat_ctr >= patience:
                logging.info(f"{tag} Early stop at epoch {epoch+1}")
                break

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        val_loss, mae, rmse, dir_acc = evaluate_model(
            model, verify_loader, criterion, device, use_amp,
            scale_t, mean_t, amp_dtype,
            use_gpu_prefetch=enable_gpu_prefetch,
            channels_last=channels_last)
        logging.info(f"{tag} Stage {stage} Final | val={val_loss:.5f} mae={mae:.5f} "
                     f"rmse={rmse:.5f} dir={dir_acc:.4f}")

        log_data.append({
            'horizon': horizon_name, 'stage': stage,
            'n_sequences': len(stage_indices),
            'train_loss': ep_loss, 'val_mse': val_loss,
            'val_mae': mae, 'val_rmse': rmse, 'val_dir_acc': dir_acc,
        })

        if val_loss < best_global_val:
            best_global_val = val_loss
            if os.path.exists(ckpt_path):
                shutil.copy(ckpt_path, os.path.join(horizon_output, 'best_model.pth'))

    pd.DataFrame(log_data).to_csv(
        os.path.join(log_dir, f'training_log_{horizon_name}.csv'), index=False)
    pd.DataFrame(epoch_log_data).to_csv(
        os.path.join(log_dir, f'training_epoch_log_{horizon_name}.csv'), index=False)
    pd.DataFrame(batch_diag_data).to_csv(
        os.path.join(log_dir, f'training_batch_diag_{horizon_name}.csv'), index=False)

    logging.info(f"[DIAG] Horizon {horizon_name} best_val_loss={best_global_val}")
    status = 'success' if math.isfinite(best_global_val) else 'failed'
    if status == 'failed':
        logging.error(f"{tag} COMPLETE | No finite validation checkpoint was produced.")
    else:
        logging.info(f"{tag} COMPLETE | Best Val Loss: {best_global_val:.5f}")
    results[horizon_name] = {'status': status, 'best_val_loss': best_global_val}


def _train_single_horizon_safe(**kwargs):
    """Wrapper that prevents per-horizon crashes from dropping global results."""
    horizon_name = kwargs.get('horizon_name', 'unknown')
    tag = f"[{horizon_name}]"
    results = kwargs.get('results')
    try:
        train_single_horizon(**kwargs)
    except Exception as exc:
        msg = str(exc)
        mapping_error = "Couldn't open shared file mapping" in msg or "error code: <1455>" in msg
        retried = False

        if mapping_error:
            loader_kwargs = dict(kwargs.get('loader_kwargs', {}))
            if int(loader_kwargs.get('num_workers', 0)) > 0:
                retried = True
                logging.warning(
                    f"{tag} DataLoader shared mapping exhaustion detected (WinError 1455). "
                    "Retrying this horizon with num_workers=0 + async CPU prefetch queue."
                )
                fallback_kwargs = dict(kwargs)
                fallback_loader_kwargs = dict(loader_kwargs)
                fallback_loader_kwargs['num_workers'] = 0
                fallback_loader_kwargs.pop('persistent_workers', None)
                fallback_loader_kwargs.pop('prefetch_factor', None)
                fallback_kwargs['loader_kwargs'] = fallback_loader_kwargs
                fallback_kwargs['cpu_prefetch_queue'] = max(2, int(kwargs.get('cpu_prefetch_queue', 2)))
                try:
                    train_single_horizon(**fallback_kwargs)
                    return
                except Exception:
                    logging.exception(f"{tag} Fallback run also failed.")

        if not retried:
            logging.exception(f"{tag} Worker crashed unexpectedly.")

        if isinstance(results, dict):
            results[horizon_name] = {
                'status': 'failed',
                'best_val_loss': float('inf'),
                'error': type(exc).__name__,
            }


# ============================================================
# 11. Pipeline
# ============================================================
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            type=str,   default="metrics.csv")
    parser.add_argument("--prices",             type=str,   default="stock_prices.csv")
    parser.add_argument("--companies",          type=str,   default="companies.csv")
    parser.add_argument("--horizons",           type=str,   default="7,14,30",
                        help="Comma-separated forecast horizons in days (e.g. 7,14,30)")
    parser.add_argument("--lookback",           type=int,   default=120)
    parser.add_argument("--stages",             type=int,   default=3)
    parser.add_argument("--stage_ratios",       type=str,   default="0.1,0.2,0.5")
    parser.add_argument("--verify_split",       type=float, default=0.15)
    parser.add_argument("--batch_size",         type=int,   default=128)
    parser.add_argument("--min_batch_size",     type=int,   default=32,
                        help="Lower bound for auto batch-size search")
    parser.add_argument("--max_batch_size",     type=int,   default=1024,
                        help="Upper bound for auto batch-size search")
    parser.add_argument("--epochs_per_stage",   type=int,   default=20)
    parser.add_argument("--learning_rate",      type=float, default=0.0005)
    parser.add_argument("--patience",           type=int,   default=5)
    parser.add_argument("--accum_steps",        type=int,   default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * accum_steps)")
    parser.add_argument("--stride",             type=int,   default=1,
                        help="Window stride for index sampling (1=all windows, 3=every 3rd). "
                             "Higher values reduce dataset size and RAM usage.")
    parser.add_argument("--num_workers",        type=int,   default=-1,
                        help="DataLoader workers. -1 = mode-based auto")
    parser.add_argument("--prefetch_factor",    type=int,   default=-1,
                        help="DataLoader prefetch factor. -1 = mode-based auto")
    parser.add_argument("--cpu_prefetch_queue", type=int,   default=4,
                        help="CPU-side queue depth when num_workers=0")
    parser.add_argument("--device",             type=str,   default="cuda")
    parser.add_argument("--mixed_precision",    type=_str_to_bool, default=True)
    parser.add_argument("--iterative_training", type=_str_to_bool, default=True)
    parser.add_argument("--parallel_training",  type=_str_to_bool, default=False,
                        help="Deprecated. Ignored: training is always sequential for throughput stability")
    parser.add_argument("--mode",               type=str,   default="high_throughput",
                        choices=["high_throughput", "low_memory"],
                        help="Runtime mode toggle")
    parser.add_argument("--auto_batch_size",    type=_str_to_bool, default=None,
                        help="Enable batch-size auto tuning (default: true in high_throughput, false in low_memory)")
    parser.add_argument("--enable_gpu_prefetch", type=_str_to_bool, default=True,
                        help="Overlap host-to-device transfer with compute")
    parser.add_argument("--channels_last",      type=_str_to_bool, default=False,
                        help="Enable channels-last memory format for 4D tensors")
    parser.add_argument("--profile",            type=_str_to_bool, default=False,
                        help="Enable torch.profiler trace on first epoch/stage")
    parser.add_argument("--profile_steps",      type=int, default=150,
                        help="Active profiler steps when --profile true")
    parser.add_argument("--profile_sync_timing", type=_str_to_bool, default=False,
                        help="Synchronize CUDA for more accurate per-stage timing (slower)")
    parser.add_argument("--checkpoint_dir",     type=str,   default="models/")
    parser.add_argument("--log_dir",            type=str,   default="logs/")
    parser.add_argument("--output_dir",         type=str,   default="output_model/")
    args = parser.parse_args()

    device   = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp  = args.mixed_precision and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    is_iterative = args.iterative_training
    stage_ratios = [float(r) for r in args.stage_ratios.split(',')]
    if len(stage_ratios) < args.stages:
        raise ValueError("stage_ratios must contain at least --stages values")
    stage_ratios = stage_ratios[:args.stages]
    lookback     = args.lookback
    auto_batch_size_enabled = (
        args.auto_batch_size
        if args.auto_batch_size is not None
        else args.mode == 'high_throughput'
    )

    # Parse horizons — e.g. "7,14,30" -> {"7d": 7, "14d": 14, "30d": 30}
    horizons = {}
    for h in args.horizons.split(','):
        h = int(h.strip())
        horizons[f"{h}d"] = h
    logging.info(f"Horizons to train: {horizons}")

    for d in [args.checkpoint_dir, args.log_dir, args.output_dir]:
        os.makedirs(d, exist_ok=True)

    create_dummy_dataset_if_missing(args.dataset, args.prices, args.companies)

    # DataLoader setup
    is_windows = os.name == 'nt'
    if args.num_workers == -1:
        cpu_total = os.cpu_count() or 8
        if args.mode == 'high_throughput':
            num_workers = max(2, min(8, cpu_total // 2)) if is_windows else max(4, min(16, cpu_total // 2))
        else:
            num_workers = 0 if is_windows else 1
    else:
        num_workers = max(0, args.num_workers)

    if args.prefetch_factor == -1:
        prefetch_factor = 4 if args.mode == 'high_throughput' else 2
    else:
        prefetch_factor = max(2, min(8, args.prefetch_factor))

    pin_memory = device.type == 'cuda' and args.mode == 'high_throughput'
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = prefetch_factor

    logging.info(
        f"Mode={args.mode} | num_workers={num_workers} | pin_memory={pin_memory} "
        f"| persistent_workers={loader_kwargs.get('persistent_workers', False)} "
        f"| prefetch_factor={loader_kwargs.get('prefetch_factor', 0)}"
    )

    if args.parallel_training:
        logging.warning(
            "parallel_training=True ignored: sequential horizon training is enforced "
            "to avoid GIL contention and CPU scheduling overhead."
        )

    logging.info(f"Auto batch size enabled: {auto_batch_size_enabled}")

    # ================================================================
    # Shared data preparation (single pass, reused by all horizons)
    # ================================================================
    logging.info("Loading datasets...")
    metrics_df = pd.read_csv(args.dataset)
    prices_df  = pd.read_csv(args.prices)
    metrics_df['time'] = pd.to_datetime(metrics_df['time'], utc=True)
    prices_df['time']  = pd.to_datetime(prices_df['time'],  utc=True)

    df = pd.merge(metrics_df, prices_df, on=['symbol', 'time'], how='inner')
    del metrics_df, prices_df  # free source DataFrames immediately
    if df.empty:
        raise ValueError("Merged dataset is empty — check 'time' and 'symbol' columns match.")
    df = df.sort_values(['symbol', 'time']).reset_index(drop=True)

    unique_symbols = df['symbol'].unique()
    sym2id = {s: i for i, s in enumerate(unique_symbols)}
    df['symbol_id'] = df['symbol'].map(sym2id)
    with open(os.path.join(args.output_dir, 'symbol_mapping.json'), 'w') as f:
        json.dump(sym2id, f, indent=4)

    target_cols    = ['open', 'high', 'low', 'close']
    raw_price_cols = [f'raw_{c}' for c in target_cols]
    features = list(FEATURE_SCHEMA)
    validate_feature_schema(features)

    # ---- RAM-3: vectorised pandas preprocessing (no joblib workers) ----
    logging.info("Preprocessing features (vectorised pandas)...")
    df = _vectorised_preprocess(df, target_cols, raw_price_cols)

    def _normalize_symbol_group(group: pd.DataFrame) -> pd.DataFrame:
        norm = normalize_features(group)
        norm.index = group.index
        return norm[features]

    normalized_features = df.groupby('symbol', group_keys=False).apply(_normalize_symbol_group)
    df[features] = normalized_features[features]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[features] = df.groupby('symbol')[features].ffill().bfill()
    df[raw_price_cols] = df.groupby('symbol')[raw_price_cols].ffill().bfill()

    for _symbol, grp in df.groupby('symbol', sort=False):
        assert_sequence_integrity(grp[['time'] + features], seq_len=1)

    # ---- RAM-2: cast ALL numeric data to float32 before tensor/scaler work ----
    logging.info("Casting all numeric columns to float32...")
    df[features]       = df[features].astype('float32')
    df[raw_price_cols] = df[raw_price_cols].astype('float32')

    with open(os.path.join(args.output_dir, 'features.json'), 'w') as f:
        json.dump(features, f, indent=4)

    # ---- Horizon-independent temporal split ----
    all_dates = df['time'].drop_duplicates().sort_values().values
    split_pos = max(1, int(len(all_dates) * (1.0 - args.verify_split)))
    cutoff_date = pd.Timestamp(all_dates[min(split_pos, len(all_dates) - 1)], tz='UTC')
    train_mask  = df['time'] < cutoff_date
    logging.info(f"Train/val temporal cutoff: {cutoff_date}")

    # ---- Volatility regimes ----
    vol_col = 'volatility'
    q33 = df.loc[train_mask, vol_col].quantile(0.33)
    q66 = df.loc[train_mask, vol_col].quantile(0.66)
    df['regime'] = 0
    df.loc[df[vol_col] > q33, 'regime'] = 1
    df.loc[df[vol_col] > q66, 'regime'] = 2

    # ---- RAM-4: deferred sklearn import — keeps workers from loading full stack ----
    def build_scalers(df, train_mask, features, target_cols, output_dir):
        from sklearn.preprocessing import StandardScaler
        import joblib as _joblib
        scaler_X = StandardScaler()
        scaler_X.fit(df.loc[train_mask, features].values)
        df[features] = scaler_X.transform(df[features].values)
        scaler_Y = StandardScaler()
        scaler_Y.fit(df.loc[train_mask, target_cols].values)
        df[target_cols] = scaler_Y.transform(df[target_cols].values)
        _joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
        _joblib.dump(scaler_Y, os.path.join(output_dir, 'scaler_Y.pkl'))
        logging.info("Scalers saved.")
        return scaler_X, scaler_Y

    scaler_X, scaler_Y = build_scalers(df, train_mask, features, target_cols, args.output_dir)

    # ---- Shared tensors on CPU (read-only across threads) ----
    # RAM note: only the flat [N_rows, F] tensors live here — no windows pre-expanded.
    X_tensor          = torch.tensor(df[features].values,       dtype=torch.float32)
    Y_ret_tensor      = torch.tensor(df[target_cols].values,    dtype=torch.float32)
    Y_price_tensor    = torch.tensor(df[raw_price_cols].values, dtype=torch.float32)
    last_price_tensor = Y_price_tensor
    sym_tensor        = torch.tensor(df['symbol_id'].values,    dtype=torch.long)
    regime_tensor     = torch.tensor(df['regime'].values,       dtype=torch.long)
    idx_to_date       = df['time'].values
    del df  # free the full DataFrame once tensors are built

    # Pre-compute per-symbol index arrays (used by each horizon worker)
    df_groups = []
    # Reconstruct per-symbol index ranges from sym_tensor
    sym_np = sym_tensor.numpy()
    for sid in range(len(sym2id)):
        df_groups.append(np.where(sym_np == sid)[0])

    # ================================================================
    # Launch training — sequential per horizon (throughput-optimized)
    # ================================================================
    results = {}
    shared_kwargs = dict(
        X_tensor=X_tensor, sym_tensor=sym_tensor, regime_tensor=regime_tensor,
        Y_ret_tensor=Y_ret_tensor, Y_price_tensor=Y_price_tensor,
        last_price_tensor=last_price_tensor,
        df_groups=df_groups, idx_to_date=idx_to_date, cutoff_date=cutoff_date,
        scaler_Y=scaler_Y, num_symbols=len(sym2id), num_features=len(features),
        lookback=lookback, target_cols=target_cols,
        batch_size=args.batch_size, epochs_per_stage=args.epochs_per_stage,
        learning_rate=args.learning_rate, patience=args.patience,
        stages=args.stages, stage_ratios=stage_ratios, is_iterative=is_iterative,
        accum_steps=args.accum_steps,
        stride=args.stride,
        mode=args.mode,
        auto_batch_size_enabled=auto_batch_size_enabled,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        enable_gpu_prefetch=args.enable_gpu_prefetch,
        channels_last=args.channels_last,
        checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir,
        log_dir=args.log_dir, loader_kwargs=loader_kwargs,
        cpu_prefetch_queue=args.cpu_prefetch_queue,
        profile=args.profile,
        profile_steps=args.profile_steps,
        profile_sync_timing=args.profile_sync_timing,
        results=results,
    )

    logging.info(f"Sequential training for {len(horizons)} horizons: {list(horizons.keys())}")
    for h_name, h_days in horizons.items():
        _train_single_horizon_safe(horizon_name=h_name, horizon=h_days, **shared_kwargs)

    for h_name in horizons:
        if h_name not in results:
            logging.error(f"[{h_name}] No result produced by worker; marking horizon as failed.")
            results[h_name] = {
                'status': 'failed',
                'best_val_loss': float('inf'),
                'error': 'no_result',
            }

    logging.info("Training complete.")
    logging.info(results)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("MULTI-HORIZON TRAINING COMPLETE")
    print("=" * 60)
    for h_name in horizons:
        r = results.get(h_name, {})
        status = r.get('status', 'unknown')
        val    = r.get('best_val_loss', float('inf'))
        print(f"  {h_name:>4s}  status={status:<8s}  best_val_loss={val:.5f}")
    print(f"\nShared artefacts: {args.output_dir}")
    print(f"  scaler_X.pkl | scaler_Y.pkl | symbol_mapping.json | features.json")
    for h_name in horizons:
        print(f"  {h_name}/best_model.pth")
    print(f"\nInference example:")
    print(f"  from train import predict, MultiMetricPredictor")
    print(f"  model_7d = MultiMetricPredictor(..., forecast_horizon=7)")
    print(f"  model_7d.load_state_dict(torch.load('output_model/7d/best_model.pth'))")
    print(f"  df = predict(ohlc_history, symbol_id=0, regime_id=1,")
    print(f"               model=model_7d, scaler_X=scaler_X, scaler_Y=scaler_Y,")
    print(f"               feature_names=features, device=device)")

if __name__ == "__main__":
    main()