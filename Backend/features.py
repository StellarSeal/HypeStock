import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Unified feature engineering pipeline ensuring strict parity between 
    train.py (training) and ml_model.py (inference).
    """
    df = df.copy()

    # 1. Volume Existence Validation
    if 'volume' not in df.columns:
        raise ValueError("Validation Error: 'volume' column is missing from the dataset.")

    # 2. Base Returns
    if is_training:
        df['Daily_Return_1d'] = df.groupby('symbol')['close'].pct_change().fillna(0)
        df['Daily_Return_5d'] = df.groupby('symbol')['close'].pct_change(5).fillna(0)
    else:
        df['Daily_Return_1d'] = df['close'].pct_change().fillna(0)
        df['Daily_Return_5d'] = df['close'].pct_change(5).fillna(0)

    df['ret_1d'] = df['Daily_Return_1d']
    df['ret_5d'] = df['Daily_Return_5d']

    # 3. Cross-sectional / Market features
    if is_training:
        df['market_return'] = df.groupby('time')['Daily_Return_1d'].transform('mean').fillna(0)
    else:
        # For single-symbol inference, isolate market impact to prevent zero-filling leakage
        df['market_return'] = 0.0
        
    df['relative_strength'] = (df['Daily_Return_1d'] - df['market_return']).fillna(0)

    # 4. Grouped or Rolling Feature Operations
    def calc_indicators(group):
        close = group['close']
        high = group['high']
        low = group['low']
        volume = group['volume']

        # Trend Metrics
        group['MA20'] = close.rolling(20).mean()
        group['MA50'] = close.rolling(50).mean()
        group['EMA20'] = close.ewm(span=20, adjust=False).mean()
        
        # Momentum
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        group['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-9))
        group['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        group['momentum_20d'] = close.pct_change(20).fillna(0)
        
        # Volatility
        group['Rolling_Vol_20d_std'] = group['Daily_Return_1d'].rolling(20).std()
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        group['ATR'] = tr.rolling(14).mean()
        
        # Volume Indicators
        group['Volume_MA20'] = volume.rolling(20).mean()
        group['Volume_Change_pct'] = volume.pct_change().fillna(0)
        vol_mean = volume.rolling(20, min_periods=1).mean()
        vol_std = volume.rolling(20, min_periods=1).std()
        group['volume_zscore'] = ((volume - vol_mean) / (vol_std + 1e-9)).fillna(0)
        
        # Additional Telemetry
        first_close = close.iloc[0] if close.iloc[0] != 0 else 1e-9
        group['Cumulative_Return'] = (close / first_close) - 1
        group['Daily_Range'] = (high - low) / close.replace(0, 1e-9)
        group['Vol_Close_Corr_20d'] = volume.rolling(20).corr(close).fillna(0)
        
        std_20 = close.rolling(20).std()
        group['BB_Width'] = (4 * std_20) / group['MA20'].replace(0, 1e-9)
        
        # ADX Proxy
        group['ADX'] = 0.0 # Retaining parity with original model's domain
        
        obv = (np.sign(delta) * volume).cumsum()
        group['OBV_Slope_5d'] = obv.diff(5) / 5.0
        
        group['Lagged_Return_t1'] = group['Daily_Return_1d'].shift(1)
        group['Lagged_Return_t3'] = group['Daily_Return_1d'].shift(3)
        group['Lagged_Return_t5'] = group['Daily_Return_1d'].shift(5)
        group['Dist_from_MA50'] = (close - group['MA50']) / group['MA50'].replace(0, 1e-9)

        return group

    if is_training:
        df = df.groupby('symbol', group_keys=False).apply(calc_indicators)
    else:
        df = calc_indicators(df)

    # Global cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df