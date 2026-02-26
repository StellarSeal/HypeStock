import pandas as pd
import numpy as np

# Suppress runtime warnings for expected division by zero (handled later by fillna)
np.seterr(divide='ignore', invalid='ignore')

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw OHLCV Pandas DataFrame and appends technical indicators."""
    
    # Ensure sequential data
    group = df.sort_values('time').copy()
    
    close = group['close']
    high = group['high']
    low = group['low']
    volume = group['volume']

    # 1. Trend Indicators
    group['MA20'] = close.rolling(window=20).mean()
    group['MA50'] = close.rolling(window=50).mean()
    group['EMA20'] = close.ewm(span=20, adjust=False).mean()
    
    # 2. Momentum Indicators
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    group['RSI'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    group['MACD'] = ema12 - ema26
    
    # 3. Volatility Indicators
    group['Rolling_Vol_20d_std'] = close.rolling(window=20).std()
    
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    group['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # 4. Volume Indicators
    group['Volume_MA20'] = volume.rolling(window=20).mean()
    group['Volume_Change_pct'] = volume.pct_change(fill_method=None) 
    
    # 5. Performance Indicators
    group['Daily_Return_1d'] = close.pct_change(periods=1, fill_method=None)
    group['Daily_Return_5d'] = close.pct_change(periods=5, fill_method=None)
    
    first_close = close.iloc[0] if len(close) > 0 else 1
    group['Cumulative_Return'] = np.where(first_close == 0, 0, (close / first_close) - 1)
    group['Daily_Range'] = high - low
    
    # 6. Correlation
    group['Vol_Close_Corr_20d'] = close.rolling(window=20).corr(volume)
    
    # 7. AI "Texture" Indicators
    bb_mavg = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_hband = bb_mavg + (2 * bb_std)
    bb_lband = bb_mavg - (2 * bb_std)
    group['BB_Width'] = np.where(bb_mavg == 0, 0, (bb_hband - bb_lband) / bb_mavg)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)
    
    safe_atr = np.where(group['ATR'] == 0, np.nan, group['ATR'])
    plus_di14 = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / safe_atr)
    minus_di14 = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / safe_atr)
    
    di_sum = (plus_di14 + minus_di14).abs()
    dx = np.where(di_sum == 0, 0, 100 * (plus_di14 - minus_di14).abs() / di_sum)
    group['ADX'] = pd.Series(dx, index=close.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    obv_change = np.where(close > prev_close, volume, np.where(close < prev_close, -volume, 0))
    obv = pd.Series(obv_change, index=close.index).cumsum()
    group['OBV_Slope_5d'] = obv.diff(periods=5) / 5
    
    # 8. Time-Series "Memory" Features
    group['Lagged_Return_t1'] = group['Daily_Return_1d'].shift(1)
    group['Lagged_Return_t3'] = group['Daily_Return_1d'].shift(3)
    group['Lagged_Return_t5'] = group['Daily_Return_1d'].shift(5)
    group['Dist_from_MA50'] = np.where(close == 0, 0, (close - group['MA50']) / close)

    # Format
    group = group.round(4)
    return group