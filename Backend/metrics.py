import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a raw OHLCV Pandas DataFrame and appends all technical indicators."""
    
    group = df.sort_values('time').copy()
    close = group['close']
    high = group['high']
    low = group['low']
    volume = group['volume']

    # Trend
    group['ma20'] = close.rolling(window=20).mean()
    group['ma50'] = close.rolling(window=50).mean()
    
    # Momentum (RSI, MACD)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    group['rsi'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    group['macd'] = ema12 - ema26
    
    # Volatility (ATR)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    group['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    group['volatility'] = group['atr'] # Alias for generic term
    
    # Format and clean
    group = group.round(4)
    group = group.replace([np.inf, -np.inf], np.nan)
    group = group.astype(object).where(pd.notnull(group), None)
    
    return group