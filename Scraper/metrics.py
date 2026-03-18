import pandas as pd
import numpy as np
import os

# Suppress runtime warnings for expected division by zero (handled later by fillna)
np.seterr(divide='ignore', invalid='ignore')

# DESIGN CHOICE: Group processing ensures indicators do not bleed across different symbols.
# Default windows (14 for RSI/ATR, 20/50 for MA) are used as standard quantitative baselines.
def calculate_indicators(group):
    # Ensure sequential data
    group = group.sort_values('time')
    
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
    
    # Standard RSI uses Wilder's Smoothing (alpha=1/window)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # Safe RSI calculation to prevent div by zero
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    group['RSI'] = rsi
    
    # MACD (Standard 12-day EMA - 26-day EMA)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    group['MACD'] = ema12 - ema26
    
    # 3. Volatility Indicators
    group['Rolling_Vol_20d_std'] = close.rolling(window=20).std()
    
    # Average True Range (ATR)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    group['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # 4. Volume Indicators
    group['Volume_MA20'] = volume.rolling(window=20).mean()
    # fill_method=None prevents pandas deprecation warnings
    group['Volume_Change_pct'] = volume.pct_change(fill_method=None) 
    
    # 5. Performance Indicators
    group['Daily_Return_1d'] = close.pct_change(periods=1, fill_method=None)
    group['Daily_Return_5d'] = close.pct_change(periods=5, fill_method=None)
    
    # Safe cumulative return
    first_close = close.iloc[0] if len(close) > 0 else 1
    group['Cumulative_Return'] = np.where(first_close == 0, 0, (close / first_close) - 1)
    
    group['Daily_Range'] = high - low
    
    # 6. Correlation
    group['Vol_Close_Corr_20d'] = close.rolling(window=20).corr(volume)
    
    # 7. AI "Texture" Indicators (Page 6)
    # Bollinger Bands (20-day, 2 Standard Deviations)
    bb_mavg = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_hband = bb_mavg + (2 * bb_std)
    bb_lband = bb_mavg - (2 * bb_std)
    # DESIGN CHOICE: BBW normalized by middle band to provide absolute compression metric
    group['BB_Width'] = np.where(bb_mavg == 0, 0, (bb_hband - bb_lband) / bb_mavg)
    
    # Average Directional Index (ADX)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)
    
    atr14 = group['ATR']  # Reuse computed ATR
    
    # Prevent div by zero in ADX components
    safe_atr = np.where(atr14 == 0, np.nan, atr14)
    plus_di14 = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / safe_atr)
    minus_di14 = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / safe_atr)
    
    di_sum = (plus_di14 + minus_di14).abs()
    dx = np.where(di_sum == 0, 0, 100 * (plus_di14 - minus_di14).abs() / di_sum)
    group['ADX'] = pd.Series(dx, index=close.index).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # On-Balance Volume (OBV)
    obv_change = np.where(close > prev_close, volume, np.where(close < prev_close, -volume, 0))
    obv = pd.Series(obv_change, index=close.index).cumsum()
    # DESIGN CHOICE: OBV slope approximated via 5-day simple moving difference
    group['OBV_Slope_5d'] = obv.diff(periods=5) / 5
    
    # 8. Time-Series "Memory" Features (Page 7)
    group['Lagged_Return_t1'] = group['Daily_Return_1d'].shift(1)
    group['Lagged_Return_t3'] = group['Daily_Return_1d'].shift(3)
    group['Lagged_Return_t5'] = group['Daily_Return_1d'].shift(5)
    group['Dist_from_MA50'] = np.where(close == 0, 0, (close - group['MA50']) / close)

    return group

def generate_mock_data(filepath):
    """Creates a sample dataset if no input file is found to ensure script execution."""
    dates = pd.date_range(start="2024-01-01", periods=100)
    data = {
        'time': dates,
        'symbol': ['NCG'] * 100,
        'open': np.random.uniform(8, 12, 100),
        'high': np.random.uniform(8.5, 12.5, 100),
        'low': np.random.uniform(7.5, 11.5, 100),
        'close': np.random.uniform(8, 12, 100),
        'volume': np.random.randint(10000, 50000, 100)
    }
    df = pd.DataFrame(data)
    # Fix High/Low boundaries
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Enforce strict column order for the database
    df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    df.to_csv(filepath, index=False)
    print(f"Generated sample data at {filepath}")

if __name__ == "__main__":
    input_file = "stock_prices.csv"
    output_file = "metrics.csv"
    
    if not os.path.exists(input_file):
        generate_mock_data(input_file)
        
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Enforce numeric types to prevent `.apply()` failing on dirty data across 1600 stocks
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- FIX 1: Enforce correct column order on the original input data ---
    # This ensures that when the database copies 'stock_prices.csv', it finds 'time' first.
    base_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    # Safely order base columns, appending any unexpected extra columns to the end
    ordered_input_cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
    df = df[ordered_input_cols]
    
    # Overwrite the input file so the database importer uses the corrected layout
    df.to_csv(input_file, index=False)
    print(f"Re-saved {input_file} with corrected column order ('time', 'symbol', ...)")

    print(f"Calculating indicators for {df['symbol'].nunique()} distinct symbols...")
    # Process each symbol independently
    metrics_df = df.groupby('symbol', group_keys=False).apply(calculate_indicators)
    
    # --- REMOVE SOURCE FIELDS ---
    # Drop the original price and volume columns, keeping only time, symbol, and new indicators
    metrics_df.drop(columns=num_cols, inplace=True, errors='ignore')
    
    # --- GLOBAL ERROR HANDLING ---
    # Convert all infinite values (from division by zero) to NaN, then replace all NaNs with 0
    metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    metrics_df.fillna(0, inplace=True)
    
    # Round to 4 decimal places to keep the CSV clean for downstream ML tasks
    metrics_df = metrics_df.round(4)
    
    # --- FIX 2: Enforce correct column order on the metrics output data ---
    # Guarantee that 'time' and 'symbol' are the first two columns.
    final_cols = ['time', 'symbol'] + [col for col in metrics_df.columns if col not in ['time', 'symbol']]
    metrics_df = metrics_df[final_cols]
    
    metrics_df.to_csv(output_file, index=False)
    print(f"Successfully computed indicators and saved to {output_file} with corrected column order.")