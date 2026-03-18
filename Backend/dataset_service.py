import logging
import time as time_module
from datetime import datetime, timedelta, timezone
from sqlalchemy import text, bindparam
from fastapi import HTTPException
from database import AsyncSessionLocal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

MAX_ROWS_RETURNED = 5000
ALLOWED_RANGES = {'1M', '3M', '6M', '1Y', '3Y', 'ALL'}

INDICATOR_MAP = {
    'ma20': 'ma20',
    'ma50': 'ma50',
    'ema20': 'ema20',
    'rsi': 'rsi',
    'macd': 'macd',
    'atr': 'atr',
    'volatility': 'rolling_vol_20d_std', 
    'volume': 'volume', 
    'volume_ma20': 'volume_ma20',
    'volume_change_pct': 'volume_change_pct',
    'daily_return_1d': 'daily_return_1d',
    'daily_return_5d': 'daily_return_5d',
    'cumulative_return': 'cumulative_return',
    'daily_range': 'daily_range',
    'vol_close_corr_20d': 'vol_close_corr_20d',
    'bb_width': 'bb_width',
    'adx': 'adx',
    'obv_slope_5d': 'obv_slope_5d',
    'lagged_return_t1': 'lagged_return_t1',
    'lagged_return_t3': 'lagged_return_t3',
    'lagged_return_t5': 'lagged_return_t5',
    'dist_from_ma50': 'dist_from_ma50'
}

def validate_range(range_val: str):
    if range_val.upper() not in ALLOWED_RANGES:
        raise HTTPException(status_code=400, detail=f"Invalid range. Allowed: {ALLOWED_RANGES}")

def get_date_threshold(range_val: str, padding_days: int = 0) -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    if range_val.upper() == 'ALL':
        return datetime(1970, 1, 1)
        
    days_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '3Y': 1095}
    days = days_map.get(range_val.upper(), 30) + padding_days
    return now - timedelta(days=days)

def safe_serialize_time(t) -> str:
    if isinstance(t, datetime):
        return t.isoformat().replace('+00:00', 'Z')
    if hasattr(t, "isoformat"):
        return t.isoformat()
    return str(t)

def downsample_rows(rows: list, max_rows: int) -> list:
    if len(rows) > max_rows:
        step = len(rows) // max_rows
        return rows[::step]
    return rows

async def get_database_stats() -> int:
    try:
        async with AsyncSessionLocal() as session:
            sql = text("SELECT sum(reltuples::bigint) AS estimate FROM pg_class WHERE relname IN ('stock_prices', 'metrics', 'companies')")
            result = await session.execute(sql)
            row = result.fetchone()
            return row.estimate if row and row.estimate else 3450000
    except Exception as e:
        logger.error(f"Failed to fetch DB stats: {e}")
        return 3450000

async def get_stock_summary(symbol: str) -> dict:
    start_dt = get_date_threshold('ALL')
    
    try:
        async with AsyncSessionLocal() as session:
            comp_sql = text("SELECT company_name FROM companies WHERE stock_code = :sym LIMIT 1")
            comp_res = await session.execute(comp_sql, {"sym": symbol})
            comp_row = comp_res.fetchone()
            
            if not comp_row:
                raise ValueError(f"Company {symbol} not found.")

            agg_sql = text("""
                SELECT 
                    MAX(close) as highest_close, 
                    MIN(close) as lowest_close, 
                    AVG(volume) as average_volume,
                    COUNT(DISTINCT "time"::date) as trading_days,
                    MIN(time) as start_date,
                    MAX(time) as end_date
                FROM stock_prices 
                WHERE symbol = :sym AND time >= :start
            """)
            agg_res = await session.execute(agg_sql, {"sym": symbol, "start": start_dt})
            agg = agg_res.fetchone()

            first_sql = text("SELECT close FROM stock_prices WHERE symbol = :sym AND time >= :start ORDER BY time ASC LIMIT 1")
            last_sql = text("SELECT close FROM stock_prices WHERE symbol = :sym AND time >= :start ORDER BY time DESC LIMIT 1")
            
            first_res = await session.execute(first_sql, {"sym": symbol, "start": start_dt})
            last_res = await session.execute(last_sql, {"sym": symbol, "start": start_dt})
            
            first_val = first_res.scalar()
            last_val = last_res.scalar()
            
            computed_cum_return = 0.0
            if first_val and last_val and first_val > 0:
                computed_cum_return = ((last_val - first_val) / first_val) * 100

            latest_sql = text("""
                SELECT rolling_vol_20d_std as volatility 
                FROM metrics WHERE symbol = :sym ORDER BY "time" DESC LIMIT 1
            """)
            latest_res = await session.execute(latest_sql, {"sym": symbol})
            latest = latest_res.fetchone()

        return {
            "company_name": comp_row.company_name,
            "symbol": symbol,
            "start_date": safe_serialize_time(agg.start_date) if agg.start_date else "N/A",
            "end_date": safe_serialize_time(agg.end_date) if agg.end_date else "N/A",
            "data_range": "ALL",
            "metrics": {
                "highest_close": float(agg.highest_close or 0),
                "lowest_close": float(agg.lowest_close or 0),
                "average_volume": float(agg.average_volume or 0),
                "volatility": float(latest.volatility if latest and latest.volatility else 0),
                "cumulative_return": float(computed_cum_return),
                "trading_days": int(agg.trading_days or 0)
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"DB Error in get_stock_summary for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="DB query failed")

async def get_stock_price(symbol: str, range_val: str) -> list:
    validate_range(range_val)
    start_dt = get_date_threshold(range_val)
    
    try:
        async with AsyncSessionLocal() as session:
            # Joining tables since metrics no longer contains OHLCV
            sql = text("""
                SELECT p."time", p.open, p.high, p.low, p.close, p.volume,
                       m.ma20, m.ma50, m.ema20, m.rsi, m.macd,
                       m.rolling_vol_20d_std as volatility,
                       m.atr, m.daily_return_1d,
                       m.lagged_return_t1, m.lagged_return_t3, m.lagged_return_t5,
                       m.dist_from_ma50
                FROM stock_prices p
                LEFT JOIN metrics m ON p.symbol = m.symbol AND p."time" = m."time"
                WHERE p.symbol = :sym AND p."time" >= :start 
                ORDER BY p."time" ASC
                LIMIT :limit
            """)
            
            fetch_limit = MAX_ROWS_RETURNED * 2 if range_val.upper() == 'ALL' else MAX_ROWS_RETURNED
            result = await session.execute(sql, {"sym": symbol, "start": start_dt, "limit": fetch_limit})
            rows = result.fetchall()

        sampled_rows = downsample_rows(rows, MAX_ROWS_RETURNED)

        return [{
            "time": safe_serialize_time(r.time),
            "open": float(r.open) if r.open is not None else 0.0,
            "high": float(r.high) if r.high is not None else 0.0,
            "low": float(r.low) if r.low is not None else 0.0,
            "close": float(r.close) if r.close is not None else 0.0,
            "volume": int(r.volume) if r.volume is not None else 0,
            "ma20": float(r.ma20) if r.ma20 is not None else None,
            "ma50": float(r.ma50) if r.ma50 is not None else None,
            "ema20": float(r.ema20) if r.ema20 is not None else None,
            "rsi": float(r.rsi) if r.rsi is not None else None,
            "macd": float(r.macd) if r.macd is not None else None,
            "volatility": float(r.volatility) if r.volatility is not None else None,
            "atr": float(r.atr) if r.atr is not None else None,
            "daily_return_1d": float(r.daily_return_1d) if r.daily_return_1d is not None else None,
            "lagged_return_t1": float(r.lagged_return_t1) if r.lagged_return_t1 is not None else None,
            "lagged_return_t3": float(r.lagged_return_t3) if r.lagged_return_t3 is not None else None,
            "lagged_return_t5": float(r.lagged_return_t5) if r.lagged_return_t5 is not None else None,
            "dist_from_ma50": float(r.dist_from_ma50) if r.dist_from_ma50 is not None else None,
        } for r in sampled_rows]
        
    except Exception as e:
        logger.error(f"DB Error in get_stock_price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="DB query failed")

async def get_stock_indicator(symbol: str, indicator_type: str, range_val: str) -> list:
    validate_range(range_val)
    db_col = INDICATOR_MAP.get(indicator_type.lower())
    if not db_col:
        return []

    start_dt = get_date_threshold(range_val)
    
    try:
        async with AsyncSessionLocal() as session:
            # Volume is retrieved directly from stock_prices
            if db_col == 'volume':
                sql = text("""
                    SELECT "time", volume as value 
                    FROM stock_prices 
                    WHERE symbol = :sym AND "time" >= :start 
                    ORDER BY "time" ASC
                    LIMIT :limit
                """)
            else:
                sql = text(f"""
                    SELECT "time", {db_col} as value 
                    FROM metrics 
                    WHERE symbol = :sym AND "time" >= :start 
                    ORDER BY "time" ASC
                    LIMIT :limit
                """)
            
            fetch_limit = MAX_ROWS_RETURNED * 2 if range_val.upper() == 'ALL' else MAX_ROWS_RETURNED
            result = await session.execute(sql, {"sym": symbol, "start": start_dt, "limit": fetch_limit})
            rows = result.fetchall()

        sampled_rows = downsample_rows(rows, MAX_ROWS_RETURNED)

        return [{
            "time": safe_serialize_time(getattr(r, 'time')), 
            "value": float(r.value) if r.value is not None else None
        } for r in sampled_rows]
        
    except Exception as e:
        logger.error(f"DB Error in get_stock_indicator for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="DB query failed")

async def get_stock_list(page: int = 0, limit: int = 24, query: str = "") -> dict:
    offset = page * limit
    try:
        async with AsyncSessionLocal() as session:
            if query:
                search_term = f"%{query.upper()}%"
                sql = text("""
                    SELECT stock_code, company_name 
                    FROM companies 
                    WHERE stock_code LIKE :search OR UPPER(company_name) LIKE :search
                    ORDER BY stock_code ASC
                    LIMIT :limit OFFSET :offset
                """)
                params = {"search": search_term, "limit": limit + 1, "offset": offset}
            else:
                sql = text("""
                    SELECT stock_code, company_name 
                    FROM companies 
                    ORDER BY stock_code ASC
                    LIMIT :limit OFFSET :offset
                """)
                params = {"limit": limit + 1, "offset": offset}
                
            result = await session.execute(sql, params)
            rows = result.fetchall()
            
        has_more = len(rows) > limit
        items_to_return = rows[:limit]
        items = []

        if items_to_return:
            symbols = tuple([r.stock_code for r in items_to_return])
            
            async with AsyncSessionLocal() as session:
                agg_sql = text("""
                    SELECT symbol, 
                            MIN(time) as start_date, 
                            MAX(time) as end_date, 
                            COUNT(DISTINCT "time"::date) as entry_count
                    FROM stock_prices
                    WHERE symbol IN :symbols
                    GROUP BY symbol
                """).bindparams(bindparam('symbols', expanding=True))
                    
                agg_res = await session.execute(agg_sql, {"symbols": symbols})
                agg_rows = agg_res.fetchall()
                agg_map = {r.symbol: r for r in agg_rows}
                
            for r in items_to_return:
                stats = agg_map.get(r.stock_code)
                start_year = str(stats.start_date.year) if stats and stats.start_date else "N/A"
                end_year = str(stats.end_date.year) if stats and stats.end_date else "N/A"
                
                items.append({
                    "stock_code": r.stock_code,
                    "company_name": r.company_name,
                    "start_date": start_year, 
                    "end_date": end_year,
                    "trading_days": stats.entry_count if stats else 0
                })

        return {
            "items": items,
            "hasMore": has_more
        }
    except Exception as e:
        logger.error(f"DB Error in get_stock_list: {str(e)}")
        raise HTTPException(status_code=500, detail="DB query failed")

async def get_comparison_data(symbols: list[str], range_val: str) -> dict:
    """Fetches, aligns, and serializes multiple stocks into a single comparison payload."""
    import pandas as pd
    import numpy as np
    
    validate_range(range_val)
    start_dt = get_date_threshold(range_val)

    try:
        async with AsyncSessionLocal() as session:
            # Joining tables since metrics no longer contains OHLCV
            sql = text("""
                SELECT p."time", p.symbol, p.open, p.high, p.low, p.close, p.volume,
                       m.ma20, m.ma50, m.ema20, m.rsi, m.macd, m.rolling_vol_20d_std as volatility,
                       m.atr, m.daily_return_1d,
                       m.lagged_return_t1, m.lagged_return_t3, m.lagged_return_t5,
                       m.dist_from_ma50
                FROM stock_prices p
                LEFT JOIN metrics m ON p.symbol = m.symbol AND p."time" = m."time"
                WHERE p.symbol IN :symbols AND p."time" >= :start
                ORDER BY p."time" ASC
            """).bindparams(bindparam('symbols', expanding=True))

            result = await session.execute(sql, {"symbols": tuple(symbols), "start": start_dt})
            rows = result.fetchall()

        if not rows:
            return {"symbols": symbols, "available_metrics": [], "available_time_ranges": list(ALLOWED_RANGES), "data": {}, "meta": {"start_date": None, "end_date": None}}

        df = pd.DataFrame([dict(r._mapping) for r in rows])
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        metrics_list = [
            'open', 'high', 'low', 'close', 'volume',
            'ma20', 'ma50', 'ema20', 'rsi', 'macd',
            'volatility', 'atr', 'daily_return_1d',
            'lagged_return_t1', 'lagged_return_t3', 'lagged_return_t5',
            'dist_from_ma50',
        ]
        available_metrics = [m for m in metrics_list if m in df.columns]

        payload_data = {}
        for m in available_metrics:
            # Pivot creates an aligned time-series table. We forward fill missing days.
            pivot = df.pivot_table(index='time', columns='symbol', values=m, aggfunc='last')
            pivot = pivot.ffill()
            pivot = pivot.reset_index()
            # Replace NaNs safely with None for JSON encoding
            pivot = pivot.replace([np.inf, -np.inf], np.nan)
            pivot = pivot.where(pd.notnull(pivot), None)
            payload_data[m] = pivot.to_dict(orient='records')

        return {
            "symbols": symbols,
            "available_metrics": available_metrics,
            "available_time_ranges": list(ALLOWED_RANGES),
            "data": payload_data,
            "meta": {
                "start_date": df['time'].min() if not df.empty else None,
                "end_date": df['time'].max() if not df.empty else None
            }
        }
    except Exception as e:
        logger.error(f"DB Error in get_comparison_data: {e}")
        raise HTTPException(status_code=500, detail="Comparison query failed")