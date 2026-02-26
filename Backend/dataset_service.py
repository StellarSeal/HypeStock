from sqlalchemy import text
import pandas as pd
import numpy as np
from database import AsyncSessionLocal

# Rule 1: No Global DataFrames. Queries are executed async.
async def get_stocks(page: int, limit: int, query: str):
    """Async Postgres query for stocks with pagination."""
    async with AsyncSessionLocal() as session:
        search_term = f"%{query}%"
        
        sql = text("""
            SELECT c.stock_code, c.company_name,
                   MIN(p.time) as start_date,
                   MAX(p.time) as end_date,
                   COUNT(p.time) as entry_count
            FROM companies c
            LEFT JOIN stock_prices p ON c.stock_code = p.symbol
            WHERE c.stock_code ILIKE :search OR c.company_name ILIKE :search
            GROUP BY c.stock_code, c.company_name
            ORDER BY c.stock_code
            LIMIT :limit OFFSET :offset
        """)
        
        result = await session.execute(sql, {"search": search_term, "limit": limit, "offset": page * limit})
        rows = result.fetchall()

        # Count total for pagination meta
        count_sql = text("SELECT COUNT(*) FROM companies WHERE stock_code ILIKE :search OR company_name ILIKE :search")
        total = (await session.execute(count_sql, {"search": search_term})).scalar()

        items = [{
            "stock_code": r.stock_code,
            "company_name": r.company_name,
            "start_date": r.start_date.strftime('%Y-%m-%d') if r.start_date else None,
            "end_date": r.end_date.strftime('%Y-%m-%d') if r.end_date else None,
            "entry_count": r.entry_count
        } for r in rows]

        return {
            "items": items,
            "total": total,
            "hasMore": (page * limit + limit) < total
        }

async def get_on_demand_indicators(symbol: str):
    """Rule 4: Fetch raw OHLCV, compute metrics, return JSON safe dict."""
    async with AsyncSessionLocal() as session:
        # Fetch OHLCV data into temporary memory
        sql = text("SELECT time, symbol, open, high, low, close, volume FROM stock_prices WHERE symbol = :sym ORDER BY time ASC")
        result = await session.execute(sql, {"sym": symbol})
        rows = result.fetchall()

    if not rows:
        return []

    # 1. Create temporary DataFrame
    df = pd.DataFrame(rows, columns=['time', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    
    # Ensure numerics
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Run mathematical indicators via metrics.py
    from metrics import calculate_indicators
    df_metrics = calculate_indicators(df)

    # 3. Rule 4: Clean output - Convert NaN / Inf to None for JSON compliance
    # Prevent db storage by directly returning it
    df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)
    df_metrics = df_metrics.astype(object).where(pd.notnull(df_metrics), None)

    # Convert timestamps back to string
    df_metrics['time'] = df_metrics['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    return df_metrics.to_dict(orient='records')