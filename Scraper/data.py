import vnstock
import pandas as pd
import time
import os
import json
import random
import math
from datetime import date

# ── Rate limit configuration ──────────────────────────────────────────────────
FETCH_START      = '2011-01-01'
FETCH_END        = '2026-03-17'
FETCH_END_DATE   = date.fromisoformat(FETCH_END)

BATCH_SIZE       = 16          # requests per batch  (≤50 req/min limit)
BATCH_PAUSE      = 60.0        # seconds to wait between batches
MAX_PER_HOUR     = 2500        # hard hourly cap
# ─────────────────────────────────────────────────────────────────────────────


def fetch_stock_history(symbol: str) -> pd.DataFrame | None:
    """Fetch historical OHLCV data for a single symbol from VCI."""
    try:
        return (
            vnstock.Vnstock()
            .stock(symbol=symbol, source='VCI')
            .quote.history(start=FETCH_START, end=FETCH_END)
        )
    except Exception as e:
        print(f"  [WARN] Error fetching {symbol}: {e}")
        return None


def is_active(frame: pd.DataFrame) -> bool:
    """
    Return True only if the stock passes both activity filters:
      1. At least one trading day where volume > 0.
      2. Has not 'crashed' — i.e. has volume > 0 activity within the last
         90 days of the fetch window (so delisted / dead tickers are excluded).
    """
    if frame is None or frame.empty:
        return False

    trading_days = frame[frame['volume'] > 0]
    if trading_days.empty:
        return False                    # Filter 1: zero trading days

    # Filter 2: last active date must be within 90 days of FETCH_END
    last_active = pd.to_datetime(trading_days['time'].max()).date()
    days_since_active = (FETCH_END_DATE - last_active).days
    if days_since_active > 90:
        return False

    return True


def fetch_data():
    """
    Fetch all stock data from VCI in rate-limited batches.

    Rate limits enforced:
      • ≤ BATCH_SIZE  requests per minute  (one BATCH_PAUSE between batches)
      • ≤ MAX_PER_HOUR requests per hour   (tracked cumulatively)
    """
    frames        = []
    metadata_log  = []
    start_time    = time.time()

    stocks       = vnstock.Listing().all_symbols()
    stocks_list  = list(stocks.index)
    random.shuffle(stocks_list)

    total        = len(stocks_list)
    processed    = 0          # successfully fetched & accepted stocks
    skipped      = 0          # fetched but filtered out
    requests_made = 0         # total HTTP requests (for hourly cap)
    hour_window_start = time.time()

    print(f"Starting batch fetch for {total} symbols "
          f"(batch={BATCH_SIZE}, pause={BATCH_PAUSE}s, cap={MAX_PER_HOUR}/hr)\n")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = stocks_list[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = math.ceil(total / BATCH_SIZE)

        print(f"── Batch {batch_num}/{total_batches} "
              f"(symbols {batch_start+1}–{min(batch_start+BATCH_SIZE, total)}) ──")

        for x in batch:
            symbol = stocks.loc[x, 'symbol']
            org    = stocks.loc[x, 'organ_name']

            # Hourly cap: if we're approaching the limit, pause until the
            # current 1-hour window has elapsed before continuing.
            if requests_made >= MAX_PER_HOUR:
                elapsed_in_window = time.time() - hour_window_start
                sleep_needed = 3600.0 - elapsed_in_window
                if sleep_needed > 0:
                    print(f"\n  [RATE] Hourly cap reached — pausing {sleep_needed:.0f}s …")
                    time.sleep(sleep_needed)
                # Reset window
                hour_window_start = time.time()
                requests_made = 0

            frame = fetch_stock_history(symbol=symbol)
            requests_made += 1
            elapsed = time.time() - start_time

            if not is_active(frame):
                skipped += 1
                reason = "empty/None" if (frame is None or frame.empty) else \
                         ("zero volume" if frame[frame['volume'] > 0].empty else "crashed/delisted")
                print(f"  [SKIP] {symbol:>8} ({org}) — {reason}")
                continue

            frame = frame.copy()
            frame['symbol']     = symbol
            frame['organ_name'] = org
            frames.append(frame)
            processed += 1

            trading_days = int((frame['volume'] > 0).sum())
            print(f"  [OK]   #{processed:>4} {symbol:>8} ({org}) — "
                  f"{frame.shape[0]} rows, {trading_days} trading days")

            metadata_log.append({
                "time_elapsed":           elapsed,
                "event":                  "stock_data_fetched",
                "symbol":                 symbol,
                "organ_name":             org,
                "entries_fetched":        frame.shape[0],
                "trading_days":           trading_days,
                "total_stocks_processed": processed,
            })

        # Pause between batches (skip after the very last batch)
        if batch_start + BATCH_SIZE < total:
            print(f"\n  [RATE] Batch complete — pausing {BATCH_PAUSE:.0f}s before next batch …\n")
            time.sleep(BATCH_PAUSE)

    # ── Merge & save ──────────────────────────────────────────────────────────
    print(f"\nFinished fetching. Accepted {processed} stocks, skipped {skipped}.")

    if not frames:
        print("No data collected — nothing to save.")
        return

    stock_market = pd.concat(frames, ignore_index=True)

    # ── stock_prices.csv: OHLCV rows, symbol only (no company name) ───────────
    prices = stock_market[['symbol', 'time', 'open', 'high', 'low', 'close', 'volume']]
    prices.to_csv('./stock_prices.csv', index=False, encoding='utf-8')
    print(f"Price data saved to ./stock_prices.csv  ({len(prices):,} rows)")

    # ── companies.csv: unique symbol → organ_name mapping ────────────────────
    companies = (
        stock_market[['symbol', 'organ_name']]
        .drop_duplicates(subset='symbol')
        .sort_values('symbol')
        .reset_index(drop=True)
    )
    companies.to_csv('./companies.csv', index=False, encoding='utf-8')
    print(f"Company map saved to ./companies.csv  ({len(companies):,} entries)")

    # ── fetch_metadata.json ───────────────────────────────────────────────────
    metadata_path = './fetch_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_log, f, indent=4)
    print(f"Animation metadata saved to {metadata_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting data pipeline …\n")
    fetch_data()
