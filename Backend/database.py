import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Target TimescaleDB Container in docker-compose
ASYNC_DB_URL = os.environ.get("DATABASE_URL_ASYNC", "postgresql+asyncpg://admin:hypestock_password_idk@db:5432/stock_data")
SYNC_DB_URL = os.environ.get("DATABASE_URL_SYNC", "postgresql://admin:hypestock_password_idk@db:5432/stock_data")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

# 1. Async Engine (For FastAPI Event Loop)
# REMEDIATION: Increased connection pool limits and timeouts to prevent connection starvation
async_engine = create_async_engine(
    ASYNC_DB_URL, 
    echo=False,
    pool_size=20,           # pool_size >= 10 constraint
    max_overflow=20,        # max_overflow >= 20 constraint
    pool_timeout=30,        # Explicit timeout constraint
    pool_recycle=1800       
)

# Use the modern async_sessionmaker (SQLAlchemy 2.0+)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# 2. Sync Engine (For Celery Workers & PandasAI Data Extraction)
sync_engine = create_engine(SYNC_DB_URL, echo=False)

# 3. Async Redis Client (For FastAPI Caching)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def init_db_indexes():
    """
    REMEDIATION: Index validation function to ensure deterministic query performance.
    Executed dynamically during FastAPI app startup.
    """
    try:
        async with async_engine.begin() as conn:
            logger.info("Validating database indexes (idx_stock_prices_symbol_time, idx_metrics_symbol_time)...")
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time
                ON stock_prices (symbol, time DESC);
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_metrics_symbol_time
                ON metrics (symbol, "time" DESC);
            """))
            logger.info("Database indexes validated successfully.")
    except Exception as e:
        logger.error(f"Failed to create database indexes: {e}")