import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# Target TimescaleDB Container in docker-compose
ASYNC_DB_URL = os.environ.get("DATABASE_URL_ASYNC", "postgresql+asyncpg://postgres:postgres@db:5432/postgres")
SYNC_DB_URL = os.environ.get("DATABASE_URL_SYNC", "postgresql://postgres:postgres@db:5432/postgres")

# 1. Async Engine (For FastAPI Event Loop)
async_engine = create_async_engine(ASYNC_DB_URL, echo=False)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# 2. Sync Engine (For Celery Workers & PandasAI Data Extraction)
sync_engine = create_engine(SYNC_DB_URL, echo=False)