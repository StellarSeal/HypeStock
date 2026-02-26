# HypeStock

<div align="center">

**HypeStock** is an *AI-powered stock analytics analysis and prediction* platform developed with `Python`, `Redis` and `TimescaleDB` - Powered by **[vnstock](https://github.com/thinh-vu/vnstock)**

</div>

> [!WARNING]
> This project is designed for a full Docker environment. Run all services together (`db`, `redis`, `ollama`, `backend`, `celery_worker`, `frontend`) to avoid runtime failures.

> [!NOTE]
> This project was made for DAP391 course (AI1905) - Spring 2026, FPT University

## ✨ Overview

HypeStock provides live stock exploration and AI-assisted analysis on top of historical market data. The backend serves Socket.IO events via **FastAPI**, computes technical indicators on demand, and offloads AI queries to Celery workers. Data is persisted in **TimescaleDB/PostgreSQL**, while **Redis** handles message brokering and distributed Socket.IO state.

HypeStock allows users to browse through live stock data computed by our systems. Users may explore the default dataset provided by the application, or request to pull the latest data via the **vnstock** library through the front-end interface.

## ✅ Requirements

- Docker + Docker Compose v2
- Python 3.11 (for local development outside Docker)
- NVIDIA GPU + NVIDIA Container Toolkit (recommended for Ollama container acceleration)
- Internet connection (for pulling Docker images and AI models)
- Optional: `GEMINI_API_KEY` for cloud AI provider mode

## 📦 Building

> [!WARNING]
> Before building, please **extract** the default sample data included in the `Data` folder

Build all containers from the project root:

```bash
docker compose build
```

## 🚀 Install / Run

1. Navigate to the project root (`HypeStock/`).
2. (Optional) Export your Gemini key:
	- PowerShell:
	```powershell
	$env:GEMINI_API_KEY="your_key_here"
	```
    - .env file
    ```env
    GEMINI_API_KEY=your_key_here
    ```
3. Start the full stack:
	```bash
	docker-compose up --build -d
	```
4. Open frontend at `http://localhost`.
5. Backend health endpoint: `http://localhost:8000/`.

To stop:

```bash
docker compose down
```

## ⚙️ Configuration

Primary runtime configuration lives in:

- `docker-compose.yml`
- `Dockerfile`

Key environment variables:

- `GEMINI_API_KEY` - Enables Gemini cloud responses
- `OLLAMA_URL` - Local LLM endpoint (default: `http://ollama:11434/api/chat`)
- `REDIS_URL` - Redis broker/manager URL (default: `redis://redis:6379/0`)

## 🔌 Tech stack - Core services/Dependencies

Core services from `docker-compose.yml`:

- `db` - TimescaleDB/PostgreSQL data store
- `redis` - Caching and Celery broker
- `ollama` - Local model serving (pulls `deepseek-r1:7b`)
- `backend` - FastAPI + Socket.IO server
- `celery_worker` - Async AI processing worker
- `frontend` - Nginx static host for `Front-End-Main/`

Python backend dependencies are listed in `requirements.txt` (FastAPI, Socket.IO, Celery, Redis, SQLAlchemy, asyncpg, pandas, google-genai, pandasai, etc.).

## 🗂️ Project Structure

- `Backend/` - FastAPI server, sockets, AI routing, data access, task queue logic
- `Data/` - CSV seed data (`companies.csv`, `stock_prices.csv`, `metrics.csv`)
- `Front-End-Main/` - Main static web UI
- `Front-End-Demo/` - Demo UI variant
- `docker-compose.yml` - Multi-service orchestration
- `Dockerfile` - Backend and worker image build

## 📝 Notes

- On first startup, Ollama pulls `deepseek-r1:7b`, so initialization may take several minutes.
- The backend container imports CSV data into PostgreSQL on startup and truncates existing `companies`/`stock_prices` to avoid duplicates.
- Default database credentials in compose are development-only and should be changed for production.

---

System recommendations (for local Ollama usage, tuned for `deepseek-r1:7b`):

- CPU: **Intel i5-12400/AMD Ryzen 5 5600** or better
- Memory: **16GB RAM** minimum (12GB+ VRAM recommended)
- GPU: **RTX 3050** or higher
- Storage: **40GB** free storage (ideally prioritize SSDs for faster I/O operations) or more
- Any OS that supports **Docker**