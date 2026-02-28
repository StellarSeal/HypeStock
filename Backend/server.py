import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import socketio

import dataset_service 
from database import redis_client, init_db_indexes
from models import ExplainPredictionRequest, build_envelope, SummaryResponse, PredictionResponse
from tasks import generate_prediction_explanation, compute_feature_importance

# --- SETUP FASTAPI & ASGI SOCKET.IO ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
mgr = socketio.AsyncRedisManager(REDIS_URL)
sio = socketio.AsyncServer(async_mode='asgi', client_manager=mgr, cors_allowed_origins='*')

# REMEDIATION: Hooks database index initialization on FastAPI startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_indexes()
    yield

app = FastAPI(title="HypeStock REST API v3.0 (Aligned Spec)", lifespan=lifespan)

# REMEDIATION: Add CORS Middleware to allow the Frontend to read REST responses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio_app = socketio.ASGIApp(socketio_server=sio, socketio_path='')
app.mount("/socket.io", sio_app)

# ==============================================================================
# I. REST ENDPOINTS (With Strict Redis Caching)
# ==============================================================================

@app.get("/stock/{symbol}/summary", response_model=SummaryResponse)
async def get_summary(symbol: str):
    cache_key = f"summary:{symbol}"
    cached = await redis_client.get(cache_key)
    if cached: 
        return json.loads(cached)
        
    data = await dataset_service.get_stock_summary(symbol)
    await redis_client.setex(cache_key, 21600, json.dumps(data)) # TTL: 6 hours
    return data

@app.get("/stock/{symbol}/price")
async def get_price(symbol: str, range: str = Query("1M", regex="^(1M|3M|6M|1Y|3Y|ALL)$")):
    cache_key = f"price:{symbol}:{range}"
    cached = await redis_client.get(cache_key)
    if cached: 
        return json.loads(cached)
        
    data = await dataset_service.get_stock_price(symbol, range)
    await redis_client.setex(cache_key, 3600, json.dumps(data)) # TTL: 1 hour
    return data

@app.get("/stock/{symbol}/indicator")
async def get_indicator(symbol: str, type: str = Query(..., description="rsi, macd, atr, etc."), range: str = Query("1M", regex="^(1M|3M|6M|1Y|3Y|ALL)$")):
    # Never compute on frontend, return only Date + Value.
    cache_key = f"indicator:{symbol}:{type}:{range}"
    cached = await redis_client.get(cache_key)
    if cached: 
        return json.loads(cached)
        
    data = await dataset_service.get_stock_indicator(symbol, type, range)
    await redis_client.setex(cache_key, 3600, json.dumps(data)) # TTL: 1 hour
    return data

@app.get("/stock/{symbol}/prediction", response_model=PredictionResponse)
async def get_prediction(symbol: str, range: str = Query("1M", regex="^(1M|3M|6M|1Y|3Y|ALL)$")):
    # Per Blueprint: Temporary Behavior
    cache_key = f"prediction:{symbol}:{range}"
    cached = await redis_client.get(cache_key)
    if cached: 
        return json.loads(cached)
        
    response_data = {"available": False, "message": "Future behavior will trigger Celery regression pipeline."}
    await redis_client.setex(cache_key, 1800, json.dumps(response_data)) # TTL: 30 minutes
    return response_data

# ==============================================================================
# II. WEBSOCKET ENDPOINTS (Strictly Async Tasks / Progress)
# ==============================================================================

@sio.event
async def connect(sid, environ):
    print(f"[WS] Client connected: {sid}")
    await sio.emit('server_ack', build_envelope('server_ack', 'sys', {'status': 'connected'}), room=sid)

@sio.on('startup')
async def handle_startup(sid, data):
    """Handles frontend startup handshake to unlock the UI immediately."""
    print(f"[WS] Startup handshake from {sid}: {data}")
    await sio.emit('startup_response', build_envelope('startup_response', data.get('request_id', 'sys'), {'status': 'ready'}), room=sid)

@sio.on('request_stocks')
async def handle_request_stocks(sid, data):
    """Handles paginated and searched stock requests from the frontend."""
    request_id = data.get('request_id', 'sys')
    page = data.get('page', 0)
    limit = data.get('limit', 24)
    query = data.get('query', '')

    try:
        result = await dataset_service.get_stock_list(page=page, limit=limit, query=query)
        await sio.emit('stock_data', build_envelope('stock_data', request_id, result), room=sid)
    except Exception as e:
        print(f"[WS] Error fetching stocks: {e}")
        await sio.emit('error', build_envelope('error', request_id, {"code": 500, "message": str(e)}), room=sid)

@sio.on('explain_prediction')
async def handle_explain_prediction(sid, data):
    """Handles offloaded LLM Explanation generation triggered from Frontend"""
    try:
        req = ExplainPredictionRequest(**(data or {}))
    except ValidationError as e:
        await sio.emit('error', build_envelope('error', "unknown", {"code": 400, "message": "Validation Error"}), room=sid)
        return

    # Dispatch to Celery Worker
    generate_prediction_explanation.delay(
        sid, req.request_id, req.symbol, req.range, req.top_features
    )