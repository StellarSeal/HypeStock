import os
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import socketio

import dataset_service 
from database import redis_client, init_db_indexes
from models import ExplainPredictionRequest, build_envelope, SummaryResponse, PredictionResponse, CompareRequest
from tasks import generate_prediction_explanation, process_ai_chat
from ml_model import predict_ensemble

# --- GLOBAL TRACKING STATE ---
BOOT_TIME = time.time()
ACTIVE_USERS = set()
REQUEST_HISTORY = deque(maxlen=2000)

# --- SETUP FASTAPI & ASGI SOCKET.IO ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:16379/0")
mgr = socketio.AsyncRedisManager(REDIS_URL)
sio = socketio.AsyncServer(async_mode='asgi', client_manager=mgr, cors_allowed_origins='*')

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_indexes()
    yield

app = FastAPI(title="HypeStock REST API v4.0", lifespan=lifespan)

@app.middleware("http")
async def track_requests(request: Request, call_next):
    REQUEST_HISTORY.append(time.time())
    response = await call_next(request)
    return response

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
# I. REST ENDPOINTS
# ==============================================================================

@app.get("/system/status")
async def get_system_status():
    now = time.time()
    buckets = {int((now - i*60)//60): 0 for i in range(30)}
    for t in list(REQUEST_HISTORY):
        b = int((now - t)//60)
        if b in buckets:
            buckets[b] += 1
            
    graph_data = [{"time": f"-{b}m", "count": buckets[b]} for b in sorted(buckets.keys(), reverse=True)]
    entries = await dataset_service.get_database_stats()
    
    return {
        "boot_time": BOOT_TIME,
        "active_users": len(ACTIVE_USERS),
        "total_entries": entries,
        "request_graph": graph_data
    }

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
    cache_key = f"indicator:{symbol}:{type}:{range}"
    cached = await redis_client.get(cache_key)
    if cached: 
        return json.loads(cached)
        
    data = await dataset_service.get_stock_indicator(symbol, type, range)
    await redis_client.setex(cache_key, 3600, json.dumps(data)) # TTL: 1 hour
    return data

@app.get("/stock/{symbol}/prediction", response_model=PredictionResponse)
async def get_prediction(symbol: str):
    """
    Evaluates the 7D Ensemble against the latest available historical data.
    """
    # Prediction engine expects at least enough history for Lookback padding (1Y is safe)
    historical_data = await dataset_service.get_stock_price(symbol, '1Y')
    prediction = predict_ensemble(symbol, historical_data)
    return prediction

@app.get("/stock/search")
async def search_stocks_rest(query: str = ""):
    return await dataset_service.get_stock_list(page=0, limit=50, query=query)

@app.post("/stock/compare")
async def init_compare(req: CompareRequest):
    data = await dataset_service.get_comparison_data(req.symbols, req.default_time_range)
    return data

# ==============================================================================
# II. WEBSOCKET ENDPOINTS
# ==============================================================================

@sio.event
async def connect(sid, environ):
    ACTIVE_USERS.add(sid)
    print(f"[WS] Client connected: {sid}. Active: {len(ACTIVE_USERS)}")
    await sio.emit('server_ack', build_envelope('server_ack', 'sys', {'status': 'connected'}), room=sid)

@sio.event
async def disconnect(sid):
    ACTIVE_USERS.discard(sid)
    print(f"[WS] Client disconnected: {sid}. Active: {len(ACTIVE_USERS)}")

@sio.on('startup')
async def handle_startup(sid, data):
    await sio.emit('startup_response', build_envelope('startup_response', data.get('request_id', 'sys'), {'status': 'ready'}), room=sid)

@sio.on('request_stocks')
async def handle_request_stocks(sid, data):
    request_id = data.get('request_id', 'sys')
    page = data.get('page', 0)
    limit = data.get('limit', 24)
    query = data.get('query', '')

    try:
        result = await dataset_service.get_stock_list(page=page, limit=limit, query=query)
        await sio.emit('stock_data', build_envelope('stock_data', request_id, result), room=sid)
    except Exception as e:
        await sio.emit('error', build_envelope('error', request_id, {"code": 500, "message": str(e)}), room=sid)

@sio.on('explain_prediction')
async def handle_explain_prediction(sid, data):
    try:
        req = ExplainPredictionRequest(**(data or {}))
    except ValidationError:
        await sio.emit('error', build_envelope('error', "unknown", {"code": 400, "message": "Validation Error"}), room=sid)
        return

    generate_prediction_explanation.delay(
        sid, req.request_id, req.symbol, req.range, req.top_features
    )

@sio.on('ai')
async def handle_ai_chat_socket(sid, data):
    request_id = data.get('request_id', 'sys')
    content = data.get('content', '')
    seed = data.get('seed', 0)
    context = data.get('context', '') 
    process_ai_chat.delay(sid, request_id, content, seed, context)