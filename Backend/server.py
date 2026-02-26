import os
from fastapi import FastAPI
import socketio
from pydantic import ValidationError

import dataset_service
from models import StartupRequest, RequestStocks, AIRequest, IndicatorRequest, build_envelope
from tasks import process_ai_query_task

# --- SETUP FASTAPI & ASGI SOCKET.IO ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Redis Manager connects to the cluster for distributed Socket.IO state
mgr = socketio.AsyncRedisManager(REDIS_URL)
sio = socketio.AsyncServer(async_mode='asgi', client_manager=mgr, cors_allowed_origins='*')

app = FastAPI(title="HypeStock Backend v2.0")
app.mount('/socket.io', socketio.ASGIApp(sio))

@app.get('/')
async def index():
    return {"status": "FastAPI Socket Backend is Running"}

# --- EVENT ROUTING ---

@sio.event
async def connect(sid, environ):
    print(f"[WS] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[WS] Client disconnected: {sid}")

@sio.on('startup')
async def handle_startup(sid, data):
    try:
        req = StartupRequest(**(data or {}))
    except ValidationError as e:
        await sio.emit('error', build_envelope('error', "unknown", {"details": e.errors()}), room=sid)
        return

    # Acknowledge readiness
    await sio.emit('startup_response', build_envelope('startup_response', req.request_id, {'status': 'ready'}), room=sid)

    # Auto-fetch stock list directly from Postgres
    try:
        initial_stocks = await dataset_service.get_stocks(page=0, limit=20, query="")
        await sio.emit('stock_data', build_envelope('stock_data', req.request_id, initial_stocks), room=sid)
    except Exception as e:
        print(f"[ERROR] Auto-fetch stocks failed: {e}")

@sio.on('request_stocks')
async def handle_request_stocks(sid, data):
    try:
        req = RequestStocks(**(data or {}))
    except ValidationError as e:
        await sio.emit('error', build_envelope('error', "unknown", {"details": e.errors()}), room=sid)
        return

    try:
        result = await dataset_service.get_stocks(req.page, req.limit, req.query)
        await sio.emit('stock_data', build_envelope('stock_data', req.request_id, result), room=sid)
    except Exception as e:
        print(f"[ERROR] Stock fetch failed: {e}")
        await sio.emit('error', build_envelope('error', req.request_id, {"message": "Database lookup failed"}), room=sid)

@sio.on('request_indicators')
async def handle_request_indicators(sid, data):
    """Rule 4: On-Demand Technical Indicators computation."""
    try:
        req = IndicatorRequest(**(data or {}))
    except ValidationError as e:
        await sio.emit('error', build_envelope('error', "unknown", {"details": e.errors()}), room=sid)
        return

    try:
        metrics_data = await dataset_service.get_on_demand_indicators(req.symbol)
        await sio.emit('indicator_data', build_envelope('indicator_data', req.request_id, metrics_data), room=sid)
    except Exception as e:
        print(f"[ERROR] On-Demand metrics failed: {e}")
        await sio.emit('error', build_envelope('error', req.request_id, {"message": "Indicator calculation failed"}), room=sid)

@sio.on('ai')
async def handle_ai_event(sid, data):
    try:
        req = AIRequest(**(data or {}))
    except ValidationError as e:
        await sio.emit('error', build_envelope('error', "unknown", {"details": e.errors()}), room=sid)
        return

    # Rule 5: Decouple AI. Offload to Celery via Redis.
    process_ai_query_task.delay(sid, req.request_id, req.content, req.seed, req.model)

    # Optional UI UX Acknowledgement
    ack = {"status": "processing", "message": "Dispatched to Celery Worker"}
    await sio.emit('ai_ack', build_envelope('ai_ack', req.request_id, ack), room=sid)