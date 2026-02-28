import os
import json
import time
from celery import Celery
import socketio
import redis
from models import build_envelope
from ai_agent import ai_gateway

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Setup Celery Worker
celery_app = Celery('ai_tasks', broker=REDIS_URL)

# Setup Sync RedisManager and standard redis client
mgr = socketio.RedisManager(REDIS_URL)
sync_redis = redis.Redis.from_url(REDIS_URL)

@celery_app.task
def generate_prediction_explanation(sid: str, request_id: str, symbol: str, range_val: str, top_features: list):
    """
    Background Task: Calls the AIGateway and emits result straight back to the socket.
    """
    try:
        explanation = ai_gateway.generate_prediction_explanation(symbol, range_val, top_features)
        
        payload = {
            "symbol": symbol,
            "range": range_val,
            "explanation_text": explanation
        }
        envelope = build_envelope("prediction_explanation_ready", request_id, payload)
        mgr.emit('prediction_explanation_ready', envelope, room=sid)

    except Exception as e:
        print(f"[CELERY] Explanation Task Failed: {e}")
        error_env = build_envelope("server_error", request_id, {"code": 500, "message": str(e)})
        mgr.emit('server_error', error_env, room=sid)

@celery_app.task
def compute_feature_importance(symbol: str, range_val: str):
    """
    Background Task: Trains lightweight RandomForestRegressor and extracts importance.
    Stores result in Redis for the REST endpoint to consume.
    """
    # Mock ML process for the immediate frontend build
    time.sleep(2) 
    importance = {
        "trend_strength": 0.45, 
        "volume_surge": 0.30, 
        "volatility": 0.25
    }
    
    # Store directly into Redis cache
    cache_key = f"importance:{symbol}:{range_val}"
    sync_redis.setex(cache_key, 3600, json.dumps(importance)) # 1 hr TTL
    
    return importance