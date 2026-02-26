import os
from celery import Celery
import socketio
import ai_agent
from models import build_envelope

# --- CELERY CONFIGURATION (Rule 5) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Setup Celery Worker
celery_app = Celery('ai_tasks', broker=REDIS_URL)

# Setup Sync RedisManager to communicate back directly to Socket.IO clients
mgr = socketio.RedisManager(REDIS_URL)

@celery_app.task
def process_ai_query_task(sid: str, request_id: str, content: str, seed: str, model_provider: str):
    """
    Background Task: Receives an AI request from Redis Queue, processes it via LLMs,
    and emits the envelope message back to the client websocket connection via RedisManager.
    """
    try:
        # 1. Heavy Blocking Call
        response_text = ai_agent.process_message(content, model_provider)

        # 2. Package
        payload = {
            "content": content,
            "response": response_text,
            "seed": seed,
            "model_used": model_provider
        }
        envelope = build_envelope("ai_response", request_id, payload)
        
        # 3. Emit straight to user sid, skipping FastAPI loop entirely
        mgr.emit('ai_response', envelope, room=sid)

    except Exception as e:
        print(f"[CELERY] Task Failed: {e}")
        error_env = build_envelope("error", request_id, {"message": f"Background Process Error: {str(e)}"})
        mgr.emit('error', error_env, room=sid)