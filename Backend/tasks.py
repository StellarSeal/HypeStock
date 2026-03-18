import os
from celery import Celery
import socketio
from ai_agent import ai_gateway
from models import build_envelope

# --- CONFIGURATION ---
# Target the updated Redis port schema for message brokering
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:26379/0")

# Initialize the Celery Application
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# Initialize a synchronous Redis Manager. 
# This allows our detached Celery workers to broadcast Socket.IO events 
# through Redis, which the FastAPI ASGI server will pick up and send to clients.
socket_manager = socketio.RedisManager(REDIS_URL)

@celery_app.task(name="tasks.process_ai_chat")
def process_ai_chat(sid: str, request_id: str, content: str, seed: int, context: str):
    """
    Processes natural language queries asynchronously.
    Offloads heavy LLM/PandasAI generation from the main event loop.
    """
    try:
        response = ai_gateway.answer_chat_query(sid, content, context)
        
        # The frontend script.js expects a payload with type="ai" and the specific seed
        payload = {
            "type": "ai",
            "seed": seed,
            "response": response
        }
        
        envelope = build_envelope("ai_response", request_id, payload)
        socket_manager.emit('ai_response', envelope, room=sid)
        
    except Exception as e:
        error_envelope = build_envelope('error', request_id, {"code": 500, "message": f"AI Processing Error: {str(e)}"})
        socket_manager.emit('error', error_envelope, room=sid)


@celery_app.task(name="tasks.generate_prediction_explanation")
def generate_prediction_explanation(sid: str, request_id: str, symbol: str, range_val: str, top_features: list):
    """
    Generates a financial explanation of the ML prediction features in the background.
    """
    try:
        explanation = ai_gateway.generate_prediction_explanation(symbol, range_val, top_features)
        
        payload = {
            "symbol": symbol,
            "explanation": explanation
        }
        
        envelope = build_envelope("explain_result", request_id, payload)
        socket_manager.emit('explain_result', envelope, room=sid)
        
    except Exception as e:
        error_envelope = build_envelope('error', request_id, {"code": 500, "message": f"Explanation Error: {str(e)}"})
        socket_manager.emit('error', error_envelope, room=sid)


@celery_app.task(name="tasks.clear_user_memory")
def clear_user_memory(sid: str):
    """
    Frees in-process AI session memory on client disconnects.
    """
    from ai_agent import session_memory_store
    session_memory_store.free_memory(sid)


@celery_app.task(name="tasks.compute_feature_importance")
def compute_feature_importance(*args, **kwargs):
    """
    Placeholder task for long-running dataset permutations 
    or recalculating feature importance matrices.
    """
    pass