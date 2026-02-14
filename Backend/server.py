# --- SOCKET ORCHESTRATOR ---
# Entry point. Routes events. No business logic.

import asyncio
import socketio
from aiohttp import web
import dataset_service
import ai_agent

# Configuration
PORT = 8000

# Setup Socket.IO Server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# Initialize Data Layer
dataset_service.load_datasets()

# --- EVENTS ---

@sio.event
async def connect(sid, environ):
    print(f"[WS] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[WS] Client disconnected: {sid}")

@sio.on('startup')
async def handle_startup(sid, data):
    """
    Handles the initial frontend handshake.
    Used to synchronize the loading screen with backend readiness.
    """
    print(f"[WS] Startup handshake received from {sid}")
    
    # Optional: Perform any per-session initialization here
    # For now, we simulate a brief check to ensure the UI feels responsive
    await asyncio.sleep(0.2) 
    
    await sio.emit('startup_response', {
        'status': 'ready',
        'server_time': asyncio.get_running_loop().time()
    }, room=sid)

@sio.on('request_stocks')
async def handle_request_stocks(sid, data):
    # data: { page, limit, query }
    data = data or {}
    page = int(data.get('page', 0))
    limit = int(data.get('limit', 20))
    query = data.get('query', "")

    try:
        # Blocking call is fast enough here, but could be threaded if dataset is huge
        result = dataset_service.get_stocks(page, limit, query)
        
        response_payload = result.copy()
        if 'request_id' in data:
            response_payload['request_id'] = data['request_id']
            
        await sio.emit('stock_data', response_payload, room=sid)
        
    except Exception as e:
        print(f"[ERROR] Stock fetch failed: {e}")
        await sio.emit('error', {'message': "Failed to fetch stocks"}, room=sid)

@sio.on('ai')
async def handle_ai_event(sid, data):
    content = data.get('content')
    seed = data.get('seed')
    # Extract model preference (default to cloud if missing)
    model_provider = data.get('model', 'cloud')

    print(f"[RX] Received: {content} (Seed: {seed} | Model: {model_provider})")

    if not content:
        await sio.emit('error', {'message': 'No content provided'}, room=sid)
        return

    # Run AI blocking task in executor to avoid freezing the event loop
    loop = asyncio.get_running_loop()
    try:
        # Pass model_provider to the agent
        ai_response = await loop.run_in_executor(None, ai_agent.process_message, content, model_provider)
        
        await sio.emit('ai_response', {
            "type": "ai",
            "content": content,
            "response": ai_response,
            "seed": seed,
            "model_used": model_provider
        }, room=sid)
        print(f"[TX] Sent response to {sid} (via {model_provider})")
        
    except Exception as e:
        print(f"[ERROR] AI processing failed: {e}")
        await sio.emit('error', {'message': 'AI Processing Error'}, room=sid)

# --- RUN ---

async def handle_index(request):
    return web.Response(text="HypeStock Python Backend Running")

app.add_routes([web.get('/', handle_index)])

if __name__ == '__main__':
    print(f"[START] Server running on http://localhost:{PORT}")
    web.run_app(app, port=PORT)