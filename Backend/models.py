from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime, timezone

# --- SOCKET ENVELOPE BUILDER (Rule 3) ---
def build_envelope(msg_type: str, request_id: str, payload: Any) -> Dict[str, Any]:
    """Wraps all outbound messages in a strict envelope format."""
    return {
        "type": msg_type,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "payload": payload
    }

# --- INBOUND PAYLOAD VALIDATION (Rule 2) ---
class BaseSocketRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{int(datetime.now().timestamp())}")

class StartupRequest(BaseSocketRequest):
    pass

class RequestStocks(BaseSocketRequest):
    page: int = 0
    limit: int = 20
    query: str = ""

class AIRequest(BaseSocketRequest):
    content: str
    seed: Optional[str] = None
    model: str = 'cloud'

class IndicatorRequest(BaseSocketRequest):
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None