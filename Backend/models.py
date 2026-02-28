from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime, timezone

# --- SOCKET ENVELOPE BUILDER ---
def build_envelope(msg_type: str, request_id: str, payload: Any) -> Dict[str, Any]:
    """Wraps all outbound messages in a strict envelope format."""
    clean_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    return {
        "type": msg_type,
        "request_id": request_id,
        "timestamp": clean_timestamp,
        "payload": payload
    }

# --- SOCKET PAYLOAD VALIDATION ---
class BaseSocketRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{int(datetime.now().timestamp())}")

class ExplainPredictionRequest(BaseSocketRequest):
    symbol: str
    range: str
    top_features: List[str]

# --- REST PAYLOAD MODELS ---
class MetricsResponse(BaseModel):
    highest_close: float
    lowest_close: float
    average_volume: float
    volatility: float
    cumulative_return: float
    trading_days: int

class SummaryResponse(BaseModel):
    company_name: str
    symbol: str
    start_date: str
    end_date: str
    data_range: str
    metrics: MetricsResponse

class PredictionResponse(BaseModel):
    available: bool
    message: Optional[str] = "Model training in progress or not yet integrated."