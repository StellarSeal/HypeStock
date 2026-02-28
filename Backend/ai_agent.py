import os
import requests
import time
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY", "dummy")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/chat")
DEFAULT_LOCAL_MODEL = "gemma3:12b"

class AIGateway:
    """
    Centralized AI routing service. 
    Handles token pooling, rate limiting, and provider fallback (Ollama -> Gemini).
    """
    def __init__(self):
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        try:
            self.gemini_client = genai.Client(api_key=API_KEY)
        except Exception as e:
            print(f"[AIGateway] Error initializing Gemini client: {e}")

    def generate_prediction_explanation(self, symbol: str, range_val: str, top_features: list, recent_data: str = "") -> str:
        prompt = (
            f"You are a financial analyst AI. Explain the stock price prediction for {symbol} "
            f"over a {range_val} range. The regression model identified these top driving features: "
            f"{', '.join(top_features)}. Provide a concise, professional explanation for an investor."
        )
        # Attempt to use lightweight local model first for simple text generation
        return self.route_request(prompt, lightweight=True)

    def route_request(self, prompt: str, lightweight: bool = True) -> str:
        """Routes request based on complexity and availability with fallback logic."""
        
        # 1. Try Local Ollama if lightweight
        if lightweight:
            try:
                payload = {
                    "model": DEFAULT_LOCAL_MODEL, 
                    "messages": [
                        {"role": "system", "content": "You are a specialized financial AI."},
                        {"role": "user", "content": prompt}
                    ], 
                    "stream": False
                }
                response = requests.post(OLLAMA_URL, json=payload, timeout=30)
                response.raise_for_status()
                return response.json().get('message', {}).get('content', 'Error: Empty response from local AI.')
            except requests.exceptions.RequestException as e:
                print(f"[AIGateway] Local Ollama unavailable/failed ({e}). Falling back to Gemini.")

        # 2. Fallback to Cloud (Gemini)
        if not self.gemini_client:
            return "System Error: No AI providers available (Local failed, Cloud not initialized)."
            
        try:
            # Basic rate limit handling wrapper (Retry mechanism)
            for attempt in range(3):
                try:
                    response = self.gemini_client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=["You are a specialized financial AI.", prompt]
                    )
                    return response.text
                except Exception as api_err:
                    if "429" in str(api_err) and attempt < 2:
                        time.sleep(2 ** attempt) # Exponential backoff
                        continue
                    raise api_err
        except Exception as e:
            return f"Agent Runtime Error (Cloud Fallback Failed): {str(e)}"

# Singleton instance
ai_gateway = AIGateway()