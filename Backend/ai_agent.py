import os
import requests
import pandas as pd
from google import genai
from database import sync_engine

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "dummy")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/chat") # Adjusted for docker DNS
DEFAULT_LOCAL_MODEL = "gemma3:12b"

SYSTEM_PROMPT = (
    "You are an LLM-based financial data analyst and market prediction assistant. "
    "Scope of responsibility: Equity markets, Stock prices, technical indicators, financial datasets. "
    "Behavior rules: Answer ONLY questions that clearly relate to the scope above. "
    "If a question is unrelated, respond with exactly the single word: banana"
)

# --- SETUP CLIENTS ---
client = None
PANDAS_AVAILABLE = False

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"[AI] Error initializing Gemini client: {e}")

try:
    import pandasai as pai
    from pandasai.llm import GoogleGemini
    
    llm = GoogleGemini(api_key=API_KEY)
    pai.config.set({'llm': llm, 'enable_cache': False})
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"[AI] PandasAI not available: {e}. Running chat-only mode.")

# --- ROUTING & LOGIC ---

def query_ollama(prompt, model=DEFAULT_LOCAL_MODEL):
    try:
        payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('message', {}).get('content', 'Error: No content from Ollama.')
    except requests.exceptions.RequestException as e:
        return f"Error connecting to local Ollama container: {e}"

def process_message(user_input, model_provider='cloud'):
    """Synchronous execution entry point designed for Celery workers."""
    key_words = ["average", "max", "min", "plot", "compare", "correlation", "trend", "price", "stock"]
    
    try:
        # PATH A: PANDAS AI (Dynamic Extractor)
        if PANDAS_AVAILABLE and any(k in user_input.lower() for k in key_words):
            # Load a recent slice of the dataset to analyze synchronously
            # Limit row count to avoid memory bloat in Celery Container
            query = "SELECT * FROM stock_prices ORDER BY time DESC LIMIT 20000"
            df = pd.read_sql(query, sync_engine)
            
            if not df.empty:
                result = df.chat(user_input)
                return str(result)

        # PATH B: OLLAMA
        if model_provider == 'local':
            return query_ollama(user_input)

        # PATH C: GEMINI (Fallback Cloud)
        if not client:
            return "System Error: Gemini Client not initialized."
            
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[SYSTEM_PROMPT, user_input]
        )
        return response.text
            
    except Exception as e:
        return f"Agent Runtime Error: {str(e)}"