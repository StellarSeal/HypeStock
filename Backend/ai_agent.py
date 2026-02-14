# --- AI AGENT MODULE ---
# Handles interactions with LLMs (Cloud & Local) and Data Analysis (PandasAI).

import os
import requests
import pandas as pd
from google import genai

# --- CONFIGURATION ---
# Fallback to the key found in previous context if env var not set
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBN6b76KcGMke4pYkKHk4ggU01_nGc8If0")
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LOCAL_MODEL = "gemma3:12b"

SYSTEM_PROMPT = (
    "You are an LLM-based financial data analyst and market prediction assistant. "
    "Scope of responsibility: "
    "Equity markets and stocks; "
    "Stock prices, historical data, and price action; "
    "Technical and statistical indicators; "
    "Financial markets and macro/micro market behavior; "
    "Company fundamentals and performance; "
    "Financial datasets, time series, and predictive analysis. "
    "Behavior rules: "
    "Answer ONLY questions that clearly relate to the scope above. "
    "If a question is unrelated, respond with exactly the single word: banana"
)

# --- SETUP CLIENTS ---
client = None
df = None
PANDAS_AVAILABLE = False

# 1. Setup Gemini
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"[AI] Error initializing Gemini client: {e}")

# 2. Setup PandasAI (Optional)
try:
    import pandasai as pai
    # Attempt to import LiteLLM wrapper if available, or fallback to standard logic if needed
    # For this implementation, we'll try to follow geminam.py pattern specifically
    try:
        from pandasai_litellm.litellm import LiteLLM
        llm = LiteLLM(model='gemini/gemini-2.5-flash', api_key=API_KEY)
    except ImportError:
        # Fallback to standard PandasAI Gemini if the specific litellm wrapper is missing
        from pandasai.llm import GoogleGemini
        llm = GoogleGemini(api_key=API_KEY)

    # Load Data
    csv_file = 'stock_prices.csv'
    if os.path.exists(csv_file):
        df = pai.read_csv(csv_file)
        print(f"[AI] Loaded data from {csv_file} for Analysis")
    else:
        print(f"[WARN] {csv_file} not found. Using mock stock data for PandasAI.")
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'AAPL': [150.0, 152.5, 151.2, 153.8],
            'GOOGL': [100.0, 101.5, 99.8, 102.1],
            'MSFT': [250.0, 255.0, 253.5, 258.0]
        })
    
    # Configure PandasAI
    pai.config.set({'llm': llm})
    PANDAS_AVAILABLE = True

except ImportError as e:
    print(f"[WARN] PandasAI/Data Analysis modules not available: {e}")
    print("[INFO] Agent will run in Chat-Only mode.")
    PANDAS_AVAILABLE = False

# --- HELPER FUNCTIONS ---

def query_ollama(prompt, model=DEFAULT_LOCAL_MODEL):
    """Sends a request to the local Ollama instance."""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        print(f"[OLLAMA] Sending request to {model}...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get('message', {}).get('content', 'Error: No content from Ollama.')
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ollama Connection Failed: {e}")
        return f"Error: Could not connect to local model ({model}). Ensure Ollama is running."

def process_message(user_input, model_provider='cloud'):
    """
    Main entry point for AI processing.
    Routes between PandasAI, Ollama, and Gemini based on input and config.
    """
    # Keywords that trigger Data Analysis mode
    key_words = ["average", "max", "min", "plot", "compare", "correlation", "trend", "price", "stock"]
    
    try:
        # PATH A: Data Analysis (PandasAI)
        # Triggered if Pandas is available, data exists, and input contains keywords
        if PANDAS_AVAILABLE and df is not None and any(k in user_input.lower() for k in key_words):
            print(f"[PANDAS] Routing to PandasAI: {user_input}")
            # PandasAI 'chat' method usually returns a result string or handle
            result = df.chat(user_input)
            return str(result)
            
        # PATH B: Local Model (Ollama)
        elif model_provider == 'local':
            print(f"[LOCAL] Routing to Ollama ({DEFAULT_LOCAL_MODEL}): {user_input}")
            return query_ollama(user_input)

        # PATH C: Remote Model (Gemini) - Default
        else:
            if not client:
                return "System Error: Gemini Client not initialized."
                
            print(f"[CLOUD] Routing to Gemini (Flash): {user_input}")
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[SYSTEM_PROMPT, user_input]
            )
            return response.text
            
    except Exception as e:
        print(f"[AI] Processing Error: {e}")
        return f"I encountered an error processing that request: {str(e)}"