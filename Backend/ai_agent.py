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

    def answer_chat_query(self, query: str, context: str = "") -> str:
        """
        Classifies and processes the chat query using Dataset (A), Hybrid (B), or LLM (C) modes.
        Includes Active State Context and Strict Guardrails.
        """
        router_prompt = (
            "You are an intent classifier for a financial AI assistant.\n"
            "Classify the following user query into exactly one of these categories:\n"
            "MODE_A: Queries requiring specific historical numerical data, statistical aggregations, or simple technical indicator lookups from the dataset (e.g., 'price', 'volume', 'MA20', 'RSI', 'unusual volume', 'gap up/down').\n"
            "MODE_B: Queries asking to interpret technical data, analyze trends, or explain indicators for a specific stock (e.g., 'Is this stock overbought?', 'Is the trend bullish?', 'Interpret MACD', 'Explain RSI for AAPL').\n"
            "MODE_C: Queries asking about general fundamentals (P/E, PEG, ROE, FCF, debt, dividends, institutional ownership), macro sentiment, analyst targets, or general market education.\n"
            "IRRELEVANT: Queries not related to stocks, finance, or the market.\n\n"
            f"Query: \"{query}\"\n\n"
            "Reply with ONLY the exact word 'MODE_A', 'MODE_B', 'MODE_C', or 'IRRELEVANT'."
        )
        route = self.route_request(router_prompt, lightweight=False).strip().upper()

        if "IRRELEVANT" in route:
            return "I'm sorry, but I can only assist you with stock-related questions."
        
        if "MODE_A" in route:
            # Pure DB Extractor Mode
            return self.query_pandasai(query, context)

        elif "MODE_B" in route:
            # Hybrid Mode: Contextual Dataset Extraction + Interpretation
            data_context = self.query_pandasai(query, context)
            hybrid_prompt = (
                f"You are an elite financial analyst AI. A user asked: '{query}'.\n"
                f"Based on the dataset extraction, we retrieved this exact data: {data_context}\n"
                f"Current App Context: {context}\n"
                "Please explain and interpret this data specifically for the user in a professional, concise manner. Do not fabricate extra numbers."
            )
            return self.route_request(hybrid_prompt, lightweight=False)
            
        else:
            # MODE_C: General Knowledge / Fundamentals / Educational with strong Guardrails
            general_prompt = (
                f"You are a helpful stock market assistant. Current App Context: {context}\n"
                "CRITICAL GUARDRAIL: Never fabricate, invent, or estimate fundamental data (e.g., P/E, PEG, ROE, FCF, debt ratios, dividends, institutional ownership, analyst targets, earnings expectations). "
                "If the user asks for fundamental data and it is not explicitly provided in the current context, you MUST state clearly that the platform's dataset does not contain this information currently.\n"
                f"User Query: {query}"
            )
            return self.route_request(general_prompt, lightweight=False)

    def query_pandasai(self, query: str, context: str = "") -> str:
        try:
            import pandasai as pai
            from pandasai_litellm.litellm import LiteLLM
            
            db_connection = {
                "host": "db",
                "port": 15432,
                "database": "stock_data",
                "user": "admin",
                "password": "hypestock_password_idk"
            }
            
            llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=API_KEY)
            pai.config.set({
                "llm": llm,
                "verbose": False
            })
            
            conn_companies = pai.create(
                path="hypestock/companies",
                source={"type": "postgres", "connection": db_connection, "table": "companies"}
            )
            
            conn_prices = pai.create(
                path="hypestock/stock-prices",
                source={"type": "postgres", "connection": db_connection, "table": "stock_prices"}
            )
            
            conn_metrics = pai.create(
                path="hypestock/metrics",
                source={"type": "postgres", "connection": db_connection, "table": "metrics"}
            )
            
            context_injection = (
                f"\n\n--- PLATFORM METADATA & DATASET SCOPE INSTRUCTIONS ---\n"
                f"You have access to 3 dataframes: 'companies' (stock symbols and names), 'stock_prices' (daily OHLCV), and 'metrics' (technical indicators).\n"
                f"- 'stock_prices' contains: Open, High, Low, Close, Volume.\n"
                f"- 'metrics' contains: MA20, MA50, EMA20, RSI, MACD, ATR, Rolling Volatility 20d, BB Width, ADX, Daily Returns, Cumulative Return, etc.\n"
            )
            
            if context:
                context_injection += f"\n- CURRENT APP CONTEXT: {context}. Use this context if the user implies \"this stock\" or \"these stocks\".\n"
            
            response = pai.chat(query + context_injection, conn_companies, conn_prices, conn_metrics)
            return str(response)
            
        except Exception as e:
            print(f"[AIGateway] PandasAI Error: {e}")
            return self.route_request(
                f"The user asked: '{query}'. However, the internal dataset query failed. "
                f"Provide a helpful general answer to their stock-related question noting that realtime specifics aren't accessible.",
                lightweight=False
            )

# Singleton instance
ai_gateway = AIGateway()