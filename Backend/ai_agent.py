import os
import json
import time
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from google import genai

GEMINI_TOKEN_ENV_KEYS = ("GEMINI_API_KEYS", "GEMINI_API_KEY")
FALLBACK_GEMINI_TOKEN = "dummy"


def _parse_token_pool(raw_value: str) -> list[str]:
    return [token.strip() for token in (raw_value or "").split(",") if token.strip()]


def _load_token_pool() -> list[str]:
    # Prefer already-exported environment variables (e.g. docker compose injection).
    for key in GEMINI_TOKEN_ENV_KEYS:
        tokens = _parse_token_pool(os.environ.get(key, ""))
        if tokens:
            return tokens

    # Fallback for local direct runs where .env exists but isn't injected.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()

                key, value = line.split("=", 1)
                if key.strip() not in GEMINI_TOKEN_ENV_KEYS:
                    continue

                clean_value = value.strip().strip("\"").strip("'")
                tokens = _parse_token_pool(clean_value)
                if tokens:
                    return tokens
        except Exception:
            pass

    return [FALLBACK_GEMINI_TOKEN]

class SessionMemory:
    """
    Lightweight in-process memory with a rolling message window per session.
    """

    def __init__(self):
        self._sessions = {}
        self._lock = RLock()
        self._max_messages = self._read_int_env("CHAT_MEMORY_MAX_MESSAGES", 24)
        self._max_chars = self._read_int_env("CHAT_MEMORY_MAX_MESSAGE_CHARS", 360)

    def _read_int_env(self, name: str, default: int) -> int:
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
            return max(1, value)
        except (TypeError, ValueError):
            return default

    def _truncate(self, text: str, limit: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 3].rstrip() + "..."

    def add_message(self, token: str, role: str, content: str, metadata: dict = None):
        if not token or not content:
            return

        role_name = "client" if role.lower() in {"user", "client"} else "backend"
        metadata_text = ""
        if metadata:
            try:
                metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                metadata_text = ""

        entry = {
            "role": role_name,
            "content": content,
            "metadata": metadata_text,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

        with self._lock:
            if token not in self._sessions:
                self._sessions[token] = deque(maxlen=self._max_messages)
            self._sessions[token].append(entry)

    def get_history(self, token: str, current_query: str = "") -> str:
        _ = current_query  # Reserved for future prioritization.

        with self._lock:
            entries = list(self._sessions.get(token, []))

        if not entries:
            return ""

        lines = []
        for entry in entries[-8:]:
            speaker = "Client" if entry["role"] == "client" else "Backend"
            content = self._truncate(entry["content"], self._max_chars)
            metadata = self._truncate(entry.get("metadata", ""), 120) if entry.get("metadata") else ""
            if metadata:
                lines.append(f"- {speaker}: {content} | meta: {metadata}")
            else:
                lines.append(f"- {speaker}: {content}")

        return "\n--- Conversation Memory (recent) ---\n" + "\n".join(lines) + "\n"

    def free_memory(self, token: str):
        with self._lock:
            self._sessions.pop(token, None)

# Global in-house session memory instance
session_memory_store = SessionMemory()

class AIGateway:
    """
    Centralized AI routing service.
    Handles token pooling and Gemini request retries.
    """
    def __init__(self):
        self._pool_lock = RLock()
        self._token_pool = _load_token_pool()
        self._token_index = 0
        self._gemini_clients: dict[str, genai.Client] = {}
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        # Pre-warm at least one client, while keeping full lazy round-robin creation.
        self.gemini_client = None
        for token in self._token_pool:
            client = self._get_or_create_client(token)
            if client:
                self.gemini_client = client
                break

        # Start rotation from the first token on the first live request.
        with self._pool_lock:
            self._token_index = 0

    def _next_token(self) -> str:
        with self._pool_lock:
            if not self._token_pool:
                return ""

            token = self._token_pool[self._token_index]
            self._token_index = (self._token_index + 1) % len(self._token_pool)
            return token

    def _get_or_create_client(self, token: str):
        with self._pool_lock:
            cached_client = self._gemini_clients.get(token)
        if cached_client is not None:
            return cached_client

        try:
            client = genai.Client(api_key=token)
        except Exception as e:
            print(f"[AIGateway] Error initializing Gemini client for token pool entry: {e}")
            return None

        with self._pool_lock:
            self._gemini_clients[token] = client
        return client

    def _next_gemini_agent(self):
        token = self._next_token()
        if not token:
            return "", None
        return token, self._get_or_create_client(token)

    def _extract_analysis_mode(self, query: str) -> tuple[str, str]:
        if not query:
            return "", "concise"

        mode = "concise"
        cleaned_query = query
        mode_match = re.search(r"\bMODE\s*=\s*(concise|detailed)\b", query, flags=re.IGNORECASE)
        if mode_match:
            mode = mode_match.group(1).lower()
            cleaned_query = re.sub(
                r"\bMODE\s*=\s*(concise|detailed)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()

        if not cleaned_query:
            cleaned_query = query.strip()

        return cleaned_query, mode

    def _build_price_only_analysis_prompt(
        self,
        user_query: str,
        elegant_payload: str,
        data_context: str,
        mode: str,
    ) -> str:
        if mode == "detailed":
            length_instruction = "Target length: 130-180 words (~30-50% shorter than a full narrative draft)."
        else:
            length_instruction = "Target length: 90-140 words (~30-50% shorter than a full narrative draft)."

        return (
            "You are refining a stock analysis response generated in a price-only environment.\n"
            "Rewrite it to be tighter, more direct, and trading-desk style.\n"
            "Objective:\n"
            "- Compress output by ~30-50% while preserving meaning and final bias.\n"
            "- Increase information density and remove academic tone.\n"
            "Strict rules:\n"
            "- Preserve section structure exactly: [1] Snapshot, [2] Interpretation, [3] Conclusion.\n"
            "- Do not add or remove sections.\n"
            "- Do not add disclaimers.\n"
            "- Do not discuss fundamentals, macroeconomics, external catalysts, or news.\n"
            "- Use only price trend, volatility, and volume behavior.\n"
            "- Keep wording concise, analytical, and confident.\n"
            f"- Mode: {mode}. {length_instruction}\n\n"
            "Compression rules:\n"
            "- Replace long phrasing with direct wording.\n"
            "- Remove filler phrases and repeated ideas across sections.\n"
            "- [1] Snapshot bullets: max 18 words each.\n"
            "- [2] Interpretation: maximum 3 sentences.\n\n"
            "Metric referencing (important):\n"
            "- Avoid vague descriptions; reference key metrics when available.\n"
            "- Prioritize only high-signal numbers: price levels, % changes, volatility behavior, and volume vs baseline.\n"
            "- Integrate metrics naturally (e.g., '34,800 VNĐ to 37,000 VNĐ', '-1.9%', '100k vs 30k baseline').\n"
            "- Do not overload with numbers.\n\n"
            "Output format (must match exactly):\n"
            "[1] Snapshot\n"
            "- Price trend: <max 18 words>\n"
            "- Volatility: <max 18 words>\n"
            "- Volume: <max 18 words>\n\n"
            "[2] Interpretation\n"
            "<up to 3 concise sentences linking trend, volume, and volatility>\n\n"
            "[3] Conclusion\n"
            "- Bias: Bullish / Bearish / Neutral\n"
            "- Reason: <one-line justification from observed signals>\n\n"
            "Decision requirement:\n"
            "- Always choose exactly one of Bullish, Bearish, or Neutral.\n"
            "- Never hedge or avoid the directional call.\n\n"
            f"User query: {user_query}\n"
            f"Context and memory: {elegant_payload}\n"
            f"Dataset extraction: {data_context}\n"
            "Return only the rewritten analysis."
        )

    def generate_prediction_explanation(self, symbol: str, range_val: str, top_features: list, recent_data: str = "") -> str:
        prompt = (
            f"You are a financial analyst AI. Explain the stock price prediction for {symbol} "
            f"over a {range_val} range. The regression model identified these top driving features: "
            f"{', '.join(top_features)}. Provide an assertive, professional explanation."
            f"CURRENCY RULE: Format any currency in thousands of VNĐ (e.g., 22.3 becomes 22,300 VNĐ)."
        )
        return self.route_request(prompt, lightweight=True)

    def route_request(self, prompt: str, lightweight: bool = True) -> str:
        """Routes requests to Gemini with retry/backoff on throttling."""
        _ = lightweight  # Kept for backward compatibility with existing callers.

        if not self._token_pool:
            return "System Error: Gemini token pool is empty."
            
        last_error = None
        max_attempts = max(3, len(self._token_pool))
        try:
            for attempt in range(max_attempts):
                _, agent_client = self._next_gemini_agent()
                if not agent_client:
                    last_error = RuntimeError("Gemini client is not initialized for selected token.")
                    continue

                try:
                    response = agent_client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=["You are a specialized financial AI.", prompt]
                    )
                    return response.text
                except Exception as api_err:
                    last_error = api_err
                    if "429" in str(api_err) and attempt < (max_attempts - 1):
                        time.sleep(2 ** min(attempt, 4))  # Exponential backoff with sane cap
                        continue
                    if any(code in str(api_err) for code in ("401", "403")) and attempt < (max_attempts - 1):
                        continue

            if last_error:
                return f"Agent Runtime Error: {str(last_error)}"
            return "Agent Runtime Error: Gemini request failed with no detailed error."
        except Exception as e:
            return f"Agent Runtime Error: {str(e)}"

    def answer_chat_query(self, sid: str, query: str, context: str = "") -> str:
        """
        Classifies and processes the chat query using Dataset (A), Hybrid (B), or LLM (C) modes.
        Includes Memory retrieval, Context formatting obfuscation and Strict Guardrails.
        """
        context_details = ""
        context_compare = ""
        context_graph = ""
        normalized_query, analysis_mode = self._extract_analysis_mode(query)
        
        try:
            if context:
                ctx_data = json.loads(context)
                context_details = ctx_data.get("details", "")
                context_compare = ctx_data.get("compare", "")
                context_graph = ctx_data.get("graph_data", "")
        except Exception:
            pass
            
        # Retrieve recent in-process session history for contextual awareness
        history = session_memory_store.get_history(sid, current_query=normalized_query)
        
        elegant_payload = f"User asked: '{normalized_query}'\n"
        if history:
            elegant_payload = history + elegant_payload
        if context_details:
            elegant_payload += f"User is looking at: '{context_details}'\n"
        if context_compare:
            elegant_payload += f"User is comparing: '{context_compare}'\n"
        if context_graph:
            elegant_payload += f"Recent Metrics (Compressed 30d): {context_graph}\n"

        router_prompt = (
            "You are an intent classifier for a financial AI assistant.\n"
            "Classify the following user query into exactly one of these categories:\n"
            "MODE_A: Queries requiring specific historical numerical data, statistical aggregations, or simple technical indicator lookups from the dataset (e.g., 'price', 'volume', 'MA20', 'RSI', 'unusual volume', 'gap up/down').\n"
            "MODE_B: Queries asking to interpret technical data, analyze trends, or explain indicators for a specific stock (e.g., 'Is this stock overbought?', 'Is the trend bullish?', 'Interpret MACD', 'Explain RSI for AAPL').\n"
            "MODE_C: Queries asking about general fundamentals (P/E, PEG, ROE, FCF, debt, dividends, institutional ownership), macro sentiment, analyst targets, or general market education.\n"
            "IRRELEVANT: Queries not related to stocks, finance, or the market.\n\n"
            f"Context and Query:\n{elegant_payload}\n\n"
            "Reply with ONLY the exact word 'MODE_A', 'MODE_B', 'MODE_C', or 'IRRELEVANT'."
        )
        route = self.route_request(router_prompt, lightweight=False).strip().upper()

        TONING_RULE = (
            "CRITICAL GUARDRAILS:\n"
            "1. From what I can tell, if fundamental data is requested and not explicitly provided, "
            "do not explicitly state the lack of a dataset. Instead, assertively state that the platform "
            "only shows price data, and that users should not base their decision alone on this.\n"
            "2. Always format currency in thousands of VNĐ. For example, convert 22.3 to 22,300 VNĐ."
        )

        final_response = ""

        if "IRRELEVANT" in route:
            final_response = "I'm sorry, but I can only assist you with stock-related questions."
        
        elif "MODE_A" in route:
            # Pure DB Extractor Mode
            final_response = self.query_pandasai(normalized_query, context, context_graph)

        elif "MODE_B" in route:
            # Hybrid Mode: Contextual Dataset Extraction + Interpretation
            data_context = self.query_pandasai(normalized_query, context, context_graph)
            hybrid_prompt = self._build_price_only_analysis_prompt(
                user_query=normalized_query,
                elegant_payload=elegant_payload,
                data_context=data_context,
                mode=analysis_mode,
            )
            final_response = self.route_request(hybrid_prompt, lightweight=False)
            
        else:
            # MODE_C: General Knowledge / Fundamentals / Educational with strong Assertive Guardrails
            general_prompt = (
                f"You are a helpful, assertive stock market assistant.\n"
                f"{elegant_payload}\n"
                f"CRITICAL GUARDRAIL: Never fabricate, invent, or estimate fundamental data.\n"
                f"{TONING_RULE}"
            )
            final_response = self.route_request(general_prompt, lightweight=False)

        # Persist both client and backend messages into in-process session memory.
        session_memory_store.add_message(
            sid,
            "client",
            query,
            metadata={
                "details": context_details,
                "compare": context_compare,
                "graph_data": context_graph,
            },
        )
        session_memory_store.add_message(
            sid,
            "backend",
            final_response,
            metadata={"route": route},
        )

        return final_response

    def query_pandasai(self, query: str, context: str = "", context_graph: str = "") -> str:
        try:
            import pandasai as pai
            from pandasai import Agent
            from pandasai_litellm.litellm import LiteLLM
            
            db_connection = {
                "host": "db",
                "port": 15432, 
                "user": "admin", 
                "password": "hypestock_password_idk",
                "database": "stock_data"
            }

            # PandasAI v3 semantic paths require slug-safe organization/model names.
            semantic_org = "stock-data"
            
            api_token = self._next_token()
            if not api_token:
                return "System Error: Gemini token pool is empty."

            llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=api_token)
            
            # Using PandasAI v3 Semantic Data Extensions for Postgres
            df_companies = pai.create(
                path=f"{semantic_org}/companies",
                description="Contains company stock symbols and names.",
                source={
                    "type": "postgres",
                    "connection": db_connection,
                    "table": "companies",
                    "columns": [
                        {"name": "stock_code", "type": "string", "description": "Stock code or ticker symbol"},
                        {"name": "company_name", "type": "string", "description": "Full company name"}
                    ]
                }
            )

            df_prices = pai.create(
                path=f"{semantic_org}/stock-prices",
                description="Daily OHLCV historical stock prices.",
                source={
                    "type": "postgres",
                    "connection": db_connection,
                    "table": "stock_prices",
                    "columns": [
                        {"name": "time", "type": "datetime", "description": "Trading date and time"},
                        {"name": "symbol", "type": "string", "description": "Stock code"},
                        {"name": "open", "type": "float", "description": "Opening price"},
                        {"name": "high", "type": "float", "description": "Highest price"},
                        {"name": "low", "type": "float", "description": "Lowest price"},
                        {"name": "close", "type": "float", "description": "Closing price"},
                        {"name": "volume", "type": "integer", "description": "Trading volume"}
                    ]
                }
            )

            df_metrics = pai.create(
                path=f"{semantic_org}/metrics",
                description="Calculated technical indicators and metrics.",
                source={
                    "type": "postgres",
                    "connection": db_connection,
                    "table": "metrics",
                    "columns": [
                        {"name": "time", "type": "datetime", "description": "Date of metrics"},
                        {"name": "symbol", "type": "string", "description": "Stock code"},
                        {"name": "ma20", "type": "float", "description": "20-day moving average"},
                        {"name": "ma50", "type": "float", "description": "50-day moving average"},
                        {"name": "ema20", "type": "float", "description": "20-day exponential moving average"},
                        {"name": "rsi", "type": "float", "description": "Relative strength index"},
                        {"name": "macd", "type": "float", "description": "MACD indicator"},
                        {"name": "rolling_vol_20d_std", "type": "float", "description": "20-day rolling volatility"},
                        {"name": "atr", "type": "float", "description": "Average true range"},
                        {"name": "volume_ma20", "type": "float", "description": "20-day volume moving average"},
                        {"name": "volume_change_pct", "type": "float", "description": "Volume change percentage"},
                        {"name": "daily_return_1d", "type": "float", "description": "1-day daily return"},
                        {"name": "daily_return_5d", "type": "float", "description": "5-day daily return"},
                        {"name": "cumulative_return", "type": "float", "description": "Cumulative return"},
                        {"name": "daily_range", "type": "float", "description": "Daily price range"},
                        {"name": "vol_close_corr_20d", "type": "float", "description": "Volume-close correlation (20 days)"},
                        {"name": "bb_width", "type": "float", "description": "Bollinger bands width"},
                        {"name": "adx", "type": "float", "description": "Average directional index"},
                        {"name": "obv_slope_5d", "type": "float", "description": "5-day OBV slope"},
                        {"name": "lagged_return_t1", "type": "float", "description": "Lagged return (t-1)"},
                        {"name": "lagged_return_t3", "type": "float", "description": "Lagged return (t-3)"},
                        {"name": "lagged_return_t5", "type": "float", "description": "Lagged return (t-5)"},
                        {"name": "dist_from_ma50", "type": "float", "description": "Distance from MA50"}
                    ]
                }
            )
            
            context_injection = (
                f"\n\n--- PLATFORM METADATA & DATASET SCOPE INSTRUCTIONS ---\n"
                f"You have access to 3 datasets: 'companies' (stock symbols and names), 'stock_prices' (daily OHLCV), and 'metrics' (technical indicators).\n"
            )
            
            try:
                if context:
                    ctx_data = json.loads(context)
                    context_details = ctx_data.get("details", "")
                    context_compare = ctx_data.get("compare", "")
                    if context_details or context_compare or context_graph:
                        context_injection += f"\n- CURRENT APP CONTEXT:\n"
                        if context_details:
                            context_injection += f"  - User is looking at: '{context_details}'\n"
                        if context_compare:
                            context_injection += f"  - User is comparing: '{context_compare}'\n"
                        if context_graph:
                            context_injection += f"  - Compressed 30-day Chart Data: '{context_graph}'\n"
                        context_injection += "Use this context if the user implies \"this stock\" or \"these stocks\".\n"
            except Exception:
                pass
            
            # Toning formatting override for PandasAI raw responses
            context_injection += (
                "\n- CRITICAL FORMATTING: From what I can tell, convert any price or currency values to thousands of VNĐ by "
                "multiplying by 1000 and appending 'VNĐ' (e.g., 22.3 becomes 22,300 VNĐ).\n"
            )

            agent = Agent(
                [df_companies, df_prices, df_metrics],
                config={
                    "llm": llm,
                    "verbose": False,
                    "enable_cache": False
                }
            )
            
            response = agent.chat(query + context_injection)
            return str(response)
            
        except Exception as e:
            print(f"[AIGateway] PandasAI Error: {e}")
            return self.route_request(
                f"The user asked: '{query}'. However, the internal dataset query failed. "
                f"Provide a helpful general answer. Assertively analyze the query from what I can tell and general market context, prioritizing insights that can be derived from the available context.\n"
                f"CRITICAL GUARDRAILS:\n"
                f"1. Do not explicitly state the lack of a dataset. Instead, assertively state that the platform only shows price data, and that users should not base their decision alone on this.\n"
                f"2. Convert currency to thousands of VNĐ (e.g., 22.3 to 22,300 VNĐ).",
                lightweight=False
            )

# Singleton instance
ai_gateway = AIGateway()