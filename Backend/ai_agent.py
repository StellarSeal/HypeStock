import os
import json
import time
import re
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from google import genai

GEMINI_TOKEN_ENV_KEY = "GEMINI_API_KEY"
FALLBACK_GEMINI_TOKEN = "dummy"


def _load_token() -> str:
    token = os.environ.get(GEMINI_TOKEN_ENV_KEY, "").strip()
    if token:
        return token

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()

                key, value = line.split("=", 1)
                if key.strip() != GEMINI_TOKEN_ENV_KEY:
                    continue

                token = value.strip().strip('"').strip("'")
                if token:
                    return token
        except Exception:
            pass

    return FALLBACK_GEMINI_TOKEN


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
        _ = current_query

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
    Handles a single Gemini token with rate limiting and quota handling.
    """

    def __init__(self):
        self._rate_lock = RLock()
        self._request_lock = RLock()
        self._api_token = _load_token()
        self._next_allowed_request_ts = 0.0
        self._min_request_interval_s = self._read_float_env(
            "GEMINI_MIN_REQUEST_INTERVAL_SECONDS",
            2.0,
            minimum=0.0,
        )
        self._quota_lock = RLock()
        self._quota_cooldown_seconds = self._read_float_env(
            "GEMINI_QUOTA_COOLDOWN_SECONDS",
            55.0,
            minimum=0.0,
        )
        self._quota_blocked_until_ts = 0.0
        self.gemini_client = None
        self._init_gemini()

    def _read_float_env(self, name: str, default: float, minimum: float = 0.0) -> float:
        raw = os.environ.get(name, str(default))
        try:
            value = float(raw)
            if value < minimum:
                return default
            return value
        except (TypeError, ValueError):
            return default

    def _wait_for_request_slot(self):
        if self._min_request_interval_s <= 0:
            return

        with self._rate_lock:
            now = time.monotonic()
            wait_time = self._next_allowed_request_ts - now
            if wait_time > 0:
                time.sleep(wait_time)
            self._next_allowed_request_ts = time.monotonic() + self._min_request_interval_s

    def _extract_retry_after_seconds(self, err: Exception) -> float:
        response = getattr(err, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            header_value = headers.get("Retry-After") or headers.get("retry-after")
            parsed_header = self._parse_retry_after_seconds(header_value)
            if parsed_header is not None:
                return parsed_header

        text = str(err)
        for pattern in (
            r"retry\s*after\s*[:=]?\s*(\d+(?:\.\d+)?)\s*s",
            r"retry\s*after\s*[:=]?\s*(\d+(?:\.\d+)?)",
            r"retryDelay\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)s?",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return max(0.0, float(match.group(1)))
            except (TypeError, ValueError):
                continue

        return 0.0

    def _is_throttled_error(self, err: Exception) -> bool:
        err_text = str(err)
        err_upper = err_text.upper()
        return (
            "429" in err_text
            or "RESOURCE_EXHAUSTED" in err_upper
            or "TOO MANY REQUESTS" in err_upper
            or "RATE LIMIT" in err_upper
        )

    def _is_quota_exhausted_error(self, err: Exception) -> bool:
        err_upper = str(err).upper()
        quota_markers = (
            "QUOTA EXCEEDED",
            "CHECK YOUR PLAN AND BILLING DETAILS",
            "GENERATIVELANGUAGE.GOOGLEAPIS.COM/GENERATE_CONTENT_FREE_TIER_REQUESTS",
            "GENERATEREQUESTSPERDAYPERPROJECTPERMODEL-FREETIER",
            "FREETIER",
            "PERDAY",
        )
        if "RESOURCE_EXHAUSTED" in err_upper and "QUOTA" in err_upper:
            return True
        return any(marker in err_upper for marker in quota_markers)

    def _start_quota_cooldown(self, err: Exception):
        retry_after = self._extract_retry_after_seconds(err)
        delay = max(self._quota_cooldown_seconds, retry_after)
        if delay <= 0:
            return

        with self._quota_lock:
            candidate_until = time.monotonic() + delay
            if candidate_until > self._quota_blocked_until_ts:
                self._quota_blocked_until_ts = candidate_until

    def _quota_cooldown_remaining_seconds(self) -> float:
        with self._quota_lock:
            remaining = self._quota_blocked_until_ts - time.monotonic()
            if remaining <= 0:
                self._quota_blocked_until_ts = 0.0
                return 0.0
            return remaining

    def _build_limit_message(
        self,
        err: Exception | None = None,
        precomputed_retry_after: float | None = None,
    ) -> str:
        retry_after = (
            precomputed_retry_after
            if precomputed_retry_after is not None
            else self._extract_retry_after_seconds(err) if err is not None else 0.0
        )
        if retry_after <= 0:
            retry_after = self._quota_cooldown_remaining_seconds()

        retry_hint = ""
        if retry_after > 0:
            retry_hint = f" Please retry in about {int(math.ceil(retry_after))} seconds."

        if err is not None and self._is_quota_exhausted_error(err):
            return (
                "The AI service quota is exhausted for the current Gemini project."
                + retry_hint
                + " Try another API key/project or wait for quota reset."
            )

        if err is not None and self._is_throttled_error(err):
            return "The AI service is temporarily rate-limited." + retry_hint

        if retry_after > 0:
            return "The AI service is temporarily unavailable due to recent quota limits." + retry_hint

        return "The AI service is temporarily unavailable. Please try again shortly."

    def _parse_retry_after_seconds(self, value) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(str(value).strip())
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            return None
        return None

    def _init_gemini(self):
        try:
            self.gemini_client = genai.Client(api_key=self._api_token)
        except Exception as e:
            print(f"[AIGateway] Error initializing Gemini client: {e}")
            self.gemini_client = None

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
        return self.route_request(prompt)

    def route_request(self, prompt: str, lightweight: bool = True) -> str:
        """Routes a request to the single configured Gemini token."""
        _ = lightweight

        cooldown_remaining = self._quota_cooldown_remaining_seconds()
        if cooldown_remaining > 0:
            return self._build_limit_message(precomputed_retry_after=cooldown_remaining)

        if not self.gemini_client:
            return "System Error: Gemini client is not initialized."

        try:
            with self._request_lock:
                self._wait_for_request_slot()
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=["You are a specialized financial AI.", prompt]
                )
            return response.text
        except Exception as api_err:
            print(f"[AIGateway] Gemini request failed: {api_err}")
            if self._is_quota_exhausted_error(api_err):
                self._start_quota_cooldown(api_err)
            return self._build_limit_message(api_err)

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
        route_raw = self.route_request(router_prompt).strip()
        route = route_raw.upper()

        route_label = ""
        for candidate in ("MODE_A", "MODE_B", "MODE_C", "IRRELEVANT"):
            if candidate in route:
                route_label = candidate
                break

        TONING_RULE = (
            "CRITICAL GUARDRAILS:\n"
            "1. From what I can tell, if fundamental data is requested and not explicitly provided, "
            "do not explicitly state the lack of a dataset. Instead, assertively state that the platform "
            "only shows price data, and that users should not base their decision alone on this.\n"
            "2. Always format currency in thousands of VNĐ. For example, convert 22.3 to 22,300 VNĐ."
        )

        final_response = ""

        if not route_label:
            if any(tag in route for tag in ("429", "RESOURCE_EXHAUSTED", "TOO MANY REQUESTS", "QUOTA", "RATE-LIMIT", "RATE LIMIT")):
                final_response = route_raw
            else:
                final_response = "The AI router is temporarily unavailable. Please retry in a moment."

        elif route_label == "IRRELEVANT":
            final_response = "I'm sorry, but I can only assist you with stock-related questions."

        elif route_label == "MODE_A":
            final_response = self.query_pandasai(normalized_query, context, context_graph)

        elif route_label == "MODE_B":
            data_context = self.query_pandasai(normalized_query, context, context_graph)
            hybrid_prompt = self._build_price_only_analysis_prompt(
                user_query=normalized_query,
                elegant_payload=elegant_payload,
                data_context=data_context,
                mode=analysis_mode,
            )
            final_response = self.route_request(hybrid_prompt)

        else:
            general_prompt = (
                f"You are a helpful, assertive stock market assistant.\n"
                f"{elegant_payload}\n"
                f"CRITICAL GUARDRAIL: Never fabricate, invent, or estimate fundamental data.\n"
                f"{TONING_RULE}"
            )
            final_response = self.route_request(general_prompt)

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
            metadata={"route": route_label or route},
        )

        return final_response

    def _is_safe_symbol(self, symbol: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9._-]{1,10}", (symbol or "").upper()))

    def _extract_symbol_hint(self, query: str, context: str = "") -> str:
        text_chunks = [query or ""]
        if context:
            try:
                ctx_data = json.loads(context)
                text_chunks.append(ctx_data.get("details", ""))
                text_chunks.append(ctx_data.get("compare", ""))
            except Exception:
                pass

        haystack = " ".join(text_chunks).upper()
        explicit_patterns = (
            r"\b(?:STOCK|TICKER|SYMBOL)\s*[:=\-]?\s*([A-Z][A-Z0-9._-]{1,9})\b",
            r"\bFOR\s+([A-Z][A-Z0-9._-]{1,9})\b",
        )
        for pattern in explicit_patterns:
            match = re.search(pattern, haystack)
            if not match:
                continue
            candidate = match.group(1).upper()
            if self._is_safe_symbol(candidate):
                return candidate

        blacklist = {
            "MODE", "MODE_A", "MODE_B", "MODE_C", "RSI", "MACD", "ADX", "ATR",
            "EMA", "EMA20", "MA", "MA20", "MA50", "OHLCV", "VNĐ", "VND", "USD",
        }
        for token in re.findall(r"\b[A-Z][A-Z0-9._-]{1,9}\b", haystack):
            if token in blacklist:
                continue
            if self._is_safe_symbol(token):
                return token

        return ""

    def _format_thousand_vnd(self, value) -> str:
        try:
            number = float(value)
            if number != number:
                return "N/A"
            return f"{number * 1000:,.0f} VNĐ"
        except (TypeError, ValueError):
            return "N/A"

    def _format_decimal(self, value, decimals: int = 2) -> str:
        try:
            number = float(value)
            if number != number:
                return "N/A"
            return f"{number:,.{decimals}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _safe_stock_details_with_execute_sql_query(self, df_prices, df_metrics, symbol: str) -> str:
        safe_symbol = (symbol or "").strip().upper()
        if not self._is_safe_symbol(safe_symbol):
            return ""

        price_query = (
            "SELECT time, open, high, low, close, volume "
            "FROM stock_prices "
            f"WHERE symbol = '{safe_symbol}' "
            "ORDER BY time DESC "
            "LIMIT 1"
        )
        metrics_query = (
            "SELECT time, ma20, ma50, ema20, rsi, macd, atr, adx, bb_width "
            "FROM metrics "
            f"WHERE symbol = '{safe_symbol}' "
            "ORDER BY time DESC "
            "LIMIT 1"
        )

        try:
            price_df = df_prices.execute_sql_query(price_query)
        except Exception:
            return f"Could not retrieve details for stock '{safe_symbol}'."

        if price_df is None or getattr(price_df, "empty", True):
            return f"Could not retrieve details for stock '{safe_symbol}'."

        latest_price = price_df.iloc[0]
        details = [
            f"Stock details for {safe_symbol} ({latest_price.get('time', 'N/A')}):",
            f"- Open: {self._format_thousand_vnd(latest_price.get('open'))}",
            f"- High: {self._format_thousand_vnd(latest_price.get('high'))}",
            f"- Low: {self._format_thousand_vnd(latest_price.get('low'))}",
            f"- Close: {self._format_thousand_vnd(latest_price.get('close'))}",
            f"- Volume: {self._format_decimal(latest_price.get('volume'), 0)}",
        ]

        try:
            metrics_df = df_metrics.execute_sql_query(metrics_query)
            if metrics_df is not None and not getattr(metrics_df, "empty", True):
                latest_metrics = metrics_df.iloc[0]
                details.extend(
                    [
                        "- Indicators:",
                        f"  MA20: {self._format_thousand_vnd(latest_metrics.get('ma20'))}",
                        f"  MA50: {self._format_thousand_vnd(latest_metrics.get('ma50'))}",
                        f"  EMA20: {self._format_thousand_vnd(latest_metrics.get('ema20'))}",
                        f"  RSI: {self._format_decimal(latest_metrics.get('rsi'))}",
                        f"  MACD: {self._format_decimal(latest_metrics.get('macd'), 4)}",
                        f"  ATR: {self._format_decimal(latest_metrics.get('atr'), 4)}",
                        f"  ADX: {self._format_decimal(latest_metrics.get('adx'))}",
                        f"  BB Width: {self._format_decimal(latest_metrics.get('bb_width'), 4)}",
                    ]
                )
        except Exception:
            pass

        return "\n".join(details)

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

            semantic_org = "stock-data"

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

            symbol_hint = self._extract_symbol_hint(query, context)
            is_detail_lookup = bool(
                symbol_hint
                and re.search(r"\b(details?|snapshot|overview|latest|current|summary)\b", query, flags=re.IGNORECASE)
            )
            if is_detail_lookup:
                safe_details = self._safe_stock_details_with_execute_sql_query(df_prices, df_metrics, symbol_hint)
                if safe_details:
                    return safe_details

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

            context_injection += (
                "\n- CRITICAL FORMATTING: From what I can tell, convert any price or currency values to thousands of VNĐ by "
                "multiplying by 1000 and appending 'VNĐ' (e.g., 22.3 becomes 22,300 VNĐ).\n"
            )
            context_injection += (
                "- SQL SAFETY: If SQL is required, use execute_sql_query with a single SELECT statement only. "
                "Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, UNION, or multiple statements in one query.\n"
            )

            cooldown_remaining = self._quota_cooldown_remaining_seconds()
            if cooldown_remaining > 0:
                return self._build_limit_message(precomputed_retry_after=cooldown_remaining)

            try:
                llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=self._api_token)
                agent = Agent(
                    [df_companies, df_prices, df_metrics],
                    config={
                        "llm": llm,
                        "verbose": False,
                        "enable_cache": False
                    }
                )

                with self._request_lock:
                    self._wait_for_request_slot()
                    response = agent.chat(query + context_injection)
                return str(response)
            except Exception as agent_err:
                if self._is_quota_exhausted_error(agent_err):
                    self._start_quota_cooldown(agent_err)
                raise

        except Exception as e:
            print(f"[AIGateway] PandasAI Error: {e}")
            return self.route_request(
                f"The user asked: '{query}'. However, the internal dataset query failed. "
                f"Provide a helpful general answer. Assertively analyze the query from what I can tell and general market context, prioritizing insights that can be derived from the available context.\n"
                f"CRITICAL GUARDRAILS:\n"
                f"1. Do not explicitly state the lack of a dataset. Instead, assertively state that the platform only shows price data, and that users should not base their decision alone on this.\n"
                f"2. Convert currency to thousands of VNĐ (e.g., 22.3 to 22,300 VNĐ).",
            )


# Singleton instance
ai_gateway = AIGateway()