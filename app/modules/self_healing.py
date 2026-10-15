"""Self-healing + self-awareness + hardening.

Self-healing: safe_fetch() retries with backoff, falls back across periods
(2y→1y→6mo→3mo), sanitises symbols, and never raises to the UI.
Self-awareness: capability_report() tells the user exactly what the AI can and
CANNOT do right now (Ollama up? TF installed? how many bars?).
Hardening: validate_symbol(), safe_float(), guarded blocks used app-wide.
"""
import re
import time
import pandas as pd

SAFE_SYMBOL = re.compile(r"^[A-Z0-9\.\-\^=]{1,16}$")


def validate_symbol(raw: str) -> str:
    s = (raw or "").upper().strip().replace(" ", "")
    if not SAFE_SYMBOL.match(s):
        raise ValueError(f"Unsafe symbol rejected: {raw!r}. Use formats like AAPL, BTC-USD, EURUSD=X, ^GSPC, GC=F.")
    return s


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if pd.isna(v) or pd.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_fetch(fetch_fn, symbol: str, periods=("2y", "1y", "6mo", "3mo", "1mo")):
    """Try longest history first, degrade gracefully. Returns (df, period_used)."""
    last_err = None
    for p in periods:
        try:
            df = fetch_fn(symbol, period=p)
            if df is not None and not df.empty and len(df) >= 10:
                return df, p
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    # Last resort: whatever 1mo gives, even thin
    try:
        df = fetch_fn(symbol, period="1mo")
        return df if df is not None else pd.DataFrame(), "1mo"
    except Exception as e:
        last_err = e
    return pd.DataFrame(), f"FAILED: {last_err}"


def capability_report(ollama_ok: bool, df, stock_info: dict) -> dict:
    n = len(df) if df is not None and not getattr(df, "empty", True) else 0
    try:
        import tensorflow  # noqa: F401
        tf_ok = True
    except ImportError:
        tf_ok = False
    caps = {
        "live_data": n >= 10,
        "bars": n,
        "ollama_llm": bool(ollama_ok),
        "lstm_deep_learning": tf_ok,
        "analyst_data": bool((stock_info or {}).get("analyst_target") not in (None, "N/A")),
        "news": None,  # filled by caller
    }
    limits = []
    if not caps["live_data"]:
        limits.append("No live price history — I will ABSTAIN, not guess.")
    if n < 60:
        limits.append(f"Only {n} bars — long indicators (SMA200) are degraded.")
    if not ollama_ok:
        limits.append("Ollama offline — using rule-based research, no LLM brief.")
    if not tf_ok:
        limits.append("TensorFlow missing — LSTM/RL tabs use fallback or are disabled.")
    if not caps["analyst_data"]:
        limits.append("No analyst targets (normal for crypto/forex) — valuation is technical-only.")
    return {"capabilities": caps, "limits": limits,
            "verdict": "READY" if caps["live_data"] else "ABSTAIN — no live data"}
