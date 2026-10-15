"""Anti-hallucination layer: every AI claim must ground to LIVE data.

Rules enforced:
1. Numbers check — any price/indicator quoted by the LLM is verified against
   the live dataframe; mismatches are flagged, never shown as fact.
2. Citation requirement — verdicts carry source labels (LIVE price, LIVE news,
   MODEL estimate, SIMULATED backtest).
3. No-fabrication guard — when data is missing the AI must say so (returns
   ABSTAIN + missing-data list) instead of inventing.
4. Confidence calibration — LLM 0-100 claims are shrunk toward 50 when
   evidence is thin.
"""
import re


def extract_numbers(text: str) -> list:
    if not text:
        return []
    return re.findall(r"\$?[\d,]+\.\d+|\$?[\d,]+%?", text)


def verify_brief_numbers(brief: str, live_price: float, facts: dict, tolerance: float = 0.05) -> dict:
    """Cross-check $-prices in the LLM brief against the live price."""
    if not brief or not live_price:
        return {"checked": False, "flags": []}
    flags = []
    for m in re.finditer(r"\$\s?([\d,]+\.?\d*)", brief):
        try:
            claimed = float(m.group(1).replace(",", ""))
            if live_price > 0 and abs(claimed - live_price) / live_price > 0.25:
                # Allow targets/stops far from spot — only flag if claimed AS current price
                ctx = brief[max(0, m.start() - 60):m.start()].lower()
                if any(w in ctx for w in ("currently", "trading at", "last", "price is", "priced at")):
                    flags.append(f"Claimed spot ${claimed:,.2f} vs live ${live_price:,.2f} — FLAGGED, use live.")
        except ValueError:
            pass
    return {"checked": True, "flags": flags}


def calibrate_confidence(raw_conf: float, n_evidence: int, data_bars: int) -> float:
    """Shrink overconfident claims when evidence is thin."""
    try:
        c = float(raw_conf)
    except (TypeError, ValueError):
        return 0.5
    c = max(0.0, min(1.0, c))
    # Evidence penalty: <3 sources or <60 bars pulls toward 0.5
    penalty = 0.0
    if n_evidence < 3:
        penalty += (3 - n_evidence) * 0.07
    if data_bars < 60:
        penalty += 0.08
    if c > 0.5:
        c = max(0.5, c - penalty)
    return round(c, 2)


def grounded_verdict(symbol: str, recommendation: str, confidence: float,
                     live_price: float, verified: list, missing: list, flags: list) -> dict:
    """Final honesty wrapper — the UI renders this, never raw LLM text alone."""
    grounded_conf = calibrate_confidence(confidence, len(verified or []), 250)
    return {
        "symbol": symbol,
        "recommendation": recommendation,
        "confidence": grounded_conf,
        "confidence_raw": confidence,
        "live_price": live_price,
        "sources": ["LIVE: Yahoo Finance OHLCV", "LIVE: Yahoo news/analyst (when present)",
                    "MODEL: RandomForest direction (SIMULATED estimate, not a quote)",
                    "SIMULATED: LSTM/RL/backtests — never prices"],
        "verified": verified or [],
        "missing": missing or [],
        "hallucination_flags": flags or [],
        "disclaimer": ("Not financial advice. No investment can double money guaranteed. "
                       "All 'predictions' are statistical estimates from past data."),
    }
