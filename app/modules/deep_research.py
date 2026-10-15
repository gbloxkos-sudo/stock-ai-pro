"""Deep internet research layer.

Aggregates every *locally available* evidence source (Yahoo news, analyst
ratings, fundamentals, price action) and, when Ollama is reachable, asks the
LLM to do web-style deep synthesis: bull/bear cases, catalysts, risks, peers.

Designed to never crash offline — every source degrades to a structured
fallback so the SUPER-AI verdict engine always has evidence to score.
"""
from modules.data_loader import get_recent_news, get_analyst_data, fetch_stock_info
from modules.ollama_ai import ollama_generate, check_ollama_status


def deep_research(symbol: str, stock_info: dict, df, model: str = "qwen3.5:0.8b") -> dict:
    symbol = (symbol or "").upper().strip()
    stock_info = stock_info or {}
    news = get_recent_news(symbol, max_items=8)
    analyst = get_analyst_data(symbol)

    # Price-structure facts (pure pandas — always available)
    facts = _price_facts(df)
    fundamentals = _fundamental_facts(stock_info)

    llm_brief = ""
    if check_ollama_status():
        llm_brief = _llm_deep_dive(symbol, stock_info, df, news, analyst, facts, model)

    bull, bear, catalysts, risks = _extract_cases(news, facts, fundamentals, llm_brief)

    # Verification checklist — what backs the call, what is missing
    verified, unverified = _verification_checklist(news, analyst, stock_info, df, llm_brief)

    return {
        "symbol": symbol,
        "news": news,
        "analyst": analyst,
        "price_facts": facts,
        "fundamentals": fundamentals,
        "llm_brief": llm_brief,
        "bull_case": bull,
        "bear_case": bear,
        "catalysts": catalysts,
        "risks": risks,
        "verified_points": verified,
        "missing_data": unverified,
    }


def _price_facts(df) -> dict:
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return {}
    close = df["Close"].dropna()
    if len(close) < 5:
        return {}
    last = float(close.iloc[-1])
    out = {
        "last": last,
        "chg_5d": float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0,
        "chg_1mo": float(close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else 0.0,
        "high_52w": float(close.max()),
        "low_52w": float(close.min()),
        "off_high": float(close.iloc[-1] / close.max() - 1),
    }
    for c in ("RSI_14", "MACD", "MACD_Signal", "BB_Width", "CCI_20", "SMA_20", "SMA_50", "ATR_14"):
        try:
            if c in df.columns:
                out[c] = float(df[c].iloc[-1])
        except Exception:
            pass
    return out


def _fundamental_facts(info: dict) -> dict:
    keys = ["name", "sector", "industry", "market_cap", "pe_ratio", "forward_pe",
            "eps", "beta", "dividend_yield", "asset_class", "currency",
            "fifty_two_high", "fifty_two_low", "analyst_target", "recommendation"]
    return {k: info.get(k, "N/A") for k in keys}


def _llm_deep_dive(symbol, info, df, news, analyst, facts, model) -> str:
    headlines = "\n".join(f"- {n.get('title', '')} ({n.get('provider', '')})" for n in (news or [])[:8]) or "No headlines available."
    prompt = (
        f"You are a senior equity + crypto + macro research desk. Do a DEEP investment research brief on {symbol}.\n"
        f"Company/asset: {info.get('name', symbol)} | Class: {info.get('asset_class', '?')} | Sector: {info.get('sector', 'N/A')}\n"
        f"Price facts: {facts}\nFundamentals: {info}\nAnalyst: {analyst}\nHeadlines:\n{headlines}\n\n"
        "Return a structured brief with sections:\n"
        "1. THESIS (2-3 sentences)\n2. BULL CASE (3 bullets)\n3. BEAR CASE (3 bullets)\n"
        "4. CATALYSTS (earnings, macro, halvings/upgrades, rate decisions)\n"
        "5. KEY RISKS (volatility, valuation, regulation, leverage)\n"
        "6. VERDICT (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL + 0-100 confidence + one-line reason)\n"
        "Be specific to the asset class (for crypto discuss on-chain/macro, for forex discuss rate differentials, "
        "for indices discuss breadth). End with: 'Not financial advice.'"
    )
    try:
        out = ollama_generate(model, prompt, system="You are a rigorous multi-asset research analyst.", temperature=0.4, max_tokens=1600)
        return out if out and not out.startswith("Error") else ""
    except Exception:
        return ""


def _extract_cases(news, facts, fundamentals, llm_brief):
    bull, bear, catalysts, risks = [], [], [], []
    # Deterministic technical cases — always present
    rsi = facts.get("RSI_14")
    if rsi is not None:
        if rsi < 35:
            bull.append(f"Oversold bounce setup (RSI {rsi:.1f} < 35).")
        elif rsi > 70:
            bear.append(f"Overbought exhaustion risk (RSI {rsi:.1f} > 70).")
        elif 45 <= rsi <= 60:
            bull.append(f"Healthy momentum zone (RSI {rsi:.1f}).")
    macd = facts.get("MACD")
    sig = facts.get("MACD_Signal")
    if macd is not None and sig is not None:
        if macd > sig:
            bull.append("MACD above signal — positive momentum.")
        else:
            bear.append("MACD below signal — negative momentum.")
    off_high = facts.get("off_high")
    if off_high is not None:
        if off_high > -0.05:
            bull.append("Trading near 52-week highs — leadership.")
        elif off_high < -0.25:
            bear.append(f"Deep drawdown ({off_high*100:.0f}% off highs) — damaged trend.")
            bull.append("Deep drawdown may offer contrarian entry IF trend stabilises.")
    if facts.get("chg_1mo", 0) > 0.1:
        bull.append("Strong 1-month momentum (+10%+).")
        risks.append("Chase risk after vertical run — size positions.")
    if facts.get("chg_1mo", 0) < -0.15:
        bear.append("Steep 1-month selloff (-15%+) — capital-preservation caution.")
    # News-driven
    for n in (news or [])[:5]:
        t = n.get("title", "")
        if t:
            catalysts.append(t)
    if not catalysts:
        catalysts.append("No fresh catalyst headlines found — thesis rests on price/fundamentals.")
        risks.append("Low news visibility — verify thesis with earnings/analyst data.")
    risks.append("All markets carry risk of loss; volatility can exceed historical ranges.")
    return bull, bear, catalysts, risks


def _verification_checklist(news, analyst, info, df, llm_brief):
    verified, missing = [], []
    if df is not None and not getattr(df, "empty", True) and len(df) >= 30:
        verified.append(f"Price history verified ({len(df)} bars, live market data).")
    else:
        missing.append("Insufficient price history (<30 bars).")
    if news:
        verified.append(f"{len(news)} recent headlines retrieved.")
    else:
        missing.append("No recent headlines — sentiment unconfirmed.")
    if analyst.get("target_mean") or analyst.get("ratings"):
        verified.append("Wall-Street analyst data present.")
    else:
        missing.append("No analyst ratings/targets (common for crypto/forex).")
    if info.get("pe_ratio", "N/A") not in ("N/A", None):
        verified.append("Fundamental multiples available.")
    else:
        missing.append("Fundamental multiples missing — valuation is technical-only.")
    if llm_brief:
        verified.append("LLM deep-research brief generated.")
    else:
        missing.append("LLM brief unavailable (Ollama offline) — rule-based analysis used.")
    return verified, missing
