"""SUPER-MEGA AI investment engine.

Combines 6 scored pillars into one 0-100 composite per symbol:
  1. Technical structure  (25%) — trend, RSI, MACD, Bollinger, SMA stack
  2. ML direction         (20%) — RandomForest prob_up * accuracy gate
  3. Momentum/return      (15%) — 1mo/5d returns, distance from highs
  4. Fundamentals/value   (15%) — PE/PEG/beta/analyst target margin
  5. Risk (inverted)       (15%) — volatility, drawdown, VaR penalty
  6. Sentiment/research   (10%) — news presence + LLM brief tone

Outputs a ranked table + a single BEST pick with confidence, position size
(Kelly-lite), entry/stop/target, bull/bear case and a verification checklist.

HONESTY CONTRACT: no asset is ever "100% guaranteed". The engine reports the
highest-expected-value pick *with* calibrated confidence and explicit risks.
"""
import numpy as np
import pandas as pd

from modules.data_loader import fetch_stock_data, fetch_stock_info, calculate_portfolio_metrics
from modules.indicators import add_all_indicators
from modules.lstm_predictor import DirectionPredictor
from modules.deep_research import deep_research
from modules.portfolio_optimizer import kelly_criterion, value_at_risk

PILLAR_WEIGHTS = {
    "technical": 0.25, "ml": 0.20, "momentum": 0.15,
    "fundamental": 0.15, "risk": 0.15, "sentiment": 0.10,
}

RECOMMENDATION_BANDS = [
    (75, "STRONG_BUY"), (60, "BUY"), (45, "HOLD"), (30, "SELL"), (0, "STRONG_SELL"),
]


def _band(score: float) -> str:
    for thresh, name in RECOMMENDATION_BANDS:
        if score >= thresh:
            return name
    return "STRONG_SELL"


def score_technical(df: pd.DataFrame) -> tuple:
    """0-100 technical score + explanation list."""
    if df is None or df.empty:
        return 50.0, ["No data"]
    last = df.iloc[-1]
    score, notes = 50.0, []
    try:
        # Trend stack
        if last["Close"] > last["SMA_50"]:
            score += 10; notes.append("Above SMA50 (+10)")
        else:
            score -= 8; notes.append("Below SMA50 (-8)")
        if last["SMA_20"] > last["SMA_50"]:
            score += 7; notes.append("SMA20>SMA50 golden stack (+7)")
        else:
            score -= 5; notes.append("SMA20<SMA50 (-5)")
        # RSI
        rsi = float(last["RSI_14"])
        if rsi < 30:
            score += 8; notes.append(f"Oversold RSI {rsi:.0f} — bounce odds (+8)")
        elif rsi < 45:
            score += 4; notes.append(f"RSI {rsi:.0f} soft (+4)")
        elif rsi <= 65:
            score += 5; notes.append(f"RSI {rsi:.0f} healthy (+5)")
        elif rsi <= 75:
            score -= 3; notes.append(f"RSI {rsi:.0f} hot (-3)")
        else:
            score -= 10; notes.append(f"RSI {rsi:.0f} overbought (-10)")
        # MACD
        if float(last["MACD"]) > float(last["MACD_Signal"]):
            score += 8; notes.append("MACD bullish (+8)")
        else:
            score -= 7; notes.append("MACD bearish (-7)")
        # Bollinger position
        up, lo, cl = float(last["BB_Upper"]), float(last["BB_Lower"]), float(last["Close"])
        pos = (cl - lo) / (up - lo) if up != lo else 0.5
        if pos < 0.15:
            score += 5; notes.append("At lower Bollinger — stretched (+5)")
        elif pos > 0.9:
            score -= 5; notes.append("At upper Bollinger — stretched (-5)")
        # CCI extremes
        cci = float(last.get("CCI_20", 0))
        if cci < -100:
            score += 4; notes.append("CCI oversold (+4)")
        elif cci > 100:
            score -= 4; notes.append("CCI overbought (-4)")
    except Exception as e:
        notes.append(f"partial: {e}")
    return float(np.clip(score, 0, 100)), notes


def score_ml(df: pd.DataFrame) -> tuple:
    try:
        rf = DirectionPredictor()
        res = rf.train(df, use_grid_search=False)
        pred = rf.predict_today(df)
        acc = float(res.get("accuracy", 0.5))
        p_up = float(pred.get("probability_up", 0.5))
        # Gate: model must beat coin-flip to earn weight
        edge = (acc - 0.5) * 2  # -1..1 roughly
        edge = float(np.clip(edge, -1, 1))
        score = 50 + (p_up - 0.5) * 100 * (0.4 + 0.6 * max(edge, 0))
        score = float(np.clip(score, 0, 100))
        return score, [f"RF acc {acc*100:.1f}%", f"P(up) {p_up*100:.0f}%"], {"accuracy": acc, "p_up": p_up, "direction": pred.get("prediction", "?")}
    except Exception as e:
        return 50.0, [f"ML unavailable: {e}"], {"accuracy": 0.5, "p_up": 0.5, "direction": "?"}


def score_momentum(df: pd.DataFrame) -> tuple:
    if df is None or df.empty or len(df) < 6:
        return 50.0, ["thin history"]
    c = df["Close"].dropna()
    r5 = float(c.iloc[-1] / c.iloc[-6] - 1)
    r1mo = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else float(c.iloc[-1] / c.iloc[0] - 1)
    off_high = float(c.iloc[-1] / c.max() - 1)
    score = 50 + r1mo * 150 + r5 * 200 + off_high * 60
    notes = [f"5d {r5*100:+.1f}%", f"~1mo {r1mo*100:+.1f}%", f"off-high {off_high*100:.0f}%"]
    return float(np.clip(score, 0, 100)), notes


def score_fundamental(info: dict, facts: dict) -> tuple:
    info = info or {}
    score, notes = 50.0, []
    try:
        pe = info.get("pe_ratio")
        if isinstance(pe, (int, float)) and pe and pe > 0:
            if pe < 15:
                score += 10; notes.append(f"Cheap PE {pe:.1f} (+10)")
            elif pe < 25:
                score += 4; notes.append(f"Fair PE {pe:.1f} (+4)")
            elif pe < 40:
                score -= 4; notes.append(f"Rich PE {pe:.1f} (-4)")
            else:
                score -= 10; notes.append(f"Very rich PE {pe:.1f} (-10)")
        else:
            notes.append("No PE (crypto/forex/growth) — neutral")
        beta = info.get("beta")
        if isinstance(beta, (int, float)):
            if beta > 1.8:
                score -= 5; notes.append(f"High beta {beta:.2f} (-5)")
            elif beta < 0.8:
                score += 3; notes.append(f"Defensive beta {beta:.2f} (+3)")
        tgt = info.get("analyst_target")
        last = (facts or {}).get("last")
        if isinstance(tgt, (int, float)) and isinstance(last, (int, float)) and last > 0:
            margin = tgt / last - 1
            score += float(np.clip(margin * 80, -12, 12))
            notes.append(f"Analyst upside {margin*100:+.0f}%")
        if (info.get("asset_class")) in ("CRYPTO", "FOREX"):
            notes.append(f"{info.get('asset_class')} — scored on structure/macro, not earnings")
    except Exception as e:
        notes.append(f"partial: {e}")
    return float(np.clip(score, 0, 100)), notes


def score_risk(df: pd.DataFrame, metrics: dict) -> tuple:
    """Higher = safer. Penalises volatility, drawdown, tail risk."""
    if df is None or df.empty:
        return 50.0, ["no data"]
    metrics = metrics or {}
    vol = float(metrics.get("volatility", 0.3))
    dd = abs(float(metrics.get("max_drawdown", 0.2)))
    rets = df["Close"].pct_change().dropna()
    var = abs(float(value_at_risk(rets))) if len(rets) else 0.03
    score = 85 - vol * 70 - dd * 60 - var * 120
    notes = [f"vol {vol*100:.0f}%", f"maxDD {dd*100:.0f}%", f"VaR95 {var*100:.1f}%"]
    return float(np.clip(score, 0, 100)), notes


def score_sentiment(research: dict) -> tuple:
    if not research:
        return 50.0, ["no research"]
    n_news = len(research.get("news") or [])
    brief = research.get("llm_brief") or ""
    score = 50.0
    notes = []
    if n_news >= 5:
        score += 8; notes.append(f"{n_news} headlines (+8)")
    elif n_news >= 2:
        score += 4; notes.append(f"{n_news} headlines (+4)")
    else:
        score -= 4; notes.append("thin news flow (-4)")
    bl = brief.lower()
    if brief:
        pos = sum(w in bl for w in ("bullish", "strong_buy", "upgrade", "record", "growth", "breakout"))
        neg = sum(w in bl for w in ("bearish", "downgrade", "sell", "risk", "warning", "miss", "lawsuit"))
        score += (pos - neg) * 3
        notes.append(f"LLM tone {pos}➕/{neg}➖")
    return float(np.clip(score, 0, 100)), notes


def analyse_one(symbol: str, period: str = "1y", model: str = "qwen3.5:0.8b",
               df_preloaded=None, info_preloaded=None) -> dict:
    symbol = (symbol or "").upper().strip()
    df = df_preloaded if df_preloaded is not None else fetch_stock_data(symbol, period=period)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "No market data", "score": 0.0, "recommendation": "HOLD"}
    try:
        df = add_all_indicators(df)
    except Exception:
        pass
    if df.empty:
        return {"symbol": symbol, "error": "Not enough history for indicators", "score": 0.0, "recommendation": "HOLD"}
    info = info_preloaded if info_preloaded is not None else fetch_stock_info(symbol)
    metrics = calculate_portfolio_metrics(df)
    research = deep_research(symbol, info, df, model=model)

    t, t_notes = score_technical(df)
    m, m_notes, ml_detail = score_ml(df)
    mo, mo_notes = score_momentum(df)
    f, f_notes = score_fundamental(info, research.get("price_facts", {}))
    r, r_notes = score_risk(df, metrics)
    s, s_notes = score_sentiment(research)

    composite = (t * PILLAR_WEIGHTS["technical"] + m * PILLAR_WEIGHTS["ml"]
                 + mo * PILLAR_WEIGHTS["momentum"] + f * PILLAR_WEIGHTS["fundamental"]
                 + r * PILLAR_WEIGHTS["risk"] + s * PILLAR_WEIGHTS["sentiment"])
    composite = round(float(composite), 1)
    rec = _band(composite)
    conf = round(float(np.clip(0.35 + abs(composite - 50) / 90 + (ml_detail.get("accuracy", 0.5) - 0.5) * 0.4, 0.3, 0.92)), 2)

    # Trade plan: ATR-based stop/target, Kelly-lite size
    last = float(df["Close"].iloc[-1])
    atr = float(df["ATR_14"].iloc[-1]) if "ATR_14" in df.columns and pd.notna(df["ATR_14"].iloc[-1]) else last * 0.02
    direction_long = rec in ("STRONG_BUY", "BUY")
    stop = round(last - 2 * atr if direction_long else last + 2 * atr, 2)
    target = round(last + 3 * atr if direction_long else last - 3 * atr, 2)
    try:
        kelly = kelly_criterion(win_rate=ml_detail.get("accuracy", 0.55), avg_win=atr, avg_loss=atr)
    except Exception:
        kelly = 0.0
    size_pct = round(float(np.clip(kelly * 100 if rec in ("STRONG_BUY", "BUY") else (3.0 if rec == "HOLD" else 0.0), 0, 15)), 1)

    return {
        "symbol": symbol, "score": composite, "recommendation": rec, "confidence": conf,
        "pillars": {"technical": round(t, 1), "ml": round(m, 1), "momentum": round(mo, 1),
                    "fundamental": round(f, 1), "risk": round(r, 1), "sentiment": round(s, 1)},
        "pillar_notes": {"technical": t_notes, "ml": m_notes, "momentum": mo_notes,
                         "fundamental": f_notes, "risk": r_notes, "sentiment": s_notes},
        "ml": ml_detail, "metrics": metrics, "info": info,
        "research": research, "price": last,
        "trade_plan": {"direction": "LONG" if direction_long else ("SHORT/AVOID" if rec in ("SELL", "STRONG_SELL") else "NEUTRAL"),
                       "entry": round(last, 2), "stop": stop, "target": target,
                       "position_pct": size_pct},
    }


def rank_universe(symbols: list, period: str = "1y", model: str = "qwen3.5:0.8b", progress=None) -> pd.DataFrame:
    rows = []
    total = max(1, len(symbols))
    for i, sym in enumerate(symbols):
        if progress is not None:
            try:
                progress.progress((i) / total, text=f"Deep-scanning {sym} ({i+1}/{total})")
            except Exception:
                pass
        try:
            a = analyse_one(sym, period=period, model=model)
            if "error" in a and "score" in a and a.get("score", 0) == 0 and "price" not in a:
                rows.append({"Symbol": sym, "Score": 0, "Verdict": "DATA GAP",
                             "Conf": "—", "Price": "—", "P(Up)": "—", "Risk Vol": "—"})
                continue
            rows.append({
                "Symbol": sym,
                "Score": a["score"],
                "Verdict": a["recommendation"],
                "Conf": f"{a['confidence']*100:.0f}%",
                "Price": f"${a['price']:.2f}",
                "P(Up)": f"{a['ml'].get('p_up', 0.5)*100:.0f}%",
                "Risk Vol": f"{a['metrics'].get('volatility', 0)*100:.0f}%",
                "_detail": a,
            })
        except Exception as e:
            rows.append({"Symbol": sym, "Score": 0, "Verdict": "ERROR", "Conf": "—",
                         "Price": "—", "P(Up)": "—", "Risk Vol": "—", "_detail": {"symbol": sym, "error": str(e)[:120]}})
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def best_pick(ranked: pd.DataFrame):
    if ranked is None or ranked.empty:
        return None
    ok = ranked[~ranked["Verdict"].isin(["ERROR", "DATA GAP"])]
    if ok.empty:
        return None
    top = ok.iloc[0]
    return top.get("_detail", None)
