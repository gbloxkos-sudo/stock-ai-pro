# GBLOXK OS AI STOCK TRADER — Complete User Guide

> Version 1.0 · © GBLOXK_OS LLC

A field-tested operator's manual for the GBLOXK OS AI Stock Trader — the one terminal that fuses live data, multi-agent consensus, self-healing fetch, and anti-hallucination engineering across every market.

---

## 1. What You See On Startup

| Area | What it is |
|---|---|
| **📊 Header** | `GBLOXK OS AI STOCK TRADER` — click to refocus tabs |
| **🌍 Sidebar top** | Market Universe selector + Manual Symbol input |
| **🤖 Model badge** | Green = Ollama online / Red = offline rule-only mode |
| **🧠 Health panel** | Live read on data bars, capabilities, self-learning stats |
| **9 tabs** | The working surfaces (see [Tabs](#tabs-12) below) |

### First-time setup checklist
- [ ] `pip install -r requirements.txt` succeeded
- [ ] Ollama installed? (`ollama serve`) → badge turns green
- [ ] Symbol accepted? (no red "unsafe symbol" error)
- [ ] LIVE timestamp visible under header (e.g. `last bar: 2026-10-15`)

---

## 2. Choosing What To Analyse

### Pick a preset universe
Dropdown "Market Universe" auto-selects 10 symbols and pre-fills the scanner.  
Use presets when you want a 30-second sector/macro pulse (e.g. `₿ Crypto Top 10`).

### Or type any symbol (mixed freely)
The input box accepts and sanitises:
- **Stocks:** `AAPL`, `TSLA`, `BRK-B`, `NVDA`
- **Crypto:** `BTC-USD`, `ETH-USD`, `SOL-USD`, `DOGE-USD`, `1000BONK-USD`
- **Forex:** `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `AUDJPY=X`
- **Indices:** `^GSPC`, `^NDX`, `^DJI`, `^FTSE`, `^GDAXI`, `^N225`, `^HSI`, `^ASX200`
- **Commodities / futures:** `GC=F` (gold), `CL=F` (oil), `SI=F` (silver), `NG=F`, `CORN=F`, `COPPER=F`, `ZC=F` (corn), `ZS=F` (soy)
- **ETFs:** `SPY`, `QQQ`, `IWM`, `GLD`, `ARKK`, `BTCC`

**Tip:** type `BTC-USD, EURUSD=X, ^GSPC, GC=F` → all 4 analysed together in Best-Investment scan.

---

## 3. Tabs

### Tab 1 — 📊 Dashboard
- 8-key summary: price, total/annualised return, volatility, Sharpe, max-DD, beta, PE
- Candlestick with volume (Plotly dark)
- AI market summary (if Ollama on) — press "Generate AI Summary"
- Key-info panel: sector, industry, market cap, logo

### Tab 2 — 💹 Technical Analysis
28 indicators live computed:
- Trend: SMA-20/50/200, EMA-12/26, BB width
- Momentum: RSI-2/7/14/21, CCI-20/50, ROC-10/20
- Volume: Vol-SMA-20, Volume ratio
- Volatility: ATR-14
- Correlation heatmap of all signals

### Tab 3 — 🔮 LSTM Prediction
> ⚠️ **SIMULATED estimate** — label clearly in the UI.
- Lookback slider (30–120 days)
- Train on click → RMSE vs actual overlayed
- If TensorFlow missing → graceful "install TF" message (no crash)

### Tab 4 — 🎯 Direction Forecast (next 2 days)
- Random Forest classifier, GridSearchCV tuning
- Accuracy + classification report + feature-importance bar chart
- Today's prediction: P(UP) / P(DOWN) / CONFIDENCE

### Tab 5 — 🤖 AI Trading Agent
> ⚠️ **SIMULATED paper-trading.**
- Deep Q-Learning (DQN) agent: BUY / SELL / HOLD
- Window size (5–20) + episodes (1–20)
- Trade history table + equity overlay
- Auto-falls back to MLPRegressor if TF absent

### Tab 6 — 💬 AI Insights Chat
Ollama-powered conversation on the selected asset:
- Type any question → streaming answer
- "Why is BTC dropping?" / "Support levels?" / "Tech earnings outlook?"
- "Clear Chat" resets history (kept in session until then)

### Tab 7 — 🤝 Multi-Agent Analysis
TradingAgents-inspired 7-agent debate:

| Agent role | Weight |
|---|---|
| Research Analyst (fundamentals) | 15 % |
| Bullish Analyst | 15 % |
| Bearish Analyst | 15 % |
| Technical Analyst (ML) | 30 % |
| Sentiment Analyst (news) | 15 % |
| Risk Manager | 20 % |
| **Portfolio Manager** (consensus) | — |

Result → consensus recommendation, position sizing (Kelly), expected returns.

### Tab 8 — 🔍 Stock Screener (old)
Rank a custom list (default 10 large-caps). Returns table: Score, Verdict, Conf, Price, P(up), Risk Vol.
→ **Prefer the new Best Investment tab (Tab 9) for full live detail.**

### Tab 9 — 🏆 Best Investment (LIVE)  *[core scanner]*
1. Choose universe (preset or freeform symbols, any mix)
2. Pick history (6 mo / 1 y / 2 y)
3. Press **"🔎 SCAN LIVE MARKETS"**
4. Watch 6-pillar progress bar update per symbol
5. Output:
   - Full ranked table
   - Single top pick with: score, confidence, trade frame, bull/bear cases
   - ✅ verified-live-points vs ❓ missing-data
   - 🛡️ anti-hallucination report (no fabricated prices)
   - 📓 journal log link (self-learning)

---

## 4. Reading A Verdict

Example output for `BTC-USD`:
```
🟢 STRONG_BUY  84/100  73 % confidence  LIVE $101,542
```

### 6-pillar breakdown (expandable)
Each pillar shows raw /100 + one-line evidence — verify before acting.

### Trade frame (ATR-based, risk-first)
```
LONG  entry $101,542  stop $99,107  target $103,977  size 6.2 %
```
- Stop = last − 2×ATR
- Target = last + 3×ATR  (≈ 1.5× R/R)
- Position % = Kelly-lite, soft-capped 15 %

### Reality check (always shown)
> Doubling money = ≈ +100 % — needs a 2× runner. Crypto/small-caps can; blue-chips rarely without multi-year holds.

---

## 5. Symbols That Return "DATA GAP"
Common causes:
- Ticker typo → fixed-input sanitiser rejects
- Delisting / bankruptcy → no data on Yahoo
- Forex pair inverted → try `JPY=X` instead of `USDJPY`

Fix: re-enter symbol or switch preset universe.

---

## 6. Ollama (LLM) Notes
- Offline mode = fully rule-based verdicts (slightly lower confidence)
- LLM brief degrades gracefully and is never the sole basis
- Temperature locked to research (0.3–0.7) to limit hallucination
- Every LLM price claim is post-checked vs live feed

**Recommended pulls** (if you want richer briefs):
```bash
ollama run deepseek-r1:8b-qwen-distilled
ollama run qwen3.5:0.8b
```

---

## 7. Self-Healing / Self-Awareness FAQ
| Symptom | Action |
|---|---|
| "No runtime found" in console | Harmless — Streamlit cache warming; ignore |
| Scan shows 0 bars | Try 1y period instead of Max |
| Symbol "rejected" | Use format from [Choosing symbols](#choosing-what-to-analyse) |
| TensorFlow import warning | Normal if TF not installed — fallback active |

---

## 8. Keyboard / Pro Tips
- Paste a comma list of 50 tickers → click scan → use browser "find" in results table
- `Ctrl + F` in any expander to jump to a symbol
- Results auto-cache 1 hour per symbol (yfinance rate-safe)
- Click column headers to re-sort the ranking table

---

## 9. Glossary
| Term | Meaning |
|---|---|
| LIVE | Fetched from Yahoo Finance at scan time |
| MODEL | Statistical estimate from RF / LSTM |
| SIMULATED | Backtested result, not a real trade |
| ABSTAIN | Engine withheld recommendation — trust it |
| Confidence | Calibrated via evidence-count shrink |
| Kelly-lite | Position sizing from Kelly criterion, capped |

---

## 10. Support / License
- Repo issues: <https://github.com/donfatzd/gblockx-os-ai-stock-trader/issues>
- © 2026 **GBLOXK_OS LLC**
- Educational license only — never trade the demo blindly
