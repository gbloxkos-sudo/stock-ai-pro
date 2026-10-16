# GBLOXK OS AI STOCK TRADER — Super Advanced Edition

> AI-powered market terminal fusing live data, multi-agent consensus, deep learning, and anti-hallucination engineering across **all markets**: stocks, crypto, forex, indices, commodities, ETFs, and global ADRs.

⚠️ **Honesty contract:** No AI can *guarantee* doubling your money or a "100 % best" investment. This tool gives you the highest-expected-value pick **from live facts**, with calibrated confidence, verified vs missing data, and explicit risks. Never invest money you cannot afford to lose.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Install / Setup](#install--setup)
3. [User Guide](#user-guide)
4. [All Markets Covered](#all-markets-covered)
5. [Architecture](#architecture)
6. [Self-Healing / Anti-Hallucination](#self-healing--anti-hallucination)
7. [Self-Learning](#self-learning)
8. [Deploy to Streamlit Sharing / GitHub](#deploy)
9. [FAQ](#faq)
10. [License & Credits](#license--credits)

---

## Quick Start

```bash
git clone https://github.com/donfatzd/gblockx-os-ai-stock-trader.git
cd gblockx-os-ai-stock-trader/app
pip install -r requirements.txt
./run.sh                      # launches on http://localhost:8501
```

Optional — deep-learning tabs (LSTM, RL):
```bash
pip install tensorflow       # app self-heals & runs fallback without it
```

The AI runs **without Ollama** (rule-based mode) or **with** (full LLM research brief):
```bash
ollama serve &                # terminal 1
ollama pull qwen3.5:0.8b      # terminal 2 (only needed once)
streamlit run app.py
```

---

## Install & Setup

### Prerequisites
- **Python 3.10+**
- **Ollama** *(optional but recommended for deep-research briefs)* — <https://ollama.com/download>

### Steps
1. Clone
2. Create env (optional but recommended):
   ```bash
   python -m venv venv && source venv/bin/activate
   ```
3. Install:
   ```bash
   pip install -r requirements.txt
   ```
4. Pull model (recommended):
   ```bash
   ollama pull qwen3.5:0.8b
   ollama serve
   ```
5. Run:
   ```bash
   streamlit run app.py
   # or  ./run.sh
   ```
6. Browser: <http://localhost:8501>

### Offline-capable?
- ✅ Price data (Yahoo Finance)
- ✅ All charts/technicals/scoring
- ✅ Multi-agent consensus (rule-based fallbacks)
- ✅ Risk/position sizing math
- 🔴 LLM deep-dive briefs (require Ollama running locally)

---

## User Guide

### 1. Sidebar — Market Selection (all markets)

| Control | How |
|---|---|
| 🌍 **Market Universe** | Pick a preset (US Mega-Caps, Crypto Top 10, Forex Majors, Global Indices, Commodities…) for the scanner, or **Manual Symbol**. |
| 🔤 **Symbol** | Any yfinance symbol, mixed freely: `AAPL`, `BTC-USD`, `ETH-USD`, `EURUSD=X`, `^GSPC`, `GC=F`, `SPY` |
| 📅 **Time Period** | 1 mo → Max (controls main chart + indicators) |
| 🤖 **AI Model** | Auto-switches if offline → shows warning |

### 2. Tabs

| # | Tab | What you get |
|---|---|---|
| 1 | 📊 **Dashboard** | Snapshot: live metrics table, candlestick, portfolio stats, AI summary |
| 2 | 💹 **Technical Analysis** | 28+ indicators — RSI, MACD, Bollinger, CCI, ATR, ROC, volatility, volumes; correlation heatmap |
| 3 | 🔮 **LSTM Prediction** | TensorFlow LSTM forecast vs actual (⚠️ SIMULATED estimate) |
| 4 | 🎯 **Direction Forecast** | 2-day up/down classifier (Random Forest + GridSearchCV) |
| 5 | 🤖 **AI Trading Agent** | Deep Q-Learning simulator (⚠️ SIMULATED paper-trading) |
| 6 | 💬 **AI Insights Chat** | Live Ollama-powered Q&A on the selected asset |
| 7 | 🤝 **Multi-Agent Analysis** | 7-agent *TradingAgents*-style debate → consensus verdict |
| 8 | 🔍 **Stock Screener** | Rank any list of symbols |
| 9 | 🏆 **Best Investment (LIVE)** | **Core scanner:** live 6-pillar scoring of a full universe → single top pick |

### 3. How the Best-Investment score works

Each symbol is scored **0–100** across 6 independently verified pillars:

| Pillar | Weight | Inputs |
|---|---|---|
| 🟦 **Technical** | 25 % | Trend stack, RSI, MACD, Bollinger position, CCI |
| 🟩 **ML Direction** | 20 % | RF model accuracy-gated probability of upside |
| 🟨 **Momentum** | 15 % | 5-day / ~1-month returns, distance from 52w high |
| 🟥 **Fundamental** | 15 % | PE, forward PE, beta, analyst target margin |
| 🟪 **Risk** | 15 % | Volatility, max drawdown, VaR(95%) — inverted |
| 🟧 **Sentiment** | 10 % | Live headlines + LLM tone (bull/bear word count) |

Final recommendation band:
```
75+ STRONG_BUY   60+ BUY   45 HOLD   30 SELL   0 STRONG_SELL
```

Confidence is **calibrated**: raw LLM/odds claims shrink toward 50 % when evidence sources are thin (`anti_hallucination.py`).

### 4. Trade plan (live)
For every verdict, the app outputs an **ATR-based frame**:
- LONG or SHORT direction
- Entry (~last live price)
- Stop = 2×ATR
- Target = 3×ATR (≈ 1.5× risk/reward)
- Position % via Kelly-lite, capped to 15 %

> Doubling money mathematically requires ≈ +100 % — only high-volatility assets (small-caps, crypto) reach that, with commensurate drawdown risk.

---

## All Markets Covered

| Category | Symbols | Example |
|---|---|---|
| 🇺🇸 US Mega-Caps | 10 | `AAPL`, `MSFT`, `NVDA` |
| 💻 Tech Leaders | 10 | `GOOGL`, `AVGO`, `AMD` |
| ₿ **Crypto Top 10** | 10 | `BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD` |
| ₿ Crypto DeFi/Alts | 8 | `MATIC-USD`, `UNI-USD` |
| 💱 **Forex Majors** | 7 | `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` |
| 📈 **Global Indices** | 6 | `^GSPC`, `^NDX`, `^DJI`, `^FTSE`, `^N225` |
| 🛢️ Commodities | 4 | `GC=F` (gold), `CL=F` (oil), `SI=F` (silver), `NG=F` |
| 🏦 Finance/Healthcare | 10 | `JPM`, `JNJ`, `UNH` |
| ⚡ Energy/Utilities | 6 | `XOM`, `CVX`, `NEE` |
| 🌍 Global ADRs | 8 | `TSM`, `ASML`, `BABA` |
| 📦 ETFs | 10 | `SPY`, `QQQ`, `GLD`, `TLT` |

Crypto markets trade 24/7 — data is always fresh.

---

## Architecture

```
app/
├── app.py                      # Streamlit front-end (9 tabs)
├── run.sh                      # Launcher
├── requirements.txt
├── .streamlit/config.toml
├── assets/                      # (CSS/icons)
├── models/                      # LSTM weights, verdict_journal.csv (self-learning)
└── modules/
    ├── data_loader.py           # LIVE fetch (stocks/crypto/forex/index/commodity) + portfolio metrics
    ├── indicators.py            # 28+ indicators, adaptive windows (no crash on short history)
    ├── lstm_predictor.py        # LSTM + RandomForest direction model
    ├── trading_agent.py         # Deep Q-Learning RL agent (DQN) w/ sklearn fallback
    ├── multi_agent_trader.py    # 7-agent consensus system
    ├── ollama_ai.py             # Local LLM API + deep-dive prompts
    ├── deep_research.py         # News + analyst + price facts + LLM brief
    ├── portfolio_optimizer.py   # Efficient frontier, Kelly, VaR, Sharpe/Sortino/Calmar
    ├── anti_hallucination.py    # Number grounding, confidence shrink, honesty wrapper
    ├── self_healing.py          # Symbol validation, period fallback, capability report
    ├── self_learning.py         # Verdict journal, hit-rate tracking (improves over time)
    ├── universe.py              # All market presets
    └── visualizations.py        # Plotly charts (Plotly Dark, 2013+ style)
```

### Self-healing & anti-hallucination
- Any bad ticker → auto-tries 2y→1y→…→1mo, sanitises input, never crashes UI
- All AI claims cross-checked: every price the LLM quotes is verified vs live feed
- Verdicts always carry source tags: `LIVE price` · `LIVE news` · `MODEL estimate (simulated)` · `SIMULATED backtest`
- Thin-evidence → confidence pulled toward 0.5, verdict may switch to **ABSTAIN**

### Self-learning
`models/verdict_journal.csv` logs every top-pick call; 5-session later outcomes update hit-rate shown in the sidebar.

---

## Deploy

### Streamlit Community Cloud
1. Push this folder to a GitHub repo
2. <https://streamlit.io/deploy> → select repo, branch `main`, file path `app/app.py`
3. Done — auto-rebuilds on each push

### Docker
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run","app.py"]
```

### Local HTTPS (for mobile)
```bash
streamlit run app.py --server.enableCORS false --server.headless true
```

---

## FAQ

**Is this investment advice?** No. Educational research tool only.  
**Can it double my money?** It finds the best *expected value* — not a guarantee. Doubling needs ~+100% moves carrying high drawdown risk.  
**Do I need Ollama?** No — full scoring works offline. Ollama only powers the LLM research briefs.  
**Do I need TensorFlow?** No — LSTM/RL tabs auto-fallback to scikit-learn or disable gracefully.  
**Why is a number in the scan flagged?** Anti-hallucination check found an LLM-quoted price ≠ live. Hover the ⚠️ to see delta.  
**Will it learn?** Every verdict is written to `models/verdict_journal.csv`; accuracy is tracked in the sidebar health panel.

---

## License & Credits

- © 2026 — **GBLOXK_OS LLC**
- Built with Streamlit, pandas, scikit-learn, Plotly, yfinance
- Multi-agent architecture inspired by the [TradingAgents paper](https://arxiv.org/abs/2412.20138)
- Data source: Yahoo Finance (live market data only)
- Distributed under the repo’s license for educational use only.

*Never substitute this tool for professional financial advice. Markets are risky — you can and will lose money.*
