import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import (
    fetch_stock_data, fetch_stock_info, calculate_portfolio_metrics,
    format_currency, get_recent_news, detect_asset_class
)
from modules.universe import UNIVERSES, SAMPLE_SYMBOLS, MARKET_HELP
from modules.indicators import add_all_indicators
from modules.lstm_predictor import train_lstm_model, predict_lstm, DirectionPredictor
from modules.trading_agent import train_agent, RLTradingAgent
from modules.ollama_ai import (
    check_ollama_status, get_available_models, ollama_generate,
    generate_ai_summary, init_chat_history, add_message, clear_chat,
    get_chat_response, get_indicator_analysis_prompt
)
from modules.visualizations import (
    create_price_chart, create_price_with_ma, create_rsi_chart,
    create_macd_chart, create_lstm_prediction_chart,
    create_feature_importance_chart, create_portfolio_chart,
    create_trading_chart_with_signals, create_correlation_heatmap
)

st.set_page_config(
    page_title="GBLOXK OS AI STOCK TRADER",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 GBLOXK OS AI STOCK TRADER</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">Advanced AI-Powered Stock Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("GBLOXK OS Settings")
st.sidebar.caption("🌍 All markets: stocks, crypto, forex, indices, commodities")

market_preset = st.sidebar.selectbox(
    "Market Universe",
    ["Manual Symbol"] + list(UNIVERSES.keys()),
    help="Pick a preset universe for the Best-Investment scanner, or type any symbol manually.",
)
if market_preset != "Manual Symbol":
    preset_syms = UNIVERSES[market_preset]
    st.sidebar.caption(f"{len(preset_syms)} symbols: {', '.join(preset_syms[:6])}…")
    DEFAULT_SYMBOL = preset_syms[0]
else:
    DEFAULT_SYMBOL = "AAPL"
selected_stock = st.sidebar.text_input(
    "Stock / Crypto / Forex Symbol", value=DEFAULT_SYMBOL,
    help="Stocks: AAPL, TSLA · Crypto: BTC-USD, ETH-USD · Forex: EURUSD=X · Index: ^GSPC · Commodity: GC=F (gold), CL=F (oil)"
).upper().strip()
# Hardening: reject unsafe symbols instead of crashing / injecting
try:
    from modules.self_healing import validate_symbol
    if selected_stock:
        selected_stock = validate_symbol(selected_stock)
except ValueError as _ve:
    st.sidebar.error(str(_ve))
    st.stop()
with st.sidebar.expander("Symbol formats"):
    st.markdown(MARKET_HELP)

period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
selected_period = st.sidebar.selectbox("Time Period", list(period_options.keys()), index=4)
period_value = period_options[selected_period]

st.sidebar.markdown("---")
ollama_available = check_ollama_status()
if ollama_available:
    models = get_available_models()
    default_idx = 0
    for i, m in enumerate(models):
        if m == "qwen3.5:0.8b":
            default_idx = i
            break
    ai_model = st.sidebar.selectbox("AI Model (Ollama)", models, index=default_idx)
    st.sidebar.success("Ollama Connected")
else:
    ai_model = "qwen3.5:0.8b"
    st.sidebar.warning("Ollama not running. Start with `ollama serve`.")

with st.sidebar.expander("🧠 AI self-awareness & health"):
    from modules.self_learning import learning_stats
    st.caption("What this AI can/can't do *right now*:")
    st.write(f"Ollama LLM: {'🟢 online' if ollama_available else '🔴 offline (rule-based mode)'}")
    try:
        import tensorflow  # noqa: F401
        st.write("Deep learning (LSTM/RL): 🟢 installed")
    except ImportError:
        st.write("Deep learning (LSTM/RL): 🟡 TensorFlow missing — those tabs are SIMULATED/fallback only")
    _ls = learning_stats()
    st.write(f"📓 Self-learning journal: {_ls.get('message', '')}")
    st.caption("Self-healing: auto-retries, period fallback, anti-hallucination number checks — always on.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
- **LSTM Prediction**: Deep learning price forecasting
- **Random Forest**: Direction classification with technical indicators  
- **RL Trading Agent**: Deep Q-Learning automated trading simulation
- **AI Insights**: Powered by local Ollama LLMs
- **Data Source**: Yahoo Finance
""")

if not selected_stock:
    st.warning("Please enter a stock symbol.")
    st.stop()

# Fetch data
with st.spinner(f"Fetching data for {selected_stock}..."):
    df = fetch_stock_data(selected_stock, period=period_value)

if df.empty:
    st.error(f"No data found for {selected_stock}. Please check the symbol and try again.")
    st.stop()

df = add_all_indicators(df)

with st.spinner(f"Fetching stock info for {selected_stock}..."):
    stock_info = fetch_stock_info(selected_stock)

# LIVE-DATA badge — proves nothing below is fake/simulated
from datetime import datetime, timezone
try:
    _last_bar = df.index[-1]
    _last_str = str(_last_bar)
except Exception:
    _last_str = "unknown"
_asset = (stock_info or {}).get("asset_class") or detect_asset_class(selected_stock, stock_info or {})
st.caption(f"🟢 LIVE market data · {selected_stock} · {_asset} · last bar: {_last_str} · fetched {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Source: Yahoo Finance")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard", "💹 Technical Analysis", "🔮 LSTM Prediction",
    "🎯 Direction Forecast", "🤖 AI Trading Agent", "💬 AI Insights Chat",
    "🤝 Multi-Agent Analysis", "🔍 Stock Screener", "🏆 Best Investment (LIVE)"
])

with tab1:
    st.header("📊 Market Dashboard")

    metrics = calculate_portfolio_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    if metrics:
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Total Return", f"{metrics['total_return']*100:.2f}%",
                      delta_color="inverse")
        with col3:
            st.metric("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")
        with col4:
            st.metric("Volatility", f"{metrics['volatility']*100:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    if metrics:
        with col5:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col6:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        with col7:
            st.metric("Beta", f"{stock_info.get('beta', 'N/A')}")
        with col8:
            pe = stock_info.get('pe_ratio', 'N/A')
            st.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else pe)

    st.plotly_chart(create_price_chart(df, title=f"{selected_stock} Price Chart"), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"{selected_stock} - Key Info")
        if stock_info:
            st.write(f"**Name:** {stock_info.get('name', 'N/A')}")
            st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** {format_currency(stock_info.get('market_cap', 0))}")
            st.write(f"**Description:** {stock_info.get('description', 'N/A')[:200]}...")

    with col_b:
        if ollama_available and stock_info and metrics:
            st.subheader("AI Market Summary")
            if st.button("Generate AI Summary"):
                with st.spinner("Analyzing..."):
                    summary = generate_ai_summary(ai_model, selected_stock, stock_info, metrics)
                    if summary and not summary.startswith("Error"):
                        st.markdown(summary)
                    else:
                        st.info("AI summary unavailable. Check Ollama connection.")
        else:
            st.info("Connect to Ollama to generate AI-powered market summaries")

    if metrics:
        st.markdown("---")
        st.subheader("Performance Metrics")
        fig = create_price_with_ma(df, f"{selected_stock} Price & Moving Averages")
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.header("💹 Technical Analysis - " + selected_stock)

    indicator_cols = st.columns(4)
    with indicator_cols[0]:
        st.metric("RSI (14)", f"{df['RSI_14'].iloc[-1]:.2f}")
    with indicator_cols[1]:
        macd_val = df['MACD'].iloc[-1]
        signal_val = df['MACD_Signal'].iloc[-1]
        st.metric("MACD", f"{macd_val:.4f}", delta=f"{macd_val - signal_val:.4f}")
    with indicator_cols[2]:
        st.metric("BB Width", f"{df['BB_Width'].iloc[-1]:.4f}")
    with indicator_cols[3]:
        st.metric("CCI (20)", f"{df['CCI_20'].iloc[-1]:.2f}")

    st.plotly_chart(create_price_with_ma(df), width='stretch')

    col_ta1, col_ta2 = st.columns(2)
    with col_ta1:
        st.subheader("RSI - Overbought/Oversold")
        st.plotly_chart(create_rsi_chart(df), width='stretch')
    with col_ta2:
        st.subheader("MACD")
        st.plotly_chart(create_macd_chart(df), width='stretch')

    st.plotly_chart(create_correlation_heatmap(df, ['Close', 'SMA_20', 'SMA_50', 'RSI_14',
                                                      'MACD', 'MACD_Signal', 'BB_Width',
                                                      'CCI_20', 'Volume_SMA_20', 'Volume_Ratio']),
                      width='stretch')

    st.subheader("Recent Data")
    st.dataframe(df[['Close', 'Open', 'High', 'Low', 'Volume',
                      'SMA_20', 'SMA_50', 'RSI_14', 'MACD']].tail(20),
                 width='stretch')

with tab3:
    st.header("🔮 LSTM Price Prediction - " + selected_stock)
    st.caption("⚠️ SIMULATED estimate from past prices — NOT a real quote, NOT a guarantee. Training data is live; the forecast is a model guess.")

    st.markdown("""
    This tab uses an LSTM (Long Short-Term Memory) neural network to predict future stock prices.
    LSTM is a type of recurrent neural network well-suited for time series forecasting.
    """)

    col_pred, col_train = st.columns([3, 1])
    with col_pred:
        lookback = st.slider("Lookback Period (days)", min_value=30, max_value=120, value=60, step=10)
    with col_train:
        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=1)

    if st.button("Train & Predict LSTM Model"):
        with st.spinner(f"Training LSTM model for {selected_stock}... This may take a moment."):
            try:
                model, scaler = train_lstm_model(df, epochs=epochs, lookback=lookback, verbose=0)

                train_df, valid_df = predict_lstm(model, df, scaler, lookback=lookback)

                predictions_clean = valid_df['Predictions'].dropna()
                close_aligned = valid_df['Close'].loc[predictions_clean.index]
                rmse = np.sqrt(np.mean((predictions_clean - close_aligned) ** 2)) if len(predictions_clean) > 0 else 0

                st.plotly_chart(create_lstm_prediction_chart(train_df, valid_df), width='stretch')

                st.info(f"**RMSE:** ${rmse:.2f}")

                pred_df = valid_df[['Close', 'Predictions']].dropna()
                st.subheader("Prediction vs Actual (Last 30 Days)")
                st.dataframe(pred_df.tail(30).style.format("${:.2f}"), width='stretch')
            except ImportError:
                st.warning("TensorFlow not available. The LSTM model requires TensorFlow.")
                st.markdown("""
                **Install TensorFlow** to use this feature:
                ```
                pip install tensorflow
                ```
                Alternatively, the model can use the pre-trained model from the original project if available.
                """)
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    else:
        st.info("Click the button above to train the LSTM model and generate predictions.")

with tab4:
    st.header("🎯 Direction Forecast - " + selected_stock)

    st.markdown("""
    This tab uses a Random Forest classifier with technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
    to predict whether the stock price will increase or decrease in the next 2 days.
    """)

    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        use_grid = st.checkbox("Use GridSearchCV for hyperparameter tuning", value=True)
    with col_rf2:
        test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20)

    if st.button("Train Random Forest Model"):
        with st.spinner("Training Random Forest classifier..."):
            try:
                rf = DirectionPredictor()
                results = rf.train(df, use_grid_search=use_grid)

                st.success(f"**Model trained successfully!**")
                st.markdown(f"- **Accuracy**: {results['accuracy']:.4f}")
                st.markdown(f"- **Best Parameters**: {results['best_params']}")
                st.markdown(f"- **Training samples**: {results['train_size']}")
                st.markdown(f"- **Test samples**: {results['test_size']}")

                prediction = rf.predict_today(df)
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric("Prediction", prediction['prediction'])
                with col_pred2:
                    st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
                with col_pred3:
                    st.metric("P(Up)", f"{prediction['probability_up']*100:.1f}%")

                st.plotly_chart(create_feature_importance_chart(results['feature_importance']),
                                width='stretch')

                st.subheader("Classification Report")
                st.code(results['classification_report'])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Click the button above to train the Random Forest model.")

with tab5:
    st.header("🤖 AI Trading Agent - " + selected_stock)
    st.caption("⚠️ SIMULATED paper-trading on historical prices — NO real orders, NO real money. For strategy research only.")

    st.markdown(f"""
    This tab uses a Deep Q-Learning (DQN) reinforcement learning agent to simulate trading
    on {selected_stock}. The agent learns to decide when to **BUY**, **SELL**, or **HOLD** based
    on price patterns. The model learns purely from price data - no indicators are used.
    """)

    col_rl1, col_rl2 = st.columns(2)
    with col_rl1:
        rl_window = st.slider("Window Size", min_value=5, max_value=20, value=10)
    with col_rl2:
        rl_episodes = st.slider("Training Episodes", min_value=1, max_value=20, value=5)

    initial_balance = st.number_input("Initial Balance ($)", min_value=1000.0, value=10000.0, step=1000.0)

    if st.button("Start AI Trading Agent"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Training RL trading agent..."):
            try:
                agent, trades, avg_profit, success_rate = train_agent(
                    selected_stock,
                    window_size=rl_window,
                    episodes=rl_episodes,
                    batch_size=32,
                    initial_balance=initial_balance,
                    progress_bar=progress_bar
                )

                progress_bar.empty()
                status_text.success("Training complete!")
                status_text = st.empty()

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Avg Profit", f"${avg_profit:.2f}")
                with col_m2:
                    st.metric("Success Rate", f"{success_rate*100:.1f}%")
                with col_m3:
                    st.metric("Total Trades", len([t for t in trades if t['action'] == 'SELL']))

                st.plotly_chart(create_trading_chart_with_signals(df, trades), width='stretch')

                if trades:
                    trade_df = pd.DataFrame(trades)
                    st.subheader("Trade History")
                    st.dataframe(trade_df, width='stretch')
            except ImportError:
                st.warning("TensorFlow not available. The RL agent will use scikit-learn fallback.")
                with st.spinner("Training with MLP fallback..."):
                    agent, trades, avg_profit, success_rate = train_agent(
                        selected_stock,
                        window_size=rl_window,
                        episodes=rl_episodes,
                        progress_bar=progress_bar
                    )
                    st.metric("Avg Profit", f"${avg_profit:.2f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    else:
        st.info("Click the button above to start the RL trading agent training simulation.")

    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Agent observes a state (price window + sigmoid differences)
    2. Decides: **Hold** (0), **Buy** (1), or **Sell** (2)
    3. Learns via Deep Q-Learning with epsilon-greedy exploration
    4. Rewards: Profit from successful sell operations
    """)

with tab6:
    st.header("💬 AI Insights Chat - " + selected_stock)

    init_chat_history()

    if not ollama_available:
        st.warning("⚠️ Ollama is not running. Start it with `ollama serve`.")

    st.markdown(f"Chat with an AI assistant about {selected_stock} stocks. Ask about price predictions, trading strategies, technical analysis, or market insights!")

    if st.button("Clear Chat", key="clear_chat"):
        clear_chat()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about stocks...")

    if prompt:
        add_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        if ollama_available:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in get_chat_response(ai_model, selected_stock, st.session_state.chat_history):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                add_message("assistant", full_response)
        else:
            with st.chat_message("assistant"):
                st.markdown("Sorry, Ollama is not running. Please start it with `ollama serve`.")

with tab7:
    st.header("🤝 Multi-Agent Analysis - " + selected_stock)

    st.markdown("""
    This advanced analysis uses a **multi-agent system** inspired by professional trading firms.
    Each agent analyzes the stock from a different perspective, then agents debate and reach a consensus.
    """)

    st.markdown("""
    **Agent Architecture (TradingAgents-inspired):**
    | Agent | Role |
    |-------|------|
    | Research Analyst | Fundamental analysis |
    | Bullish Analyst | Positive signals focus |
    | Bearish Analyst | Negative signals focus |
    | Technical Analyst | ML-based technical analysis |
    | Sentiment Analyst | News sentiment (LLM) |
    | Risk Manager | Portfolio risk assessment |
    | Portfolio Manager | Consensus & position sizing |
    """)

    if st.button("Run Multi-Agent Analysis"):
        with st.spinner("Running 7-agent analysis..."):
            try:
                from modules.multi_agent_trader import MultiAgentTradingSystem
                from modules.ollama_ai import ollama_generate

                mas = MultiAgentTradingSystem(model=ai_model)

                with st.spinner("Agents analyzing..."):
                    analysis = mas.analyze_stock(selected_stock, df, stock_info)

                st.success("Analysis complete!")

                rec = analysis['recommendation']
                conf = analysis['confidence']

                rec_colors = {
                    'STRONG_BUY': '🟢', 'BUY': '🟢',
                    'HOLD': '🟡',
                    'SELL': '🔴', 'STRONG_SELL': '🔴'
                }

                st.markdown(f"## 🎯 Consensus Recommendation: {rec_colors.get(rec, '')} {rec}")
                st.metric("Confidence", f"{conf*100:.1f}%", delta=None)

                consensus = analysis['consensus']
                if 'position_size' in consensus:
                    st.info(f"**Position Sizing:** {consensus['position_size']['reasoning']}")
                    col_ps1, col_ps2, col_ps3 = st.columns(3)
                    with col_ps1:
                        st.metric("Base Allocation %", f"{consensus['position_size']['base_allocation']:.1f}%")
                    with col_ps2:
                        st.metric("Optimal Allocation %", f"{consensus['position_size']['optimal_allocation']:.1f}%")
                    with col_ps3:
                        st.metric("Max Suggested %", f"{consensus['position_size']['max_suggested']:.1f}%")

                if 'expected_return' in consensus:
                    st.success(f"""
                    **Expected Returns:**
                    - Monthly: {consensus['expected_return']['monthly']*100:.2f}%
                    - Quarterly: {consensus['expected_return']['quarterly']*100:.2f}%
                    - Annual: {consensus['expected_return']['annual']*100:.2f}%
                    """)

                st.markdown("---")
                st.subheader("🤖 Individual Agent Votes")

                for decision in analysis['agent_decisions']:
                    with st.expander(f"{decision.agent_name} - {decision.recommendation} ({decision.confidence*100:.0f}%)"):
                        st.markdown(f"**Reasoning:** {decision.reasoning}")
                        if decision.supporting_evidence:
                            st.markdown(f"**Evidence:** {', '.join(decision.supporting_evidence[:5])}")

                st.markdown("---")
                st.subheader("📈 Technical Prediction Details")
                tp = analysis['technical_prediction']
                st.markdown(f"- **ML Direction Prediction:** {tp['direction']}")
                st.markdown(f"- **Probability Up:** {tp['prob_up']*100:.1f}%")
                st.markdown(f"- **Probability Down:** {tp['prob_down']*100:.1f}%")
                st.markdown(f"- **Model Accuracy:** {analysis['rf_accuracy']*100:.1f}%")

            except Exception as e:
                import traceback
                st.error(f"Error in multi-agent analysis: {str(e)}")
                st.text(traceback.format_exc())
    else:
        st.info("Click the button above to run the full multi-agent analysis with all 7 AI agents.")

with tab8:
    st.header("🔍 Stock Screener & Recommendations — LIVE DATA ONLY")
    st.caption("✅ Every price below is fetched live from Yahoo Finance at scan time. Nothing is simulated or cached as 'real'. Simulations (LSTM/RL/backtests) are labelled SIMULATED in their own tabs.")

    st.markdown("""
    Screen multiple stocks simultaneously and get AI-powered investment recommendations.
    The screener uses the multi-agent system to analyze each stock and rank them.
    """)

    default_symbols = "AAPL,MSFT,GOOGL,TSLA,AMZN,META,NVDA,JNJ,JPM,V"
    symbols_input = st.text_input("Stock Symbols (comma-separated)", value=default_symbols)
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    if st.button("Run Stock Screener"):
        if not symbols:
            st.warning("Please enter at least one stock symbol.")
        else:
            with st.spinner(f"Analyzing {len(symbols)} stocks with multi-agent system..."):
                try:
                    from modules.multi_agent_trader import MultiAgentTradingSystem

                    mas = MultiAgentTradingSystem(model=ai_model)
                    results_df = mas.screen_and_recommend(symbols)

                    st.dataframe(results_df, width='stretch')

                    buy_stocks = results_df[results_df['Recommendation'].isin(['STRONG_BUY', 'BUY'])]
                    if len(buy_stocks) > 0:
                        st.success(f"🟢 **Recommended Buys: {', '.join(buy_stocks['Symbol'].tolist())}**")
                        if ollama_available:
                            if st.button("Get AI Justification for Top Buy"):
                                top_stock = buy_stocks.iloc[0]['Symbol']
                                with st.spinner("Generating AI justification..."):
                                    prompt = f"Explain why {top_stock} stock is a good investment right now in 3-4 sentences."
                                    justification = ollama_generate(ai_model, prompt, temperature=0.4, max_tokens=256)
                                    st.markdown(justification)
                    else:
                        st.info("No strong buy recommendations from the multi-agent system.")

                except Exception as e:
                    import traceback
                    st.error(f"Error: {str(e)}")
                    st.text(traceback.format_exc())
    else:
        st.info("Enter stock symbols above and click **Run Stock Screener** to analyze.")

    st.markdown("---")
    st.subheader("Quick Sector Performance")
    sectors = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "Auto": ["TSLA", "F", "GM"],
        "Finance": ["JPM", "V", "MA", "PYPL"],
        "Healthcare": ["JNJ", "PFE", "MRK", "ABBV"],
        "Energy": ["XOM", "CVX", "COP", "NEE"]
    }
    selected_sector = st.selectbox("Compare Sector", list(sectors.keys()))
    if selected_sector:
        sector_symbols = sectors[selected_sector]
        sector_data = []
        for sym in sector_symbols:
            try:
                d = fetch_stock_data(sym, "1mo", "1d")
                if not d.empty:
                    ret = (d['Close'].iloc[-1] / d['Close'].iloc[0] - 1) * 100
                    sector_data.append({"Stock": sym, "1M Return (%)": f"{ret:.2f}%"})
            except Exception:
                pass

        if sector_data:
            sector_df = pd.DataFrame(sector_data)
            st.dataframe(sector_df, width='stretch')

    st.markdown("---")
    st.subheader("🌍 All-Market Presets (live)")
    st.caption("Crypto trades 24/7 · Forex trades Sun–Fri · Stocks/ETFs follow exchange hours. All quotes below are LIVE.")
    for _uname, _syms in {
        "₿ Crypto Top 10": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
        "💱 Forex Majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "📈 Global Indices": ["^GSPC", "^NDX", "^DJI"],
        "🛢️ Commodities": ["GC=F", "SI=F", "CL=F"],
    }.items():
        with st.expander(_uname):
            _rows = []
            for _s in _syms:
                try:
                    _d = fetch_stock_data(_s, "1mo", "1d")
                    if not _d.empty:
                        _ret = (float(_d['Close'].iloc[-1]) / float(_d['Close'].iloc[0]) - 1) * 100
                        _rows.append({"Symbol": _s, "Price": f"${float(_d['Close'].iloc[-1]):,.2f}", "1M Return (%)": f"{_ret:.2f}%"})
                except Exception:
                    pass
            if _rows:
                st.dataframe(pd.DataFrame(_rows), width="stretch")

with tab9:
    st.header("🏆 Best Investment Right Now — LIVE, Real Data Only")
    st.caption("✅ All prices/indicators/news fetched LIVE from Yahoo Finance at scan time. ❌ No fake or pre-baked prices. Simulations (LSTM/RL) are NEVER presented as real.")
    st.warning(
        "Honest truth: **no AI can guarantee doubling your money or a '100% best' investment.** "
        "Below is the highest-scoring opportunity *right now* from LIVE data, with confidence, risks, "
        "and exactly what was verified — so you decide with real facts."
    )

    st.markdown("### 1️⃣ Choose what to scan")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        scan_preset = st.selectbox("Universe preset", list(UNIVERSES.keys()), index=0)
        use_preset = st.checkbox(f"Scan full '{scan_preset}' preset", value=True)
    with c2:
        scan_period = st.selectbox("History for scoring", ["6mo", "1y", "2y"], index=1)
    with c3:
        max_syms = st.slider("Max symbols", 3, 30, 10)

    if use_preset:
        scan_symbols = UNIVERSES[scan_preset][:max_syms]
    else:
        _default = ",".join(UNIVERSES[scan_preset][:10])
        _txt = st.text_input("Symbols (comma-separated, any market)", value=_default,
                             help="Mix anything: AAPL, BTC-USD, ETH-USD, EURUSD=X, ^GSPC, GC=F, SPY")
        scan_symbols = [s.strip().upper() for s in _txt.split(",") if s.strip()][:max_syms]

    st.markdown(f"**Scanning {len(scan_symbols)} live symbols:** {', '.join(scan_symbols)}")

    if st.button("🔎 SCAN LIVE MARKETS — find the best investment now", type="primary"):
        from modules.super_ai import rank_universe, best_pick
        from datetime import datetime, timezone
        prog = st.progress(0, text="Starting live scan…")
        t0 = datetime.now(timezone.utc)
        ranked = rank_universe(scan_symbols, period=scan_period, model=ai_model, progress=prog)
        prog.empty()
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        st.caption(f"🟢 LIVE scan finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · {len(scan_symbols)} symbols · took {dt:.0f}s · Source: Yahoo Finance (real quotes)")

        show = ranked.drop(columns=["_detail"], errors="ignore") if ranked is not None else None
        if show is not None and not show.empty:
            st.subheader("📊 Live ranking (0-100 composite, 6 pillars)")
            st.dataframe(show, width="stretch")

        pick = best_pick(ranked) if ranked is not None else None
        if not pick:
            st.error("No live data returned for these symbols. Try fewer symbols or a different universe (e.g. Crypto Top 10).")
        else:
            rec, score, conf = pick["recommendation"], pick["score"], pick["confidence"]
            emoji = {"STRONG_BUY": "🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴"}.get(rec, "⚪")
            st.markdown(f"## {emoji} BEST RIGHT NOW: **{pick['symbol']}** — {rec} ({score}/100, {conf*100:.0f}% confidence)")
            st.caption(f"LIVE price ${pick['price']:.2f} · {(pick.get('info') or {}).get('name', pick['symbol'])} · {(pick.get('info') or {}).get('asset_class', '?')}")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Composite", f"{score}/100")
            with k2:
                st.metric("ML P(up)", f"{pick['ml'].get('p_up', 0.5)*100:.0f}%")
            with k3:
                st.metric("Volatility", f"{pick['metrics'].get('volatility', 0)*100:.0f}%")
            with k4:
                st.metric("Max DD", f"{pick['metrics'].get('max_drawdown', 0)*100:.0f}%")

            tp = pick["trade_plan"]
            st.info(f"**Live trade frame (ATR-based, risk-first):** {tp['direction']} · entry ~${tp['entry']:.2f} · stop ${tp['stop']:.2f} · target ${tp['target']:.2f} · size ≤{tp['position_pct']}% of portfolio. Doubling money needs ~+100%: position so a stop-out costs 1-2% max.")

            with st.expander("📐 Why this one — 6 pillar scores (live)", expanded=True):
                st.write({k: f"{v}/100" for k, v in pick["pillars"].items()})
                for pillar, notes in pick["pillar_notes"].items():
                    st.markdown(f"**{pillar}:** " + "; ".join(notes))

            with st.expander("🐂 Bull case vs 🐻 Bear case (live evidence)", expanded=True):
                st.markdown("**Bull:**")
                for b in pick["research"].get("bull_case", []):
                    st.markdown(f"- {b}")
                st.markdown("**Bear:**")
                for b in pick["research"].get("bear_case", []):
                    st.markdown(f"- {b}")

            with st.expander("✅ Verified live facts vs ❓ missing data"):
                st.markdown("**Verified:**")
                for v in pick["research"].get("verified_points", []):
                    st.markdown(f"- ✅ {v}")
                if pick["research"].get("missing_data"):
                    st.markdown("**Not confirmed — do not treat as fact:**")
                    for m in pick["research"].get("missing_data", []):
                        st.markdown(f"- ❓ {m}")

            with st.expander("📰 Live headlines used"):
                for n in pick["research"].get("news", [])[:8]:
                    st.markdown(f"- {n.get('title', '')} — _{n.get('provider', '')}_")
                if not pick["research"].get("news"):
                    st.caption("No headlines returned live — score relied on price + fundamentals.")

            llm = (pick["research"].get("llm_brief") or "").strip()
            if llm and not llm.startswith("Error"):
                with st.expander("🤖 Deep-research brief (Ollama, live inputs)"):
                    st.markdown(llm)

            st.error(
                "⚠️ **Doubling-money reality check (live math):** a 2× needs either a ~100% runner (typical only in "
                "high-volatility crypto/small-caps, with 50-80% drawdown risk) or years of compounding. "
                "This pick is the *best expected value now* — NOT a guarantee. Never invest money you can't afford to lose."
            )

            # ── Anti-hallucination grounding + self-learning journal ──
            from modules.anti_hallucination import verify_brief_numbers, grounded_verdict
            from modules.self_learning import log_verdict, resolve_journal
            _chk = verify_brief_numbers(llm, pick["price"], pick["research"].get("price_facts", {}))
            _g = grounded_verdict(pick["symbol"], rec, conf, pick["price"],
                                  pick["research"].get("verified_points", []),
                                  pick["research"].get("missing_data", []), _chk.get("flags", []))
            with st.expander("🛡️ Anti-hallucination report (grounded to LIVE data)"):
                st.write(f"Calibrated confidence: {_g['confidence']*100:.0f}% (raw {conf*100:.0f}%)")
                st.caption("Sources: " + " · ".join(_g["sources"]))
                if _g["hallucination_flags"]:
                    for _f in _g["hallucination_flags"]:
                        st.warning(f"⚠️ {_f}")
                else:
                    st.success("✅ No fabricated prices detected — numbers match live feed.")
                if _g["missing"]:
                    st.caption("❓ Not claimed as fact: " + "; ".join(_g["missing"][:4]))
                st.caption(_g["disclaimer"])
            try:
                log_verdict(pick["symbol"], rec, score, conf, pick["price"])
                resolve_journal(pick["symbol"], pick["price"])
                st.caption("📓 Verdict logged to self-learning journal — future scans measure whether this call was right.")
            except Exception:
                pass
    else:
        st.info("Pick a universe and press **SCAN LIVE MARKETS**. Every number shown after the scan is fetched live — no simulated prices, ever.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    GBLOXK OS AI STOCK TRADER | Copyright © 2026 — GBLOXK_OS LLC | Powered by Streamlit + Ollama |
    Prices/news: LIVE Yahoo Finance (timestamped in-app) · Forecasts/backtests: SIMULATED estimates, never quotes |
    <strong>Disclaimer:</strong> Educational tool. Not investment advice. No returns guaranteed — you can lose money.
</div>
""", unsafe_allow_html=True)

if ollama_available:
    def _ai_analysis_button():
        if st.button("Generate AI Analysis"):
            with st.spinner("Generating..."):
                prompt = get_indicator_analysis_prompt(
                    f"Stock: {selected_stock}\nRSI: {df['RSI_14'].iloc[-1]:.2f}\nMACD: {df['MACD'].iloc[-1]:.4f}",
                    selected_stock
                )
                result = ollama_generate(ai_model, prompt, temperature=0.6)
                if result and not result.startswith("Error"):
                    with st.expander(f"AI Analysis for {selected_stock}"):
                        st.markdown(result)
    st.markdown("---")
    with st.expander("Generate AI Technical Analysis"):
        _ai_analysis_button()