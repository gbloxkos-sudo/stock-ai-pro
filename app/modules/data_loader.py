"""Universal market data loader: stocks, crypto, forex, indices, commodities, global markets.

Supports any yfinance-compatible symbol:
  Stocks:      AAPL, MSFT, TSLA, NVDA
  Crypto:      BTC-USD, ETH-USD, SOL-USD, DOGE-USD
  Forex:       EURUSD=X, GBPUSD=X, JPY=X, DX-Y.NYB
  Indices:     ^GSPC, ^NDX, ^DJI, ^FTSE, ^N225
  Commodities: GC=F (gold), SI=F (silver), CL=F (oil), NG=F
  ETFs:        SPY, QQQ, ARKK
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly flatten yfinance OHLCV frames (single or multi-ticker)."""
    if df is None or df.empty:
        return pd.DataFrame()
    # Newer yfinance: MultiIndex columns (Price x Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Deduplicate columns (can happen with duplicate levels)
    df = df.loc[:, ~df.columns.duplicated()]
    # Keep only what we need, case-insensitive
    colmap = {c.lower(): c for c in df.columns}
    wanted = ["open", "high", "low", "close", "volume"]
    out = pd.DataFrame(index=df.index)
    for w in wanted:
        if w in colmap:
            out[w.capitalize()] = pd.to_numeric(df[colmap[w]], errors="coerce")
        elif w == "volume":
            out["Volume"] = 0.0  # forex/crypto-spot may lack volume
    # Drop rows with no price
    if "Close" in out.columns:
        out = out.dropna(subset=["Close"])
    return out


@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return pd.DataFrame()
    # Crypto trades 24/7 — yfinance fine with same call, but prefer longer history
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=True)
        df = _flatten_ohlcv(df)
        if not df.empty and set(["Open", "High", "Low", "Close"]).issubset(df.columns):
            return df
    except Exception:
        pass
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, auto_adjust=False)
        hist = _flatten_ohlcv(hist)
        if not hist.empty:
            return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    st.error(f"No data found for {symbol}. Check the symbol (e.g. AAPL, BTC-USD, EURUSD=X, ^GSPC, GC=F).")
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def fetch_stock_info(symbol: str) -> dict:
    symbol = (symbol or "").upper().strip()
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        fast = {}
        try:
            fast = {"last_price": ticker.fast_info.get("lastPrice") if hasattr(ticker, "fast_info") else None}
        except Exception:
            pass
        asset = detect_asset_class(symbol, info)
        return {
            'name': info.get('longName') or info.get('shortName') or symbol,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'description': info.get('description', 'N/A'),
            'logo_url': info.get('logo_url', ''),
            'asset_class': asset,
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', info.get('fullExchangeName', 'N/A')),
            'fifty_two_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'analyst_target': info.get('targetMeanPrice', 'N/A'),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'last_price': fast.get('last_price'),
        }
    except Exception:
        return {'name': symbol, 'asset_class': detect_asset_class(symbol, {})}


def detect_asset_class(symbol: str, info: dict) -> str:
    s = (symbol or "").upper()
    quote_type = (info.get('quoteType') or '').upper()
    if quote_type == 'CRYPTOCURRENCY' or (s.endswith('-USD') and len(s) <= 12):
        return 'CRYPTO'
    if quote_type == 'CURRENCY' or s.endswith('=X'):
        return 'FOREX'
    if s.startswith('^') or quote_type == 'INDEX':
        return 'INDEX'
    if s.endswith('=F') or quote_type in ('FUTURE', 'COMMODITY'):
        return 'COMMODITY'
    if quote_type == 'ETF':
        return 'ETF'
    return 'STOCK'


@st.cache_data(ttl=3600)
def fetch_multiple_stocks(symbols: list, period: str = "1y") -> dict:
    result = {}
    for symbol in symbols:
        result[symbol] = fetch_stock_data(symbol, period=period)
    return result


def calculate_portfolio_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty or 'Close' not in df.columns:
        return {}
    close_col = df['Close'].dropna()
    if len(close_col) < 2:
        return {}
    returns = close_col.pct_change().dropna()
    total_return = float(close_col.iloc[-1] / close_col.iloc[0] - 1)
    n = len(close_col)
    annualized_return = float((1 + total_return) ** (252 / n) - 1) if n > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    # Sharpe vs 2% risk-free
    excess = returns - 0.02 / 252
    sharpe_ratio = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() not in (0, np.nan) else 0.0
    max_drawdown = calculate_max_drawdown(close_col)
    # Sortino (downside only)
    downside = returns[returns < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() != 0 else 0.0
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'current_price': float(close_col.iloc[-1]),
        'n_days': n,
    }


def calculate_max_drawdown(prices: pd.Series) -> float:
    if prices is None or len(prices) == 0:
        return 0.0
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max.replace(0, np.nan)
    return float(drawdown.min()) if len(drawdown) else 0.0


def get_recent_news(symbol: str, max_items: int = 5) -> list:
    try:
        ticker = yf.Ticker(symbol)
        try:
            news = ticker.get_news()
        except Exception:
            news = getattr(ticker, 'news', []) or []
        out = []
        for item in (news or [])[:max_items]:
            if isinstance(item, dict):
                content = item.get('content', item)
                if isinstance(content, dict):
                    out.append({
                        'title': content.get('title', ''),
                        'summary': content.get('summary', content.get('description', '')),
                        'url': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else content.get('link', ''),
                        'provider': (content.get('provider', {}) or {}).get('displayName', '') if isinstance(content.get('provider'), dict) else '',
                    })
                else:
                    out.append({'title': item.get('title', ''), 'summary': item.get('summary', ''), 'url': item.get('link', ''), 'provider': ''})
        return out
    except Exception:
        return []


def get_analyst_data(symbol: str) -> dict:
    """Analyst ratings, price targets, earnings — best-effort, never crashes."""
    out = {'ratings': [], 'target_mean': None, 'earnings': None}
    try:
        t = yf.Ticker(symbol)
        try:
            recs = t.recommendations
            if recs is not None and len(recs):
                out['ratings'] = recs.tail(5).to_dict('records')
        except Exception:
            pass
        try:
            info = t.info or {}
            out['target_mean'] = info.get('targetMeanPrice')
            out['recommendation'] = info.get('recommendationKey')
            out['num_analysts'] = info.get('numberOfAnalystOpinions')
        except Exception:
            pass
    except Exception:
        pass
    return out


def format_currency(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"
