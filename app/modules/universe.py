"""Market universes: every asset class the app can analyse.

All symbols are yfinance-compatible — stocks, crypto, forex, indices,
commodities/futures, ETFs and global markets.
"""
from modules.data_loader import detect_asset_class

UNIVERSES = {
    "🇺🇸 US Mega Caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JNJ", "V"],
    "💻 Tech Leaders": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "CRM", "ADBE", "INTC"],
    "₿ Crypto Top 10": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD"],
    "₿ Crypto DeFi / Alts": ["MATIC-USD", "ATOM-USD", "UNI-USD", "LTC-USD", "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD"],
    "💱 Forex Majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"],
    "📈 Global Indices": ["^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^N225", "^HSI"],
    "🛢️ Commodities & Futures": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F"],
    "🏦 Finance + Healthcare": ["JPM", "V", "MA", "BAC", "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH"],
    "⚡ Energy + Utilities": ["XOM", "CVX", "COP", "NEE", "DUK", "SHEL", "TTE"],
    "🌍 Global ADRs": ["TSM", "ASML", "NVO", "BABA", "TM", "SONY", "SAP", "UL"],
    "📦 ETFs": ["SPY", "QQQ", "DIA", "IWM", "ARKK", "VTI", "EEM", "GLD", "TLT", "USO"],
}

SAMPLE_SYMBOLS = {
    "Stock": "AAPL",
    "Crypto": "BTC-USD",
    "Forex": "EURUSD=X",
    "Index": "^GSPC",
    "Commodity": "GC=F",
    "ETF": "SPY",
}

MARKET_HELP = """**Symbol formats:**
- Stocks: `AAPL`, `TSLA`, `BRK-B`
- Crypto: `BTC-USD`, `ETH-USD`, `SOL-USD`
- Forex: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
- Indices: `^GSPC` (S&P500), `^NDX` (Nasdaq100), `^DJI` (Dow)
- Commodities: `GC=F` (gold), `CL=F` (oil), `SI=F` (silver)
- ETFs: `SPY`, `QQQ`, `GLD`"""


def guess_universe(symbols: list) -> str:
    classes = {detect_asset_class(s, {}) for s in symbols}
    if classes == {"CRYPTO"}:
        return "crypto"
    if classes == {"FOREX"}:
        return "forex"
    if len(classes) == 1:
        return list(classes)[0].lower()
    return "mixed"
