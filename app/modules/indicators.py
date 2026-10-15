import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = sma(series, window)
    std = series.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, ma, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period).mean()
    return (tp - sma_tp) / (0.015 * mad)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # Adaptive windows: never require more data than we have (critical for
    # crypto/1mo views where SMA_200 would wipe the whole frame).
    w20 = min(20, max(2, n // 3)) if n < 20 else 20
    w50 = min(50, max(5, n // 2)) if n < 50 else 50
    w200 = 200 if n >= 210 else 0  # only if enough history

    df['SMA_20'] = sma(df['Close'], w20)
    df['SMA_50'] = sma(df['Close'], w50)
    if w200:
        df['SMA_200'] = sma(df['Close'], 200)
    else:
        df['SMA_200'] = df['Close'].expanding().mean()
    df['EMA_12'] = ema(df['Close'], min(12, max(2, n // 2)))
    df['EMA_26'] = ema(df['Close'], min(26, max(3, n // 2)))

    df['RSI_2'] = rsi(df['Close'], 2)
    df['RSI_7'] = rsi(df['Close'], 7)
    df['RSI_14'] = rsi(df['Close'], 14)
    df['RSI_21'] = rsi(df['Close'], 21)

    bb_upper, bb_ma, bb_lower = bollinger_bands(df['Close'], 20)
    df['BB_Upper'] = bb_upper
    df['BB_MA'] = bb_ma
    df['BB_Lower'] = bb_lower
    df['BB_Width'] = (bb_upper - bb_lower) / bb_ma

    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist

    df['CCI_20'] = cci(df['High'], df['Low'], df['Close'], 20)
    df['CCI_50'] = cci(df['High'], df['Low'], df['Close'], 50)

    df['Daily_Returns'] = df['Close'].pct_change()
    df['Volatility_20'] = df['Daily_Returns'].rolling(window=20).std()

    df['Price_Change_1D'] = df['Close'].pct_change(1)
    df['Price_Change_3D'] = df['Close'].pct_change(3)
    df['Price_Change_5D'] = df['Close'].pct_change(5)

    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    df['ATR_14'] = average_true_range(df, 14)

    df['ROC_10'] = rate_of_change(df['Close'], 10)
    df['ROC_20'] = rate_of_change(df['Close'], 20)

    # Don't nuke short histories (crypto / 1mo). Only drop rows where the
    # *core* trend/momentum features are NaN, keep the rest forward-filled.
    core = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal']
    core = [c for c in core if c in df.columns]
    df = df.dropna(subset=core)
    df = df.ffill().bfill()
    return df


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100