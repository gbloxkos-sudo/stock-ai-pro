import numpy as np
import pandas as pd
from itertools import combinations
import streamlit as st


def calculate_efficient_frontier(returns: pd.DataFrame, n_portfolios: int = 10000):
    """Calculate the efficient frontier using Monte Carlo simulation."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    results = np.zeros((3 + num_assets, n_portfolios))
    max_sharpe_ratio = 0
    max_sharpe_port = None

    for i in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)
        portfolio_sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = portfolio_sharpe
        results[3:, i] = weights

        if portfolio_sharpe > max_sharpe_ratio:
            max_sharpe_ratio = portfolio_sharpe
            max_sharpe_port = results[:, i]

    min_vol_idx = np.argmin(results[1, :])
    min_vol_port = results[:, min_vol_idx]

    return results, max_sharpe_port, min_vol_port


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly Criterion for optimal position size.
    
    K% = W - (1-W)/R where W = win rate, R = avg win / avg loss
    """
    if avg_loss == 0 or avg_loss == -1:
        return 0.0

    kelly = win_rate - (1 - win_rate) / (abs(avg_win) / abs(avg_loss)) if avg_loss != 0 else 0
    return max(0, min(kelly, 0.25))  # Cap at 25%


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) using historical simulation."""
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional VaR)."""
    if returns.empty:
        return 0.0
    var = value_at_risk(returns, confidence)
    beyond_var = returns[returns <= var]
    return float(beyond_var.mean()) if len(beyond_var) > 0 else var


def max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown."""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if len(excess_returns) < 2 or excess_returns.std() == 0:
        return 0.0
    return float((excess_returns.mean() / excess_returns.std()) * np.sqrt(252))


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    excess_returns = returns - risk_free_rate / 252
    downside_deviation = np.sqrt(
        (np.minimum(excess_returns, 0) ** 2).mean()
    ) * np.sqrt(252)
    if downside_deviation == 0:
        return 0.0
    return float((excess_returns.mean() * np.sqrt(252)) / downside_deviation)


def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    annual_return = (1 + returns.mean()) ** 252 - 1
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return 0.0
    return float(annual_return / mdd)


def optimize_portfolio_weights(returns: pd.DataFrame, objective: str = "sharpe"):
    """Optimize portfolio weights based on objective.
    
    Options: 'sharpe' (max Sharpe), 'min_vol' (min volatility), 'risk_parity'
    """
    results, max_sharpe_port, min_vol_port = calculate_efficient_frontier(returns)

    if objective == "min_vol":
        weights = min_vol_port[3:]
    elif objective == "risk_parity":
        weights = _risk_parity_weights(returns)
    else:
        weights = max_sharpe_port[3:]

    return dict(zip(returns.columns, weights))


def _risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """Calculate risk parity weights (inverse volatility weighting)."""
    vol = returns.std() * np.sqrt(252)
    inv_vol = 1 / vol
    return (inv_vol / inv_vol.sum()).values


def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001
):
    """Simple backtesting engine.
    
    Args:
        df: DataFrame with 'Close' column
        signals: Series with 1 (buy), -1 (sell), 0 (hold)
        initial_capital: Starting cash
        transaction_cost: Fractional transaction cost per trade
    """
    position = 0
    cash = initial_capital
    portfolio_values = []
    trades = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0

        if signal == 1 and position == 0:
            shares_to_buy = cash * (1 - transaction_cost) / price
            cash -= shares_to_buy * price
            position += shares_to_buy
            trades.append({'date': df.index[i], 'action': 'BUY', 'price': price, 'shares': shares_to_buy})
        elif signal == -1 and position > 0:
            cash += position * price * (1 - transaction_cost)
            trades.append({'date': df.index[i], 'action': 'SELL', 'price': price, 'shares': position})
            position = 0

        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value / initial_capital - 1)

    portfolio_series = pd.Series(portfolio_values, index=df.index)
    returns = portfolio_series.pct_change().dropna()

    metrics = {
        'total_return': portfolio_values[-1] if portfolio_values else 0,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': max_drawdown(pd.Series(portfolio_values, index=df.index)),
        'var_95': value_at_risk(returns),
        'expected_shortfall': expected_shortfall(returns),
        'num_trades': len(trades),
    }

    return portfolio_series, metrics, trades


def screen_stocks(symbols: list, criteria: dict = None) -> pd.DataFrame:
    """Screen stocks based on fundamental and technical criteria."""
    import yfinance as yf

    if criteria is None:
        criteria = {
            'min_market_cap': 0,
            'min_beta': 0,
            'max_pe': 1000,
            'min_volume': 0,
        }

    results = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")

            if hist.empty:
                continue

            market_cap = info.get('marketCap', 0)
            pe = info.get('trailingPE', float('inf'))
            beta = info.get('beta', 0)
            volume = hist['Volume'].mean()

            pass_criteria = (
                market_cap >= criteria.get('min_market_cap', 0) and
                pe <= criteria.get('max_pe', float('inf')) and
                beta >= criteria.get('min_beta', 0) and
                volume >= criteria.get('min_volume', 0)
            )

            if pass_criteria:
                results.append({
                    'symbol': symbol,
                    'name': info.get('shortName', ''),
                    'market_cap': format_mcap(market_cap),
                    'pe_ratio': round(pe, 2) if pe != float('inf') else 'N/A',
                    'beta': round(beta, 2),
                    'volume': int(volume),
                    'current_price': round(hist['Close'].iloc[-1], 2),
                    'pct_change_3mo': round((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100, 2),
                })
        except Exception:
            continue

    return pd.DataFrame(results)


def format_mcap(value: float) -> str:
    """Format market cap into human-readable string."""
    if value >= 1e12:
        return f"${value/1e12:.1f}T"
    elif value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    return f"${value:,.0f}"