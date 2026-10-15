import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_price_chart(df: pd.DataFrame, title: str = "Stock Price Chart"):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Volume')
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='#90caf9',
        ),
        row=2, col=1
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_price_with_ma(df: pd.DataFrame, title: str = "Price with Moving Averages"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))

    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='20-day SMA',
            line=dict(color='#ff7f0e', width=1, dash='dot')
        ))

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='50-day SMA',
            line=dict(color='#2ca02c', width=1, dash='dot')
        ))

    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_200'],
            name='200-day SMA',
            line=dict(color='#d62728', width=1, dash='dot')
        ))

    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='rgba(255,127,0,0.5)', width=1),
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='rgba(255,127,0,0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,127,0,0.05)'
        ))

    fig.update_layout(
        title=title,
        height=500,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )

    return fig


def create_rsi_chart(df: pd.DataFrame):
    fig = go.Figure()

    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI_14'],
            name='RSI (14)',
            line=dict(color='#9467bd', width=2)
        ))

    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")

    fig.update_layout(
        title='Relative Strength Index (RSI)',
        height=300,
        template='plotly_dark',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
    )

    return fig


def create_macd_chart(df: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('MACD', 'Histogram'),
        row_heights=[0.7, 0.3]
    )

    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            name='MACD',
            line=dict(color='#2ca02c', width=2)
        ), row=1, col=1)

    if 'MACD_Signal' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            name='Signal',
            line=dict(color='#ff7f0e', width=1, dash='dot')
        ), row=1, col=1)

    if 'MACD_Hist' in df.columns:
        colors = ['red' if v < 0 else 'green' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_Hist'],
            name='Histogram',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(height=400, template='plotly_dark', title='MACD')
    return fig


def create_lstm_prediction_chart(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_df.index,
        y=train_df['Close'],
        name='Train Actual',
        line=dict(color='#2196F3', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=valid_df.index,
        y=valid_df['Close'],
        name='Validation Actual',
        line=dict(color='#4CAF50', width=2)
    ))

    if 'Predictions' in valid_df.columns:
        fig.add_trace(go.Scatter(
            x=valid_df.index,
            y=valid_df['Predictions'],
            name='LSTM Predicted',
            line=dict(color='#FF5722', width=2, dash='dot')
        ))

    fig.update_layout(
        title='LSTM Price Prediction',
        height=500,
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )

    return fig


def create_trading_chart_with_signals(df: pd.DataFrame, trades: list):
    fig = create_price_with_ma(df, title='RL Trading Agent - Trade Signals')

    buy_signals = [t for t in trades if t['action'] == 'BUY']
    sell_signals = [t for t in trades if t['action'] == 'SELL']

    if buy_signals:
        buy_dates = [df.index[int(t['step'])] for t in buy_signals]
        buy_prices = [t['price'] for t in buy_signals]
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up'),
        ))

    if sell_signals:
        sell_dates = [df.index[int(t['step'])] for t in sell_signals]
        sell_prices = [t['price'] for t in sell_signals]
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down'),
        ))

    return fig


def create_feature_importance_chart(feature_importance_df: pd.DataFrame):
    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'].head(15),
        y=feature_importance_df['Feature'].head(15),
        orientation='h',
        marker_color='#42A5F5'
    ))

    fig.update_layout(
        title='Top 15 Feature Importances (Random Forest)',
        height=500,
        template='plotly_dark',
        yaxis=dict(autorange='reversed')
    )

    return fig


def create_portfolio_chart(trades: list, initial_balance: float = 10000.0):
    if not trades:
        return None

    fig = go.Figure()

    balance = initial_balance
    balances = [initial_balance]
    timestamps = ['Start']

    for trade in trades:
        if trade['action'] == 'SELL':
            balance += trade.get('profit', 0)
        balances.append(balance)
        timestamps.append(f"Step {trade['step']}")

    fig.add_trace(go.Scatter(
        x=range(len(balances)),
        y=balances,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#66BB6A', width=2),
    ))

    fig.update_layout(
        title='Portfolio Value Over Trades',
        xaxis_title='Trade Steps',
        yaxis_title='Portfolio Value ($)',
        height=300,
        template='plotly_dark',
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: list = None):
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in df.columns]

    corr_df = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu',
        zmid=0,
    ))

    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=500,
        template='plotly_dark',
    )

    return fig