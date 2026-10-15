import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

from modules.ollama_ai import ollama_generate, ollama_chat, check_ollama_status
from modules.indicators import add_all_indicators, sma, ema, rsi, bollinger_bands, macd
from modules.lstm_predictor import DirectionPredictor
from modules.trading_agent import RLTradingAgent, sigmoid
from modules.portfolio_optimizer import kelly_criterion


@dataclass
class AgentDecision:
    agent_name: str
    recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    supporting_evidence: list = field(default_factory=list)


class ResearchAgent:
    """Fundamental analysis agent using Ollama LLM."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.name = "Research Analyst"

    def analyze(self, symbol: str, stock_info: dict, df: pd.DataFrame) -> AgentDecision:
        if not check_ollama_status():
            return self._fallback_analysis(symbol, stock_info, df)

        prompt = f"""
        You are a senior equity research analyst. Analyze {symbol} based on the following data:
        
        Company: {stock_info.get('name', symbol)}
        Sector: {stock_info.get('sector', 'N/A')}
        Industry: {stock_info.get('industry', 'N/A')}
        Market Cap: {stock_info.get('market_cap', 'N/A')}
        P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
        Forward P/E: {stock_info.get('forward_pe', 'N/A')}
        EPS: {stock_info.get('eps', 'N/A')}
        Beta: {stock_info.get('beta', 'N/A')}
        
        Recent price data (last 5 days):
        {df[['Close', 'Volume']].tail(5).to_string()}
        
        Technical indicators:
        - Current RSI(14): {df['RSI_14'].iloc[-1]:.2f}
        - Current MACD: {df['MACD'].iloc[-1]:.4f}
        - Current price trend vs 50-day MA: {(df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1)*100:.1f}%
        
        Provide:
        1. A brief fundamental assessment (overvalued/undervalued/fair)
        2. Key risks and opportunities
        3. Overall recommendation (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
        4. Confidence level (0-100%)
        
        Format: JSON with keys: assessment, risks, opportunities, recommendation, confidence
        """

        system = "You are a financial research analyst. Provide analysis in JSON format."
        result = ollama_generate(self.model, prompt, system=system, temperature=0.3, max_tokens=512)

        try:
            parsed = json.loads(result.strip().split('\n')[-1])
        except (json.JSONDecodeError, IndexError):
            return self._fallback_analysis(symbol, stock_info, df)

        return AgentDecision(
            agent_name=self.name,
            recommendation=parsed.get('recommendation', 'HOLD'),
            confidence=float(parsed.get('confidence', 50)) / 100.0,
            reasoning=f"{parsed.get('assessment', '')} | Risks: {parsed.get('risks', '')} | Opportunities: {parsed.get('opportunities', '')}",
            supporting_evidence=[
                f"Market Cap: {stock_info.get('market_cap', 'N/A')}",
                f"P/E: {stock_info.get('pe_ratio', 'N/A')}",
                f"Beta: {stock_info.get('beta', 'N/A')}",
            ]
        )

    def _fallback_analysis(self, symbol, stock_info, df):
        close = df['Close'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        pct_vs_ma = (close / sma_50 - 1) * 100 if sma_50 else 0
        rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50

        trend = "up" if pct_vs_ma > 2 else "down" if pct_vs_ma < -2 else "neutral"

        if trend == "up" and rsi < 70:
            rec = "BUY"
            conf = 0.6
        elif trend == "down" and rsi > 30:
            rec = "SELL"
            conf = 0.6
        else:
            rec = "HOLD"
            conf = 0.5

        return AgentDecision(
            agent_name=self.name,
            recommendation=rec,
            confidence=conf,
            reasoning=f"Price is {pct_vs_ma:.1f}% vs SMA50 ({trend}), RSI={rsi:.1f}",
            supporting_evidence=["Technical trend analysis"]
        )


class BullishAnalystAgent:
    """Optimistic analysis agent - focuses on positive signals."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.name = "Bullish Analyst"

    def analyze(self, symbol: str, df: pd.DataFrame, rf_accuracy: float) -> AgentDecision:
        if not check_ollama_status():
            return self._fallback(symbol, df, rf_accuracy)

        prompt = f"""
        You are an optimistic stock analyst. Your job is to find bullish arguments for {symbol}.
        
        Current data:
        - Price: ${df['Close'].iloc[-1]:.2f}
        - 52W High/Low: {df['Close'].max():.2f} / {df['Close'].min():.2f}
        - RSI(14): {df['RSI_14'].iloc[-1]:.2f}
        - MACD: {df['MACD'].iloc[-1]:.4f}
        - MACD Signal: {df['MACD_Signal'].iloc[-1]:.4f}
        - BB Width: {df['BB_Width'].iloc[-1]:.4f}
        - 50-day SMA trend vs price: {(df['Close'].iloc[-1]/df['SMA_50'].iloc[-1]-1)*100:.1f}%
        - ML direction prediction accuracy: {rf_accuracy*100:.1f}%
        
        List all bullish factors and give a recommendation from: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        Also provide confidence (0-100). Output as JSON.
        """

        system = "You are a bullish analyst looking for positive signals."
        result = ollama_generate(self.model, prompt, system=system, temperature=0.4, max_tokens=384)

        try:
            parsed = json.loads(result.strip().split('\n')[-1])
        except (json.JSONDecodeError, IndexError):
            return self._fallback(symbol, df, rf_accuracy)

        return AgentDecision(
            agent_name=self.name,
            recommendation=parsed.get('recommendation', 'BUY'),
            confidence=float(parsed.get('confidence', 55)) / 100.0,
            reasoning=parsed.get('analysis', ''),
            supporting_evidence=list(parsed.get('bullish_factors', []))
        )

    def _fallback(self, symbol, df, rf_accuracy):
        price_vs_sma = (df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1) * 100
        macd_hist = df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]

        bullish_signals = sum([
            price_vs_sma > 0, macd_hist > 0, rsi < 60,
            rf_accuracy > 0.5
        ])

        rec = "STRONG_BUY" if bullish_signals >= 3 else "BUY" if bullish_signals >= 2 else "HOLD"
        return AgentDecision(
            agent_name=self.name,
            recommendation=rec,
            confidence=min(0.4 + bullish_signals * 0.15, 0.95),
            reasoning=f"Bullish signals: {bullish_signals}/4 (price above SMA50, MACD hist, RSI<60, ML accuracy)",
            supporting_evidence=[f"Price vs SMA50: {price_vs_sma:.1f}%", f"MACD hist: {macd_hist:.4f}", f"RSI: {rsi:.1f}"]
        )


class BearishAnalystAgent:
    """Pessimistic analysis agent - focuses on negative signals."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.name = "Bearish Analyst"

    def analyze(self, symbol: str, df: pd.DataFrame, rf_accuracy: float) -> AgentDecision:
        if not check_ollama_status():
            return self._fallback(symbol, df, rf_accuracy)

        prompt = f"""
        You are a bearish/pessimistic stock analyst. Your job is to find bearish arguments for {symbol}.
        
        Current data:
        - Price: ${df['Close'].iloc[-1]:.2f}
        - 52W High/Low: {df['Close'].max():.2f} / {df['Close'].min():.2f}
        - RSI(14): {df['RSI_14'].iloc[-1]:.2f}
        - MACD: {df['MACD'].iloc[-1]:.4f}
        - BB Width: {df['BB_Width'].iloc[-1]:.4f}
        - 50-day SMA trend vs price: {(df['Close'].iloc[-1]/df['SMA_50'].iloc[-1]-1)*100:.1f}%
        - ML direction prediction accuracy: {rf_accuracy*100:.1f}%
        
        List all bearish factors and give a recommendation from: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        Also provide confidence (0-100). Output as JSON.
        """

        system = "You are a bearish analyst looking for negative signals."
        result = ollama_generate(self.model, prompt, system=system, temperature=0.4, max_tokens=384)

        try:
            parsed = json.loads(result.strip().split('\n')[-1])
        except (json.JSONDecodeError, IndexError):
            return self._fallback(symbol, df, rf_accuracy)

        return AgentDecision(
            agent_name=self.name,
            recommendation=parsed.get('recommendation', 'SELL'),
            confidence=float(parsed.get('confidence', 55)) / 100.0,
            reasoning=parsed.get('analysis', ''),
            supporting_evidence=list(parsed.get('bearish_factors', []))
        )

    def _fallback(self, symbol, df, rf_accuracy):
        price_vs_sma = (df['Close'].iloc[-1] / df['MA_50' if 'MA_50' in df.columns else 'SMA_50'].iloc[-1] - 1) * 100
        rsi = df['RSI_14'].iloc[-1]
        macd_hist = df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]
        bb_pos = (df['Close'].iloc[-1] - df['BB_Lower'].iloc[-1]) / (df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1])

        bearish_signals = sum([
            price_vs_sma < 0, macd_hist < 0, rsi > 60,
            bb_pos > 0.8, rf_accuracy < 0.55
        ])

        rec = "STRONG_SELL" if bearish_signals >= 3 else "SELL" if bearish_signals >= 2 else "HOLD"
        return AgentDecision(
            agent_name=self.name,
            recommendation=rec,
            confidence=min(0.4 + bearish_signals * 0.15, 0.95),
            reasoning=f"Bearish signals: {bearish_signals}/5",
            supporting_evidence=[f"Price vs SMA50: {price_vs_sma:.1f}%", f"RSI: {rsi:.1f}", f"BB Position: {bb_pos:.2f}"]
        )


class TechnicalAnalystAgent:
    """Technical analysis agent using LSTM + indicators."""

    def __init__(self):
        self.name = "Technical Analyst"

    def analyze(self, symbol: str, df: pd.DataFrame) -> AgentDecision:
        try:
            rf = DirectionPredictor()
            results = rf.train(df, use_grid_search=False)
            prediction = rf.predict_today(df)

            confidence = prediction['confidence']
            raw_pred = prediction['prediction']
            if 'UP' in raw_pred:
                rec = 'BUY'
            elif 'DOWN' in raw_pred:
                rec = 'SELL'
            else:
                rec = 'HOLD'

            evidence = [
                f"RSI(14): {df['RSI_14'].iloc[-1]:.2f}",
                f"MACD: {df['MACD'].iloc[-1]:.4f}",
                f"BB Width: {df['BB_Width'].iloc[-1]:.4f}",
                f"ML Accuracy: {results['accuracy']*100:.1f}%",
                f"Top signal: {results['feature_importance']['Feature'].iloc[0]}"
            ]

            return AgentDecision(
                agent_name=self.name,
                recommendation=rec,
                confidence=confidence,
                reasoning=f"ML model predicts {rec} with {confidence*100:.1f}% confidence. Accuracy: {results['accuracy']*100:.1f}%",
                supporting_evidence=evidence
            )
        except Exception as e:
            rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
            rec = "BUY" if rsi < 50 else "SELL" if rsi > 60 else "HOLD"
            return AgentDecision(
                agent_name=self.name,
                recommendation=rec,
                confidence=0.5,
                reasoning=f"Fallback: RSI={rsi:.1f}",
                supporting_evidence=["Technical indicator analysis"]
            )


class SentimentAgent:
    """News and sentiment analysis using Ollama."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.name = "Sentiment Analyst"

    def analyze(self, symbol: str, news_items: list, stock_info: dict) -> AgentDecision:
        if not news_items:
            return AgentDecision(
                agent_name=self.name,
                recommendation="HOLD",
                confidence=0.3,
                reasoning="No news data available for sentiment analysis.",
                supporting_evidence=[]
            )

        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', item.get('description', '')[:200])}"
            for item in news_items[:5]
        ])

        prompt = f"""
        You are a market sentiment analyst. Analyze the sentiment of these news items for {symbol}:
        
        {news_text}
        
        Also consider: Company = {stock_info.get('name', symbol)}, Sector = {stock_info.get('sector', 'N/A')}
        
        Provide:
        1. Overall sentiment (BULLISH, NEUTRAL, BEARISH)
        2. Key news themes
        3. Impact on stock price
        4. Recommendation
        5. Confidence (0-100)
        
        Format: JSON
        """

        system = "You are a sentiment analysis expert for financial markets."
        result = ollama_generate(self.model, prompt, system=system, temperature=0.5, max_tokens=512)

        try:
            parsed = json.loads(result.strip().split('\n')[-1])
        except (json.JSONDecodeError, IndexError):
            return AgentDecision(
                agent_name=self.name,
                recommendation="HOLD",
                confidence=0.4,
                reasoning="Sentiment analysis unavailable.",
                supporting_evidence=["Fallback to neutral"]
            )

        sentiment = parsed.get('sentiment', 'NEUTRAL')
        if sentiment == 'BULLISH':
            rec = "BUY"
        elif sentiment == 'BEARISH':
            rec = "SELL"
        else:
            rec = "HOLD"

        return AgentDecision(
            agent_name=self.name,
            recommendation=rec,
            confidence=float(parsed.get('confidence', 50)) / 100.0,
            reasoning=f"Sentiment: {sentiment} | {parsed.get('analysis', '')}",
            supporting_evidence=parsed.get('key_themes', [])
        )


class RiskManagementAgent:
    """Risk assessment and portfolio management agent."""

    def __init__(self):
        self.name = "Risk Manager"

    def analyze(self, symbol: str, df: pd.DataFrame, initial_capital: float, model_predictions: dict) -> AgentDecision:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        max_dd = (df['Close'].max() - df['Close'].min()) / df['Close'].max()
        var_95 = np.percentile(returns, 5)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        rf_acc = model_predictions.get('rf_accuracy', 0)
        confidence = model_predictions.get('confidence', 0.5)

        risk_factors = []
        risk_score = 0

        if volatility > 0.4:
            risk_score += 2
            risk_factors.append(f"High volatility ({volatility*100:.1f}%)")
        if max_dd > 0.3:
            risk_score += 1
            risk_factors.append(f"High drawdown ({max_dd*100:.1f}%)")
        if abs(var_95) > 0.05:
            risk_score += 1
            risk_factors.append(f"High VaR ({var_95*100:.1f}%)")

        if risk_score >= 2:
            rec = "SELL"
            conf = 0.8
            reasoning = "High risk profile detected. Consider reducing exposure."
        elif risk_score >= 1:
            rec = "HOLD"
            conf = 0.6
            reasoning = "Moderate risk. Proceed with caution."
        else:
            rec = "BUY" if rf_acc > 0.5 else "HOLD"
            conf = min(rf_acc * 0.8, 0.8)
            reasoning = "Favorable risk metrics. Position sizing recommended."

        return AgentDecision(
            agent_name=self.name,
            recommendation=rec,
            confidence=conf,
            reasoning=reasoning,
            supporting_evidence=[
                f"Volatility: {volatility*100:.1f}%",
                f"Max DD: {max_dd*100:.1f}%",
                f"VaR(95): {var_95*100:.1f}%",
                f"Sharpe: {sharpe:.2f}",
            ] + risk_factors
        )


class PortfolioManagerAgent:
    """Final consensus decision maker using weighted voting."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.name = "Portfolio Manager"
        self.risk_weights = {
            'Research Analyst': 0.15,
            'Bullish Analyst': 0.15,
            'Bearish Analyst': 0.15,
            'Technical Analyst': 0.30,
            'Sentiment Analyst': 0.15,
            'Risk Manager': 0.20,
        }

    def make_decision(self, symbol: str, decisions: List[AgentDecision]) -> Tuple[str, dict]:
        recommendation_scores = {'STRONG_BUY': 0.0, 'BUY': 0.0, 'HOLD': 0.0, 'SELL': 0.0, 'STRONG_SELL': 0.0}

        total_confidence = 0.0
        for decision in decisions:
            weight = self.risk_weights.get(decision.agent_name, 0.15)
            recommendation_scores[decision.recommendation] += weight * decision.confidence
            total_confidence += weight * decision.confidence

        total_weight = sum(
            self.risk_weights.get(d.agent_name, 0.15) * d.confidence
            for d in decisions if d.agent_name in self.risk_weights
        )

        best_rec = max(recommendation_scores, key=recommendation_scores.get)
        normalized_confidence = min(
            recommendation_scores[best_rec] / total_weight if total_weight > 0 else 0.5,
            1.0
        )

        position_size = self._calculate_position_size(
            symbol, best_rec, normalized_confidence,
            decisions[-1].supporting_evidence
        )

        return best_rec, {
            'recommendation': best_rec,
            'confidence': normalized_confidence,
            'all_votes': {d.agent_name: {'rec': d.recommendation, 'conf': d.confidence} for d in decisions},
            'details': decisions,
            'position_size': position_size,
            'expected_return': self._estimate_expected_return(best_rec, normalized_confidence),
        }

    def _calculate_position_size(self, symbol: str, recommendation: str, confidence: float, evidence: list) -> dict:
        if recommendation in ('STRONG_BUY', 'BUY'):
            base_size = 0.1 * confidence  # 10% base, scaled by confidence
        elif recommendation == 'HOLD':
            base_size = 0.05
        else:
            base_size = 0.0  # Don't buy on negative recommendation

        kelly_pct = 0.0
        for e in evidence:
            if 'win_rate' in str(e).lower() or 'kelly' in str(e).lower():
                try:
                    kelly_pct = float(str(e).split(':')[1].strip().rstrip('%')) / 100
                except (IndexError, ValueError):
                    pass

        return {
            'base_allocation': round(base_size * 100, 2),
            'optimal_allocation': round(max(base_size, kelly_pct) * 100, 2) if kelly_pct > 0 else round(base_size * 100, 2),
            'max_suggested': round(base_size * 0.25 * 100, 2),
            'reasoning': f"Allocation based on {recommendation} signal with {confidence*100:.0f}% confidence"
        }

    def _estimate_expected_return(self, recommendation: str, confidence: float) -> dict:
        return_map = {
            'STRONG_BUY': {'monthly': 0.08, 'quarterly': 0.24, 'annual': 0.40},
            'BUY': {'monthly': 0.03, 'quarterly': 0.09, 'annual': 0.15},
            'HOLD': {'monthly': 0.005, 'quarterly': 0.015, 'annual': 0.05},
            'SELL': {'monthly': -0.02, 'quarterly': -0.06, 'annual': -0.12},
            'STRONG_SELL': {'monthly': -0.05, 'quarterly': -0.15, 'annual': -0.30},
        }
        base = return_map.get(recommendation, return_map['HOLD'])
        return {
            'monthly': round(base['monthly'] * confidence, 4),
            'quarterly': round(base['quarterly'] * confidence, 4),
            'annual': round(base['annual'] * confidence, 4),
        }


class MultiAgentTradingSystem:
    """Orchestrates all agents to produce a consensus trading recommendation."""

    def __init__(self, model: str = "qwen3.5:0.8b"):
        self.model = model
        self.research = ResearchAgent(model)
        self.bull = BullishAnalystAgent(model)
        self.bear = BearishAnalystAgent(model)
        self.tech = TechnicalAnalystAgent()
        self.sentiment = SentimentAgent(model)
        self.risk = RiskManagementAgent()
        self.portfolio = PortfolioManagerAgent(model)

    def analyze_stock(self, symbol: str, df: pd.DataFrame, stock_info: dict = None,
                      news_items: list = None) -> dict:
        stock_info = stock_info or {}
        news_items = news_items or []

        if df.empty:
            return {'error': 'No data available'}

        rf_pred = DirectionPredictor()
        rf_results = rf_pred.train(df, use_grid_search=False)
        prediction = rf_pred.predict_today(df)
        rf_accuracy = rf_results['accuracy']

        decisions = []

        decisions.append(self.research.analyze(symbol, stock_info, df))
        decisions.append(self.bull.analyze(symbol, df, rf_accuracy))
        decisions.append(self.bear.analyze(symbol, df, rf_accuracy))
        decisions.append(self.tech.analyze(symbol, df))
        decisions.append(self.sentiment.analyze(symbol, news_items, stock_info))
        decisions.append(self.risk.analyze(symbol, df, 100000, {
            'rf_accuracy': rf_accuracy,
            'confidence': prediction['confidence']
        }))

        recommendation, details = self.portfolio.make_decision(symbol, decisions)

        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': details['confidence'],
            'agent_decisions': decisions,
            'consensus': details,
            'technical_prediction': {
                'direction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'prob_up': prediction['probability_up'],
                'prob_down': prediction['probability_down'],
            },
            'rf_accuracy': rf_accuracy,
        }

    def screen_and_recommend(self, symbols: list) -> pd.DataFrame:
        import yfinance as yf
        results = []
        for symbol in symbols:
            try:
                df = yf.download(symbol, period="2y", progress=False)
                if df.empty:
                    continue
                df_ind = add_all_indicators(df)
                if df_ind.empty:
                    continue

                analysis = self.analyze_stock(symbol, df_ind)
                results.append({
                    'Symbol': symbol,
                    'Recommendation': analysis['recommendation'],
                    'Confidence': f"{analysis['confidence']*100:.0f}%",
                    'RF Accuracy': f"{analysis['rf_accuracy']*100:.1f}%",
                    'Current Price': f"${df_ind['Close'].iloc[-1]:.2f}",
                    'Direction': analysis['technical_prediction']['direction'],
                    'P(Up)': f"{analysis['technical_prediction']['prob_up']*100:.0f}%",
                })
            except Exception as e:
                results.append({'Symbol': symbol, 'Recommendation': 'ERROR', 'Confidence': str(e)[:50],
                               'RF Accuracy': 'N/A', 'Current Price': 'N/A',
                               'Direction': 'N/A', 'P(Up)': 'N/A'})

        return pd.DataFrame(results)