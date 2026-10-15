import requests
import json
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "qwen3.5:0.8b"


def check_ollama_status() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_URL}/tags", timeout=10)
        if r.status_code == 200:
            data = r.json()
            model_names = [m['name'] for m in data.get('models', [])]
            if model_names:
                return model_names
    except Exception:
        pass
    return [DEFAULT_MODEL]


def _prepare_payload(model, messages=None, prompt=None, system="", temperature=0.7, max_tokens=2048, stream=False):
    payload = {"model": model, "options": {"temperature": temperature, "num_predict": max_tokens}}
    if prompt is not None:
        payload["prompt"] = prompt
        payload["system"] = system
        payload["stream"] = stream
        payload["think"] = False
    if messages is not None:
        payload["messages"] = messages
        payload["stream"] = stream
        payload["think"] = False
    return payload


def ollama_generate(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False
) -> str:
    payload = _prepare_payload(model, prompt=prompt, system=system, temperature=temperature,
                               max_tokens=max_tokens, stream=stream)

    try:
        r = requests.post(f"{OLLAMA_URL}/generate", json=payload, timeout=300)
        if r.status_code == 200:
            if stream:
                return r
            data = r.json()
            response = data.get('response', '')
            if not response and data.get('thinking'):
                pass
            return response
        else:
            return f"Error: {r.status_code} - {r.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure `ollama serve` is running."
    except requests.exceptions.ReadTimeout:
        return "Error: Ollama request timed out. Model may be loading."
    except Exception as e:
        return f"Error: {str(e)}"


def ollama_chat(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    payload = _prepare_payload(model, messages=messages, temperature=temperature,
                               max_tokens=max_tokens, stream=True)

    try:
        r = requests.post(f"{OLLAMA_URL}/chat", json=payload, stream=True, timeout=300)
        if r.status_code == 200:
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
        else:
            yield f"Error: {r.status_code}"
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to Ollama."
    except requests.exceptions.ReadTimeout:
        yield "Error: Ollama request timed out. Model may be loading."
    except Exception as e:
        yield f"Error: {str(e)}"


def get_stock_analysis_prompt(stock_data: str, stock_symbol: str) -> str:
    return f"""
You are a professional stock market analyst. Analyze the following stock data for {stock_symbol} and provide:
1. A summary of recent price action and trend
2. Key technical levels (support/resistance)
3. Risk assessment
4. Investment recommendation (with disclaimer)

Data:
{stock_data}
"""


def get_indicator_analysis_prompt(indicators: str, stock_symbol: str) -> str:
    return f"""
As a financial analyst, explain what the following technical indicators suggest for {stock_symbol}:
- Moving averages and their crossovers
- RSI readings (overbought/oversold signals)
- MACD signals
- Bollinger Bands position
- Volume trends

Provide a comprehensive analysis:
{indicators}
"""


def get_prediction_explanation_prompt(prediction: str, stock_symbol: str) -> str:
    return f"""
You are an AI stock prediction assistant. For the stock {stock_symbol}, provide:
1. Explanation of the prediction methodology used
2. Interpretation of the results
3. Confidence level assessment
4. Factors that might affect accuracy
5. Practical trading considerations

Prediction details:
{prediction}
"""


def get_trading_advice_prompt(stock_symbol: str, context: str) -> str:
    return f"""
Act as a senior trading advisor. Given the market context for {stock_symbol}, provide:
1. Clear entry and exit signals
2. Position sizing recommendations
3. Stop-loss levels
4. Risk/reward ratio
5. Time horizon assessment

Context:
{context}

IMPORTANT: Include disclaimers about trading risks.
"""


def generate_ai_summary(
    model: str,
    stock_symbol: str,
    stock_info: dict,
    recent_metrics: dict
) -> str:
    prompt = f"""
    Provide a concise (3-4 paragraphs) professional market summary for {stock_symbol}.
    
    Stock Information:
    - Name: {stock_info.get('name', 'N/A')}
    - Sector: {stock_info.get('sector', 'N/A')}
    - Industry: {stock_info.get('industry', 'N/A')}
    - Market Cap: {stock_info.get('market_cap', 'N/A')}
    - P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
    - Beta: {stock_info.get('beta', 'N/A')}
    
    Recent Performance Metrics:
    - Current Price: ${recent_metrics.get('current_price', 0):.2f}
    - Total Return: {recent_metrics.get('total_return', 0)*100:.2f}%
    - Annualized Return: {recent_metrics.get('annualized_return', 0)*100:.2f}%
    - Volatility: {recent_metrics.get('volatility', 0)*100:.2f}%
    - Sharpe Ratio: {recent_metrics.get('sharpe_ratio', 0):.2f}
    - Max Drawdown: {recent_metrics.get('max_drawdown', 0)*100:.2f}%
    
    Be concise but informative. Include key investment considerations.
    """

    system = "You are a professional financial analyst providing concise market summaries."

    return ollama_generate(
        model=model,
        prompt=prompt,
        system=system,
        temperature=0.5,
        max_tokens=1024
    )


def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def clear_chat():
    st.session_state.chat_history = []


def get_chat_response(model: str, stock_symbol: str, chat_history: list) -> str:
    messages = [
        {"role": "system", "content": f"You are a helpful AI stock analysis assistant. You provide financial insights and analysis for stocks. Stock being discussed: {stock_symbol}. Keep responses concise and include disclaimers where appropriate."}
    ] + chat_history

    return ollama_chat(model=model, messages=messages, temperature=0.7, max_tokens=2048)