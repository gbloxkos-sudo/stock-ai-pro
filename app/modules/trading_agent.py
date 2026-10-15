import numpy as np
import pandas as pd
import random
import math
from collections import deque
import streamlit as st


def sigmoid(x):
    return 1 / (1 + (np.exp(-x) if x >= 0 else 1 / np.exp(x)))


def stocks_price_format(n):
    if n < 0:
        return "- ${:.2f}".format(abs(n))
    else:
        return "${:.2f}".format(abs(n))


def dataset_loader(stock_name: str, period: str = "1y") -> pd.Series:
    from modules.data_loader import fetch_stock_data
    df = fetch_stock_data(stock_name, period=period)
    if df.empty:
        import yfinance as yf
        dataset = yf.download(stock_name, period=period, progress=False)
        if isinstance(dataset.columns, pd.MultiIndex):
            try:
                dataset.columns = dataset.columns.get_level_values(0)
            except Exception:
                pass
        close = dataset['Close'] if 'Close' in dataset.columns else dataset.iloc[:, 0]
    else:
        close = df['Close']
    # Squeeze to Series (yfinance sometimes returns single-col DataFrame)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def state_creator(data: pd.Series, timestep: int, window_size: int) -> np.ndarray:
    # Window of `window_size` prices ending at `timestep`, padded at the start.
    # Returns window_size-1 sigmoided diffs as the state vector.
    timestep = max(0, min(int(timestep), len(data) - 1))
    window_size = max(2, int(window_size))
    end = timestep + 1
    start = end - window_size
    if start >= 0:
        windowed_data = list(data.iloc[start:end])
    else:
        pad_count = -start
        padded = [data.iloc[0]] * pad_count
        windowed_data = padded + list(data.iloc[0:end])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state])


class RLTradingAgent:
    def __init__(self, state_size: int, action_space: int = 3):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def _build_model(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dense(32, activation='relu'),
                Dense(self.action_space, activation='linear')
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
            return model
        except ImportError:
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            dummy_X = np.zeros((1, self.state_size))
            dummy_y = np.zeros((1, self.action_space))
            model.fit(dummy_X, dummy_y)
            return model

    def _update_target_model(self):
        try:
            import tensorflow as tf
            self.target_model.set_weights(self.model.get_weights())
        except (ImportError, AttributeError):
            import copy
            self.target_model = copy.deepcopy(self.model)

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        try:
            import tensorflow as tf
            q_values = self.model.predict(state, verbose=0)
        except (ImportError, AttributeError):
            q_values = np.array([self.model.predict(state)])

        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                try:
                    import tensorflow as tf
                    q_next = np.amax(self.target_model.predict(next_state, verbose=0)[0])
                except (ImportError, AttributeError):
                    try:
                        q_next = np.amax(self.model.predict(next_state)[0])
                    except Exception:
                        q_next = 0
                target = reward + self.gamma * q_next

            try:
                import tensorflow as tf
                target_f = self.model.predict(state, verbose=0)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            except (ImportError, AttributeError):
                pass

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        try:
            self.model.save(path)
        except (AttributeError, OSError):
            import joblib
            joblib.dump(self.model, path)

    def load(self, path: str):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path)
            self._update_target_model()
        except Exception:
            import joblib
            self.model = joblib.load(path)


def train_agent(
    stock_name: str,
    window_size: int = 10,
    episodes: int = 5,
    batch_size: int = 32,
    initial_balance: float = 10000.0,
    progress_bar=None
):
    data = dataset_loader(stock_name, period="2y")
    data = data.dropna()
    data_values = data.values
    data_samples = len(data_values) - window_size

    agent = RLTradingAgent(state_size=window_size)
    total_profit = 0.0
    inventory = []
    trades = []

    for episode in range(1, episodes + 1):
        if progress_bar:
            progress_bar.progress(episode / episodes, text=f"Training episode {episode}/{episodes}")

        state = state_creator(data, 0, window_size + 1)
        episode_profit = 0.0
        inventory = []

        for t in range(data_samples):
            action = agent.act(state)
            next_state = state_creator(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:
                inventory.append(data_values[t])
                trades.append({
                    'step': t,
                    'date': str(data.index[t]),
                    'action': 'BUY',
                    'price': float(data_values[t]),
                    'balance': initial_balance + episode_profit
                })

            elif action == 2 and len(inventory) > 0:
                buy_price = inventory.pop(0)
                reward = max(data_values[t] - buy_price, 0)
                profit = data_values[t] - buy_price
                episode_profit += profit
                trades.append({
                    'step': t,
                    'date': str(data.index[t]),
                    'action': 'SELL',
                    'price': float(data_values[t]),
                    'buy_price': float(buy_price),
                    'profit': float(profit),
                    'balance': initial_balance + episode_profit
                })

            done = (t == data_samples - 1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                pass

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        total_profit += episode_profit

        if progress_bar:
            progress_bar.text(f"Episode {episode} complete. Profit: {stocks_price_format(episode_profit)}")

    avg_profit = total_profit / episodes
    success_rate = len([t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) > 0]) / max(1, len([t for t in trades if t['action'] == 'SELL']))

    return agent, trades, avg_profit, success_rate