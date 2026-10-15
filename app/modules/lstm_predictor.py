import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st


@st.cache_resource
def get_scaler():
    return MinMaxScaler(feature_range=(0, 1))


def prepare_lstm_data(data: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm_model(input_shape: tuple):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(df: pd.DataFrame, epochs: int = 1, lookback: int = 60, verbose: int = 1):
    scaler = get_scaler()
    dataset = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(dataset)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]

    X_train, y_train = prepare_lstm_data(train_data, lookback)

    model = build_lstm_model((X_train.shape[1], 1))

    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=verbose)

    return model, scaler


def predict_lstm(model, df: pd.DataFrame, scaler, lookback: int = 60):
    dataset = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(dataset)

    train_size = int(len(scaled_data) * 0.8)

    # Build prediction inputs starting from train_size - lookback (need lookback context)
    inputs = scaled_data[train_size - lookback:]
    X_test = []
    for i in range(lookback, len(inputs)):
        X_test.append(inputs[i - lookback:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    train_df = df.iloc[:train_size].copy()
    valid_df = df.iloc[train_size:].copy()

    pred_col = 'Predictions'
    valid_df[pred_col] = np.nan
    pred_values = predictions.flatten()

    n_valid = len(valid_df)
    valid_df.loc[valid_df.index, pred_col] = np.nan
    pred_len = min(len(pred_values), n_valid)
    valid_df.iloc[-pred_len:, valid_df.columns.get_loc(pred_col)] = pred_values[:pred_len]

    return train_df, valid_df


class DirectionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.best_params = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from modules.indicators import (
            sma, ema, rsi, bollinger_bands, macd, cci
        )

        features = pd.DataFrame(index=df.index)

        features['Close'] = df['Close']
        features['Open'] = df['Open']
        features['High'] = df['High']
        features['Low'] = df['Low']
        features['Volume'] = df['Volume']

        features['SMA_5'] = sma(df['Close'], 5)
        features['SMA_10'] = sma(df['Close'], 10)
        features['SMA_20'] = sma(df['Close'], 20)
        features['SMA_50'] = sma(df['Close'], 50)
        features['SMA_200'] = sma(df['Close'], 200)

        features['EMA_12'] = ema(df['Close'], 12)
        features['EMA_26'] = ema(df['Close'], 26)

        features['RSI_2'] = rsi(df['Close'], 2)
        features['RSI_7'] = rsi(df['Close'], 7)
        features['RSI_14'] = rsi(df['Close'], 14)
        features['RSI_21'] = rsi(df['Close'], 21)

        bb_upper, bb_ma, bb_lower = bollinger_bands(df['Close'], 20)
        features['BB_Upper'] = bb_upper
        features['BB_Lower'] = bb_lower
        features['BB_MA'] = bb_ma
        features['BB_Width'] = (bb_upper - bb_lower) / bb_ma

        macd_line, signal_line, hist = macd(df['Close'])
        features['MACD'] = macd_line
        features['MACD_Signal'] = signal_line
        features['MACD_Hist'] = hist

        features['CCI_20'] = cci(df['High'], df['Low'], df['Close'], 20)
        features['CCI_50'] = cci(df['High'], df['Low'], df['Close'], 50)

        features['Daily_Return'] = df['Close'].pct_change()
        features['Volatility_20'] = features['Daily_Return'].rolling(window=20).std()

        features['Price_Change_1D'] = df['Close'].pct_change(1)
        features['Price_Change_3D'] = df['Close'].pct_change(3)
        features['Price_Change_5D'] = df['Close'].pct_change(5)
        features['Price_Change_10D'] = df['Close'].pct_change(10)

        features['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        features['Volume_Ratio'] = df['Volume'] / features['Volume_SMA_20']

        features['Target'] = np.where(
            features['Close'].shift(-2) > features['Close'].shift(-1), 1, 0
        )

        features = features.dropna()

        return features

    def train(self, df: pd.DataFrame, use_grid_search: bool = True):
        features_df = self._create_features(df)

        if len(features_df) < 10:
            dummy_X = np.zeros((1, 10))
            dummy_y = np.array([0])
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(dummy_X, dummy_y)
            self.scaler = MinMaxScaler()
            self.scaler.fit(dummy_X)
            self.feature_columns = features_df.columns.tolist()[:-1] if len(features_df.columns) > 1 else ['Close']
            return {
                'accuracy': 0.0,
                'classification_report': 'Insufficient data for training',
                'feature_importance': pd.DataFrame({'Feature': self.feature_columns, 'Importance': [0.0] * len(self.feature_columns)}),
                'best_params': {},
                'n_samples': len(features_df),
                'train_size': 0,
                'test_size': 0,
            }

        self.feature_columns = [col for col in features_df.columns if col != 'Target']
        X = features_df[self.feature_columns].values
        y = features_df['Target'].values

        if len(X) < 10:
            min_split = max(1, int(len(X) * 0.1))
            X_train, X_test = X[:-min_split], X[-min_split:]
            y_train, y_test = y[:-min_split], y[-min_split:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
            )

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            self.model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            self.best_params = {'n_estimators': 200, 'max_depth': 20}

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': feature_importance,
            'best_params': self.best_params,
            'n_samples': len(features_df),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

    def predict_today(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            raise ValueError("Model must be trained first")

        features_df = self._create_features(df)

        if len(features_df) == 0 or len(self.feature_columns) == 0:
            return {
                'prediction': 'HOLD',
                'probability_up': 0.5,
                'probability_down': 0.5,
                'confidence': 0.5,
            }

        try:
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)

            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]

            return {
                'prediction': '📈 UP' if prediction == 1 else '📉 DOWN',
                'probability_up': probability[1],
                'probability_down': probability[0],
                'confidence': max(probability),
            }
        except Exception:
            return {
                'prediction': 'HOLD',
                'probability_up': 0.5,
                'probability_down': 0.5,
                'confidence': 0.5,
            }