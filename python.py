import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import gym
from stable_baselines3 import DQN
import os

# Fix GPU configuration
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"  # Adjust path if needed

# Step 1: Fetch Data
ticker = "TATASTEEL.NS"
data = yf.download(ticker, start="2015-01-01", end="2024-02-29")
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
data = data[['Close']]
data.dropna(inplace=True)

# Step 2: Feature Engineering
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
print("Columns before dropping NaN:", data.columns)
print("Number of NaN values before drop:\n", data.isna().sum())
data.dropna(inplace=True)
print("Columns after dropping NaN:", data.columns)

# Step 3: Prepare Data for ML
expected_columns = ['SMA_10', 'SMA_50', 'RSI']
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}. Check feature engineering steps.")

X = data[['SMA_10', 'SMA_50', 'RSI']]
y = data['Close'].shift(-1)
data_cleaned = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)
X = data_cleaned[['SMA_10', 'SMA_50', 'RSI']]
y = data_cleaned['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Step 5: LSTM Model
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])
X_lstm, y_lstm = [], []
for i in range(60, len(data_scaled)):
    X_lstm.append(data_scaled[i-60:i])
    y_lstm.append(data_scaled[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

model_lstm = Sequential([
    Input(shape=(60, 1)),
    LSTM(50, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='tanh'),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=1)
lstm_preds = model_lstm.predict(X_test_lstm)

# Step 6: Reinforcement Learning (DQN) Setup
try:
    env = gym.make("stocks-v0", df=data, frame_bound=(50, len(data)), window_size=10)
    model_dqn = DQN("MlpPolicy", env, verbose=1)
    model_dqn.learn(total_timesteps=10000)
except Exception as e:
    print(f"DQN failed: {e}. Skipping RL step.")

# Step 7: Plot Results
lstm_preds_unscaled = scaler.inverse_transform(lstm_preds.reshape(-1, 1))
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Prices", color='black')
plt.plot(xgb_preds, label="XGBoost Predictions", linestyle='dashed')
plt.plot(lstm_preds_unscaled, label="LSTM Predictions", linestyle='dotted')
plt.legend()
plt.title("Stock Price Prediction (Tata Steel)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price (INR)")
plt.show()

print("Stock Price Prediction Completed!")