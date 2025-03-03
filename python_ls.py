import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import gym
from stable_baselines3 import DQN
import os

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
    raise ValueError(f"Missing columns: {missing_columns}")

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

# Step 5: Reinforcement Learning (DQN) Setup (Commented Out)
# try:
#     env = gym.make("stocks-v0", df=data, frame_bound=(50, len(data)), window_size=10)
#     model_dqn = DQN("MlpPolicy", env, verbose=1)
#     model_dqn.learn(total_timesteps=10000)
# except Exception as e:
#     print(f"DQN failed: {e}. Skipping RL step.")

# Step 6: Plot Results (XGBoost only)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Prices", color='black')
plt.plot(xgb_preds, label="XGBoost Predictions", linestyle='dashed')
plt.legend()
plt.title("Stock Price Prediction (Tata Steel) - XGBoost Only")
plt.xlabel("Time Step")
plt.ylabel("Stock Price (INR)")
plt.savefig("stock_prediction.png")
print("Plot saved as 'stock_prediction.png'")

print("Stock Price Prediction Completed (without LSTM)!")