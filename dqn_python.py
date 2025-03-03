import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

# Step 1: Fetch Data
ticker = "TATAMOTORS.NS"
data = yf.download(ticker, start="2015-01-01", end="2025-03-03", auto_adjust=False)
time.sleep(2)  # Ensure latest data
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
data = data[['Close']]
data.dropna(inplace=True)
print("Sample Close Prices:", data['Close'].head())  # Verify prices
print("Latest Close Price:", data['Close'].iloc[-1])  # Verify latest

# Adjust prices based on current market price (138.57 INR as of March 3, 2025)
current_market_price = 138.57  # Update with exact value if needed
adjustment_factor = current_market_price / data['Close'].iloc[-1]
data['Close'] = data['Close'] * adjustment_factor
print(f"Adjustment Factor: {adjustment_factor:.2f}")
print("Adjusted Close Prices:", data['Close'].head())
print("Latest Adjusted Close Price:", data['Close'].iloc[-1])

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

# Step 3: Prepare Data for XGBoost
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

# Evaluate XGBoost
mse = mean_squared_error(y_test, xgb_preds)
mae = mean_absolute_error(y_test, xgb_preds)
r2 = r2_score(y_test, xgb_preds)
print(f"XGBoost Mean Squared Error (MSE): {mse:.2f}")
print(f"XGBoost Mean Absolute Error (MAE): {mae:.2f}")
print(f"XGBoost R-squared (R2): {r2:.2f}")

# Predict next day (March 4, 2025)
last_features = X_test.iloc[-1].values.reshape(1, -1)  # Last test set features
next_day_pred = xgb_model.predict(last_features)[0]
last_actual = y_test.iloc[-1]  # Last actual price from test set
print(f"Predicted Price for Next Day (March 4, 2025): {next_day_pred:.2f} INR")
print(f"Last Actual Price (March 3, 2025): {last_actual:.2f} INR")

# Step 5: DQN Model with gym_anytrading
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create and train DQN environment
env = DummyVecEnv([lambda: StocksEnv(df=train_data, frame_bound=(10, len(train_data)), window_size=10)])
model_dqn = DQN("MlpPolicy", env, verbose=1)
model_dqn.learn(total_timesteps=10000)

# Test DQN on the test set
test_env = DummyVecEnv([lambda: StocksEnv(df=test_data, frame_bound=(10, len(test_data)), window_size=10)])
obs = test_env.reset()
dqn_profits = []
done = False
while not done:
    action, _states = model_dqn.predict(obs)
    obs, reward, done, info = test_env.step(action)
    dqn_profits.append(info[0]['total_profit'])
    done = done[0]

# Step 6: Plot Results
plt.figure(figsize=(12, 6))

# Subplot 1: XGBoost Predictions vs Actual Prices
plt.subplot(2, 1, 1)
plt.plot(y_test.values, label="Actual Prices", color='black')
plt.plot(xgb_preds, label="XGBoost Predictions", linestyle='dashed')
plt.legend()
plt.title("XGBoost: Stock Price Prediction (Tata Steel)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price (INR)")

# Subplot 2: DQN Cumulative Profit
plt.subplot(2, 1, 2)
plt.plot(dqn_profits, label="DQN Total Profit", color='blue')
plt.legend()
plt.title("DQN: Cumulative Trading Profit (Tata Steel)")
plt.xlabel("Time Step")
plt.ylabel("Profit (INR)")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("stock_prediction_with_dqn.png")
print("Plot saved as 'stock_prediction_with_dqn.png'")

print("Stock Price Prediction Completed (XGBoost + DQN)!")