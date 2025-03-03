import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

# Step 1: Fetch Data with Correct Prices
ticker = "TATAMOTORS.NS"
data = yf.download(ticker, start="2015-01-01", end="2025-03-03", auto_adjust=True)  # ✅ Auto Adjust ON
time.sleep(2)  # Ensure latest data is fetched

# Keep only the 'Close' prices
data = data[['Close']]
data.dropna(inplace=True)

# ✅ Print first & last prices to verify correctness
print("Sample Close Prices:\n", data.head())
print("Latest Close Price:", data['Close'].iloc[-1])  # ✅ Should match Yahoo Finance

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

# Remove NaN values caused by rolling windows
data.dropna(inplace=True)

# Step 3: Prepare Data for XGBoost
X = data[['SMA_10', 'SMA_50', 'RSI']]
y = data['Close'].shift(-1)  # Predict next day's price
data_cleaned = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)
X = data_cleaned[['SMA_10', 'SMA_50', 'RSI']]
y = data_cleaned['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, xgb_preds)
mae = mean_absolute_error(y_test, xgb_preds)
r2 = r2_score(y_test, xgb_preds)
print(f"✅ XGBoost MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# Predict Next Day Price
last_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_pred = xgb_model.predict(last_features)[0]
print(f"🎯 Predicted Price for Next Day (March 4, 2025): {next_day_pred:.2f} INR")
