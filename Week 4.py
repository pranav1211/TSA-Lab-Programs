# ============================================================================
# LAB 4: FORECASTING TECHNIQUES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('AirPassengers.csv')
data = df['Passengers']

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# i. Different forecasting techniques

# Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train).fit()
ses_pred = ses_model.forecast(len(test))

# Simple Moving Average (SMA)
window = 3
sma_pred = train.rolling(window=window).mean().iloc[-1]
sma_forecast = pd.Series([sma_pred] * len(test), index=test.index)

# Holt-Winters Smoothing
hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()
hw_pred = hw_model.forecast(len(test))

# ii. Calculate evaluation metrics
def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

print("SES Metrics:", calc_metrics(test, ses_pred))
print("SMA Metrics:", calc_metrics(test, sma_forecast))
print("HW Metrics:", calc_metrics(test, hw_pred))

# iii. Identify trends and seasonal patterns
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(test.index, ses_pred, label='SES', marker='x')
plt.plot(test.index, hw_pred, label='Holt-Winters', marker='s')
plt.legend()
plt.show()
