# ============================================================================
# LAB 10: ARIMA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# i. Initialize ARIMA model with p, d, q parameters
# p=1 (AR terms), d=1 (differencing), q=1 (MA terms)
arima_model = ARIMA(train, order=(1, 1, 1))

# ii. Train the model
arima_fit = arima_model.fit()
print(f"ARIMA(1,1,1) Summary:\n{arima_fit.summary()}")

# iii. Generate forecasts
arima_pred = arima_fit.forecast(steps=len(test))
print("ARIMA Metrics:", calc_metrics(test, arima_pred))

plt.plot(test.values, label='Actual')
plt.plot(arima_pred.values, label='ARIMA(1,1,1)')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.show()

