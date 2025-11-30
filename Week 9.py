# ============================================================================
# LAB 9: ARMA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# i. Initialize ARMA model
arma_model = ARIMA(train, order=(1, 0, 1))

# ii. Train the model
arma_fit = arma_model.fit()
print(f"ARMA(1,1) Summary:\n{arma_fit.summary()}")

# iii. Generate forecasts
arma_pred = arma_fit.forecast(steps=len(test))
print("ARMA Metrics:", calc_metrics(test, arma_pred))

plt.plot(test.values, label='Actual')
plt.plot(arma_pred.values, label='ARMA(1,1)')
plt.legend()
plt.title('ARMA Model Forecast')
plt.show()