# ============================================================================
# LAB 8: MA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('AirPassengers.csv')
data = df['Passengers']

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# i. Plot ACF and PACF (already done in Lab 6)

# ii. Fit MA(1) model
ma1_model = ARIMA(train, order=(0, 0, 1)).fit()
ma1_pred = ma1_model.forecast(steps=len(test))
print(f"MA(1) AIC: {ma1_model.aic}")

# iii. Fit higher lag MA model
ma3_model = ARIMA(train, order=(0, 0, 3)).fit()
ma3_pred = ma3_model.forecast(steps=len(test))
print(f"MA(3) AIC: {ma3_model.aic}")

# iv. Compare performances
print("MA(1) Metrics:", calc_metrics(test, ma1_pred))
print("MA(3) Metrics:", calc_metrics(test, ma3_pred))

plt.figure(figsize=(12, 4))
plt.plot(test.values, label='Actual')
plt.plot(ma1_pred.values, label='MA(1)')
plt.plot(ma3_pred.values, label='MA(3)')
plt.legend()
plt.title('MA Model Comparison')
plt.show()
