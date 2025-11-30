# ============================================================================
# LAB 6: TREND DETECTION AND ACF/PACF
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('AirPassengers.csv')
data = df['Passengers']

# i. Detect trends using moving averages
ma_trend = data.rolling(window=12).mean()
plt.plot(data, label='Original')
plt.plot(ma_trend, label='Trend (MA-12)', linewidth=2)
plt.show()

# ii. Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data.dropna(), lags=40, ax=axes[0])
plot_pacf(data.dropna(), lags=40, ax=axes[1])
plt.show()