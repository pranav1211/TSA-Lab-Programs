# ============================================================================
# LAB 5: WHITE NOISE AND STATIONARITY
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

df = pd.read_csv('AirPassengers.csv')
data = df['Passengers']

# i. Generate white noise
white_noise = np.random.normal(0, 1, len(data))

# ii. Compare graphs
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(white_noise)
axes[0].set_title('White Noise')
axes[1].plot(data)
axes[1].set_title('Time Series Data')
plt.show()

# iii. Statistical tests for stationarity
# Augmented Dickey-Fuller Test
adf_result = adfuller(data)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Stationary: {adf_result[1] < 0.05}")

# KPSS Test
kpss_result = kpss(data)
print(f"\nKPSS Statistic: {kpss_result[0]}")
print(f"p-value: {kpss_result[1]}")