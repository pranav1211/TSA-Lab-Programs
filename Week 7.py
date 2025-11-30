#============================================================================
# LAB 7: AR MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

df = pd.read_csv('AirPassengers.csv')
data = df['Passengers']

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# ii. Fit AR(1) model
ar1_model = AutoReg(train, lags=1).fit()
ar1_pred = ar1_model.predict(start=len(train), end=len(train)+len(test)-1)
print(f"AR(1) AIC: {ar1_model.aic}")

# iii. Fit higher lag AR models
ar3_model = AutoReg(train, lags=3).fit()
ar3_pred = ar3_model.predict(start=len(train), end=len(train)+len(test)-1)
print(f"AR(3) AIC: {ar3_model.aic}")

plt.plot(test.values, label='Actual')
plt.plot(ar1_pred, label='AR(1)')
plt.plot(ar3_pred, label='AR(3)')
plt.legend()
plt.title('AR Model Comparison')
plt.show()