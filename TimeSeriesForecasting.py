import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Example data
date_rng = pd.date_range(start='2015-01-01', end='2024-12-01', freq='M')
data = pd.Series(range(len(date_rng)), index=date_rng)

# Fit ARIMA model (p,d,q = (1,1,1) just as example)
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(data, label="History")
plt.plot(forecast_series, label="Forecast", color="red")
plt.legend()
plt.show()

# Facebook Prophet (Great for Trend + Seasonality)
from prophet import Prophet

# Create dataframe in Prophet format
df = data.reset_index()
df.columns = ["ds", "y"]

# Fit Prophet model
model = Prophet()
model.fit(df)

# Future 12 months
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.show()
