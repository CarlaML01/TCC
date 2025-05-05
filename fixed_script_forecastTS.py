import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Load and prepare data
df = pd.read_csv('combined_gaming_data_all_years_cleaned.csv', 
                 parse_dates=['date'], 
                 index_col='date')

# Handle missing values and ensure weekly frequency
df['averageplaytime'] = df['averageplaytime'].ffill().dropna()
df = df.asfreq('W-SUN')

# Convert numeric columns
numeric_cols = ['numberofgames', 'totalrevenue', 'averagerevenue', 
               'medianrevenue', 'averageprice', 'averageplaytime',
               'top25', 'top5', 'bottom30']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.'), errors='coerce')

# Visualization settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 8)

# [Previous visualization code remains the same...]

# Stationarity testing and differencing
def test_stationarity(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'   {k}: {v:.3f}')
    
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original', color='#1f77b4')
    plt.plot(rolling_mean, label='Rolling Mean', color='#d62728')
    plt.plot(rolling_std, label='Rolling Std', color='#2ca02c')
    plt.legend()
    plt.title('Rolling Statistics for Stationarity Check')
    plt.grid(alpha=0.3)
    plt.show()

# Differencing
ts_diff = df['averageplaytime'].diff().dropna()

# SARIMA Modeling
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")

final_model = SARIMAX(
    df['averageplaytime'],
    order=(0, 1, 1),
    seasonal_order=(1, 1, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_results = final_model.fit(maxiter=100, disp=True)
print(final_results.summary())

# Forecasting
forecast = final_results.get_forecast(steps=52)

# Plot results
fig, ax = plt.subplots(figsize=(12,6))
df['averageplaytime'].plot(ax=ax, label='Observed')
forecast.predicted_mean.plot(ax=ax, style='r--', label='Forecast')
ax.fill_between(forecast.conf_int().index,
                forecast.conf_int().iloc[:,0],
                forecast.conf_int().iloc[:,1],
                color='r', alpha=0.1)
ax.set_title('Optimized SARIMA Forecast of Average Play Time')
ax.set_ylabel('Hours')
ax.legend()
plt.grid(alpha=0.3)
plt.show()

# Residual diagnostics
final_results.plot_diagnostics(figsize=(12,8))
plt.tight_layout()
plt.show()

