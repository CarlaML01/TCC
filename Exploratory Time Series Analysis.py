import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the combined dataset (saved from previous steps)
df = pd.read_csv('combined_gaming_data_all_years_cleaned.csv', parse_dates=['date'], index_col='date')

# Handle missing values - forward fill then drop remaining
df['averageplaytime'] = df['averageplaytime'].ffill().dropna()

# Ensure weekly frequency explicitly
df = df.asfreq('W-SUN')

# Check structure
print(df.info())
print(df.head())
# 2. Fix numeric columns that were read as objects
numeric_cols = ['numberofgames', 'totalrevenue', 'averagerevenue', 
               'medianrevenue', 'averageprice', 'averageplaytime',
               'top25', 'top5', 'bottom30']
for col in numeric_cols:
    if col in df.columns:
        # Handle European decimals and empty strings
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.'), errors='coerce')
        
#Visualizing Key Metrics Over Time
##We'll plot the core metrics to identify trends, seasonality, and anomalies.
# 3. Set up plotting (using available style)
available_styles = plt.style.available
print("Available styles:", available_styles)  # Check what's actually available

# Use a style that exists (commonly available ones: 'ggplot', 'seaborn', 'seaborn-darkgrid')
plt.style.use('seaborn-v0_8')  # Newer matplotlib versions use this
plt.rcParams['figure.figsize'] = (14, 8)

# 4. Plot key metrics with proper error handling
def safe_plot(series, title, ylabel):
    try:
        series.plot(title=title)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Could not plot {title}: {str(e)}")

# Plot metrics (only for columns with data)
if 'totalrevenue' in df.columns:
    safe_plot(df['totalrevenue'], 'Weekly Total Revenue', 'Revenue (Millions)')
    
if 'averageplaytime' in df.columns:
    safe_plot(df['averageplaytime'], 'Weekly Average Play Time', 'Hours')

if 'numberofgames' in df.columns:
    safe_plot(df['numberofgames'], 'Weekly Number of Games', 'Count')

# 5. Additional visualizations
# Monthly patterns
if 'month_name' in df.columns and 'averageplaytime' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='month_name', y='averageplaytime', 
                order=['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December'])
    plt.title('Monthly Distribution of Play Time')
    plt.xticks(rotation=45)
    plt.show()

# Yearly comparison
if 'year' in df.columns and 'averageplaytime' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=df.index.month, y='averageplaytime', hue='year')
    plt.title('Year-over-Year Comparison of Play Time')
    plt.xlabel('Month')
    plt.ylabel('Average Play Time (Hours)')
    plt.show()
    
# another color 
# Custom color palette for better contrast
yearly_palette = {
    2021: '#1f77b4',  # Muted blue
    2022: '#ff7f0e',  # Safety orange
    2023: '#2ca02c',  # Cooked asparagus green
    2024: '#d62728'   # Brick red
}

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df, 
    x=df.index.month, 
    y='averageplaytime', 
    hue='year',
    palette=yearly_palette,
    linewidth=2.5
)

plt.title('Year-over-Year Comparison of Average Play Time', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Play Time (Hours)', fontsize=12)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Select the target metric - we'll use average play time
ts_data = df['averageplaytime'].dropna()

# Multiplicative decomposition (better for data where seasonality grows with trend)
decomposition = seasonal_decompose(
    ts_data, 
    model='multiplicative', 
    period=52,  # Weekly data (52 weeks in a year)
    extrapolate_trend='freq'  # Handle edge effects
)

# Plot decomposition components with custom styling
plt.rcParams.update({'font.size': 12})
fig = decomposition.plot()
fig.set_size_inches(12, 8)
fig.suptitle('Time Series Decomposition of Average Play Time', y=1.02)
plt.tight_layout()
plt.show()

# Alternatively, plot components separately with enhanced visuals
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))

# Original Series
ax1.plot(ts_data, color='#1f77b4')
ax1.set_ylabel('Original')
ax1.grid(True, alpha=0.3)

# Trend Component
ax2.plot(decomposition.trend, color='#d62728')
ax2.set_ylabel('Trend')
ax2.grid(True, alpha=0.3)

# Seasonal Component
ax3.plot(decomposition.seasonal, color='#2ca02c')
ax3.set_ylabel('Seasonal')
ax3.grid(True, alpha=0.3)

# Residual Component
ax4.plot(decomposition.resid, color='#9467bd')
ax4.set_ylabel('Residual')
ax4.grid(True, alpha=0.3)

plt.suptitle('Detailed Decomposition of Average Play Time', y=1.02)
plt.tight_layout()
plt.show()

#Stationarity Testing
##To prepare for ARIMA modeling, we’ll test if the time series is stationary (mean and variance constant over time).

#Augmented Dickey-Fuller (ADF) Test
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series):
    # ADF Test
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    print('Critical Values:')
    for k, v in result[4].items():
        print(f'   {k}: {v:.3f}')
    
    # Plot rolling statistics
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

# Test for Average Play Time
test_stationarity(df['averageplaytime'])

## If p-value ≤ 0.05: Series is stationary (reject null hypothesis of non-stationarity).

## If p-value > 0.05: Differencing is needed.

#ADF Statistic: -2.453
#p-value: 0.127
#Critical Values:
#   1%: -3.464
#   5%: -2.876
#   10%: -2.575

#Next Steps: Making the Series Stationary
## First-Order Differencing
# Apply differencing
ts_diff = df['averageplaytime'].diff().dropna()

# Re-test stationarity
test_stationarity(ts_diff)

# Plot differenced series
plt.figure(figsize=(12, 6))
plt.plot(ts_diff, color='#1f77b4')
plt.title('First-Order Differenced Series')
plt.ylabel('Difference in Play Time (Hours)')
plt.grid(alpha=0.3)
plt.show()

#ADF Statistic: -10.860
#p-value: 0.000
#Critical Values:
 #  1%: -3.464
 #  5%: -2.876
 #  10%: -2.575
 
#Next Steps: Model Building
#ARIMA/SARIMA Parameter Selection
#using ACF/PACF plots on the differenced series to identify (p,d,q) and seasonal (P,D,Q,s)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts_diff, lags=40, ax=ax1, color='#1f77b4', title='ACF of Differenced Series')
plot_pacf(ts_diff, lags=40, ax=ax2, color='#d62728', title='PACF of Differenced Series')
plt.tight_layout()
plt.show()

#How to Interpret:

##ACF: Cuts off after lag q → MA term.

##PACF: Cuts off after lag p → AR term.

##Seasonal Peaks at lag 52 → SAR/SMA terms.

#SARIMA modeling
# Optimized model based on your diagnostics
model = SARIMAX(
    df['averageplaytime'],
    order=(0, 1, 1),            # Only MA(1) term
    seasonal_order=(1, 1, 0, 52), # Only SAR(1) term
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit with robust settings
results = model.fit(
    maxiter=100,
    method='powell',  # More stable than default
    disp=True
)

print(results.summary())

#Try Auto-ARIMA
#from pmdarima import auto_arima
#auto_model = auto_arima(
#    df['averageplaytime'],
#    seasonal=True, m=52,
#    stepwise=True, trace=True
#)
#print(auto_model.summary())

#since auto-arima confirms our initial model. removing insignificant terms (AR-L1 and SMA-L52). 
# trying SARIMA(0,1,1)(1,1,0,52)
#from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optimized model keeping only significant terms
#final_model = SARIMAX(
#    df['averageplaytime'],
#    order=(0, 1, 1),            # Only MA(1) term (highly significant)
#    seasonal_order=(1, 1, 0, 52), # Only SAR(1) term (highly significant)
#    enforce_stationarity=False,
#    enforce_invertibility=False
#)


import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress only the frequency warning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")

# Now run your model fitting

# Fit model with increased iterations
final_results = final_model.fit(
    maxiter=100,
    disp=True,
    method='nm'  # Nelder-Mead optimization
)

print(final_results.summary())

#Model Validation and forecasting
import matplotlib.pyplot as plt

# Check residuals
results.plot_diagnostics(figsize=(12,8))
plt.show()

# Forecast
# Plot forecasts
fig, ax = plt.subplots(figsize=(12,6))
df['averageplaytime'].plot(ax=ax, label='Observed')
forecast.predicted_mean.plot(ax=ax, style='r--', label='Forecast')
ax.fill_between(forecast.conf_int().index,
                forecast.conf_int().iloc[:,0],
                forecast.conf_int().iloc[:,1],
                color='r', alpha=0.1)
ax.set_title('Optimized SARIMA Forecast')
ax.legend()
plt.show()
