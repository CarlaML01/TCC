import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load all datasets
file_path_2021 = 'C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/2021_per_week.csv'
file_path_2022 = 'C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/2022_per_week.csv'
file_path_2023 = 'C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/2023_per_week.csv'
file_path_2024 = 'C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/2024_per_week_updated_with_month_name.csv'

# Read the CSV files
data_2021 = pd.read_csv(file_path_2021, delimiter=';')
data_2022 = pd.read_csv(file_path_2022, delimiter=';')
data_2023 = pd.read_csv(file_path_2023, delimiter=';')
data_2024 = pd.read_csv(file_path_2024, delimiter=',')

# Function to clean and process each dataset
def clean_dataset(df, year):
    # Drop rows with missing or invalid data (e.g., rows with ';;;;;;;;;')
    df = df.replace(';;;;;;;;;', np.nan).dropna()

    # Convert 'totalRevenue' to numeric (replace commas with periods for decimal separators)
    df['totalRevenue'] = pd.to_numeric(df['totalRevenue'].str.replace(',', '.'), errors='coerce')

    # Add a 'year' column
    df['year'] = year

    # Add a 'date' column by calculating the start of each week
    start_date = pd.to_datetime(f"{year}-01-01")
    df['date'] = start_date + pd.to_timedelta((df['weekNumber'] - 1) * 7, unit='D')

    # Add a 'month' column based on the 'date' column
    df['month'] = df['date'].dt.month

    # Add a 'month_name' column based on the 'date' column
    df['month_name'] = df['date'].dt.month_name()

    return df

# Clean and process each dataset
data_2021 = clean_dataset(data_2021, 2021)
data_2022 = clean_dataset(data_2022, 2022)
data_2023 = clean_dataset(data_2023, 2023)

# For 2024, the columns are already added, so no need to process again

# Combine all datasets into one
data = pd.concat([data_2021, data_2022, data_2023, data_2024], ignore_index=True)

# Convert 'date' column to datetime and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Set the frequency of the date index to weekly
data = data.asfreq('W')

# Ensure 'totalRevenue' is numeric (in case of commas as decimal separators)
data['totalRevenue'] = pd.to_numeric(data['totalRevenue'].str.replace(',', '.'), errors='coerce')

# Scale totalRevenue to millions
data['totalRevenue'] = data['totalRevenue'] / 1_000_000

# Drop rows with missing values in 'totalRevenue'
data.dropna(subset=['totalRevenue'], inplace=True)

# Print the date range to verify
print("Date range in the dataset:", data.index.min(), "to", data.index.max())

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data['totalRevenue'], label='Actual')
plt.title('Total Revenue Over Time (2021-2024)')
plt.xlabel('Date')
plt.ylabel('Total Revenue (Millions)')
plt.legend()
plt.show()

# Check for stationarity using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("Series is not stationary. Differencing may be required.")
    else:
        print("Series is stationary.")

check_stationarity(data['totalRevenue'])

# Calculate the maximum number of lags (50% of the sample size)
max_lags = int(len(data) * 0.5)
print("Maximum number of lags allowed:", max_lags)

# Plot ACF and PACF to identify ARIMA parameters
plot_acf(data['totalRevenue'].dropna(), lags=max_lags)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['totalRevenue'].dropna(), lags=max_lags)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Manually specify ARIMA parameters (p, d, q)
order = (1, 1, 1)  # Non-seasonal order (p, d, q)

# Fit the ARIMA model (without seasonal component)
model = SARIMAX(data['totalRevenue'], order=order)
model_fit = model.fit(disp=False)

# Print model summary
print(model_fit.summary())

# Plot residuals to check for patterns
residuals = model_fit.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()

# Plot ACF/PACF of residuals to ensure no significant patterns remain
plot_acf(residuals.dropna(), lags=max_lags)
plt.title('ACF of Residuals')
plt.show()

plot_pacf(residuals.dropna(), lags=max_lags)
plt.title('PACF of Residuals')
plt.show()

# Forecast future values
forecast_steps = 12  # Forecast the next 12 weeks
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create a date range for the forecast
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')

# Plot the forecast alongside the actual data
plt.figure(figsize=(14, 7))
plt.plot(data['totalRevenue'], label='Actual (2021-2024)')
plt.plot(forecast_dates, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast (2021-2024)')
plt.xlabel('Date')
plt.ylabel('Total Revenue (Millions)')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model using Mean Squared Error (MSE)
train_size = int(len(data) * 0.8)
train, test = data['totalRevenue'][:train_size], data['totalRevenue'][train_size:]

# Fit the model on the training data
model_train = SARIMAX(train, order=order)
model_train_fit = model_train.fit(disp=False)

# Forecast on the test data
test_forecast = model_train_fit.get_forecast(steps=len(test))
test_forecast_mean = test_forecast.predicted_mean

# Calculate MSE
mse = mean_squared_error(test, test_forecast_mean)
print(f"Mean Squared Error (MSE) on Test Data: {mse}")