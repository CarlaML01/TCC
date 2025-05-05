import pandas as pd
import numpy as np
from pathlib import Path

# Load all datasets with proper error handling
data_dir = Path('C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/')

def load_and_clean_dataset(file_path, year):
    try:
        # Read CSV with proper delimiter detection
        df = pd.read_csv(file_path, delimiter=';' if '202' in file_path.name else ',')
        
        # Add year column
        df['year'] = year
        
        # Standardize column names (convert to lowercase and handle variations)
        df.columns = df.columns.str.lower()
        
        # Handle specific column name variations
        column_mapping = {
            'weeknumber': 'weeknumber',
            'weeknum': 'weeknumber',
            'week_number': 'weeknumber',
            'week': 'weeknumber',
            'numberofgames': 'numberofgames',
            'totalrevenue': 'totalrevenue',
            'averagerevenue': 'averagerevenue',
            'medianrevenue': 'medianrevenue',
            'averageprice': 'averageprice',
            'averageplaytime': 'averageplaytime',
            'top25': 'top25',
            'top5': 'top5',
            'bottom30': 'bottom30'
        }
        
        # Rename columns according to our standard
        df = df.rename(columns={k.lower(): v for k, v in column_mapping.items() 
                               if k.lower() in df.columns.str.lower()})
        
        return df
    except Exception as e:
        print(f"Error loading {file_path.name}: {str(e)}")
        return None

data_2021 = load_and_clean_dataset(data_dir / '2021_per_week.csv', 2021)
data_2022 = load_and_clean_dataset(data_dir / '2022_per_week.csv', 2022)
data_2023 = load_and_clean_dataset(data_dir / '2023_per_week.csv', 2023)
data_2024 = load_and_clean_dataset(data_dir / '2024_per_week.csv', 2024)

# Function to safely convert numeric columns
def convert_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            # First replace 'undefined' with NaN
            df[col] = df[col].replace('undefined', np.nan)
            df[col] = df[col].replace('', np.nan)  # Also handle empty strings
            
            # Then handle European decimal format if needed
            if df[col].dtype == object:
                try:
                    df[col] = df[col].str.replace(',', '.').astype(float)
                except AttributeError:
                    # If not a string type
                    df[col] = df[col].astype(float)
    return df

# Process each dataframe
def process_dataframe(df, year):
    if df is None:
        return None
    
    # Convert numeric columns safely
    numeric_cols = ['totalrevenue', 'averagerevenue', 'medianrevenue', 
                   'averageprice', 'averageplaytime', 'top25', 'top5', 'bottom30']
    df = convert_numeric(df, numeric_cols)
    
    # Ensure weeknumber column exists (create if missing)
    if 'weeknumber' not in df.columns:
        if 'date' in df.columns:
            # Extract week number from date if available
            df['weeknumber'] = df['date'].dt.isocalendar().week
        else:
            # Create sequential weeks if no other info available
            df['weeknumber'] = np.arange(1, len(df)+1)
    
    # Create date from week number and year
    df['date'] = pd.to_datetime(df['year'].astype(str)) + pd.to_timedelta((df['weeknumber'] - 1) * 7, unit='D')
    
    # Add month and month_name if not present
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    if 'month_name' not in df.columns:
        df['month_name'] = df['date'].dt.month_name()
    
    return df

# Process all dataframes
data_2021 = process_dataframe(data_2021, 2021)
data_2022 = process_dataframe(data_2022, 2022)
data_2023 = process_dataframe(data_2023, 2023)
data_2024 = process_dataframe(data_2024, 2024)

# Combine all valid dataframes
valid_dfs = [df for df in [data_2021, data_2022, data_2023, data_2024] if df is not None]
combined_data = pd.concat(valid_dfs, ignore_index=True)

# Sort by date and set as index
combined_data = combined_data.sort_values('date').reset_index(drop=True)
combined_data.set_index('date', inplace=True)

# Convert to weekly frequency (forward fill missing weeks)
combined_data = combined_data.asfreq('W', method='ffill')

# Scale totalRevenue to millions
if 'totalrevenue' in combined_data.columns:
    combined_data['totalrevenue'] = combined_data['totalrevenue'] / 1_000_000

# Save the final combined dataset
output_path = data_dir / 'combined_gaming_data_all_years_cleaned.csv'
combined_data.to_csv(output_path)

print(f"Combined dataset saved to {output_path}")
print("\nFinal dataset info:")
print(combined_data.info())
print("\nFirst 5 rows:")
print(combined_data.head())
print("\nLast 5 rows:")
print(combined_data.tail())

# Set up plotting style
plt.style.use('seaborn')
plt.figure(figsize=(15, 20))

# Plot key metrics over time
metrics = ['numberOfGames', 'totalRevenue', 'averageRevenue', 'averagePlayTime', 'top25', 'bottom30']
titles = ['Number of Games', 'Total Revenue', 'Average Revenue', 'Average Play Time', 'Top 25 Players', 'Bottom 30 Players']

for i, metric in enumerate(metrics):
    plt.subplot(len(metrics), 1, i+1)
    plt.plot(df_all['date'], df_all[metric])
    plt.title(f'{titles[i]} Over Time')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.grid(True)

plt.tight_layout()
plt.show()



