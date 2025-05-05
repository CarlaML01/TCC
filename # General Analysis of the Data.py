# General Analysis of the Data
#1. Time Series Analysis (Trend Analysis). Let’s explore the trends over time using graphs, focusing on key features:

import matplotlib.pyplot as plt
import pandas as pd


file_path = 'C:/Users/lmota/OneDrive/Mac + old laptop/Previous files/Documentos/TCC MBA/scripts tcc/2024_per_week_updated_with_month_name.csv'

data = pd.read_csv(file_path)

data.describe()


# Plot total revenue over time (week number)
plt.figure(figsize=(14, 8))
plt.plot(data['weekNumber'], data['totalRevenue'], marker='o', linestyle='-', color='blue')
plt.title('Total Revenue Over Time (Weekly)', fontsize=16)
plt.xlabel('Week Number', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Plot average price over time (week number)
plt.figure(figsize=(14, 8))
plt.plot(data['weekNumber'], data['averagePrice'], marker='o', linestyle='-', color='green')
plt.title('Average Price Over Time (Weekly)', fontsize=16)
plt.xlabel('Week Number', fontsize=14)
plt.ylabel('Average Price', fontsize=14)
plt.grid(True)
plt.show()

#2. Revenue and Other Variables by Month. Since you now have the month_name and month columns, we can compare how revenue and other key variables change by month:

# Define the custom order for months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert the month_name column to a categorical type with custom order
data['month_name'] = pd.Categorical(data['month_name'], categories=month_order, ordered=True)

# Aggregating by month without sorting the results by totalRevenue
monthly_data = data.groupby('month_name').agg({
    'totalRevenue': 'sum',
    'numberOfGames': 'sum',
    'averagePrice': 'mean',
    'averagePlayTime': 'mean'
})

# Plot total revenue by month
monthly_data['totalRevenue'].plot(kind='bar', figsize=(14, 8), color='purple')
plt.title('Total Revenue by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()

#3. Analyzing Price and Revenue Correlation. To better understand how price and revenue correlate, you can plot a scatter plot:
# Scatter plot: Average Price vs Total Revenue
plt.figure(figsize=(10, 6))
plt.scatter(data['averagePrice'], data['totalRevenue'], color='orange')
plt.title('Average Price vs Total Revenue', fontsize=16)
plt.xlabel('Average Price', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

#4. Correlation Matrix. We can also generate a correlation matrix to see how strongly different variables are related:

from scipy.stats import spearmanr

# Create a DataFrame (replace 'data' with your actual DataFrame name)
# Assuming your 'data' DataFrame is already loaded

# List of columns to calculate Spearman correlation
columns = ['numberOfGames', 'totalRevenue', 'averageRevenue', 'averagePrice', 'averagePlayTime', 
           'top25', 'top5', 'bottom30', 'month', 'weekNumber']

# Calculate Spearman correlation matrix
spearman_corr, _ = spearmanr(data[columns])

# Convert Spearman correlation matrix to DataFrame for better readability
spearman_corr_df = pd.DataFrame(spearman_corr, columns=columns, index=columns)

# Plot the Spearman correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr_df, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Spearman Correlation Matrix', fontsize=16)
plt.show()

#pearson cor for linear relationships
# Example of Pearson correlation for two variables
pearson_corr, p_value = pearsonr(data['averagePlayTime'], data['totalRevenue'])

print(f"Pearson correlation: {pearson_corr}")
print(f"P-value: {p_value}")



#5. dentifying Outliers. For revenue and price analysis, detecting outliers is critical:
# Checking for outliers in total revenue by month_name
plt.figure(figsize=(12, 6))
sns.boxplot(x='month_name', y='totalRevenue', data=data)
plt.title('Outliers in Total Revenue by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()


#6. Analyzing Top Performers. Finally, we can look into how the top 25% of games (based on total revenue) behave:

# Extract the top 25% games based on revenue
top_25_percent = data[data['top25'] >= data['top25'].quantile(0.75)]

# Plot total revenue for top 25%
plt.figure(figsize=(14, 8))
plt.plot(top_25_percent['weekNumber'], top_25_percent['totalRevenue'], marker='o', linestyle='-', color='red')
plt.title('Top 25% Games: Total Revenue Over Time', fontsize=16)
plt.xlabel('Week Number', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

#7. Comparing the Bottom 30% Performers Based on Revenue or Number of Games Over Time. To identify and compare the bottom 30% performers, we can calculate the threshold for the bottom 30% based on the totalRevenue and numberOfGames, and then filter the data. Then, we will plot their performance over time.
# Calculate the threshold for the bottom 30% performers based on totalRevenue and numberOfGames
bottom_30_revenue_threshold = data['totalRevenue'].quantile(0.30)
bottom_30_games_threshold = data['numberOfGames'].quantile(0.30)

# Filter for bottom 30% performers by revenue
bottom_30_revenue = data[data['totalRevenue'] <= bottom_30_revenue_threshold]

# Filter for bottom 30% performers by numberOfGames
bottom_30_games = data[data['numberOfGames'] <= bottom_30_games_threshold]

# Plot bottom 30% performers (by revenue) over time
plt.figure(figsize=(14, 8))
plt.plot(bottom_30_revenue['date'], bottom_30_revenue['totalRevenue'], marker='o', linestyle='-', color='blue')
plt.title('Bottom 30% Performers (Revenue) Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()

# Plot bottom 30% performers (by number of games) over time
plt.figure(figsize=(14, 8))
plt.plot(bottom_30_games['date'], bottom_30_games['totalRevenue'], marker='o', linestyle='-', color='green')
plt.title('Bottom 30% Performers (Games) Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()

#8. Comparing Pricing Strategies (using averagePrice) with totalRevenue Over Time. This analysis will help us understand if there is any correlation between the average price of games and the total revenue over time. We can plot the two variables together.
# Plot averagePrice vs. totalRevenue over time
plt.figure(figsize=(14, 8))
plt.plot(data['date'], data['averagePrice'], marker='o', linestyle='-', label='Average Price', color='orange')
plt.plot(data['date'], data['totalRevenue'], marker='x', linestyle='-', label='Total Revenue', color='purple')
plt.title('Average Price vs. Total Revenue Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.show()

# If totalRevenue has much higher values than averagePrice, the averagePrice line might appear almost flat because it is being overshadowed by the larger values of totalRevenue. You can try plotting them on different y-axes to better visualize both variables:
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(data['date'], data['averagePrice'], marker='o', linestyle='-', label='Average Price', color='orange')
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Average Price', fontsize=14, color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

ax2 = ax1.twinx()  # Create a second y-axis
ax2.plot(data['date'], data['totalRevenue'], marker='x', linestyle='-', label='Total Revenue', color='purple')
ax2.set_ylabel('Total Revenue', fontsize=14, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title('Average Price vs. Total Revenue Over Time', fontsize=16)
plt.grid(True)

# Tilt the date labels
plt.xticks(rotation=45)

# Ensure that the x-axis is using the correct date format
fig.autofmt_xdate()

plt.show()


#9. Examining Game Quantity (numberOfGames) Against Revenue to See If More Games Correlate with Higher Revenue. This analysis helps us determine whether there is a positive correlation between the number of games and total revenue. A scatter plot can show us this relationship.
# Scatter plot to examine the correlation between number of games and total revenue
plt.figure(figsize=(14, 8))
plt.scatter(data['numberOfGames'], data['totalRevenue'], color='blue')
plt.title('Number of Games vs. Total Revenue', fontsize=16)
plt.xlabel('Number of Games', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Calculate the correlation between numberOfGames and totalRevenue
correlation = data['numberOfGames'].corr(data['totalRevenue'])
print(f"Correlation between numberOfGames and totalRevenue: {correlation:.2f}")

#10. Revenue per Game. How much revenue is generated on average per game.
data['revenue_per_game'] = data['totalRevenue'] / data['numberOfGames']
# Plot revenue per game over time
plt.figure(figsize=(14, 8))
plt.plot(data['date'], data['revenue_per_game'], marker='o', linestyle='-', color='blue')
plt.title('Revenue Per Game Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Revenue Per Game', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

#11. Revenue Growth Rate. This can indicate periods of rapid growth or decline.Calculate the percentage change in totalRevenue week-over-week or month-over-month.
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate the week-over-week revenue growth rate
data['revenue_growth_rate_weekly'] = data['totalRevenue'].pct_change() * 100

# Ensure month_name column is ordered correctly (from January to December)
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Calculate the total revenue per month
monthly_total_revenue = data.groupby('month_name')['totalRevenue'].sum().reset_index()

# Calculate the percentage change in total revenue month-over-month
monthly_total_revenue['percent_change'] = monthly_total_revenue['totalRevenue'].pct_change() * 100

# Plot week-over-week growth rate
plt.figure(figsize=(14, 8))
plt.plot(data['date'], data['revenue_growth_rate_weekly'], marker='o', linestyle='-', color='blue', label='Week-over-Week Growth Rate')
plt.title('Week-over-Week Revenue Growth Rate', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Growth Rate (%)', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plot the percentage change month-over-month
plt.figure(figsize=(14, 8))
sns.barplot(x='month_name', y='percent_change', data=monthly_total_revenue, palette='coolwarm', order=month_order)

# Add percentage values on top of each bar
for index, value in enumerate(monthly_total_revenue['percent_change']):
    plt.text(index, value + 0.5, f'{value:.2f}%', ha='center', fontsize=12, color='black')

# Title and labels
plt.title('Month-over-Month Percentage Change in Total Revenue', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Percentage Change (%)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


#12. Price Sensitivity Analysis. Investigate the relationship between changes in averagePrice and revenue growth to understand price sensitivity. Understanding how price changes impact revenue can guide pricing strategies. 
plt.figure(figsize=(14, 8))
plt.scatter(data['averagePrice'], data['totalRevenue'], color='green')
plt.title('Price Sensitivity: Average Price vs Total Revenue', fontsize=16)
plt.xlabel('Average Price', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

#13. Seasonality Analysis. Check for any seasonal patterns in revenue and number of games. Seasonality is crucial for predicting peak times (holidays, special events, etc.). Use time series decomposition or monthly trends to identify seasonal patterns.
# Aggregating monthly total revenue
monthly_data = data.groupby('month_name').agg({
    'totalRevenue': 'sum'
})

monthly_data['totalRevenue'].plot(kind='bar', figsize=(14, 8), color='skyblue')
plt.title('Total Revenue by Month (Seasonality)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()

#14. Game Performance Relative to Top and Bottom Groups. Compare the performance (revenue, playtime, etc.) of the top performers (top 5%, top 25%) against the bottom performers (bottom 30%). This analysis will help identify specific features that differentiate high-performing games from low-performing ones. 
# Plot comparison of total revenue between top 5% and bottom 30% performers with dual y-axes

top_5_percent = data[data['totalRevenue'] >= data['totalRevenue'].quantile(0.95)]
bottom_30_revenue = data[data['totalRevenue'] <= data['totalRevenue'].quantile(0.30)]

# Plot comparison of total revenue between top 5% and bottom 30% performers with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Ensure top_5_percent is not empty
if not top_5_percent.empty:
    ax1.plot(top_5_percent['date'], top_5_percent['totalRevenue'], marker='o', linestyle='-', color='red', label='Top 5% Performers')
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Total Revenue (Top 5%)', fontsize=14, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
else:
    print("No data available for Top 5% Performers.")

# Create a second y-axis for bottom 30% performers
if not bottom_30_revenue.empty:
    ax2 = ax1.twinx()
    ax2.plot(bottom_30_revenue['date'], bottom_30_revenue['totalRevenue'], marker='o', linestyle='-', color='blue', label='Bottom 30% Performers')
    ax2.set_ylabel('Total Revenue (Bottom 30%)', fontsize=14, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
else:
    print("No data available for Bottom 30% Performers.")

# Title and formatting
plt.title('Top 5% vs Bottom 30% Performers: Total Revenue Over Time', fontsize=16)
ax1.grid(True)
plt.xticks(rotation=45)
plt.show()




#15. Engagement Analysis (Playtime vs Revenue). Understand the relationship between averagePlayTime and totalRevenue. Higher playtime might indicate higher user engagement, which could correlate with revenue. This could reveal whether more engaged players (longer playtime) generate more revenue.
plt.figure(figsize=(14, 8))
plt.scatter(data['averagePlayTime'], data['totalRevenue'], color='purple')
plt.title('Playtime vs Total Revenue', fontsize=16)
plt.xlabel('Average Play Time', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

#16. Revenue Contribution by Top and Bottom Performers. Investigate how much revenue comes from the top performers versus the bottom performers. This helps understand how revenue is distributed. Identifying how much revenue is generated by the top performers can guide business decisions regarding marketing and resource allocation. Compare the revenue share between top 25%, top 5%, and bottom 30%.

total_revenue = data['totalRevenue'].sum()

# Calculate the revenue share from top 25%, top 5%, and bottom 30%
top_25_percent_revenue = top_25_percent['totalRevenue'].sum()
top_5_percent_revenue = top_5_percent['totalRevenue'].sum()
bottom_30_revenue_share = bottom_30_revenue['totalRevenue'].sum()

print(f"Top 25% Revenue Share: {top_25_percent_revenue / total_revenue * 100:.2f}%")
print(f"Top 5% Revenue Share: {top_5_percent_revenue / total_revenue * 100:.2f}%")
print(f"Bottom 30% Revenue Share: {bottom_30_revenue_share / total_revenue * 100:.2f}%")

#17. Customer Segmentation (Based on Playtime or Revenue). Segment customers based on averagePlayTime or totalRevenue and study their characteristics. This can help identify different user groups.  Customer segmentation can guide personalized marketing and product offerings. Cluster the data into groups (e.g., high, medium, low revenue) and analyze how different factors contribute to each segment.
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Clustering based on playtime and revenue
kmeans = KMeans(n_clusters=3)
data['customer_segment'] = kmeans.fit_predict(data[['averagePlayTime', 'totalRevenue']])

# Plotting the clusters
plt.figure(figsize=(14, 8))
scatter = plt.scatter(data['averagePlayTime'], data['totalRevenue'], c=data['customer_segment'], cmap='viridis')

# Add a color legend
legend1 = plt.legend(*scatter.legend_elements(), title="Customer Segments")
plt.gca().add_artist(legend1)

# Title and labels
plt.title('Customer Segments: Playtime vs Total Revenue', fontsize=16)
plt.xlabel('Average Play Time', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

#18. Player User Behavior Analysis ----> Player Activity vs. Revenue. How does player activity (numberOfGames or averagePlayTime) correlate with totalRevenue?
# Scatter plot: Average Play Time vs. Total Revenue
plt.figure(figsize=(14, 8))
plt.scatter(data['averagePlayTime'], data['totalRevenue'], color='purple')
plt.title('Average Play Time vs Total Revenue', fontsize=16)
plt.xlabel('Average Play Time', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Correlation between averagePlayTime and totalRevenue
correlation = data['averagePlayTime'].corr(data['totalRevenue'])
print(f"Correlation between Average Play Time and Total Revenue: {correlation:.2f}")

#19.  Player Segmentation Based on Activity. Segment players into different activity levels (e.g., low, medium, high activity) based on metrics like numberOfGames or averagePlayTime. Then, analyze the relationship between their activity and revenue.
# Create bins for activity levels based on numberOfGames
activity_bins = pd.cut(data['numberOfGames'], bins=[0, 100, 500, 1000, 5000], labels=['Low', 'Medium', 'High', 'Very High'])

# Create a new column in the data for player activity levels
data['activity_level'] = activity_bins

# Group by activity level and aggregate total revenue
activity_revenue = data.groupby('activity_level')['totalRevenue'].agg('mean')

# Plot activity level vs. average total revenue
activity_revenue.plot(kind='bar', figsize=(14, 8), color='orange')
plt.title('Average Total Revenue by Activity Level', fontsize=16)
plt.xlabel('Activity Level', fontsize=14)
plt.ylabel('Average Total Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.show()

#20. Price Sensitivity Analysis. This analysis investigates how different price points (averagePrice) influence revenue and player behavior.
# Scatter plot: Average Price vs Total Revenue
plt.figure(figsize=(14, 8))
plt.scatter(data['averagePrice'], data['totalRevenue'], color='green')
plt.title('Average Price vs Total Revenue', fontsize=16)
plt.xlabel('Average Price', fontsize=14)
plt.ylabel('Total Revenue', fontsize=14)
plt.grid(True)
plt.show()

# Correlation between averagePrice and totalRevenue
correlation_price = data['averagePrice'].corr(data['totalRevenue'])
print(f"Correlation between Average Price and Total Revenue: {correlation_price:.2f}")

#21. ANOVA
import scipy.stats as stats
# Assuming the 'data' DataFrame contains the relevant columns 'totalRevenue', 'month_name', and 'date'

# Ensure 'month_name' and 'date' are treated as categorical variables
# For example, converting 'month_name' and 'date' to categorical data type if not already
data['month_name'] = data['month_name'].astype('category')
data['date'] = data['date'].astype('category')

# ANOVA for 'month_name' (checking if there's a significant difference in totalRevenue across months)
month_groups = [data[data['month_name'] == month]['totalRevenue'] for month in data['month_name'].cat.categories]
month_groups2 = [data[data['month_name'] == month]['averagePlayTime'] for month in data['month_name'].cat.categories]
f_stat_month, p_val_month = stats.f_oneway(*month_groups)
f_stat_month2, p_val_month2 = stats.f_oneway(*month_groups2)
# ANOVA for 'date' (checking if there's a significant difference in totalRevenue across different days)
date_groups = [data[data['date'] == day]['totalRevenue'] for day in data['date'].cat.categories]
date_groups2 = [data[data['date'] == month]['averagePlayTime'] for month in data['date'].cat.categories]
f_stat_date, p_val_date = stats.f_oneway(*date_groups)
f_stat_date2, p_val_date2 = stats.f_oneway(*date_groups2)
# Output the results
print(f"ANOVA for month_name and total revenue:")
print(f"F-statistic: {f_stat_month}, P-value: {p_val_month}")
print(f"ANOVA for month_name and averageplaytime:")
print(f"F-statistic: {f_stat_month2}, P-value: {p_val_month2}")
print("\nANOVA for date:")
print(f"F-statistic: {f_stat_date}, P-value: {p_val_date}")
print("\nANOVA for date and avgplaytime:")
print(f"F-statistic: {f_stat_date2}, P-value: {p_val_date2}")

#22. PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data[['totalRevenue', 'averagePrice', 'averagePlayTime', 'numberOfGames']])
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 7))
plt.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=data['totalRevenue'], 
    cmap='viridis', 
    edgecolor='k', 
    s=50  # Adjust point size for better visibility
)
plt.colorbar(label='Total Revenue')
plt.xlabel('Principal Component 1 (PC1)', fontsize=12)
plt.ylabel('Principal Component 2 (PC2)', fontsize=12)
plt.title('PCA Visualization of Key Variables', fontsize=14)
plt.grid(alpha=0.3)  # Add a light grid for better reference
plt.tight_layout()
plt.show()


#t Test
#Comparing Means Between Two Groups: If you want to compare two groups (for example, top 25% vs bottom 30%), you can run a t-test to check if there is a significant difference in their totalRevenue or averagePlayTime.
from scipy.stats import ttest_ind
top_25_revenue = data[data['totalRevenue'] >= data['totalRevenue'].quantile(0.75)]['totalRevenue']
bottom_30_revenue = data[data['totalRevenue'] <= data['totalRevenue'].quantile(0.30)]['totalRevenue']
t_stat, p_val = ttest_ind(top_25_revenue, bottom_30_revenue)
print(f"T-Test: t_stat={t_stat}, p_val={p_val}")


#XX. Time Series Predictive Model. 
