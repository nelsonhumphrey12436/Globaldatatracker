
import pandas as pd

# Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Basic inspection
print(df.head())
print(df.info())

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Drop rows with too many missing values
df_cleaned = df.dropna(subset=['total_cases', 'total_deaths', 'total_vaccinations'])


import matplotlib.pyplot as plt

# Global data
global_data = df[df['location'] == 'World']

# Plot total cases
plt.figure(figsize=(10, 5))
plt.plot(global_data['date'], global_data['total_cases'], label='Total Cases')
plt.plot(global_data['date'], global_data['total_deaths'], label='Total Deaths')
plt.plot(global_data['date'], global_data['total_vaccinations'], label='Total Vaccinations')
plt.legend()
plt.title('Global COVID-19 Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Find max total cases by country
top_countries = df.groupby('location')['total_cases'].max().sort_values(ascending=False).head(5)

# Filter for these countries
top_df = df[df['location'].isin(top_countries.index)]

# Plot cases over time
plt.figure(figsize=(12, 6))
for country in top_countries.index:
    country_df = top_df[top_df['location'] == country]
    plt.plot(country_df['date'], country_df['total_cases'], label=country)

plt.legend()
plt.title("Top 5 Countries by Total COVID-19 Cases")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.grid(True)
plt.show()

import seaborn as sns

# Pivot table for heatmap
heatmap_data = df[df['date'] == df['date'].max()]
heatmap_data = heatmap_data[['location', 'total_cases']].dropna().sort_values('total_cases', ascending=False).head(10)

sns.barplot(data=heatmap_data, x='total_cases', y='location')
plt.title('Top 10 Countries with Most Cases (Latest Date)')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.show()
