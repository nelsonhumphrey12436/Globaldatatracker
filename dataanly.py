# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style
sns.set(style='whitegrid')

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # Converts to a pandas DataFrame

    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    # No missing values here, but if there were:
    # df.dropna(inplace=True) or df.fillna(value=..., inplace=True)

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute the mean for each numerical column
grouped = df.groupby('target').mean()
print("\nMean values grouped by target (species):")
print(grouped)

# Map target numbers to species names for easier analysis
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# Task 3: Data Visualization

# Line Chart: Simulated time-series of sepal length
df['Index'] = df.index
plt.figure(figsize=(10, 4))
sns.lineplot(data=df, x='Index', y='sepal length (cm)', hue='species')
plt.title('Trend of Sepal Length by Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend(title='Species')
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x='sepal width (cm)', bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.show()

# Scatter Plot: Sepal length vs. Petal length
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Findings & Observations (example - write more for your submission)
print("\nFindings:")
print("- Setosa species tends to have shorter petal and sepal lengths.")
print("- Virginica generally has longer petals and sepals.")
print("- Sepal width shows the most variation among all features.")
print("- Petal length has a strong positive correlation with sepal length.")