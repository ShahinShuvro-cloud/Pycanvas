import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
from scipy import stats

df=pd.read_csv(r"C:\Users\User\eda_sales_dataset.csv")
print(df)

print(df.head(5)) # Display first 5 rows
print(df.info()) # Display information about the DataFrame
print(df.tail(5)) # Display last 5 rows
print(df.describe())   # Display summary statistics
print(df.dtypes)   # Display data types of each column
print(df.columns)  # Display column names
print(df.rename(columns={"Region":"Continent",})) # Rename column 'LastName' to 'SurName'
print(df.shape) # Display the shape of the DataFrame
print(df.size) # Display the size of the DataFrame

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
results = {}
for col in numerical_columns:
    try:
        results[col] = {
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Mode': stats.mode(df[col], keepdims=True)[0][0]  # Mode might return multiple values, take first
        }
    except Exception as e:
        print(f"Error processing column {col}: {e}")

# Display results
print("\nSummary Statistics:")
for col, stats in results.items():
    print(f"\nColumn: {col}")
    print(f"  Mean: {stats['Mean']:.2f}")
    print(f"  Median: {stats['Median']:.2f}")
    print(f"  Mode: {stats['Mode']:.2f}")


# calculating IQR
df= df.select_dtypes(include=['int64', 'float64'])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("IQR for int columns:")
print('IQR for float columns:')
print(IQR)

# bar diagram
df=df.head(10)
duplicate_rows_df = df[df.duplicated()] # Find duplicate rows
print("number of duplicate rows: ", duplicate_rows_df.shape)
df_clean = df.dropna(subset=["CustomerID", "Income"]) 
plt.figure(figsize=(12,8))
plt.bar(df_clean["CustomerID"], df_clean["Income"], color="green") 
plt.xlabel("CustomerID") # Set x-axis label
plt.ylabel("Income") # Set y-axis label
plt.title("Income by CustomerID") # Set plot title
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to prevent clipping of tick-labels
plt.show() # Display the plot

# box plot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Income'])
plt.show()

# pie chart
df=df.head(11)
print(df.isnull().sum())
df_clean = df.dropna(subset=["CustomerID", "Income"])
df=df.dropna()
df.count()
plt.figure(figsize=(8, 8))
plt.pie(df['Income'], labels=df['CustomerID'], autopct='%1.1f%%')
plt.title('Pie Chart from CSV Data')
plt.axis('equal')
plt.show()
