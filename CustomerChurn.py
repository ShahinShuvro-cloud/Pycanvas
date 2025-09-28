import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

# Add necessary imports for ColumnTransformer and Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

df=pd.read_csv(r"C:\Users\User\customer_churn_data_100.csv")
print(df)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# EDA
print(df.info())
print(df.describe())
print(df['Churn'].value_counts())
ax = sns.countplot(x = 'Churn', data = df)
plt.title("Count of Customers by Churn")
ax.bar_label(ax.containers[0])
plt.show()
plt.figure(figsize = (5,6))
gb = df.groupby("Churn").agg({'Churn':"count"})
plt.pie(gb['Churn'], labels = gb.index, autopct = "%1.2f%%")
plt.title("Percentage of Churned Customers", fontsize = 10)
plt.show()
print(df['Gender'].value_counts())

# Gender Based Churned
plt.figure(figsize=(4,5))
ax = sns.countplot(x="Gender", data=df, hue="Churn")
plt.title("Churn by Gender")

# Add counts on each bar
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.0f}",
            ha="center", va="bottom", size=8)

plt.show()

print('The count of Male customers according to Churn is :')
print(df.groupby(df['Gender'] == 'Male')['Churn'].value_counts())
print('---------------------------------------------------')
print('The count of Female customers according to Churn is :')
print(df.groupby(df['Gender'] == 'Female')['Churn'].value_counts())

print(df['Age'].value_counts())
sns.countplot(x='Age', data=df, hue='Churn')
plt.title("Churn by Age Status")
plt.show()
df.groupby('Age')['Churn'].value_counts()
df.groupby('Age')['Churn'].value_counts(normalize=True) * 100

print(df['TechSupport'].value_counts())
print(df.groupby('TechSupport')['Churn'].value_counts())
sns.countplot(x='TechSupport', data=df, hue='Churn')
plt.title("Churn by TechSupport Status")
plt.show()
print(df.groupby(['TechSupport', 'Gender'])['Churn'].value_counts())

plt.figure(figsize = (9,4))
sns.histplot(x = "Tenure", data = df, bins = 72, hue = "Churn")
plt.show()

avg_tenure = df.groupby('Churn')['Tenure'].mean()
avg_tenure.plot(kind='bar')
plt.title("Average Tenure by Churn Status")
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(data=df, x='Tenure', hue='Churn', fill=True)
plt.title("Tenure by Churn Status")
plt.show()

columns = ['Tenure', 'Gender',
            'TechSupport', 'InternetService']
print("DataFrame columns:", df.columns)
print("Columns to plot:", columns)
n_cols = 3
n_rows = (len(columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()
for i, col in enumerate(columns):
    if col in df.columns:
        sns.countplot(x=col, data=df, ax=axes[i], hue="Churn")  # Fixed hue parameter
        axes[i].set_title(f'Count Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    else:
        print(f"Skipping column '{col}' as it does not exist in DataFrame")
        axes[i].set_title(f"Invalid column: {col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
print(df['ContractType'].value_counts())
sns.countplot(x='ContractType', data=df, hue='Churn')
plt.title("Churn by Contract Status")
plt.show()

sns.barplot(x='Churn', y='MonthlyCharges', data=df,hue='Gender')
plt.show()
sns.barplot(x='Churn', y='MonthlyCharges', data=df,hue='Gender')
plt.show()

plt.figure(figsize = (9,4))
sns.histplot(x = "MonthlyCharges", data = df, bins = 50, hue = "Churn")
plt.show()
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True)
plt.title("Monthly Charges by Churn Status")
plt.show()

# Data Preprocessing
print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")  
plt.title("Missing Values Heatmap", fontsize=14)
plt.show()