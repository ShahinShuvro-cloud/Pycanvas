import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\User\Downloads\Housing.csv")
print(data.head())

print(data.info())
print(data.describe()) 

sns.pairplot(data)
plt.show()

sns.heatmap(data.select_dtypes(include=['float64','int64']).corr(), annot=True, cmap='coolwarm')
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.show()

X = data[['area', 'bedrooms', 'bathrooms', 'stories']]  # Independent variables
y = data['price']  # Target variable

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make prediction
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# Visualize results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.show()

# Predicted new house
new_house = pd.DataFrame({'area': [2500], 'bedrooms': [3], 'bathrooms': [2], 'stories': [5]})
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price[0])