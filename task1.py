import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\Meghana\Documents\Skill_craft_training\TASK1\house_prices (1).csv")

# Print first 5 rows
print(data.head())

# Features and target
X = data[['Area_SqFt', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

print(predictions)

# Calculate error
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# Predict new house price
new_house = pd.DataFrame({
    'Area_SqFt': [1600],
    'Bedrooms': [3],
    'Bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])