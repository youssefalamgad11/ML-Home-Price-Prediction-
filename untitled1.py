# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 05:56:58 2024

@author: youss
"""
# Numpy to create data as array
import numpy as np
# Pandas to Get data and develop it
import pandas as pd
# Matplotlib to vis data
import matplotlib.pyplot as plt
# Split data into train and test
from sklearn.model_selection import train_test_split
# Linear regression model
from sklearn.linear_model import LinearRegression
# Evaluate Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score


# Read dataset 
df = pd.read_csv('data.csv')
# Print First 5 rows
df.head(10)


# Create copy of dataset to visualize it 
viz = df.copy()


# Check if there null cell or not
df.isnull().sum()

# Print shape of dataset (Number of rows, columns) 
df.shape

# Prnit data info like column and number of rows of each column and if it null or not and data type of it
df.info()

# Print data describtion
df.describe().T

# split data into train and test set with 20% to test and 80% of train
train, test = train_test_split(df, test_size = 0.2)
test_pred = test.copy()
train.head(10)

test.head(10)


# x Train and test with 4 column ['bedrooms', 'bathrooms', 'sqft_living', 'yr_built'], ['price']
x_train = train[['bedrooms', 'bathrooms', 'sqft_living', 'yr_built']].values
x_test = test[['bedrooms', 'bathrooms', 'sqft_living', 'yr_built']].values
# Create y train and test with close columns
y_train = train['price'].values
y_test = test['price'].values


# Set Linear Regression model with name (model_lnr)
model_lnr = LinearRegression()
# Fit Training data
model_lnr.fit(x_train, y_train)
LinearRegression()
# Predict Data with x_test
y_pred = model_lnr.predict(x_test)
# Test Model
result = model_lnr.predict([[4, 2, 3000, 2000]])
print(result)
# Get accuracy of model
print("MSE",round(mean_squared_error(y_test,y_pred), 3))
print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
print("R2 Score : ", round(r2_score(y_test,y_pred), 3) * 100)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(f'R-squared (R2): {r2}')

# Visualize the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
plt.title('Linear Regression Model - Housing Price Prediction')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()




def predict_home_price(bedrooms, bathrooms, sqft_living, yr_built):
    input_data = np.array([bedrooms, bathrooms,sqft_living , yr_built]).reshape(1, -1)
    predicted_price = model_lnr.predict(input_data)
    return predicted_price[0]
# User input
home_bedrooms = float(input("Enter the Num of bedrooms: "))
home_bathrooms = float(input("Enter the num of bathrooms: "))
home_sqft_living = float(input("Enter the sqft area (like 3000): "))
home_yr_built = float(input("Enter the year of built: "))

# Predict using user input
predicted_home_price = predict_home_price(home_bedrooms, home_bathrooms, home_sqft_living, home_yr_built)
print(f"Predicted Home Price: {predicted_home_price}")

