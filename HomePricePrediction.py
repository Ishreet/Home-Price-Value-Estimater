import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
data = pd.read_csv("home_data.csv")
sqft = data["sqft_living"]
price = data["price"]

# Create the arrays
x = np.array(sqft).reshape(-1,1)
y = np.array(price)

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Fit linear regression to training set
reg = LinearRegression()
reg.fit(xtrain, ytrain)

# Predict prices
prediction = reg.predict(xtest)

# Plot the findings
plt.scatter(xtrain, ytrain, color = "red")
plt.plot(xtrain, reg.predict(xtrain), color = "blue")
plt.title("Home Price Prediction")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.show()