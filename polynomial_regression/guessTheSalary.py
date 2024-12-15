import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

current_path = os.getcwd()
dataset_path = f"{current_path}/polynomial_regression/Position_Salaries.csv"
dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# simple linear regression for comparing regression models(simple and poly)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial linear regression

from sklearn.preprocessing import PolynomialFeatures

# create a vector of features from our features that has degree of 2.
poly_reg = PolynomialFeatures(degree=19)

X_poly = poly_reg.fit_transform(X)

# create polynomial regressor from our simple linear regressor by using the feature of polynomial with degree 2.
lin_reg_sec = LinearRegression()
poly_lin_reg = lin_reg_sec.fit(X_poly, y)

# create a plot
plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X, lin_reg.predict(X), color="blue", label="Simple Linear Regression")
plt.plot(
    X,
    poly_lin_reg.predict(X_poly),
    color="green",
    label="Polynomial Regression (degree=2)",
)
plt.title("Regression Comparison")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.show()
