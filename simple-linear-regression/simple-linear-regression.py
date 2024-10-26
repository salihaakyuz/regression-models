import pandas as pd
import numpy as np

dataset = pd.read_csv(
    "/Users/lahzatoz/regression-models/simple-linear-regression/Salary_Data.csv"
)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color="blue")
plt.plot(X_train, y_pred, color="black")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Training graph")
plt.scatter(X_test, y_test, color="purple")
plt.plot(X_train, y_pred, color="black")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Test graph")
plt.show()
