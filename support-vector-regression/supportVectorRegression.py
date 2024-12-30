import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

current_path = os.getcwd()
dataset_path = f"{current_path}/polynomial_regression/Position_Salaries.csv"
dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
# reshaping into len(y) rows and 1 column.
y = y.reshape(len(y), 1)
# now the vectors look alike, we can apply feature scaling to the both.
from sklearn.preprocessing import StandardScaler

scX = StandardScaler()
scY = StandardScaler()

y = scY.fit_transform(y)
print(y)
X = scX.fit_transform(X)
print(X)
from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(X, y)
# reverse the predicting to not scaled version
predictValue = scX.transform([[6.5]])
predictedSalaryFor6Level = regressor.predict(predictValue)
inversedSalary = scY.inverse_transform(predictedSalaryFor6Level.reshape(-1, 1))[0, 0]
print(inversedSalary)
plt.scatter(
    scX.inverse_transform(X), scY.inverse_transform(y), color="red", label="Actual Data"
)
plt.plot(
    scX.inverse_transform(X),
    scY.inverse_transform(regressor.predict(X).reshape(-1, 1)),
    color="blue",
    label="SVR",
)
plt.title("Regression Comparison")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.show()
