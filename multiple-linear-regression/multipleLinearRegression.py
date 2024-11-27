import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# import dataset
current_path = os.getcwd()
startup_dataset = pd.read_csv(
    f"{current_path}/multiple-linear-regression/50_Startups.csv"
)
X = startup_dataset.iloc[:, :-1].values
y = startup_dataset.iloc[:, -1].values
# 4th column needs to have one hot encoding
transformers = [("encoder", OneHotEncoder(), [-1])]
remainder = "passthrough"
ct = ColumnTransformer(transformers=transformers, remainder=remainder)
X = np.array(ct.fit_transform(X))
# feature scaling does not need to be applying
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted")


plt.scatter(y_test, y_pred, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--",
    label="Perfect Prediction (y=x)",
)

plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (y_pred)")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.grid()
plt.show()
