import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import os

current_path = os.getcwd()
dataset_path = pd.read_csv(f"{current_path}/data-preprocessing/Data.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(dataset.dtypes)
