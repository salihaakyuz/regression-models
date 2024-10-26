import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataset = pd.read_csv("/Users/lahzatoz/regression-models/data-preprocessing/Data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(dataset.dtypes)
