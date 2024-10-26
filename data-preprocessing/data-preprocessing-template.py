import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataset = pd.read_csv("/Users/lahzatoz/regression-models/data-preprocessing/Data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
