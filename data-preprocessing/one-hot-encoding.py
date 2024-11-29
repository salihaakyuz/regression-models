from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import numpy as np
import pandas as pd

dataset = pd.read_csv(f"{os.getcwd()}/data-preprocessing/adult.data", header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
transformers = [
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        [1, 3, 5, 6, 7, 8, 9, -1],  # the indexes that got categorical data in X
    )
]
remainder = "passthrough"
ct = ColumnTransformer(transformers=transformers, remainder=remainder)
X = np.array(ct.fit_transform(X))
