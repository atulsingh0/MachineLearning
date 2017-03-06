# -*- coding: utf-8 -*-
"""
importing library
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# data processing
data = pd.read_csv('dataset/Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

# taking care of missing values
# strategy one : Remove the missing values
# strategy two : Replace the missing value with mean
impute = Imputer(missing_values="NaN", strategy="mean", axis=0)
impute.fit(X[:, 1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])

# Encoding categorical data
labelencode_X = LabelEncoder()
X[:, 0] = labelencode_X.fit_transform(X[:, 0])

onehotencode_X = OneHotEncoder(categorical_features=[0])
X = onehotencode_X.fit_transform(X).toarray()

labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)


# splitting the data into train test data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)


# standarize the data
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)




