# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:56:47 2017

@author: Atul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# data processing
data = pd.read_csv('dataset/50_Startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


# Encoding the State columns
lb_X = LabelEncoder()
X[:, 3] = lb_X.fit_transform(X[:, 3])

ohe_X = OneHotEncoder(categorical_features=[3])
X = ohe_X.fit_transform(X).toarray()


# Avoiding the dummy variable trap
X = X[:, 1:]  # removing the first column

# splitting the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)


# fitting the data
clf = LinearRegression()
clf.fit(train_X, train_y)

# predicting the data
y_pred = clf.predict(test_X)


# priting the details
print("Training Score: ",clf.score(train_X, train_y))
print("Test Score: ",clf.score(test_X, test_y))


