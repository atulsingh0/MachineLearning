# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 22:43:20 2017

@author: Atul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# data processing
data = pd.read_csv('dataset/Salary_Data.csv')
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# splitting the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)
train_X = train_X.reshape(-1, 1)
test_X = test_X.reshape(-1, 1)
train_y = train_y.reshape( -1, 1)
test_y = test_y.reshape(-1, 1)

# creating a object
clf = LinearRegression()
clf.fit(train_X, train_y)

# predecting
y_pred = clf.predict(test_X)

# plotting the data
plt.scatter(train_X, train_y, marker='o', color='r')
plt.plot(train_X, clf.predict(train_X), color='b')
plt.grid(True)
plt.margins(0.15)
plt.title("training data")
plt.xlabel("Experience")
plt.ylabel("Salary")


# last, printing the Linear func
print("Coefficient : ",clf.coef_)
print("Intercept : ",clf.intercept_)


