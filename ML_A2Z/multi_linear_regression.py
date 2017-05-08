# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:56:47 2017

@author: Atul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


# data processing
data = pd.read_csv('dataset/50_Startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


# Encoding the State columns
# this will encode the categorical value to numerical as 0,1,2....
lb_X = LabelEncoder()
X[:, 3] = lb_X.fit_transform(X[:, 3])
# this will create binary like representation from categorical value
# set 1 for true and 0 for false
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


# Implementing backward eliminations
# as we can see in the X matrix, columns actually represent the dependent variables
# x1, x2...xn, but we dont have x0 (=1) for y = b0x0 + b1x2......
# for that we have to add a columb with 1's as first columns
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
# now, our first column reprensent the variable x0

# creating X_opt for optimum fetures
X_opt = X[:,:]  # or
X_opt = X[:, [0,1,2,3,4,5]]

# fillting the data
reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3,4,5]]
reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3,5]]
reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()
X_opt = X[:, [0,3]]
reg_OLS = sm.OLS(endog=y, exog=X_opt).fit()
reg_OLS.summary()

# now we are good, 

# Avoiding the dummy variable trap
X = X_opt  # removing the first column

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





