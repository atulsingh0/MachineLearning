# -*- coding: utf-8 -*-
"""
Created on Sun May  7 09:39:47 2017
# Polynomial Regression
@author: Atul
"""

# import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


# data processing
data = pd.read_csv('dataset/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

"""
# Encoding the State columns
# this will encode the categorical value to numerical as 0,1,2....
lb_X = LabelEncoder()
X[:, 0] = lb_X.fit_transform(X[:, 0])
# this will create binary like representation from categorical value
# set 1 for true and 0 for false
ohe_X = OneHotEncoder(categorical_features=[0])
X = ohe_X.fit_transform(X).toarray()


# Avoiding the dummy variable trap
X = X[:, 1:]  # removing the first column

# splitting the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
"""

#fitting the data
clf = LinearRegression()
clf.fit(X, y)

# polynomial, degree = 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

clf_poly = LinearRegression()
clf_poly.fit(X_poly, y)


# polynomial, degree = 4
poly4 = PolynomialFeatures(degree=4)
X_poly4 = poly4.fit_transform(X)

clf_poly4 = LinearRegression()
clf_poly4.fit(X_poly4, y)



# visualizting the data (Linear Regressions)
plt.scatter(X, y, c='blue')
plt.plot(X, clf.predict(X), c='red')
plt.xlabel("Positions")
plt.ylabel("Salary")


# visualizting the data (Linear Regressions)
plt.scatter(X, y, c='blue')
plt.plot(X, clf_poly.predict(X_poly), c='red')
plt.xlabel("Positions")
plt.ylabel("Salary")


# visualizting the data (Linear Regressions)
#X_grid = np.arange(min(X), max(X), 0.1)
plt.scatter(X, y, c='blue')
plt.plot(X, clf_poly4.predict(X_poly4), c='red')
plt.xlabel("Positions")
plt.ylabel("Salary")

# more smoother prediction
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='blue')
plt.plot(X_grid, clf_poly4.predict(poly4.fit_transform(X_grid)), c='green')
plt.grid(True)
plt.xlabel("Positions")
plt.ylabel("Salary")



