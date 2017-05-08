# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:54:11 2017

# Random Tree Regression

@author: Atul
"""

# import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import RandomForestRegressor



# data processing
data = pd.read_csv('dataset/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# creating regressor
clf = RandomForestRegressor(n_estimators=300, random_state=0)
clf.fit(X, y)

# predicting the value
y_pred = clf.predict([[6.5]])


# Visualizing the data
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='blue')
plt.plot(X_grid, clf.predict(X_grid), c='blue')
plt.xlabel("Positions")
plt.ylabel("Salary")
plt.grid(True)
plt.show()


