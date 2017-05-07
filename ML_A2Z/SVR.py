# -*- coding: utf-8 -*-
"""
Created on Sun May  7 15:14:37 2017

@author: Atul
"""

# import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.svm import SVR



# data processing
data = pd.read_csv('dataset/Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


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
clf = SVR(kernel='rbf')
clf.fit(X, y)

# visualizting the data 
plt.scatter(X, y, c='blue')
plt.plot(X, clf.predict(X), c='red')
plt.xlabel("Positions")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# predicting
sc_y.inverse_transform(clf.predict(sc_X.transform([[6.5]])))