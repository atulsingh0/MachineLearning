# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:23:12 2017

# KNN Classifier

@author: Atul
"""

# import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



# reading dataset
data = pd.read_csv('dataset/Social_Network_Ads.csv')
X = data.iloc[:,[2,3]].values
y = data.iloc[:,4].values

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25)

# standard Scaling the data
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

# Implemeting Logistic Reg
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_X, train_y)

# predict
pred_y = clf.predict(test_X)

# confusion prediction
print("Confusion Metrix:\n", confusion_matrix(test_y, pred_y))
print("Accuracy Score: ", accuracy_score(test_y, pred_y))