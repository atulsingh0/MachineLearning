# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:46:47 2017

Apriori Association 

@author: Atul
"""

import pandas as pd
#from sklearn.

# reading the data file
data = pd.read_csv('dataset/Market_Basket_Optimisation.csv', header=None)

transaction=[]
for i in range(len(data)):
    transaction.append(list(data.iloc[i].values))
