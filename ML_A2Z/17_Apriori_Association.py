# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:46:47 2017

Apriori Association 

@author: Atul
"""

import pandas as pd
from apyori import apriori

# reading the data file
data = pd.read_csv('dataset/Market_Basket_Optimisation.csv', header=None)

transaction=[]
for i in range(0, 7501):#range(len(data)):
    #transaction.append(list(data.iloc[i].values))
    transaction.append([str(data.values[i,j]) for j in range(0,20)])
    
# implementing
clf = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

result = list(clf)
#print(result)
print(type(result))

for i in result:
    print(i)