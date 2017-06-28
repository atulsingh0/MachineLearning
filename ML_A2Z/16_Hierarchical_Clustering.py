# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:07:24 2017

Hierarchical Clustering - Dendogram

@author: Atul
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# reading the dataset
data = pd.read_csv('dataset/Mall_Customers.csv')
#print(data[:4])

X = data.iloc[:, [3,4]].values
#print(X[:4])

# Generating Dendogram
plt.figure()
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendogram")


# implementing Agglomerative Clustering
clf = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')
y_pred = clf.fit_predict(X)


# Visualising the clusters
plt.figure()
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



