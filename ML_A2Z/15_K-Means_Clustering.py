# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 22:36:41 2017

K-Means Clustering

@author: Atul
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# reading the dataset
data = pd.read_csv('dataset/Mall_Customers.csv')
print(data[:4])

X = data.iloc[:, [3,4]].values
print(X[:4])

plt.figure()
wcss = []
for i in range(1, 11):
    clf = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    clf.fit(X)
    wcss.append(clf.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
clf = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = clf.fit_predict(X)


# Visualising the clusters
plt.figure()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.legend()
plt.show()



