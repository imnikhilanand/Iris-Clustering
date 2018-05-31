# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:57:28 2018

@author: Nikhil
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#creating the dataset
dataset = pd.read_csv("data.csv", header=None)

x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


#extracting the diagonal values for every iris flower
x_new1 = (((x[:,0])*(x[:,0]))+((x[:,1])*(x[:,1])))
x_new1 = np.sqrt(x_new1)
x_new2 = (((x[:,2])*(x[:,2]))+((x[:,3])*(x[:,3])))
x_new2 = np.sqrt(x_new2)


#clustering the data into three seperate clusters
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)


# Visualising the clusters
plt.scatter(x_new1[0:50,], x_new2[0:50,], s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(x_new1[50:100,], x_new2[50:100,], s = 50, c = 'blue', label = 'Iris-versicolor')
plt.scatter(x_new1[100:150,], x_new2[100:150,], s = 50, c = 'green', label = 'Iris-virginica')
plt.title('Clusters of Iris')
plt.xlabel('Sepal diagonal length(cm)')
plt.ylabel('Petal diagonal length(cm)')
plt.legend()
plt.show()

