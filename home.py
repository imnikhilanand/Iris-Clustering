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



""" 
Ananlysis of  plots for each parameter

"""


# Visualising the sepal vs Petal (diagonally)
plt.scatter(x_new1[0:50,], x_new2[0:50,], s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(x_new1[50:100,], x_new2[50:100,], s = 50, c = 'blue', label = 'Iris-versicolor')
plt.scatter(x_new1[100:150,], x_new2[100:150,], s = 50, c = 'green', label = 'Iris-virginica')
plt.title('Clusters of Iris')
plt.xlabel('Sepal diagonal length(cm)')
plt.ylabel('Petal diagonal length(cm)')
plt.legend()
plt.show()



#scatter for sepal length vs sepal width
x_coor = dataset.iloc[:,0]
y_coor = dataset.iloc[:,1]
plt.scatter(x_coor[0:50,],y_coor[0:50,],s=10,c='red',label = 'Iris-setosa')
plt.scatter(x_coor[50:100,],y_coor[50:100,],s=10,c='blue', label = 'Iris-versicolor')
plt.scatter(x_coor[100:150,],y_coor[100:150,],s=10,c='green', label = 'Iris-virginica')
plt.title('Sepal length vs Width')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()



#scatter for petal length vs petal width
x_coor2 = dataset.iloc[:,2]
y_coor2 = dataset.iloc[:,3]

plt.scatter(x_coor2[0:50,],y_coor2[0:50,],s=10,c='red',label = 'Iris-setosa')
plt.scatter(x_coor2[50:100,],y_coor2[50:100,],s=10,c='blue', label = 'Iris-versicolor')
plt.scatter(x_coor2[100:150,],y_coor2[100:150,],s=10,c='green', label = 'Iris-virginica')
plt.title('Petal length vs Width')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
plt.show()


#visualising some boxplot data  

#sepal length
pos = ['Iris-setosa','Iris-versicolor','Iris-virginca']
data_new = [x_coor[0:50,],x_coor[50:100,],x_coor[100:150,]]
plt.boxplot(data_new, labels = pos)
plt.ylabel('Sepal Length')
plt.title("Box Plot for Sepal length")
plt.show()

#sepal width
pos = ['Iris-setosa','Iris-versicolor','Iris-virginca']
data_new = [y_coor[0:50,],y_coor[50:100,],y_coor[100:150,]]
plt.boxplot(data_new, labels = pos)
plt.ylabel('Sepal Width')
plt.title("Box Plot for Sepal Width")
plt.show()

#Petal length
pos = ['Iris-setosa','Iris-versicolor','Iris-virginca']
data_new = [x_coor2[0:50,],x_coor2[50:100,],x_coor2[100:150,]]
plt.boxplot(data_new, labels = pos)
plt.ylabel('Petal Length')
plt.title("Box Plot for Petal length")
plt.show()

#Petal width
pos = ['Iris-setosa','Iris-versicolor','Iris-virginca']
data_new = [y_coor2[0:50,],y_coor2[50:100,],y_coor2[100:150,]]
plt.boxplot(data_new, labels = pos)
plt.ylabel('Petal Width')
plt.title("Box Plot for Petal Width")
plt.show()


#visualising voilin plot

#sepal length
data_new = [x_coor[0:50,],x_coor[50:100,],x_coor[100:150,]]
plt.violinplot(data_new,showmeans = True,showmedians=True)
plt.ylabel('Sepal length')
plt.title("Violin Plot")
plt.show()

#sepal width
data_new = [y_coor[0:50,],y_coor[50:100,],y_coor[100:150,]]
plt.violinplot(data_new,showmeans = True,showmedians=True)
plt.ylabel('Sepal Width')
plt.title("Violin Plot")
plt.show()


#petal length
data_new = [x_coor2[0:50,],x_coor2[50:100,],x_coor2[100:150,]]
plt.violinplot(data_new,showmeans = True,showmedians=True)
plt.ylabel('Sepal length')
plt.title("Violin Plot")
plt.show()

#petal width
data_new = [y_coor2[0:50,],y_coor2[50:100,],y_coor2[100:150,]]
plt.violinplot(data_new,showmeans = True,showmedians=True)
plt.ylabel('Sepal Width')
plt.title("Violin Plot")
plt.show()




#clustering the data into three seperate clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)












