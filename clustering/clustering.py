#! /usr/bin/python
#
#data from:
#https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, preprocessing

#extract data from file
docBuff = []
data = []

with open("data.txt") as dataFile:
    for row in dataFile:
        docBuff.append(row.rstrip('\n'))

for i in docBuff:
    data.append(i.split(","))


#clustering fase
k = 3

km = cluster.KMeans(n_clusters = k)
km.fit(preprocessing.maxabs_scale(data))

labels = km.labels_
centroids = km.cluster_centers_
print("k: " + str(k))
print(labels)

cluster_centers = np.sort(km.cluster_centers_, axis=0)
print(cluster_centers)

#plot clustered data
dataMod = np.array(data)

for i in range(k):
    data = dataMod[np.where(labels==i)]
    plt.plot(data[:,0],data[:,3],'o')
    lines = plt.plot(centroids[i,0],centroids[i,1],centroids[i,2], centroids[i,3],'kx')

plt.show()
