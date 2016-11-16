#! /usr/bin/python
#
# data samples from https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
#
#module csv for reading from csv file
#module "random" for random number
#mosule numpy for Numpy library
#module svm from sklearn for classification algorithm
import csv
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


#read from file and create an Numpy array
with open("wine.data") as dataFile:
    data = np.array([list(map(str,row)) for row in csv.reader(dataFile, delimiter=',')])

#assign data and targets
X, y = data[:,1:], data[:,:1]
#print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#load algorithm and fit data and targets into it
clf = svm.SVC()
clf.fit(X,y.ravel())

predictions = clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
