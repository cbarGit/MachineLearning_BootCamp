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

data = []

#read from file and create an Numpy array
with open("wine.data") as dataFile:
    data = np.array([list(map(str,row)) for row in csv.reader(dataFile, delimiter=',')])


#create 3 random number for selecting test samples
#test, for the random samples to choose, removing them from data
test = []
for i in range(0,3):
    dataL = data.shape[0]
    r = random.randint(0,dataL)
    test.append(data[r])
    print("test n° " + str(r) + " class n° " + test[0][0])
    print(test[0])
print()

#transform list to nparray
test = np.array(test)
#clean the test samples removing first feature (rapresents class)
test = test[:,1:]

#instList, for the classes or labels of the sklearn's targets
instList = data[:,:1].reshape(-1,).tolist()

#clean data as test samples
data = data[:,1:]
#assign data and targets
X, y = data, instList

#load algorithm and fit data and targets into it
clf = svm.SVC()
clf.fit(X,y)

#predict on every test samples
for n,i in enumerate(test):
    print(clf.predict([test[n]]))
