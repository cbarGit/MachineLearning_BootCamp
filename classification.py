#! /usr/bin/python
#
# data samples from https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
#
#module "random" for random number
#module svm from sklearn for classification algorithm
import random
from sklearn import svm

#Cleaning part:
#to refine the data samples
buff = []
data = []

#read from file and append data to a list, removing "\n" char at the end of
#every line
with open("wine.data") as dataFile:
    for row in dataFile:
        buff.append(row.rstrip('\n'))

#cleaning further, creating an array of arrays
for row in buff:
    data.append(row.split(','))

#create 3 random number for selecting test samples
rNum0 = random.randrange(0,177)
rNum1 = random.randrange(0,177)
rNum2 = random.randrange(0,177)

#instList, for the classes or labels of the sklearn's targets
#test, for the random samples to choose
instList = []
test = []

#choose the samples
#remove the first element of every sample, that describe the latter class
for n,sample in enumerate(data):
    if(n == rNum0 or n == rNum1 or n == rNum2):
        print(sample)
        test.append(sample)
    instList.append(sample[0])
    del sample[0]

#assign data and targets
X, y = data, instList

#show chosen samples
print()
for i in test:
    print(i)
print()

#load algorithm and fit data and targets into it
clf = svm.SVC()
clf.fit(X, y)

#predict on every test samples
for n,i in enumerate(test):
    print(clf.predict([test[n]]))
