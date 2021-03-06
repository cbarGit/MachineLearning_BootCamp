#! /usr/bin/python
#
# data samples from https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
#
#module numpy for Numpy library
#module pandas for Pandas library
#module svm from sklearn for classification algorithm
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('wine.data', delimiter=',', header=None)

y, X = data[[0]], data.drop(0, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#load algorithm and fit data and targets into it
clf = svm.SVC()
clf.fit(X,y.values.ravel())

predictions = clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
