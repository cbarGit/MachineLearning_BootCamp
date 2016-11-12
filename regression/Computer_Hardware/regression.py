#! /usr/bin/python
#
#data from:
#https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#import data from file
machines = pd.read_csv('machine.data')
machines = machines.drop(['ERP','PRP'], 1)

#create pairplot and linear plot between MMAX and CACH
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
pairplot = sns.pairplot(machines)
ln = sns.lmplot(x='MMAX',y='CACH',data=machines)

#split data
y = machines['CACH']
X = machines[['MYCT','MMIN','MMAX','CHMIN','CHMAX']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#fit data
lm = LinearRegression()
lmFit = lm.fit(X_train,y_train)

#calc coeffecients
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
print()

#create predictions
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#calc Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#plot residuals
sns.distplot((y_test-predictions),bins=50);

sns.plt.show()
