#dataset from https://www.kaggle.com/hobako1993/sp1-factor-binding-sites-on-chromosome1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#load data
data = pd.read_csv('numeric sequence.csv')

#remove 'label' column
nlData = pd.DataFrame(data,columns=data.columns[:-1])

X_train, X_test, y_train, y_test = train_test_split(nlData,data['label'],test_size=0.25)

#test various k
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

#draw a plot to select the right 'k'
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# try with k = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('K=1')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('\n')

# try with k = 17
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('K=17')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('\n')

# try with k = 23
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('K=23')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('\n')

# try with k = 26
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('K=26')
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('\n')

plt.show()
