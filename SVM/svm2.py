import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

iris = sns.load_dataset('iris')

sns.pairplot(iris,hue='species')
plt.show()

X = iris.drop('species',axis=1)
y = iris['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101,test_size=0.3)

model = SVC()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print('classification_report for SVC')
print(classification_report(y_test,predictions))