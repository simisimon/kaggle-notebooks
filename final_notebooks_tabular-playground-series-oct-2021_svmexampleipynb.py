#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import io

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA


# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
np.random.seed(0)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")
data.head()
data.shape


# In[ ]:


# Checking distribution of target variables 
sns.countplot(x = 'fetal_health' ,data = data)


# In[ ]:


X = data.drop(['fetal_health'], axis=1)
y = data['fetal_health']
# define oversampling strategy
oversample = RandomOverSampler()
# fit and apply the transform
X, y = oversample.fit_resample(X, y)
# reducing the dimensions to 2 
pca = PCA(n_components=2)
X = pd.DataFrame(pca.fit_transform(X))
y.value_counts()


# In[ ]:


# functions to plot decision boundary

def make_meshgrid(x, y, h = 0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
clf1 = SVC(kernel='linear', C = 0.1)

clf1.fit(x_train, y_train)

y_pred_train1 = clf1.predict(x_train)
print('Accuracy:', accuracy_score(y_train,y_pred_train1))

y_pred_test1 = clf1.predict(x_test)
print('\nAccuracy:', accuracy_score(y_test,y_pred_test1))


# In[ ]:


X1 = scaler.fit_transform(X)
fig, ax = plt.subplots(figsize=(7,5))
title = ('Decision surface of linear SVC with C = 0.1')
X0, X1 = X1[:, 0], X1[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf1, xx, yy, cmap=plt.cm.Paired, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.Paired, s=40, edgecolors='k')
ax.set_ylabel('y label')
ax.set_xlabel('x label')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

