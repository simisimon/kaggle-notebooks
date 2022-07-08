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


# # Importing Libraries

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import random
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().system('ls ../input/tabular-playground-series-feb-2022')


# In[ ]:


train = pd.read_csv("../input/tabular-playground-series-feb-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-feb-2022/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train.shape


# In[ ]:


train.isnull().sum().any()


# **No Null Values**

# In[ ]:


print(train['target'].value_counts())
plt.figure(figsize = (25,5))
sns.countplot(x = train['target'])
plt.xlabel('Target', size = 12)
plt.ylabel('Count', size = 12)
plt.title('Vaariation in Target Variabel', size = 12)


# **Target Class is Balanced**

# In[ ]:


la = LabelEncoder()
train['target'] = la.fit_transform(train['target'])


# In[ ]:


train['target'].head()


# In[ ]:


train[train.duplicated()]


# **No Duplicate Values Found**

# # Visualization

# In[ ]:


col = train.columns
plt.figure(figsize = (25,280))
color = ['r','g','b','c','y']
for i in range (len(train.columns)-1):
    plt.subplot(72, 4, i+1)
#     train[col[i]].plot(kind = 'bar')
    c = random.choice(color)
    sns.distplot(train[col[i]], hist = True, color = c)


# In[ ]:


cor = train.corr()
rel = cor['target'].sort_values(ascending = False)


# In[ ]:


l = []
for i in range (len(rel)):
    if rel[i] > 0:
        l.append(rel.index[i])
len(l)


# In[ ]:


# x = train.loc[:,l]
# x.drop('target', inplace = True, axis = 1)
# x.head()


# In[ ]:


y = train.loc[:, 'target']
y.head()


# In[ ]:


x = train.loc[:,:]
x.drop('target', inplace = True, axis = 1)
x.head()


# # Building Models

# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 42)


# # Random Forest

# In[ ]:


rcla = RandomForestClassifier(n_estimators = 450)
rcla.fit(xtrain, ytrain)


# In[ ]:


predicted = rcla.predict(xtrain)
print("Accuracy using Random Forest on training data is {} %".format(accuracy_score(predicted, ytrain)*100))


# In[ ]:


predicted = rcla.predict(xtest)
print("Accuracy using Random Forest testing data is {} %".format(accuracy_score(predicted, ytest)*100))


# # Decision Tree

# In[ ]:


dcla = RandomForestClassifier()
dcla.fit(xtrain, ytrain)


# In[ ]:


predicted = dcla.predict(xtrain)
print("Accuracy using Random Forest on training data is {} %".format(accuracy_score(predicted, ytrain)*100))


# In[ ]:


predicted = dcla.predict(xtest)
print("Accuracy using Random Forest testing data is {} %".format(accuracy_score(predicted, ytest)*100))


# # XGBoost

# In[ ]:


xcla = XGBClassifier()
xcla.fit(xtrain, ytrain)


# In[ ]:


predicted = xcla.predict(xtrain)
acu = accuracy_score(predicted, ytrain)
print("Accuracy using XGBoost on training is {} %".format(acu*100))


# In[ ]:


predicted = xcla.predict(xtest)
acu = accuracy_score(predicted, ytest)
print("Accuracy using XGBoost on testing is {} %".format(acu*100))


# # Making Prediction On Testing Data

# In[ ]:


test.head()


# In[ ]:


x_test = test.loc[:, :]
pred_test = rcla.predict(x_test)
pred_test


# In[ ]:


predicted_test = la.inverse_transform(pred_test)
predicted_test


# In[ ]:


submit = pd.DataFrame(data = {'row_id': test['row_id'], 'target': predicted_test})


# In[ ]:


submit.head()


# In[ ]:


submit.to_csv('submission8', index = False)
submit.head()


# In[ ]:


sample_sub = pd.read_csv("../input/tabular-playground-series-feb-2022/sample_submission.csv")
sample_sub.head()


# In[ ]:




