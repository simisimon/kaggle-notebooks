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


# dir(np)


# In[ ]:


# help(np)


# In[ ]:


test = pd.read_csv("../input/spaceship-titanic/test.csv")
print(test.head())
test.shape
# (4277, 13)


# In[ ]:


train = pd.read_csv("../input/spaceship-titanic/train.csv")
print(train)
train.shape 
# (8693, 14)


# In[ ]:


train.dtypes


# In[ ]:


train.isnull ().sum ()


# In[ ]:


test_id = train.PassengerId 
test_id


# In[ ]:


train = train.drop(columns = ["PassengerId", "Cabin", "Name"])


# In[ ]:


train = train.dropna(axis=0)
print(train.columns)
print(train.shape)


# In[ ]:


train = pd.get_dummies(train)
print(train.columns)
print(train.shape)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.head()


# In[ ]:


x_train = train.drop("Transported", axis=1)
x_train.head()


# In[ ]:


y_train = train.Transported
y_train.head()


# In[ ]:


test.dtypes


# In[ ]:


test.isnull ().sum()


# In[ ]:


test = test.dropna(axis=0)
test.columns


# In[ ]:


test_id = test.PassengerId
print(test_id)


# In[ ]:


x_test = test.drop(columns = ["PassengerId", "Cabin", "Name"])
x_test.head()


# In[ ]:


x_test.shape
# (3438, 10)


# In[ ]:


x_test = pd.get_dummies(x_test)
print(x_test.columns)
print(x_test.shape)


# In[ ]:


x_test.isnull ().sum ()


# In[ ]:


# import pandas_profiling
# pandas_profiling.ProfileReport(train)


# In[ ]:


# import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0) # default max_depth=3, learning_rate=0.1
gbc.fit(x_train, y_train)

score_gbc = gbc.score(x_train, y_train) # train set accuracy
print(score_gbc)
# 81.395 %


# In[ ]:


# dir(gbc)
# help(gbc.score)


# In[ ]:


gbc.predict(x_test)


# In[ ]:


# 결정트리, Random Forest, 로지스틱 회귀
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=0)
rf_clf = RandomForestClassifier(random_state=0)
lr_clf = LogisticRegression(random_state=0)


# In[ ]:


# DecisionTreeClassifier 학습평가
dt_clf.fit(x_train, y_train)
score_dt = dt_clf.score(x_train, y_train) # train set accuracy
print(score_dt)

# DecisionTreeClasifier 정확도: 93.1821


# In[ ]:


# Random Forest 학습평가
rf_clf.fit(x_train, y_train)
score_rf = rf_clf.score(x_train, y_train) # train set accuracy
print(score_rf)
# Random Forest 정확도: 93.18214


# In[ ]:


# 로지스틱 회귀 학습/예측/평가
lr_clf.fit(x_train, y_train)
score_lr = lr_clf.score(x_train, y_train) # train set accuracy
print(score_lr)
# logistic regression 정확도: 78.7520


# In[ ]:


pd.DataFrame(
    {'PassengerId': test_id, 
     'Transported': rf_clf.predict(x_test).astype(bool)}
).to_csv('/kaggle/working/submission.csv', index=False)

