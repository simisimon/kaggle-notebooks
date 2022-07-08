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


buffer_df = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv('../input/titanic/test.csv')
test_df.head()


# In[ ]:


train_df = train_df.drop(['PassengerId','Name','Ticket', 'Embarked', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId','Name','Ticket', 'Embarked', 'Cabin'], axis=1)


# In[ ]:


print(train_df.head())
print("###############################################################################################################")
print(test_df.head())


# In[ ]:


train_df.describe()


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())


# In[ ]:


print(train_df.isnull().sum())
print("-------------------------------------------------------------------------------------------------------")
print(test_df.isnull().sum())


# In[ ]:


train_df['Sex'].value_counts()


# In[ ]:


train_df['Sex'] = train_df['Sex'].replace({'male':0, 'female':1})
test_df['Sex'] = test_df['Sex'].replace({'male':0, 'female':1})
print(train_df.info())
print("=="*50)
print(test_df.info())


# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[ ]:


sb.pairplot(train_df, hue='Survived')


# In[ ]:


X = train_df.drop(['Survived'], axis =1)
y = train_df['Survived']


# In[ ]:


columns_list = list(test_df.columns)


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
scale = make_column_transformer(
    (MinMaxScaler(), columns_list),
    (StandardScaler(), columns_list),
    remainder = 'passthrough'
)

X= scale.fit_transform(X)
test_df = scale.transform(test_df)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rd = RandomForestClassifier(n_estimators = 101, n_jobs=-1)
rd.fit(X,y)


# In[ ]:


preds = rd.predict(test_df)
preds


# In[ ]:


ids = buffer_df['PassengerId']
ids


# In[ ]:


subs = pd.DataFrame(ids)
subs['Survived'] = preds
subs.head()


# In[ ]:


subs.to_csv('Submission.csv', index = False)


# In[ ]:




