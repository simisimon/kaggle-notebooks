#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Training Dataset :-**

# In[ ]:


df_train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
df_train.head(10)


# In[ ]:


# Renaming some column names for convineience
for i in range(10):
    df_train=df_train.rename(columns = {f'f_0{i}':f'f_{i}'})


# In[ ]:


df_train.head(5)


# In[ ]:


df_train.shape


# In[ ]:


df_train.info()


# EDA

# In[ ]:


sns.displot(df_train,x='target',bins=5)
# So it is a balanced dataset
df_train['target'].value_counts()


# In[ ]:


sns.displot(df_train, x='f_0', hue='target', bins=10)


# In[ ]:


sns.displot(df_train, x='f_0', col='target', bins=10)


# In[ ]:


del df_train['f_27']


# In[ ]:


y = df_train['target']


# In[ ]:


del df_train['target']


# **Building Model**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(df_train, y, test_size=0.2, random_state=42)


# In[ ]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# In[ ]:


model = AdaBoostClassifier()


# In[ ]:


model.fit(x_train_scaled,y_train)


# In[ ]:


model.score(x_test_scaled, y_test)


# In[ ]:


df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
df_test.head(10)


# In[ ]:


del df_test['f_27']


# In[ ]:


for i in range(10):
    df_test=df_test.rename(columns = {f'f_0{i}':f'f_{i}'})


# In[ ]:


predictions = model.predict(df_test)


# In[ ]:


submission = pd.DataFrame({
    'id':df_test['id'],
    'target':predictions
})


# In[ ]:


submission.to_csv('submission_2.csv', index=False)


# In[ ]:




