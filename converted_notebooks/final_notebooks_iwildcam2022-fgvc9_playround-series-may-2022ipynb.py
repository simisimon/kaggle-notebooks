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


train_data=pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_data=pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")


# In[ ]:


train_data


# In[ ]:


test_data


# In[ ]:


train_data.isnull().sum()


# In[ ]:


categorical_cols = [cname for cname in train_data.columns if
                    train_data[cname].nunique() < 20 and 
                    train_data[cname].dtype == "object"]
categorical_cols


# In[ ]:


x_train=train_data.target
x_train


# In[ ]:


train_data1=train_data.drop('f_27',axis=1)
train_data1


# In[ ]:


test_data1=test_data.drop('f_27',axis=1)
test_data1


# In[ ]:


x_train=train_data1.iloc[:,1:31]
x_train


# In[ ]:


x_test=test_data1.iloc[:,1:31]
x_test


# In[ ]:


y_train=train_data.target
y_train


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# In[ ]:


'''
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)'''


# In[ ]:


from sklearn.linear_model import SGDClassifier
model1=SGDClassifier()
model1.fit(x_train,y_train)
predictions1 = model1.predict(x_test)
predictions1
score1=model1.score(x_train,y_train)
print(score1)


# In[ ]:


'''predictions = model.predict(x_test)
predictions'''


# In[ ]:


'''
score=model.score(x_train,y_train)
print(score)'''


# In[ ]:





# In[ ]:


output = pd.DataFrame({'id': test_data.id, 'target': predictions1})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

