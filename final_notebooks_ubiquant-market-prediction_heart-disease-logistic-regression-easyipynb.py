#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-cleveland-uci/heart_cleveland_upload.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(17,7))
correlation = (df.corr())
sns.heatmap(correlation,annot=True,cmap='Blues')
plt.show()


# In[ ]:


df.sample(4)


# In[ ]:


X = df.drop("condition",axis="columns")
y = df.condition


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()


# In[ ]:


clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)


# In[ ]:


y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)


# In[ ]:


y_pred1,y_pred2


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy score of logistic regression",accuracy_score(y_test,y_pred1))
print("Accuracy score of decision tree",accuracy_score(y_test,y_pred2))


# In[ ]:


confusion_matrix(y_test,y_pred1)


# In[ ]:




