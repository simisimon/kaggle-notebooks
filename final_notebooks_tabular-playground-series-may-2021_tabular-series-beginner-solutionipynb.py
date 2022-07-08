#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action="ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_df = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/test.csv")
#display(train_df.head(10))
#display(test_df.head(10))
print(train_df.columns)
print(test_df.columns)


# In[ ]:


print(train_df.target.nunique())
unique_values = {}
mx = -1
mn = 10000001
for feature in train_df.columns.to_list():
    if feature != "id":
        unique_values["feature"] = train_df[feature].nunique()
        #print(unique_values["feature"])
        mx = max(mx, (unique_values["feature"]))
        mn = min(mn, (unique_values["feature"]))
print(mx, mn)


# In[ ]:


unique_values = {}
mx = -1
mn = 10000001
for feature in test_df.columns.to_list():
    if feature != "id":
        unique_values["feature"] = test_df[feature].nunique()
        #print(unique_values["feature"])
        mx = max(mx, (unique_values["feature"]))
        mn = min(mn, (unique_values["feature"]))
print(mx, mn)


# Null Values Check

# In[ ]:


#test_df.isnull().sum()
# okay, we dont;t have any null values


# Correlation Matrix

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

cor = train_df.corr()
#f, ax = plt.subplots(figsize=(20, 20))
#sns.heatmap(cor, vmax=.8, square=True, annot= True);


# In[ ]:


le = LabelEncoder()

X = train_df.drop(["id", "target"], axis = 1)
y =le.fit_transform(train_df.target)
print(y[0:100])


# In[ ]:


test_df_without_id = test_df.drop("id", axis = 1)


# Splitting the data

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pca = PCA(n_components=47)
# prepare transform on dataset
pca.fit(X)
# apply transform to dataset

transformed = pca.transform(X)
print(transformed.shape)
#test_df_without_id = pca.transform(test_df_without_id)



# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)
print(X_train.shape)
print(X_val.shape)


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix, log_loss

xg = XGBClassifier(n_estimators = 150, learning_rate = 0.1, random_state = 0)
lgbm = LGBMClassifier(random_state=12, num_iterations = 60, learning_rate = 0.1
                     )
cbc = CatBoostClassifier(random_state= 12)
#rc = RandomForestClassifier(max_features= 7, random_state= 0, n_estimators = 150)
"""
rc.fit(X_train, y_train)
y_pred=rc.predict_proba(X_val)
lost = log_loss(y_val, y_pred)
print(lost)
cbc.fit(X_train, y_train)
y_pred=cbc.predict_proba(X_val)
cm = log_loss(y_val, y_pred)
print(cm)
xg.fit(X_train, y_train)
y_pred=xg.predict_proba(X_val)
cm = log_loss(y_val, y_pred)
print(cm)
"""

lgbm.fit(X_train, y_train)
y_pred=lgbm.predict_proba(X_val)
cm = log_loss(y_val, y_pred)
print(cm)


# In[ ]:


res = lgbm.predict_proba(test_df_without_id)


# In[ ]:


print(res[300:400])


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/tabular-playground-series-may-2021/sample_submission.csv")
sample_submission.columns


# In[ ]:


submission = pd.DataFrame({'id': sample_submission['id'],
                           'Class_1': res[:, 0],
                           'Class_2': res[:, 1],
                           'Class_3': res[:, 2],
                           'Class_4': res[:, 3],})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




