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
print("Fetching the path and directory of the Datasets: ")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 
# ## **Importing Initial Libraries**
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

# Metrices for Evlaution
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# models
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

print("Imported")


# In[ ]:


# Unabling autocomplete in kaggle kernel
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[ ]:


print("Fetching the required datasets!")
train = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2021/train.csv')
test = pd.read_csv("/kaggle/input/tabular-playground-series-jan-2021/test.csv")
sample_sub = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2021/sample_submission.csv')
print("Read Train, test and Sample submission dataset from kaggle dir!")


# ## **Exploring the data**
# 
# Understanding below things:
# 
# 1. Shape
# 2. Stats Summary
# 3. Missing values
# 4. info of dataset
# 5. unique values
# 

# **Exploring train dataset**

# In[ ]:


print(f'Shape of the train dataset: {train.shape}')
print('Stats Description of dataset:')
train.describe()


# In[ ]:


print(f'Missing values in train dataset: {train.isnull().sum().sum()}')
print('\n-------------------')
print('Train Detailed overall information:')
train.info()
print('\n-------------------\n')
print("Nunique values are: ")
train.nunique()


# In[ ]:


train.head()


# **Conclusions**

# *  Training data has 300000 records and 16 features.
# *  Column 'id'is the primary key.
# *   It's a regression problem since we need to predict the 'target' feature which is continous in nature.
# *   There are 14 numerical features which are already scaled.
# *   There is no missing value in this data and all features are numerical.
# 

# **Exploring the Test Dataset**

# In[ ]:


print(f'Shape of the train dataset: {test.shape}')
print('Stats Description of dataset:')
test.describe()


# In[ ]:


print(f'Missing values in train dataset: {test.isnull().sum().sum()}')
print('\n-------------------')
print('Train Detailed overall information:')
test.info()
print('\n-------------------\n')
print("Nunique values are: ")
test.nunique()


# In[ ]:


test.head()


# **Conclusion**
# *  Test data has 200000 records and 15 features. 'Target' feature is absent as expected.
# *  Column 'id' is the primary key.
# *  There are 14 numerical features which are already scaled.
# *  There is no missing value in this data and all features are numerical.
# 
# 

# ## **Pre- Modeling**
# 
# 1. Defining the Dependent/target and Independent/Feature variables

# **Train Test Split**

# In[ ]:


# Separating the target variable and removing the 'id' column
X = train.drop(['target','id'],axis=1)
y = train['target']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=40)


# 
# ## **Naive Model**
# 
# This Naive model will 'predict' the median target value for all the records in test data.
# 
# This step is important to set up a benchmark score and improve further on that.
# 

# In[ ]:


# Model instance
model_dummy = DummyRegressor(strategy='mean')
# Model fitting to train dataset
model_dummy.fit(X_train,y_train)

# Prediciting the model
y_dummy = model_dummy.predict(X_test)

# Performance evaluation of model
score_dummy = mean_squared_error(y_test,y_dummy, squared=False)
print(f'{score_dummy:0.5f}')


# In[ ]:


test.shape, train.shape


# ## **Simple ML model Evalution**
# 
# 
# Let's start with some simple ML models to see how well they perform with respect to the naive model score.
# 

# In[ ]:


model_names = ['Linear','Lasso', 'Ridge', 'Decision Tree']

models = [
    LinearRegression(fit_intercept=True),
    Lasso(fit_intercept=True),
    Ridge(fit_intercept=True),
    DecisionTreeRegressor()
]

for name, model in zip(model_names,models):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    print(f'{name}: RMSE:{score}')


# In[ ]:


#Submitting the results from the best performing model so far.
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
sample_sub['target'] = model.predict(test.drop('id', axis = 1))
sample_sub.to_csv('simple_ml_model.csv', index = False)

