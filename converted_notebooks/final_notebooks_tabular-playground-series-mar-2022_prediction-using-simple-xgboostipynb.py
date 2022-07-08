#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
import sys
import warnings
import zipfile
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import optuna
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Exploring Dataset

# In[ ]:


# Declaring data directory
data_dir = os.path.join('/kaggle/input', 'tabular-playground-series-mar-2022')


# In[ ]:


# Declaring path of training and testing data
train_path = os.path.join(data_dir, 'train.csv') 
test_path = os.path.join(data_dir, 'test.csv')

# Creating dataframes for training and testing data
df_train = pd.read_csv(train_path, encoding='latin1')
df_test = pd.read_csv(test_path, encoding='latin1')


# In[ ]:


df_train.head()


# In[ ]:


# Shape of training data
df_train.shape


# In[ ]:


df_test.head()


# In[ ]:


# Shape of testing data
df_test.shape


# In[ ]:


# Checking for null values in the dataset

if (df_train.isnull().values.any() == False) :
    print('There are no null values in Training data')
else :
    print("There are null values in Training data")


# In[ ]:


# Checking for null values in the dataset

if (df_test.isnull().values.any() == False) :
    print('There are no null values in Testing data')
else :
    print("There are null values in Testing data")


# In[ ]:


# Info of df_train

df_train.info()


# ## Preparing the dataset

# In[ ]:


# Converting Dtype of df_train['time'] and df_test['time'] to datetime64[ns]

df_train['time'] = pd.to_datetime(df_train['time'])
df_test['time'] = pd.to_datetime(df_test['time'])


# In[ ]:


# Info of df_train

df_train.info()


# In[ ]:


# Info of df_test

df_test.info()


# In[ ]:


# Splitting 'time' column into individual columns of 'dayofweek', 'hour' and 'minute'
# And dropping 'time' column

df_train['dayofweek'] = df_train['time'].dt.dayofweek
df_train['hour'] = df_train['time'].dt.hour
df_train['minute'] = df_train['time'].dt.minute
df_train.drop('time', inplace=True, axis=1)

df_test['dayofweek'] = df_test['time'].dt.dayofweek
df_test['hour'] = df_test['time'].dt.hour
df_test['minute'] = df_test['time'].dt.minute
df_test.drop('time', inplace=True, axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ## Prepare data for training

# In[ ]:


# Let us consider features as X and labels as Y
# Dropping row_id, dayofweek and congestion

X = df_train.drop(['row_id', 'congestion', 'dayofweek'], axis=1)
Y = df_train['congestion']


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


# As 'direction' is in string format, we have to encode them into integers to train the model
# Creating hash for directions
dict = {
    
    'EB' : 0,
    'NB' : 1,
    'NE' : 2,
    'NW' : 3,
    'SB' : 4,
    'SE' : 5,
    'SW' : 6,
    'WB' : 7
}


# In[ ]:


# Encoding the values of 'direction' column

X['direction'] = X['direction'].apply(lambda i : dict[i])


# In[ ]:


X


# ## Training the Model

# In[ ]:


# Splitting training and testing data with test size = 0.3 as we have enough data to do so

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# In[ ]:


# Preparing the regressor and fitting data with some parameters
xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
                          colsample_bytree = 1, 
                          learning_rate = 0.3,
                          max_depth = 15, 
                          alpha = 10, 
                          n_estimators = 100, 
                          verbose=1, 
                          min_child_weight = 1, 
                          colsample_bylevel = 1, 
                          reg_alpha = 2) 

xg_reg.fit(X_train, Y_train)


# In[ ]:


# Modifying df_test to make predictions

df_test['direction'] = df_test['direction'].apply(lambda i : dict[i])
df_test.drop(['row_id', 'dayofweek'], axis=1, inplace=True)


# In[ ]:


# Making predictions

pred = xg_reg.predict(df_test)


# In[ ]:


# Submitting the predictions

df_sam = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'), encoding='latin1')
df_sam['congestion'] = pred
df_sam.to_csv('submission-1.csv', index=False)

