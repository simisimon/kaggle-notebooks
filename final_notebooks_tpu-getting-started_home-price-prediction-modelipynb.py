#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#get data
home_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df = pd.get_dummies(home_df)
df


# In[ ]:


corr = df.corr()
corr['SalePrice'].sort_values(ascending=False).head(15)


# In[ ]:


#columns of new df
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
           'TotalBsmtSF', '1stFlrSF', 'FullBath', 'BsmtQual', 'TotRmsAbvGrd',
           'YearBuilt', 'YearRemodAdd', 'KitchenQual', 'Foundation']


# In[ ]:


home_df.info()


# In[ ]:


#new dataframe
df = home_df[columns]
df


# In[ ]:


df_new = pd.get_dummies(df)
df_new.info()


# In[ ]:


# Import Deep Learning packages
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


model = keras.Sequential([
  layers.Dense(50, activation='relu', input_dim=24),
  layers.Dense(50, activation='relu', input_dim=64),
  layers.Dense(128, activation='relu', input_dim=128),
  layers.Dense(64, activation='relu', input_dim=64),
    layers.Dropout(0.2),

  layers.Dense(1, activation='linear')
])


# In[ ]:


optimizer = tensorflow.keras.optimizers.Adam(
    learning_rate=0.01)


# In[ ]:


model.compile(loss='mse', optimizer=optimizer)


# In[ ]:


model.summary()


# In[ ]:


y = df_new['SalePrice']
X = df_new.drop('SalePrice', axis=1)


# In[ ]:


X_array = np.array(X)
X_array


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_array, y, test_size=.20, random_state=42)


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


history = model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_val, y_val))


# In[ ]:


preds = model.predict(X_val)
from sklearn import metrics
metrics.r2_score(y_val, preds)  #r2 score 0.85


# In[ ]:





# In[ ]:




