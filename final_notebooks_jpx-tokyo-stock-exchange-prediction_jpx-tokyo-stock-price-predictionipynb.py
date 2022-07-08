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


import pandas as pd
import numpy as np
df = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


import math
import pandas as pd
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.columns = df.columns.str.replace('EffectiveDate','datetime')


# In[ ]:


df['datetime'] = pd.to_datetime(df['datetime'],format='%Y%m%d')


# In[ ]:


df.head(2)


# In[ ]:


df = df.set_index('datetime')


# In[ ]:


df.head()


# In[ ]:


# Create a new dataframe with only the close column
data = df.filter(['Close'])


# In[ ]:


# convet the dataframe to a numpy array
dataset = data.values


# In[ ]:


# get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
training_data_len


# In[ ]:


# Scale the Data
scaler = MinMaxScaler(feature_range=(0,1))


# In[ ]:


scaled_data = scaler.fit_transform(dataset)


# In[ ]:


scaled_data


# In[ ]:


# Create the training dataset
# Create the scaled dataset
train_data = scaled_data[0:training_data_len,:]


# In[ ]:


scaled_data[0:training_data_len,:]


# In[ ]:


X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=61:
        print(X_train)
        print(y_train)
        print()


# In[ ]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[ ]:


# Reshape tha data because LSTM model expect 3D data
X_train.shape


# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# In[ ]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(50 ,return_sequences=True,  input_shape = (X_train.shape[1], 1)))
model.add(LSTM(50 ,return_sequences=True))
model.add(LSTM(35, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


from keras import metrics
# Train the model
history = model.fit(X_train, y_train, epochs=5)


# In[ ]:


# Create the testing dataset
# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60 : , :]
# Create the daatset X_test and y_test
X_test = []
y_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])


# In[ ]:


# COnvert the data to a numpy array
X_test = np.array(X_test)


# In[ ]:


# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[ ]:


# Get the models predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(predictions, y_test)


# In[ ]:


# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse


# In[ ]:




