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


# importing relevant libraraies
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import tensorflow


# In[ ]:


df=pd.read_csv("../input/pump-sensor-data/sensor.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# ### 2,20,320 rows and 55 columns 
# #### 52 sensors data 1 timestamp and 1 index and 1 condition column are present
# ##### data is captured from 1 april 2018 to 31 august 2018 every minute

# In[ ]:


for i in df.isnull().sum().index:
    print(df.isnull().sum()[i])


# if a cloumn is having more than 1000 null values then it will be droped

# In[ ]:


l=['Unnamed: 0']
for i in df.isnull().sum().index:
    #print (df.isnull().sum()[i], end=" ")
    if (df.isnull().sum()[i]>1000):
        l.append(i)
l


# In[ ]:


df.drop(labels=l,axis=1,inplace=True)


# In[ ]:


df.columns


# ## Machine status distribution

# In[ ]:


df['machine_status'].value_counts()


# In[ ]:


plt.plot(df['machine_status'])


# - mapping normal, recovering and broken to 1,0.5,0

# In[ ]:


m={'NORMAL':1,'RECOVERING':0.5,'BROKEN':0}
df['stat']=df['machine_status'].map(m)
df['rol']=df['stat'].rolling(2).mean()


# In[ ]:


sns.set(rc={"figure.figsize":(15, 4)})
plt.plot(df['stat'])


# ### Correlation to see all the relevant columns

# In[ ]:


co=df.corr()


# In[ ]:


sns.heatmap(co)


# removing redundancy in the heatmap and using suitable colour pallete

# In[ ]:


mask = np.triu(np.ones_like(co))
sns.heatmap(co,cmap='vlag',mask=mask)


# high coorelations with machine status

# In[ ]:


co['stat']


# taking the higer correlating columns for ML model

# In[ ]:


l=[]
for i in co['stat'].index:
    if co['stat'][i]>0.75:
        print (i)
        l.append(i)


# 

# In[ ]:


l


# In[ ]:


df[l].plot(subplots =True, sharex = True, figsize = (20,30))


# ### removing last 2 items from the list

# In[ ]:


l.pop()
l.pop()


# In[ ]:


l


# In[ ]:


df.fillna(df.mean(),inplace=True)


# In[ ]:


X=df[['sensor_01', 'sensor_02', 'sensor_03', 'sensor_04',
       'sensor_05', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13',
       'sensor_14', 'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19',
       'sensor_20', 'sensor_21', 'sensor_22', 'sensor_23', 'sensor_24',
       'sensor_25', 'sensor_26', 'sensor_27', 'sensor_28', 'sensor_29',
       'sensor_30', 'sensor_31', 'sensor_32', 'sensor_33', 'sensor_34',
       'sensor_35', 'sensor_36', 'sensor_37', 'sensor_38', 'sensor_39',
       'sensor_40', 'sensor_41', 'sensor_42', 'sensor_43', 'sensor_44',
       'sensor_45', 'sensor_46', 'sensor_47', 'sensor_48', 'sensor_49']]
y=df['stat']


# In[ ]:


X_train=X[:50000]
y_train=y[:50000]
X_test=X[50000:]
y_test=y[50000:]


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


plt.plot(lr.predict(X_test))


# here we can see that there is lot of disturbance and we will not able to predict when a machine is going to fail. 

# In[ ]:


plt.plot(lr.predict(X_test))
plt.plot(y_test.reset_index(drop=True))


# there are 4 instance where model is predicting a failure but it turns out to be inaccurate

# Now we try with only variables of high coorelations

# In[ ]:


X=df[l]
y=df['stat']

X_train=X[:50000]
y_train=y[:50000]
X_test=X[50000:]
y_test=y[50000:]


# In[ ]:


X


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


plt.plot(lr.predict(X_test))
plt.plot(y_test.reset_index(drop=True))


# we can see that disturbance is reduced by a large extent and blue line nearly matches with the red line meeaning this turns out to be a better predicting model. Further Accuracy can be increased using Lstm from Tensorflow.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X=df[['sensor_02', 'sensor_04', 'sensor_10', 'sensor_11', 'sensor_12', ]]
y=df['stat']

X_train=X[:50000]
y_train=y[:50000]
X_test=X[50000:]
y_test=y[50000:]


# In[ ]:


scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential


# In[ ]:


y_train


# In[ ]:


df=df[['sensor_02', 'sensor_04', 'sensor_10', 'sensor_11', 'sensor_12','stat' ]]


# In[ ]:


df.fillna(df.mean(),inplace=True)


# In[ ]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]        
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg


# In[ ]:


X=df[['sensor_02', 'sensor_04', 'sensor_10', 'sensor_11', 'sensor_12']]
y=df['stat' ]


# In[ ]:





# In[ ]:


from sklearn.preprocessing import MinMaxScaler

values = X.values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)


# In[ ]:


reframed = series_to_supervised(scaled, 1, 1)
r = list(range(X.shape[1]+1, 2*X.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed


# In[ ]:


values = reframed.values
n_train_time = 50000
X_train = values[:n_train_time]
X_test = values[n_train_time:]
y_train= y[:n_train_time]
y_test = y[n_train_time:]


# In[ ]:


X_train=X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test=X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


y_train.shape


# In[ ]:


y_test=y_test[:-1]


# In[ ]:


model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


history=model.fit(X_train,y_train, epochs=50, batch_size=70, validation_data=(X_test, y_test), verbose=1, shuffle=False)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


pre=model.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_squared_error(pre,y_test)**0.5


# In[ ]:


plt.plot(pre)
plt.plot(y_test.reset_index(drop=True))


# In[ ]:


plt.plot(pre)


# LSTM model turns out to be more accurate and best amonmg the Three

# In[ ]:




