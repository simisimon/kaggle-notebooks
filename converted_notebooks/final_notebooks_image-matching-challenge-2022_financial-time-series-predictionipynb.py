#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input data files are available in the read-only '../input/' directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
AAPL = '/kaggle/input/historical-prices/AAPL.csv'
GOOG = '/kaggle/input/historical-prices/GOOG.csv'
JPM = '/kaggle/input/historical-prices/JPM.csv'
JNJ = '/kaggle/input/historical-prices/JNJ.csv'

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using 'Save & Run All' 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Train-test Data Handling

def data_read(fn):
    return pd.read_csv(fn)

def data_cleansing(tb):
    datetime = pd.to_datetime(tb['Date'])
    tb['year'] = datetime.dt.year
    tb['month'] = datetime.dt.month
    tb['day'] = datetime.dt.day
    tb['dayofweek'] = datetime.dt.dayofweek
    return tb.drop(['Date'], axis=1)

def data_normalise(tb):
    tb[tb.columns] = scaler.fit_transform(tb)
    return tb

# def data_inverse_normalise(tb):
#     tb[tb.columns] = scaler.inverse_transform(tb)
#     return tb

# def build():
#     pass

def build_all(tb, timestep, predictstep):
    x, y = [], []
    for i in range(tb.shape[0]-timestep-predictstep):
        x.append(np.array(tb.iloc[i: i+timestep]))
        y.append(np.array(tb.iloc[i+timestep: i+timestep+predictstep]["Adj Close"]))
        
    return np.array(x), np.array(y)

def build_train_test(fn, timestep, predictstep, ratio=0.9):
    tb = data_read(fn)
    tb = data_cleansing(tb)
    tb = data_normalise(tb)
    
    x_train, y_train = build_all(tb[:int(tb.shape[0]*ratio)], timestep, predictstep)
    x_test, y_test = build_all(tb[int(tb.shape[0]*ratio):], timestep, predictstep)
    
    return x_train, y_train, x_test, y_test


# In[ ]:


# Plotting Methods

def plot_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    
def plot_res(predict, actual):
    date_index = list(pd.read_csv(AAPL)['Date'][-416:])
    
    plt.figure(figsize=(8,4))
    plt.plot(actual)
    plt.plot(predict)
    plt.title('Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price (normalised)')
    plt.xticks(list(range(0, len(date_index), 100)), [date_index[i] for i in range(0, len(date_index), 100)])
    plt.legend(['Actual', 'Predict'], loc='upper left')


# In[ ]:


# Evaluation Metric

def percentage_difference(result, target):
    res = (result - target) / target
    return abs(res)

def evaluation(results, targets):
    acc, maxd, mind = 0, 0, float('inf')
    base = len(results)
    for i in range(base):
        diff = percentage_difference(results[i], targets[i])
        acc += diff
        maxd = max(maxd, diff)
        mind = min(mind, diff)
        
    return acc/base, maxd, mind


# In[ ]:


# Model Construction/ Train

def model_build(timestep, predictstep):
    model = Sequential()
#     model.add(LSTM(32, input_shape=(timestep, 10), return_sequences=True))
#     model.add(Dropout(0.2))
    model.add(LSTM(10, input_shape=(timestep, 10)))
#     model.add(Dropout(0.2))
    model.add(Dense(predictstep))
    model.compile(loss="mse", optimizer='adam')
    model.summary()
    return model

def model_train(fn, timestep=100, predictstep=1):
    # clear_session()
    x_train, y_train, x_test, y_test = build_train_test(fn, timestep, predictstep)
    model = model_build(timestep, predictstep)

    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    checkpoint = ModelCheckpoint(f"{fn}_best", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1, validation_data=(x_test, y_test), callbacks=[callback, checkpoint])
    
    return {'model': model, 'history': history, 'x': x_test, 'y': y_test}


# In[ ]:


# clear_session()
aapl = model_train(AAPL)
plot_loss(aapl['history'])
aaplres = aapl['model'].predict(aapl['x'])
plot_res(aaplres, aapl['y'])
print(evaluation(aaplres, aapl['y']))


# In[ ]:


# clear_session()
goog = model_train(GOOG)
plot_loss(goog['history'])
googres = goog['model'].predict(goog['x'])
plot_res(googres, goog['y'])
print(evaluation(googres, goog['y']))


# In[ ]:


# clear_session()
jpm = model_train(JPM)
plot_loss(jpm['history'])
jpmres = jpm['model'].predict(jpm['x'])
plot_res(jpmres, jpm['y'])
print(evaluation(jpmres, jpm['y']))


# In[ ]:


# clear_session()
jnj = model_train(JNJ)
plot_loss(jnj['history'])
jnjres = jnj['model'].predict(jnj['x'])
plot_res(jnjres, jnj['y'])
print(evaluation(jnjres, jnj['y']))


# In[ ]:


plt.plot(jpm['y'][-50:])
plt.plot([x for x in jpmres[-50:]])
plt.legend(['Actual', 'Predict'], loc='upper right')


# In[ ]:





# In[ ]:




