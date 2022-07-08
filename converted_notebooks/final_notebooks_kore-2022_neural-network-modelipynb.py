#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from keras import models
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow.keras import regularizers


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
sample_sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# EDA and feature selection can be found in my other [notebook](https://www.kaggle.com/code/robertturro/tps-may-2022-eda-feature-selection)

# From [Engineering the top three feature interactions](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/discussion/323892)

# In[ ]:


train['i_02_21'] = (train.f_21 + train.f_02 > 5.2).astype(int) - (train.f_21 + train.f_02 < -5.3).astype(int)
train['i_05_22'] = (train.f_22 + train.f_05 > 5.1).astype(int) - (train.f_22 + train.f_05 < -5.4).astype(int)
i_00_01_26 = train.f_00 + train.f_01 + train.f_26
train['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
test['i_02_21'] = (test.f_21 + test.f_02 > 5.2).astype(int) - (test.f_21 + test.f_02 < -5.3).astype(int)
test['i_05_22'] = (test.f_22 + test.f_05 > 5.1).astype(int) - (test.f_22 + test.f_05 < -5.4).astype(int)
i_00_01_26 = test.f_00 + test.f_01 + test.f_26
test['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)


# In[ ]:


y_train = train['target']
train = train.drop(['target'],axis=1)


# From [TPS May: Decoding f_27](https://www.kaggle.com/code/abhishek123maurya/tps-may-decoding-f-27)

# In[ ]:


def add_letters_count(data):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for char in letters:
        data[char] = data['f_27'].str.count(char)
        
    return data

train = add_letters_count(train)
test = add_letters_count(test)
train.head()


# In[ ]:


def remove_zero(data):
    cross_check = 'UVWXYZ'
    for char in cross_check:
        if data[char].sum() == 0:
            data = data.drop([char], axis=1)
            
    return data

train = remove_zero(train)
test = remove_zero(test)
train.head()


# In[ ]:


def add_pos(data):
    for i in range(10):
        data['pos' + str(i)] = (data['f_27'].str[i]).apply(lambda x: ord(x)) - 75
        
    return data

train = add_pos(train)
test = add_pos(test)
train.head()


# In[ ]:


train = train.drop('f_27',axis=1)
test = test.drop('f_27',axis=1)


# In[ ]:


train = train.set_index('id')
test = test.set_index('id')

train_columns = train.columns
test_columns = test.columns


# In[ ]:


scaler = StandardScaler()
le = preprocessing.LabelEncoder()

train = scaler.fit_transform(train)
test = scaler.fit_transform(test)


# In[ ]:


train = pd.DataFrame(data=train,columns=train_columns)
test = pd.DataFrame(data=test,columns=test_columns)


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


model2 = models.Sequential()
model2.add(layers.Dense(750, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation="relu",input_shape=(63,)))
model2.add(layers.BatchNormalization())
model2.add(layers.Dense(512, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation="relu"))
model2.add(layers.Dropout(0.1))
model2.add(layers.Dense(200,kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation="relu"))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(60,kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation="relu"))
model2.add(layers.Dropout(0.3))
model2.add(layers.Dense(16,kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation="relu"))
model2.add(layers.Dense(1,activation='sigmoid'))

model2.summary()


# In[ ]:


model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])


# In[ ]:


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
model2.fit(train, y_train, validation_split=0.3, shuffle=True, epochs=200,batch_size=2000,callbacks=[callback])


# In[ ]:


preds = model2.predict(test)


# In[ ]:


sample_sub['target'] = preds


# In[ ]:


sample_sub.to_csv('submission.csv', index=False)

