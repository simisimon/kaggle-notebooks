#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau

from sklearn.model_selection import KFold,GroupKFold
from tensorflow.keras import layers


# In[ ]:


from sklearn.preprocessing import RobustScaler,StandardScaler
rb = RobustScaler()
sc = StandardScaler()


# In[ ]:


train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')


# In[ ]:


dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')


# In[ ]:


dataset.reset_index(inplace = True)


# In[ ]:


dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')


# In[ ]:


dataset.fillna(0,inplace = True)


# In[ ]:


dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)


# In[ ]:


dataset.shape


# In[ ]:


# X we will keep all columns execpt the last one 
X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)
# the last column is our label
y_train = dataset.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

# lets have a look on the shape 
print(X_train.shape,y_train.shape,X_test.shape)


# In[ ]:


save_best = tf.keras.callbacks.ModelCheckpoint("Model.h5", monitor='val_loss',verbose=1, save_best_only=True)


# In[ ]:


def build_model():
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(33, 1)))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='uniform'))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002), loss = 'mse', metrics=['mse'])

    model.summary()
    
    return model


# In[ ]:


model = build_model()


# In[ ]:


model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=512, verbose=1, callbacks=[save_best])


# In[ ]:


model.evaluate(X_train, y_train)


# In[ ]:


model = tf.keras.models.load_model('./Model.h5')


# In[ ]:


# creating submission file 
submission = model.predict(X_test, verbose=1)
# we will keep every value between 0 and 20
submission = submission.clip(0,20)
# creating dataframe with required columns 
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission.ravel()})
# creating csv file from dataframe
submission.to_csv('submission.csv',index = False)


# In[ ]:


submission['item_cnt_month'].value_counts()


# In[ ]:




