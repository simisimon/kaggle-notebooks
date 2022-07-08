#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from keras import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sub_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df = [train_df,test_df]


# In[ ]:


train_df.head()


# In[ ]:


test_df.head(10)


# resize pixels

# In[ ]:


X_train = np.array(train_df.drop(['label'],axis=1)).reshape(len(train_df),28,28,1)
X_test = np.array(test_df).reshape(len(test_df),28,28,1)
X_train.shape


# In[ ]:


y_train = np.array(train_df['label']).reshape(len(train_df),1)
y_train.shape


# Normalize

# In[ ]:


#max color value
print(max(X_train.max(),X_test.max()))


# In[ ]:


X_train = X_train/255.
X_test = X_test/255.


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)


# In[ ]:


plt.imshow(X_train[0])
print(y_train[0])


# # Model

# In[ ]:


model = keras.Sequential([
        keras.layers.Input(shape=[28,28,1]),
        keras.layers.ZeroPadding2D(padding=(1,1),input_shape =[28,28,1]),
        keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=2,strides=2),
        keras.layers.ZeroPadding2D(padding=(1,1)),
        keras.layers.Dropout(0.1),
        
        keras.layers.Conv2D(filters = 64, kernel_size = 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=2,strides=2),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(filters = 256, kernel_size = 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=2,strides=2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dropout(0.1),
        
        keras.layers.Dense(10,activation='softmax')
    ])

model.summary()


# Compile

# In[ ]:


model.compile(
    optimizer='nadam',#nadam
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],#sparse_categorical_accuracy
)


# Callbacks for model

# In[ ]:


EPOCHS = 100

def scheduler(epoch):
    if epoch <= 25:
        return 0.001

    elif epoch <= 50:
        return (0.001) / 10
    
    elif epoch <= 75:
        return ((0.001/10) / 10)

    else:
        return (((0.001/10)/10) / 2.5)

def exponential_lr(epoch,
                   start_lr = 0.001, min_lr = 0.00001, max_lr = 0.001,
                   rampup_epochs = 0, sustain_epochs = 5,
                   exp_decay = 0.94):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay**(epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr
    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)

rng = [i for i in range(EPOCHS)]
y1 = [exponential_lr(x) for x in rng]
y2 = [scheduler(x) for x in rng]
plt.plot(rng, y1)
plt.plot(rng,y2)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y1[0], max(y1), y1[-1]))


lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=0)


es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0.0,
    patience=100,
    restore_best_weights=True,
    verbose =2,
)


# In[ ]:


history = model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    epochs=EPOCHS,
    batch_size=64,
    callbacks=[lr_callback,es_callback],
)


# In[ ]:


y_preds = model.predict(X_test)

preds = pd.Series([np.argmax(i) for i in y_preds], name='Label')
preds


# In[ ]:


sub_df['Label'] = preds
sub_df.head()


# In[ ]:


sub_df.to_csv('./submission.csv', index=False)

