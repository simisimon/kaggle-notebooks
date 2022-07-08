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

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Concatenate, LSTM, GRU
from tensorflow.keras.layers import Bidirectional, Multiply
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# set up tpu to acclerate model training >> not available

tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/train.csv')
df_train_labels = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/train_labels.csv')
df_test = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/test.csv')
df_smpl = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/sample_submission.csv')


# In[ ]:


# add lagged feature and difference features

features = df_train.columns.tolist()[3:]

def preprocessing(df):
    for feature in features:
        df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']
        
    #return df


# In[ ]:


preprocessing(df_train)


# In[ ]:


preprocessing(df_test)


# In[ ]:


# minmax scaling 

features = df_train.columns.tolist()[3:]
sc = MinMaxScaler()
df_train[features] = sc.fit_transform(df_train[features])
df_test[features] = sc.transform(df_test[features])


# In[ ]:


# standard scaling

features = df_train.columns.tolist()[3:]
std_sc = StandardScaler()
df_train[features] = std_sc.fit_transform(df_train[features])
df_test[features] = std_sc.transform(df_test[features])


# In[ ]:


# reshape for lstm model input
# The LSTM needs data with the format of [samples, time steps and features].

groups = df_train['sequence']
labels = df_train_labels['state']

df_train = df_train.drop(['sequence', 'subject', 'step'], axis=1).values
df_train = df_train.reshape(-1, 60, df_train.shape[-1])

df_test = df_test.drop(['sequence', 'subject', 'step'], axis=1).values
df_test = df_test.reshape(-1, 60, df_test.shape[-1])


# In[ ]:


# create model

def lstm_0422():
    with tpu_strategy.scope():
        x_input = Input(shape=(df_train.shape[-2:])) # (60,39)

        x1 = Bidirectional(LSTM(units=512, return_sequences=True))(x_input)
        x2 = Bidirectional(LSTM(units=256, return_sequences=True))(x1)
        z1 = Bidirectional(GRU(units=256, return_sequences=True))(x1)

        c = Concatenate(axis=2)([x2, z1])

        x3 = Bidirectional(LSTM(units=128, return_sequences=True))(c)

        x4 = GlobalMaxPooling1D()(x3)
        x5 = Dense(units=128, activation='selu')(x4)
        x_output = Dense(1, activation='sigmoid')(x5)

        model = Model(inputs=x_input, outputs=x_output, name='lstm_model')
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[AUC(name = 'auc')])
        
    return model


# In[ ]:


model = lstm_0422()
model.summary()


# In[ ]:


plot_model(model, show_shapes=True)


# In[ ]:


scores = []
test_preds = []
kf = GroupKFold(n_splits=5)


# In[ ]:


for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(df_train, df_train_labels, groups.unique())):
    
    print('\n')
    print('*'*15, f'↓ Fold {fold_idx+1} ↓', '*'*15)
    
    # Separate into train data and validation data
    X_train, X_valid = df_train[train_idx], df_train[valid_idx]
    y_train, y_valid = labels.iloc[train_idx].values, labels.iloc[valid_idx].values
    
    # Train the model
    model = lstm_0422()
    model.fit(X_train, y_train, 
              validation_data=(X_valid, y_valid), 
              epochs=15, 
              batch_size=256, 
              callbacks=[EarlyStopping(monitor='val_auc', patience=7, mode='max', 
                                       restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_auc', factor=0.6, 
                                           patience=4, verbose=False)]
             )
    
    # Save score
    score = roc_auc_score(y_valid, model.predict(X_valid, batch_size=512).squeeze())
    scores.append(score)
    
    # Predict
    test_preds.append(model.predict(df_test, batch_size=512).squeeze())
    
    print(f'Fold {fold_idx+1} | Score: {score}')
    print('*'*15, f'↑ Fold {fold_idx+1} ↑', '*'*15)
    
print(f'Mean accuracy on {kf.n_splits} folds {np.mean(scores)}')


# In[ ]:


submission_0422 = df_smpl

submission_0422['state'] = np.average(test_preds, axis = 0)
submission_0422.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




