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
        
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


import matplotlib.pyplot as plt

from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import normalize


# In[ ]:





# In[ ]:


train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_features_enc = pd.get_dummies(train_features, columns=['cp_type', 'cp_dose'], drop_first=True)
X = train_features_enc.iloc[:,1:].to_numpy()
y = train_targets.iloc[:,1:].to_numpy()


# In[ ]:


device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# In[ ]:


def plot_hist(hist, last = None):
    if last == None:
        last = len(hist.history["loss"])
    plt.plot(hist.history["loss"][-last:])
    plt.plot(hist.history["val_loss"][-last:])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


# In[ ]:


#0.0161
X_normalized = normalize(X)

def l3_res_model_2(input_shape, no_classes, lr):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(128, activation='sigmoid')(inputs)
    x = layers.BatchNormalization()(x)
    b_1 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='sigmoid')(b_1)
    x = layers.BatchNormalization()(x)
    b_2 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='sigmoid')(b_2)
    x = layers.BatchNormalization()(x)
    b_3 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='sigmoid')(b_3)
    x = layers.BatchNormalization()(x)
    b_4 = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='sigmoid')(b_4)
    x = layers.BatchNormalization()(x)
    b_5 = layers.Dropout(0.2)(x)
    
    tot_op = tf.keras.layers.add([b_1, b_2, b_3, b_4, b_5])
    outputs = layers.Dense(no_classes, activation='sigmoid')(tot_op)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['binary_crossentropy'])
    return model

losses_NN=[]
kf = KFold(n_splits=10)
tf.random.set_seed(1010)
np.random.seed(1010)

for train_index, test_index in kf.split(X):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    control_vehicle_mask = X_train[:,-2] == 0
    X_train = X_train[~control_vehicle_mask,:]
    y_train = y_train[~control_vehicle_mask]

    nnclf = l3_res_model_2((875,),206,0.0005)
    hist = nnclf.fit(X_train, y_train, batch_size=512, epochs=50, validation_data=(X_test, y_test), verbose=0)
    plot_hist(hist, last = 20)

    preds = nnclf.predict(X_test) # list of preds per class

    control_mask = X_test[:,-2]==0
    preds[control_mask] = 0

    loss = log_loss(np.ravel(y_test), np.ravel(preds))
    print('Loss: '+str(loss))
    losses_NN.append(loss)

print('Average Loss: '+str(np.average(losses_NN))) 


# In[ ]:


test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
test_features_enc = pd.get_dummies(test_features, columns=['cp_type', 'cp_dose'], drop_first=True)
X_test = test_features_enc.iloc[:,1:].to_numpy()


# In[ ]:


test_X_normalized = normalize(X_test)
nnclf = l3_res_model_2((875,),206,0.0005)
X_norm = normalize(X)
hist = nnclf.fit(X_norm, y, batch_size=512, epochs=50, verbose=0)
preds = nnclf.predict(test_X_normalized)


# In[ ]:


submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
test_submission = pd.DataFrame(preds)
test_submission.insert(0, 'sig_id', submission['sig_id'])
test_submission.columns = list(submission.columns)
test_submission.to_csv("submission.csv", index=False)

