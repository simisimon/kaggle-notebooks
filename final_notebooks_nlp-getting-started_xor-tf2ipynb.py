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


import tensorflow as tf

AB_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
X_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
tf.model.add(tf.keras.layers.Activation('sigmoid'))


# In[ ]:


tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
tf.model.summary()


# In[ ]:


history=tf.model.fit(AB_data, X_data, epochs=100000)


# In[ ]:


predictions=tf.model.predict(AB_data)>0.5
print('Prediction: \n', predictions)

score=tf.model.evaluate(AB_data, X_data)
print('Acuracy: ', score[1])

