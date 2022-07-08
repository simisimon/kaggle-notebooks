#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from tensorflow.keras import layers, models
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


train_X = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train_Y = train_X.pop('label')


# In[ ]:


train_X /= 255


# In[ ]:


train_X = np.array(train_X)
train_Y = np.array(train_Y)


# In[ ]:


print(train_X.shape)
print(train_Y.shape)


# In[ ]:


np.sqrt(784)


# In[ ]:


plt.imshow(train_X[100].reshape(28, 28))


# In[ ]:


model = models.Sequential()

'''
......    
......    ... 
...... -> ... -> ......
......
'''

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 1 1 1 1 1
# 0.2 0.2 0.2 0.2 0.2


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


# In[ ]:


train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
train_Y = keras.utils.to_categorical(train_Y)


# In[ ]:


model.fit(train_X, train_Y, epochs=10)


# In[ ]:


test_X = np.array(pd.read_csv('/kaggle/input/digit-recognizer/test.csv'))


# In[ ]:


test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))


# In[ ]:


preds = model.predict(test_X)


# In[ ]:


results = pd.DataFrame({'Label': preds.argmax(axis=1)}, index=range(1, 28001))


# In[ ]:


results.index.name = 'ImageId'
results


# In[ ]:


results.to_csv('submission.csv')


# In[ ]:




