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


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


# In[ ]:


(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()


# In[ ]:


X_test.shape


# In[ ]:


y_train


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(X_train[0])


# In[ ]:


X_train[0]


# In[ ]:


# to convert all value in between 0-1 we require to divide it 255
X_train = X_train/255
X_test = X_test/255


# In[ ]:


X_train[0] #now we can see all values are in range of 0-1 for 1st photograph


# In[ ]:


#Flattaning layer operation will be require to give 784 pixel values as input
model = Sequential()

model.add(Flatten(input_shape=(28,28))) #It will convert data into 1D --784inputs
model.add(Dense(128,activation="relu"))#no need to give inputs flatten layer will automatically gives  #here 128nodes for input layer
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation="softmax"))  #softmax bz we are having more than one nodes in output


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics=["accuracy"]) #in sparse categorical crossentropy we dont need to do one hot encoding


# In[ ]:


history= model.fit(X_train,y_train,epochs=25,validation_split=0.2)


# In[ ]:


history.history["accuracy"]


# In[ ]:


y_prob = model.predict(X_test)


# In[ ]:


y_prob


# In[ ]:


y_pred = y_prob.argmax(axis=1) #Taking only one higher probablity


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])


# In[ ]:


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])


# In[ ]:


plt.imshow(X_test[0])


# In[ ]:


model.predict(X_test[0].reshape(1,28,28)).argmax(axis=1)


# In[ ]:


plt.imshow(X_test[1])


# In[ ]:


model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)


# In[ ]:




