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
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


(X_train,y_train),(X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# In[ ]:


y_train[:5] #this is 2d array


# In[ ]:


y_train = y_train.reshape(-1,) #we are converting 2d array to 1d array
y_train[:5]


# In[ ]:


y_test = y_test.reshape(-1,)


# In[ ]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[ ]:


index=0
def plt_image(X,y,index):
    for index in range(10):
        plt.figure(figsize=(15,2))
        plt.imshow(X[index])
        plt.xlabel(classes[y[index]])


# In[ ]:


plt_image(X_train,y_train,0)


# In[ ]:


X_train[0] #here we can see our values are varry between 0-255  


# ## **Normalize the training data**

# In[ ]:


# pixel value is in ranges between 0-255 for each of channels, so we need to normalize our dataset in(0-1)
X_train = X_train / 255
X_test  = X_test / 255


# In[ ]:


X_test[0]


# ### **Checking with simple artificial neural network for image classification**

# In[ ]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# In[ ]:


# here we can see with help of ann we are getting very less accuracy that is 49.79% in 5 epochs
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# ## CNN model building

# In[ ]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[ ]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


cnn.fit(X_train, y_train, epochs=20)


# In[ ]:


cnn.evaluate(X_test,y_test)


# In[ ]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[ ]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[ ]:


y_test[:5]


# In[ ]:


plt_image(X_test, y_test,3)


# In[ ]:


classes[y_classes[3]]


# In[ ]:





# In[ ]:





# In[ ]:




