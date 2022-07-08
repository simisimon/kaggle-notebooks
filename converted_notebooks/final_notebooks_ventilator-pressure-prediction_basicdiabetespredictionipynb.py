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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[ ]:


df=pd.read_csv('../input/diabetes-data-set/diabetes.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


for column in X.columns:
    plt.figure()
    sns.distplot(df[column])


# In[ ]:


for column in X.columns:
    plt.figure()
    df[column].plot.box(figsize=(16,5))


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# XG BOOST MODEL
# 

# In[ ]:


from xgboost import XGBClassifier
model=XGBClassifier(gamma=0)
model.fit(X_train,y_train)


# In[ ]:


xg_pred=model.predict(X_test)


# In[ ]:


from sklearn import metrics 
print("Accuracy=",format(metrics.accuracy_score(y_test,xg_pred)))


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation


# In[ ]:


model1 = Sequential()
model1.add(Dense(32,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(64,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(64,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(128,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(128,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(256,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(256,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(2))
model1.add(Activation('softmax'))


# In[ ]:


model1.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",metrics=['accuracy'])
history=model1.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, y_test))
loss, accuracy = model1.evaluate(X_test,y_test, verbose=0)


# In[ ]:


plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

