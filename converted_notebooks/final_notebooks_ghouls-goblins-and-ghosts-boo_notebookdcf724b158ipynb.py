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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train_data = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip")
train_data

train_data.shape

test_data = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip")
test_data

test_data.shape

train_data.info()

train_data.isnull().sum()

test_data.info()

test_data.isna().sum()

train_data.type.value_counts()

train_data.color.value_counts()

test_data.color.value_counts()

train_data=pd.concat([train_data,pd.get_dummies(train_data['color'])],axis=1)
train_data.drop('color',axis=1,inplace=True)
train_data.head()

test_data = pd.concat([test_data, pd.get_dummies(test_data['color'])], axis=1)
test_data.drop('color', axis=1, inplace=True)
test_data.head()

X=train_data.drop(['id', 'type'], axis=1)
y=pd.get_dummies(train_data['type'])

X.shape

y.shape

np.unique(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout

# model = Sequential()

# model.add(Dense(64,input_shape=(X.shape[1],)))

# model.add(Dense(100))
# model.add(Activation("relu"))

# model.add(Dense(100))
# model.add(Activation("relu"))

# model.add(Dense(10))
# model.add(Activation("softmax"))

# model.summary()



model=Sequential()
model.add(Dense(100,input_shape=(X.shape[1],)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()

model.compile(loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"])

model.fit(X_train, y_train,
         validation_data = (X_test, y_test),
         batch_size = 16,
         epochs = 50,
         verbose = 2)

pred=model.predict(test_data.drop('id',axis=1))

pred_final=[np.argmax(i) for i in pred]

submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})
submission.head()

submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
submission.head()


# In[ ]:


submission.to_csv('../working/submission.csv', index=False)

