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


get_ipython().system('pip install flaml')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, MaxPool2D, Conv2D
from sklearn.model_selection import train_test_split
from flaml import AutoML


# In[ ]:


TitanicTrainData = pd.read_csv('/kaggle/input/titanic/train.csv')
TitanicTestData = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


TrainData = TitanicTrainData.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
TestData = TitanicTestData.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# In[ ]:


TrainData = pd.get_dummies(TrainData)
TestData = pd.get_dummies(TestData)


# In[ ]:


TrainData.fillna(TrainData.Age.mean(), inplace=True)
TrainData['Age'] = TrainData.Age.astype('int64')

TrainData


# In[ ]:


X = np.array(TrainData.loc[:, ['Age', 'Fare', 'Sex_female', 'Sex_male']])
y = np.asarray(TrainData.iloc[:, 0]).astype('int64')


# In[ ]:


class BaseModel(tf.keras.Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.D1 = Dense(16, activation='relu')
        self.D2 = Dense(2, activation='sigmoid')
        
    def call(self, x):
        x = self.D1(x)
        y = self.D2(x)
        return y


# In[ ]:


TFmodel = BaseModel()

TFmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['sparse_categorical_accuracy'])

history = TFmodel.fit(X, y, epochs=50, batch_size=32, validation_split=0.3)

TFmodel.summary()


# In[ ]:


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Acc')
plt.plot(val_acc, label='Validation Acc')
plt.title('Training And Validation Acc')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training And Validation Loss')

plt.legend()
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = AutoML()

model.fit(X_train, y_train, task='classification', metric='accuracy', time_budget=20)


# In[ ]:


print('Best ML Model:', model.best_estimator)
print('Best hyperparmeter config:', model.best_config)
print('Best Accuracy on validation data: %f'%(1 - model.best_loss))
print('Training duration of best run: %f s'%(model.best_config_train_time))


# In[ ]:


y_pred = model.predict(np.array(TestData.loc[:, ['Age', 'Fare', 'Sex_female', 'Sex_male']]))
TestData['Survived'] = y_pred


# In[ ]:


TestData


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df['Survived'] = TestData['Survived']


# In[ ]:


df


# In[ ]:


df.to_csv('Titanic_Submit.csv', index=False)


# In[ ]:




