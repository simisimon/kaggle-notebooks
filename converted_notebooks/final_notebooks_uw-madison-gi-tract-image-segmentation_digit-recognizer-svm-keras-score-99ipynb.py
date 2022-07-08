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


import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler


# In[ ]:


train=pd.read_csv('../input/digit-recognizer/train.csv')
test=pd.read_csv('../input/digit-recognizer/test.csv')
sample=pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[ ]:


train.head()


# **Do We have Enough Data for Each number ?**

# In[ ]:


train.label.value_counts().plot.bar()


# **lets plot random number**

# In[ ]:


im=np.reshape(train.iloc[np.random.randint(0,train.shape[0]),1:].values,(28,28))
from matplotlib import pyplot as plt
plt.imshow(im)


# **lest look at the statistics ...**

# In[ ]:


print(train.describe().T.sort_values(by='std' , ascending=False))


# **Do We have null values ?******

# In[ ]:


train.isna().sum().values.sum()


# **Scale the Data**

# In[ ]:


train.iloc[:,1:]=(train.iloc[:,1:])/255
test=test/255


# In[ ]:


print(train.describe().T.sort_values(by='std' , ascending=False))


# # SVM_model

# In[ ]:


X=train.drop(['label'] , axis=1)
y=train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=True, random_state=42)


# In[ ]:


svm_model=svm.SVC(kernel='poly')
svm_model.fit(X_train,y_train)
y_pred=svm_model.predict(X_test)
report=classification_report(y_true=y_test,y_pred=y_pred)
print(report)   


# In[ ]:


y_pred_final=svm_model.predict(test)
sample['Label']=y_pred_final
sample.to_csv('submission_SVM.csv', index=False)


# # Keras_model

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical


# In[ ]:


X=train.drop(['label'] , axis=1)
y=train['label']


# In[ ]:


X=X.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# In[ ]:


y=to_categorical(y,num_classes=10)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=42)


# In[ ]:





# In[ ]:


input_shape = (28, 28, 1)
batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])


# In[ ]:


hist = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test))
print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")


# In[ ]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


results = model.predict(test)


# In[ ]:


results = np.argmax(results,axis = 1)


# In[ ]:


sample['Label']=results
sample.to_csv('submission_keras.csv', index=False)


# In[ ]:





# In[ ]:




