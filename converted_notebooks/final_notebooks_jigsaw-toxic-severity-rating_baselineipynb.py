#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U tensorflow==2.8')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# загруска файлов

# In[ ]:


train_dataset = image_dataset_from_directory('/kaggle/input/nti-urfu-competitions/train/train',
                                             subset='training',
                                             seed=22,
                                             validation_split=0.1,
                                             batch_size=128,
                                             image_size=(256, 256))


# In[ ]:


validation_dataset = image_dataset_from_directory('/kaggle/input/nti-urfu-competitions/train/train',
                                             subset='validation',
                                             seed=22,
                                             validation_split=0.1,
                                             batch_size=128,
                                             image_size=(256, 256))


# In[ ]:


class_names = train_dataset.class_names
class_names


# In[ ]:


plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# In[ ]:


model = Sequential()
model.add(Flatten(input_shape=(256,256,3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(train_dataset, 
                    validation_data=validation_dataset,
                    epochs=3)


# ниже загрузка решения

# In[ ]:


sample_submission=pd.read_csv('/kaggle/input/nti-urfu-competitions/sample_submission.csv')


# In[ ]:


test_dataset = image_dataset_from_directory('/kaggle/input/nti-urfu-competitions/test/',
                                             shuffle=False,
                                             label_mode=None,
                                             image_size=(256, 256))


# In[ ]:


y_test=model.predict(test_dataset) 


# In[ ]:


i=0
for path in test_dataset.file_paths:
    p_arr=path.split('/')
    p=p_arr[-2]+'/'+p_arr[-1][:-4]
    sample_submission.loc[sample_submission['file'] == p, 'label'] = 0 if y_test[i]>=0.5 else 1
    i+=1
sample_submission


# In[ ]:


sample_submission.to_csv('sample_submission.csv',index=None)

