#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[ ]:


seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


# In[ ]:


with zipfile.ZipFile("../input/dogs-vs-cats/train.zip",'r') as z:
    z.extractall(".")


# In[ ]:


path = '/kaggle/working/train/'
filenames = os.listdir(path)
filenames[:10]


# In[ ]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
load_img(path+'dog.4841.jpg')


# In[ ]:


label = []
for filename in filenames:
    if filename.split('.')[0] =='cat':
        label.append('cat')
    else:
        label.append('dog')


# In[ ]:


df = pd.DataFrame({'name':filenames,'label':label})
df.head(5)


# In[ ]:


with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip",'r') as z:
    z.extractall(".")


# In[ ]:


path1 = '/kaggle/working/test1/'
filenames_test = os.listdir(path1)
filenames_test[:5]


# In[ ]:


load_img(path1+'5225.jpg')


# In[ ]:


train, test_val = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=17)
test, val = train_test_split(test_val, test_size=0.5,  stratify=test_val['label'], random_state=17)


# In[ ]:


aug_gen = ImageDataGenerator(rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest'
                              )
train_data = aug_gen.flow_from_dataframe(train,
                                         directory=path,
                                         x_col='name',
                                         y_col='label',
                                         class_mode='binary',
                                         target_size=(224,224),
                                         seed=17
                                        )


# In[ ]:


val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_dataframe(val,
                                       directory=path,
                                       x_col='name',
                                       y_col='label',
                                       class_mode='binary',
                                       target_size=(224,224),
                                       seed=17  
                                      )


# In[ ]:


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')


# In[ ]:


history = model.fit(train_data,validation_data = val_data, epochs = 60,callbacks = [EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, verbose=1)])


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure(figsize=(15,8))
plt.plot(acc, label='Train acc')
plt.plot(val_acc,'--', label='Val acc')
plt.title('Training and validation accuracy')
plt.legend();

