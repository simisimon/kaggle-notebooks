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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importation des librairies
# 

# In[ ]:


import numpy as np
import cv2
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D 
from keras.models  import Sequential #base du réseau de neurones 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img 
import random,os,glob 
import matplotlib.pyplot as plt
import pandas as pd
import re
import random
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow import keras


# # Préparation du dataset

# In[ ]:


os.listdir("/kaggle/input/garbage-classification/Garbage classification/Garbage classification")

data = pd.DataFrame()
path="/kaggle/input/garbage-classification/Garbage classification/Garbage classification"
for category in os.listdir(path):
    temp = pd.DataFrame()
    temp['path'] = np.nan
    temp['target'] = category
    i = 0
    for photo in os.listdir(os.path.join(path, category)):
        temp.loc[i, 'path'] =  os.path.join(path, category, photo)
        temp.loc[i, 'filename'] =  os.path.join(category, photo)
        temp.loc[i, 'target'] = category
        i += 1
    data = pd.concat([data, temp], ignore_index=True)
    del temp
    
print(len(data))
print(data['path'][0])
print(data['filename'][0])
print(data['target'][0])


# # Préparation du dataset
# 

# In[ ]:


train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,) #augmentations que l'on va appliquer

test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)

# à partir de notre fichier, création paquets de 30 images, augmentés, de taille 300/300
train_generator=train.flow_from_directory(path,
                                          target_size=(256,256),
                                          batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory(path,
                                        target_size=(256,256),
                                        class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)


# # Exemple d'image

# In[ ]:


def show_image_samples(gen ):
    t_dict=gen.class_indices
    classes=list(t_dict.keys())    
    images,labels=next(gen) # get a sample batch from the generator 
    plt.figure(figsize=(20, 20))
    length=len(labels)
    if length<25:   #show maximum of 25 images
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image=images[i]
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name)
        plt.axis('off')
    plt.show()
    
show_image_samples(train_generator)


# # Construction du réseau neuronale

# In[ ]:


early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.00001)

callbacks = [early, learning_rate_reduction]


def build_model(num_classes):
    # Loading pre-trained ResNet model
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    base_model.trainable = False
        
    return model

model = build_model(num_classes=6)

    


# # Compilation du modèle

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = 0.001), metrics=['accuracy'])


# # Entraînement

# In[ ]:


history = model.fit(train_generator, epochs=5, verbose=1, validation_data=test_generator, callbacks=[callbacks])


# # Résultats

# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()


# # Test :

# In[ ]:


def test_image_samples(gen ):
    
        t_dict=gen.class_indices
        classes=list(t_dict.keys())    
        images,labels=next(gen) # get a sample batch from the generator 
        preds = model.predict(images)
        preds = preds.argmax(1)
        plt.figure(figsize=(20, 20))
        length=len(labels)
        if length<25:   #show maximum of 25 images
            r=length
        else:
            r=25
        for i in range(r):
            plt.subplot(5, 5, i + 1)
            image=images[i]
            plt.imshow(image)
            index=np.argmax(labels[i])
            class_name=classes[index]
            class_name_pred=classes[preds[i]]
            plt.title('Actual: {}      Pred: {}'.format(class_name,class_name_pred),  color='blue', fontsize=12)
            plt.axis('off')
        plt.show()

test_image_samples(test_generator)


# In[ ]:




