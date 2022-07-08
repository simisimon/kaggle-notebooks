#!/usr/bin/env python
# coding: utf-8

# ![Modifiyed.jpg](attachment:a76b37c4-7e3e-446d-9aa3-29c95940839c.jpg)
# 
# # Plan :
# 
# 1. Importation des bibliothèques
# 2. Fonction de création d'un modèle
# 3. Chargement/Préparation du dataset
# 4. Augmentation du dataset
# 5. Création du fichier contenant les étiquettes
# 6. Exemple d'image
# 7. Construction/Compilation du réseau neuronale RESNET
# 8. La phase d'entraînement
# 9. Résultats
# 10. Évaluation 
# 11. Sauvegarde du modèle

# # 1. Importation des bibliothèques

# In[ ]:


import numpy as np
import pandas as pd
import re
import os
import random

#Tensorflow + Keras et le générateur d'images + Earlystopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

#différents types de neurones
import cv2 #librairie python pour le traitement d'images
from keras.callbacks import ModelCheckpoint,EarlyStopping #callbacks, arrêt prématuré
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D 
from keras.models  import Sequential #base du réseau de neurones 

#Traitement d'image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img 
from PIL import Image

#Pour les graphes
import random,os,glob 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# # 2. Fonction de création d'un modèle

# In[ ]:


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


# # 3. Chargement/Préparation du dataset

# In[ ]:


#chemin d'accès
file_path = '../input/manon-str-cleaned-dataset/Dataset' 
img_list = glob.glob(os.path.join(file_path, '*/*.png')) #on cherche toutes les images avec liées au chemin d'accès
print("nombre d'images :", len(img_list))

# setting the path and the labels list for classification of targets on the basis in human understandable form

train_dir = os.path.join('../input/manon-str-cleaned-dataset/Dataset')
labels = ['Carton', 'Masques', 'Metal', 'Papier', 'Plastique']

# checking the size of data available to us for training out model

for label in labels:
    directory = os.path.join(train_dir, label)
    print("Images of label \"" + label + "\":\t", len(os.listdir(directory)))


# # 4. Augmentation du dataset

# In[ ]:


#Générateur d'images
train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,) #augmentations que l'on va appliquer

test=ImageDataGenerator(rescale=1/255,validation_split=0.1)

# à partir de notre fichier, création paquets de 30 images, augmentés, de taille 300/300
train_generator=train.flow_from_directory(file_path,target_size=(256,256),batch_size=32,
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory(file_path,target_size=(256,256),class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)


# # 5. Création du fichier contenant les étiquettes

# In[ ]:


print (train_generator.class_indices)

Labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
    f.write(Labels)


# # 6. Exemple d'image

# In[ ]:


for image_batch, label_batch in train_generator:
    break
image_batch.shape, label_batch.shape

plt.imshow(image_batch[1])
print(labels[list(label_batch[1]).index(1)])


# # 7. Construction/Compilation du réseau neuronale RESNET

# In[ ]:


early = EarlyStopping(monitor="val_loss",mode="min",patience=3)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.0001)

callbacks_list = [early, learning_rate_reduction]

callbacks = callbacks_list

model = build_model(num_classes=5)


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = 0.001), metrics=['accuracy'])


# # 8. La phase d'entraînement

# In[ ]:


history = model.fit(train_generator, epochs=50, verbose=1, validation_data=test_generator, callbacks=[callbacks])


# # 9. Résultats

# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()


# # 10. Évaluation 

# In[ ]:


train_dir = os.path.join('../input/manon-str-cleaned-dataset/Dataset')
labels = ['Carton', 'Masques', 'Metal', 'Papier', 'Plastique']

for label in labels:
    directory = os.path.join(train_dir, label)
    print("Images of label \"" + label + "\":\t", len(os.listdir(directory)))
    
#cat = int(input('Enter any category by index: '))
#ind = int(input('Enter any index to test: '))

cat = 0
ind = 0

directory = os.path.join(train_dir, labels[cat % 6])
try:
    path = os.path.join(directory, os.listdir(directory)[ind])
    img = mpimg.imread(path)
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    pred = labels[np.argmax(classes)]
    
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Actual: {}      Pred: {}'.format(labels[cat], pred))
    
except:
    print('Invalid Value')


# # 11. sauvegarde du modèle

# In[ ]:


model.save('Model-RESNET-Classification-déchets.h5')

