#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Flatten, Dense, Input, concatenate, GlobalAveragePooling2D, Dropout
)
from tensorflow.keras.applications import ResNet101, Xception, InceptionV3


# In[ ]:


get_ipython().system(' git clone https://github.com/FuadHamdiBahar/EKGCroppedData.git')


# In[ ]:


df = pd.read_csv('./EKGCroppedData/CSV_PING.csv')
df = df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'use', 'Gender', 'Age', 'age_55'])
df = df[['ID', 'Sinus', 'Tachy']]
df.head()


# In[ ]:


classs = df.groupby('Tachy').count()
classs


# In[ ]:


class_1 = df.loc[df["Sinus"] == 1, :]
number_of_1s = len(class_1)

class_2 = df.loc[df["Sinus"] == 2, :]
sampled_2 = class_2.sample(number_of_1s)

class_3 = df.loc[df["Sinus"] == 3, :]
sampled_3 = class_3.sample(number_of_1s)

balance_df = pd.concat([class_1, sampled_2, sampled_3], ignore_index = True)
balance_df.head()


# In[ ]:


# img, y = df['ID'], df['Sinus']
img, sinus, tachy = balance_df['ID'], balance_df['Sinus'], balance_df['Tachy']


# In[ ]:


sinus_onehot = pd.get_dummies(sinus, prefix='sinus')
sinus_onehot.tail()


# In[ ]:


tachy_onehot = pd.get_dummies(tachy, prefix='tachy')
tachy_onehot.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, sinus_train, sinus_test, tachy_train, tachy_test = train_test_split(img, sinus_onehot, tachy_onehot, test_size=0.2, stratify=sinus_onehot, random_state=42)


# In[ ]:


# y_train = np.array(y_train)
# y_test = np.array(y_test)
sinus_train, sinus_test = np.array(sinus_train), np.array(sinus_test)
tachy_train, tachy_test = np.array(tachy_train), np.array(tachy_test)


# In[ ]:


(X_train.shape, sinus_train.shape, tachy_train.shape), (X_test.shape, sinus_test.shape, tachy_test.shape)


# In[ ]:


from tensorflow.keras.utils import img_to_array
def preprocessing(BASE_DIR, IMG_NAME):
    l = []
    for i in tqdm(IMG_NAME): 
        arr = cv2.imread(BASE_DIR + i + ".jpeg")
        arr = cv2.resize(arr, (299, 299)) 
        arr = arr.astype('float16') / 255.
        arr = img_to_array(arr)
        l.append(arr)
    return np.array(l)


# In[ ]:


BASE_DIR = './EKGCroppedData/CROPPED_12_PING_IMG/'


# In[ ]:


aVF_train = preprocessing(BASE_DIR + 'aVF/', X_train)
aVL_train = preprocessing(BASE_DIR + 'aVL/', X_train)
aVR_train = preprocessing(BASE_DIR + 'aVR/', X_train)
I_train = preprocessing(BASE_DIR + 'I/', X_train)
II_train = preprocessing(BASE_DIR + 'II/', X_train)
III_train = preprocessing(BASE_DIR + 'III/', X_train)
V1_train = preprocessing(BASE_DIR + 'V1/', X_train)
V2_train = preprocessing(BASE_DIR + 'V2/', X_train)
V3_train = preprocessing(BASE_DIR + 'V3/', X_train)
V4_train = preprocessing(BASE_DIR + 'V4/', X_train)
V5_train = preprocessing(BASE_DIR + 'V5/', X_train)
V6_train = preprocessing(BASE_DIR + 'V6/', X_train)


# In[ ]:


aVF_test = preprocessing(BASE_DIR + 'aVF/', X_test)
aVL_test = preprocessing(BASE_DIR + 'aVL/', X_test)
aVR_test = preprocessing(BASE_DIR + 'aVR/', X_test)
I_test = preprocessing(BASE_DIR + 'I/', X_test)
II_test = preprocessing(BASE_DIR + 'II/', X_test)
III_test = preprocessing(BASE_DIR + 'III/', X_test)
V1_test = preprocessing(BASE_DIR + 'V1/', X_test)
V2_test = preprocessing(BASE_DIR + 'V2/', X_test)
V3_test = preprocessing(BASE_DIR + 'V3/', X_test)
V4_test = preprocessing(BASE_DIR + 'V4/', X_test)
V5_test = preprocessing(BASE_DIR + 'V5/', X_test)
V6_test = preprocessing(BASE_DIR + 'V6/', X_test)


# # BATAS

# In[ ]:


def Cabang_IV3():
    b_model = Xception(weights='imagenet', include_top = False)
    x = b_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation = 'relu') (x)
    p = Dense(1, activation = 'sigmoid') (x)
    
    model = Model(inputs = b_model.input, outputs = p)
    
    b_model.trainable = False
    
    return model

base_model = Cabang_IV3()

pretrained_model = Model(base_model.input, base_model.layers[-3].output)

def combined_net():

    i1 = Input((299, 299, 3))
    i2 = Input((299, 299, 3))
    i3 = Input((299, 299, 3))
    i4 = Input((299, 299, 3))
    i5 = Input((299, 299, 3))
    i6 = Input((299, 299, 3))
    i7 = Input((299, 299, 3))
    i8 = Input((299, 299, 3))
    i9 = Input((299, 299, 3))
    i10 = Input((299, 299, 3))
    i11 = Input((299, 299, 3))
    i12 = Input((299, 299, 3))
    
    c1 = pretrained_model(i1)
    c2 = pretrained_model(i2)
    c3 = pretrained_model(i3)
    c4 = pretrained_model(i4)
    c5 = pretrained_model(i5)
    c6 = pretrained_model(i6)
    c7 = pretrained_model(i7)
    c8 = pretrained_model(i8)
    c9 = pretrained_model(i9)
    c10 = pretrained_model(i10)
    c11 = pretrained_model(i11)
    c12 = pretrained_model(i12)
    
    concat = concatenate([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])
    x = Dense(1024, activation='relu')(concat)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation = 'relu')(x)
    sinus = Dense(3, activation = 'softmax', name='sinus')(x)
    
    y = Dense(1024, activation='relu')(concat)
    y = Dropout(0.7)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(256, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(64, activation = 'relu')(y)
    tachy = Dense(2, activation = 'softmax', name='tachy')(y)

    model = Model(inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12], outputs = [sinus, tachy])
    
    from tensorflow.keras.optimizers import Adam
    init_lr = 1e-4
    epochs = 50
    opt = Adam(learning_rate=init_lr, decay=init_lr / epochs)
    
    model.compile(
        loss = {
            'sinus': 'categorical_crossentropy',
            'tachy': 'categorical_crossentropy',
        }, 
        metrics = {
            "sinus": 'accuracy',
            "tachy": 'accuracy',
        },
        optimizer=opt
    )
    return model
    
main_model = combined_net()
# main_model.summary()


# In[ ]:


tf.keras.utils.plot_model(main_model)


# In[ ]:


history = main_model.fit(
    [aVF_train, aVL_train, aVR_train, I_train, II_train, III_train, V1_train, V2_train, V3_train, V4_train, V5_train, V6_train],
    [sinus_train, tachy_train],
    validation_data = (
        [aVF_test, aVL_test, aVR_test, I_test, II_test, III_test, V1_test, V2_test, V3_test, V4_test, V5_test, V6_test], 
        [sinus_test, tachy_test]),
    epochs = 50,
    batch_size = 5,
    validation_batch_size=2,
)


# In[ ]:


plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['sinus_loss'], label='Sinus Loss')
plt.plot(history.history['tachy_loss'], label='Tachy Loss')
plt.legend()
plt.title('Training - Loss Function')

plt.subplot(2, 2, 2)
plt.plot(history.history['sinus_accuracy'], label='Sinus Accuracy')
plt.plot(history.history['val_sinus_accuracy'], label='Sinus Validation Accuracy')
plt.plot(history.history['val_tachy_accuracy'], label='Tachy Validation Accuracy')
plt.plot(history.history['tachy_accuracy'], label='Tachy Accuracy')
plt.legend()
plt.title('Train - Accuracy')


# In[ ]:




