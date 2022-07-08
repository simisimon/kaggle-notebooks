#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# TO PRINT ALL OUTPUTS
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# INSTALLATIONS
# Uncomment these in case an error occurs due to import

#!pip install opencv-python
#!pip install opencv-contrib-python
#!pip install pytesseract


# In[ ]:


# IMPORTS
import os, sys, time, random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # computer vision and image processing
from matplotlib import pyplot as plt # visualization
import imagehash # hashing
from dask import bag, diagnostics # parallelizable operations
from PIL import Image # Image processing

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.metrics import top_k_categorical_accuracy
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.losses import CategoricalCrossentropy
import keras.backend as K
from keras.models import Sequential
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# PATHS
TRAIN_CSV = '../input/humpback-whale-identification/train.csv'
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'


# In[ ]:


# Finding Duplicates in the dataset through hashing algorithms.
# Reused Code for Dask From https://www.kaggle.com/jpmiller/basic-eda-with-images
# phash_simple didn't pick duplicates correctly; phash, on the other hand was better.
# https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563
# had multiple steps to narrow down images to only duplicates

def get_hash(file):     
    return imagehash.phash(Image.open(file))

def build_hash_table(train_df):
    filelist = [TRAIN + f for f in train_df['Image']]
    hash_bag = bag.from_sequence(filelist).map(get_hash)
    with diagnostics.ProgressBar():
        hash_codes = hash_bag.compute()       
    hash_codes_df = pd.DataFrame(hash_codes, columns=['hash_code'])
    train_df['hc']  = hash_codes_df.astype('str')
    train_df = train_df.drop_duplicates(subset=['hc'])
    train_df = train_df.drop(columns = ['hc'], axis = 1)
    
    return train_df


# In[ ]:


# Read Train Data Reference
train_df = pd.read_csv(TRAIN_CSV)

# build_hash_table & remove duplicates
train_df = build_hash_table(train_df)


# In[ ]:


# Read Test Data Reference
test_df = pd.DataFrame(os.listdir(TEST),columns = ['Image'])
test_df['Id'] = ""


# In[ ]:


# prepare images for Keras Model
# Requires 4 dimensional / 3 dimensional input 
# (samples, size, size, channel)

IMG_SIZE = 100

def prepare_images (df, dataset):
    X_train = np.zeros((df.shape[0],IMG_SIZE, IMG_SIZE))
    c = 0
    
    for item in df['Image']:
        image = cv2.imread(dataset + item, 0)
        image = cv2.resize(image,(IMG_SIZE, IMG_SIZE),cv2.INTER_NEAREST) 
        X_train[c] = image
        c = c + 1
        
        if c%5000 ==1: 
            print(f'Processing Image : {c}')
            
    return X_train

def prepare_enc_labels(df):
    le_enc = LabelEncoder()
    y = df['Id']
    encoded = le_enc.fit_transform(y)
    ohe = OneHotEncoder(sparse=False)
    y_train = ohe.fit_transform(encoded.reshape(len(encoded),1))
    return y_train, le_enc


# In[ ]:


# Prepare Training 
# Takes More than 5 Minutes to completely load
X_train = prepare_images(train_df, TRAIN)
X_train = X_train / 255  
X_train = X_train.reshape((train_df.shape[0], IMG_SIZE, IMG_SIZE, 1))


# In[ ]:


# Prepare Labels with Encoding
y_train, le_enc = prepare_enc_labels (train_df)


# In[ ]:


# Simple CNN Model
# Chosen this network for hyperparameter tuning
# Mostly adopted from https://www.tensorflow.org/tutorials/keras/classification
# and https://www.kaggle.com/pestipeti/keras-cnn-starter

import tensorflow as tf
import tensorflow as tf
tuned_model = Sequential()
tuned_model.add(Conv2D(32, (5,5), strides = (1, 1), name = 'conv0', input_shape = (IMG_SIZE, IMG_SIZE, 1)))
tuned_model.add(BatchNormalization(axis = 3, name = 'bn0'))
tuned_model.add(Activation('tanh'))
tuned_model.add(MaxPooling2D((2, 2), name='max_pool_1'))

tuned_model.add(Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1'))
tuned_model.add(BatchNormalization(axis = 3, name = 'bn1'))
tuned_model.add(Activation('tanh'))
tuned_model.add(MaxPooling2D((2, 2), name='max_pool_2'))

tuned_model.add(Flatten())
tuned_model.add(Dense(units = 896, activation="tanh", name='rl'))
tuned_model.add(Dropout(0.8))

tuned_model.add(Dense(y_train.shape[1], activation='softmax', name='sm'))

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


tuned_model.compile(optimizer= opt,loss='categorical_crossentropy',metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
tuned_model.summary()


# In[ ]:


history = tuned_model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_split = 0.1)


# In[ ]:


# Prepare Test Data

X_test = prepare_images(test_df, TEST)
X_test = X_test/ 255
X_test = X_test.reshape((test_df.shape[0], IMG_SIZE, IMG_SIZE, 1))


# In[ ]:


# Predict
predictions = tuned_model.predict(np.array(X_test), verbose=1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(le_enc.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:


tuned_model.save("./Z5213413_FINAL.pkl")

