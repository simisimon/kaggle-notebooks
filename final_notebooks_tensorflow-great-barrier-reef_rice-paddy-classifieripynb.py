#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
import os


# In[ ]:


train_df=pd.read_csv('../input/paddy-disease-classification/train.csv')
train_df.head()


# In[ ]:


all_images={}
train_images_path='../input/paddy-disease-classification/train_images/'
for category in os.listdir(train_images_path):
    for img in os.listdir(train_images_path+category):
        all_images[img]=train_images_path+category+'/'+img
categories=os.listdir(train_images_path)
label_dict={}
for category in categories:
    label_dict[category]=categories.index(category)
label_dict


# In[ ]:


def get_label(label):
    return label_dict[label]


# In[ ]:


num_label={
    0:'tungro',
    1:'hispa',
    2:'downy_mildew',
    3:'bacterial_leaf_streak', 
    4:'bacterial_leaf_blight',
    5:'brown_spot',
    6:'blast',
    7:'normal',
    8:'dead_heart',
    9:'bacterial_panicle_blight'
}


# In[ ]:


def get_name(x):
    return num_label[x]


# In[ ]:


img_size=128
training=[]
for i in range(len(train_df)):
    img_array=cv2.imread(all_images[train_df.iloc[i,0]])
    new_array=cv2.resize(img_array,(img_size,img_size))
    label=get_label(train_df.iloc[i,1])
    training.append([new_array,label])
training[0]


# In[ ]:


random.shuffle(training)


# In[ ]:


X=[]
y=[]
for features, label in training:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,img_size,img_size,3)


# In[ ]:


X=X.astype('float32')
X/=255
from keras.utils import np_utils
Y=np_utils.to_categorical(y,10)
print(Y[100])
print(Y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


# In[ ]:


model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),

    
tf.keras.layers.Conv2D(32, (3,3), activation=None, use_bias = True),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),
    
tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),
    
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),

tf.keras.layers.Conv2DTranspose(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),

        
tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),

tf.keras.layers.Conv2DTranspose(128, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LeakyReLU(0.2),
    
tf.keras.layers.Flatten(),

#tf.keras.layers.LSTM(32, activation = 'relu', return_sequences = True),

tf.keras.layers.Dense(8192, activation='relu'),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()



# In[ ]:


from keras.callbacks import *

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
valid_ds=tf.data.Dataset.from_tensor_slices((X_valid,y_valid))
mc=ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

history=model.fit(train_ds.batch(128),
         epochs=150,
         validation_data=valid_ds.batch(128), callbacks = [mc])


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training Data')
plt.plot(history.history['val_accuracy'],label='Validation Data')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'],label='Training Data')
plt.plot(history.history['val_loss'],label='Validation Data')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


submission_df=pd.read_csv('../input/paddy-disease-classification/sample_submission.csv')
test_path='../input/paddy-disease-classification/test_images/'
test_images=[]
for i in range(len(submission_df)):
    img_array=cv2.imread(test_path+submission_df.iloc[i,0])
    new_array=cv2.resize(img_array,(img_size,img_size))
    test_images.append(new_array)
test_images[0]
submission_df


# In[ ]:


X_test=[]
for features in test_images:
    X_test.append(features)
X_test=np.array(X_test).reshape(-1,img_size,img_size,3)
X_test=X_test.astype('float32')
X_test/=255
y_pred=model.predict(X_test)
y_pred


# In[ ]:


labels=[]
for i in y_pred:
    i=np.array(i)
    labels.append(np.argmax(i,axis=0))
submission_df['label']=labels
submission_df


# In[ ]:


model.evaluate(X_valid,y_valid)


# In[ ]:


submission_df.to_csv('submission.csv',index=False)
import os
os.chdir(r'/kaggle/working')
from IPython.display import FileLink
FileLink(r'submission.csv')

