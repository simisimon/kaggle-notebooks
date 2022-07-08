#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import img_to_array


# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df=pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


target_df = train_df[['healthy', 'multiple_diseases', 'rust', 'scab']]
test_ids = test_df['image_id']


# **Image processing**

# In[ ]:


img_size = 300

train_imgs = []

for name in train_df['image_id']:
    path = '../input/plant-pathology-2020-fgvc7/images/' + name + '.jpg'
    img = cv2.imread(path)
    image = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
    train_imgs.append(image)


# In[ ]:


fig, ax = plt.subplots(1, 4, figsize=(15, 15))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(train_imgs[i])


# In[ ]:


test_imgs = []
for name in test_df['image_id']:
    path = '../input/plant-pathology-2020-fgvc7/images/' + name + '.jpg'
    img = cv2.imread(path)
    image = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
    test_imgs.append(image)


# In[ ]:


fig, ax = plt.subplots(1, 4, figsize=(15, 15))
for i in range(4):
    ax[i].set_axis_off()
    ax[i].imshow(test_imgs[i])    


# In[ ]:


X_train = np.ndarray(shape=(len(train_imgs), img_size, img_size, 3),dtype = np.float32)

for i, image in enumerate(train_imgs):
    X_train[i] = img_to_array(image)
    X_train[i] = train_imgs[i]

X_train = X_train/255
print('Train Shape: {}'.format(X_train.shape))


# In[ ]:


X_test = np.ndarray(shape=(len(test_imgs), img_size, img_size, 3),dtype = np.float32)

for i, image in enumerate(test_imgs):
    X_test[i] = img_to_array(image)
    X_test[i] = test_imgs[i]
    
X_test = X_test/255
print('Test Shape: {}'.format(X_test.shape))


# **Split training set**

# In[ ]:


y_train = train_df.copy()
del y_train['image_id']
y_train.head()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train.shape, X_val.shape


# **Set Callbacks**

# In[ ]:


from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

lr_reduce=ReduceLROnPlateau(monitor='val_accuracy',
                            factor=.25,
                            patience=3,
                            min_lr=.000001,
                            verbose=1)

es_monitor=EarlyStopping(monitor='val_loss',
                          patience=5)

mdl_check = ModelCheckpoint('best_model.h5', 
                            monitor='accuracy', 
                            verbose=0, 
                            save_best_only=True, 
                            mode='max')


# In[ ]:


get_ipython().system('pip install efficientnet')


# In[ ]:


import efficientnet.keras as efn 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


# In[ ]:


net =efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size,img_size,3))
x = net.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(4, activation= 'softmax')(x)
model = Model(inputs = net.input, outputs = predictions)


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# **Image Data Augmentation and fit model**

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=.1,
    rescale=1/255,
    fill_mode='nearest',
    shear_range=0.1,
    brightness_range=[0.5, 1.5])


# In[ ]:


history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                              epochs = 20,
                              steps_per_epoch = X_train.shape[0] // 16,
                              verbose = 1,
                              callbacks = [es_monitor,lr_reduce, mdl_check],
                              validation_data = datagen.flow(X_val, y_val,batch_size=16),
                              validation_steps = X_val.shape[0] // 16)


# In[ ]:


h = history.history

offset = 5
epochs = range(offset, len(h['loss']))

plt.figure(1, figsize=(12, 12))

plt.subplot(211)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(epochs, h['loss'][offset:], label='train')
plt.plot(epochs, h['val_loss'][offset:], label='val')
plt.legend()

plt.subplot(212)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(h[f'accuracy'], label='train')
plt.plot(h[f'val_accuracy'], label='val')
plt.legend()

plt.show()


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


sub_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
sub_df.loc[:, 'healthy':] = y_pred
sub_df.to_csv('submission.csv', index=False)
sub_df.head()

