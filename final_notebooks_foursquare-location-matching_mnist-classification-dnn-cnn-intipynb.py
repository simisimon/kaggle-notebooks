#!/usr/bin/env python
# coding: utf-8

# # Installing & Importing Packages

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
from keras import layers, models

from tensorflow.keras.layers import Conv2D, Dense, Dropout, InputLayer, Softmax, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.datasets import mnist
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


# # Importing the data

# In[ ]:


df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")

# Removing and storing the class labels in another variable
Y = df_train['label']
df_train.drop(["label"], axis=1, inplace=True)

print(df_train.shape, df_test.shape, Y.shape)


# In[ ]:


df_train.head()


# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)
df_train = np.concatenate([df_train, x_train, x_test], axis = 0)
Y = np.concatenate([Y, y_train, y_test], axis = 0)
print(df_train.shape, Y.shape)


# # Visualizing & Processing the Data

# In[ ]:


# Scaling the pixel values to be between 0 and 1
df_train = df_train.astype('float32')
df_test = df_test.astype('float32')
df_train = df_train / 255
df_test = df_test / 255


# In[ ]:


# The image is having dimensions 28*28
im_dim = 28

# Reshaping the dataset, so that we can display the individual images, and model them
df_train = tf.reshape(df_train, (-1, im_dim, im_dim, 1))
df_test = tf.reshape(df_test, (-1, im_dim, im_dim, 1))
print(df_train.shape, df_test.shape)


# In[ ]:


fig,axes = plt.subplots(5, 5, figsize = (6,6))
axes = axes.ravel()

for i in np.arange(0,25):
    axes[i].imshow(df_train[i])
    axes[i].axis("off")


# # Training the Model

# In[ ]:


# Defining some of the key parameters
num_classes = 10
batch_size = 128
epochs = 10


# In[ ]:


input_shape = (28, 28, 1)

model = tf.keras.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3, 3),padding='same', activation='relu'))
model.add(Conv2D(64,kernel_size=(3, 3),activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


# model = tf.keras.Sequential(layers = [
#     Conv2D(filters=64, kernel_size=5, activation="relu", padding='Same', input_shape=(28, 28, 1)),
#     BatchNormalization(),
    
#     Conv2D(filters=64, kernel_size=5, activation="relu", padding='Same'),
#     BatchNormalization(),
#     MaxPool2D(pool_size=(2, 2)),
#     Dropout(rate=0.25),
    
#     Conv2D(filters=64, kernel_size=3, activation="relu", padding='Same'),
#     BatchNormalization(),
    
#     Conv2D(filters=64, kernel_size=3, activation="relu", padding='Same'),
#     BatchNormalization(),
#     MaxPool2D(pool_size=(2, 2), strides=(2,2)),
#     Dropout(rate=0.25),
    
#     Conv2D(filters=64, kernel_size=3, activation="relu", padding='Same'),
#     BatchNormalization(),
#     Dropout(rate=0.25),
    
#     Flatten(),
#     Dense(256, activation="relu"),
#     BatchNormalization(),
#     Dropout(rate=0.25),
#     Dense(10, activation="softmax")
# ])


# In[ ]:


model.summary()


# In[ ]:


plot_model(model, show_shapes=True, show_layer_names=True)


# In[ ]:


# Defining the Adam Optimizer
# sgd = SGD(lr=2e-2, decay=1e-6, momentum=0.9)
rms = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Defining the callbacks
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_acc', factor = 0.5, patience = 3, 
    min_lr = 1e-5, verbose = 1
)
# early_st = EarlyStopping(
#     monitor='val_loss', min_delta=1e-3,
#     patience=5, verbose=1, restore_best_weights=True, mode = 'min'
# )

# Compiling the model
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=rms)


# In[ ]:


# Converting the class labels into one-hot form
Y_oh = to_categorical(Y, num_classes=num_classes)


# In[ ]:


# Using real-time Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center = False, samplewise_center=False, featurewise_std_normalization = False,
    samplewise_std_normalization = False, rotation_range = 10, zoom_range = 0.1,
    width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = False, 
    vertical_flip=False,  validation_split = 0.1
)
datagen.fit(df_train)

train_generator = datagen.flow(df_train, Y_oh, batch_size = batch_size, subset='training')
val_generator = datagen.flow(df_train, Y_oh, batch_size = batch_size, subset = 'validation')

# Training the model using generators
history = model.fit(
    train_generator, batch_size = batch_size,
    epochs = epochs, verbose = 1, validation_data = val_generator,
    steps_per_epoch = df_train.shape[0] // batch_size,
    use_multiprocessing = True, callbacks = [reduce_lr]
)


# In[ ]:


# Without using any augmentation
# model.fit(
#     df_train, Y_oh, batch_size = batch_size,
#     epochs = epochs, verbose = 1, validation_split = 0.2,
#     use_multiprocessing = True, callbacks = [reduce_lr]
# )


# In[ ]:


fig,axes = plt.subplots(1,2, figsize=(15,8))
fig.suptitle("The model 's evaluation ",fontsize=20)
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_title('Model Loss')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train','Test'])


axes[1].plot(history.history['accuracy'])
axes[1].plot(history.history['val_accuracy'])
axes[1].set_title('Model Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train','Test'])
plt.show()


# In[ ]:


y_pred = model.predict(df_train)
y_pred = np.argmax(y_pred, axis = 1)
acc = metrics.Accuracy()
print(acc(Y, y_pred))


# # Prediction

# In[ ]:


y_sub = model.predict(df_test)
y_sub = tf.math.argmax(y_sub, axis = 1)
y_sub = pd.Series(y_sub)
print(y_sub.shape, type(y_sub))


# In[ ]:


df_sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
df_sub.loc[ : , 'Label'] = y_sub


# In[ ]:


df_sub.to_csv("submission.csv", index = False)

