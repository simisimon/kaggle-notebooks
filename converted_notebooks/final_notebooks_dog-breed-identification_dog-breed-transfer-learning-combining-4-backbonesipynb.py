#!/usr/bin/env python
# coding: utf-8

# ## Dog Breed Identification
# 
# In this competition, given an image of a dog we are asked to predict a probability for each of the different 120 breeds of the dogs.
# 
# Notebook summary:
# * The Notebook is created on Dog Breed Dataset.
# * As there are 120 Different Breeds of Dog present in the Dataset it becomes difficult for a single pretrained model to give Good results.
# * Here we used different pretrained models to extract features from the images and combined them and then trained a DNN model on these features.
# 
# So let's get started.

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


# !pip install livelossplot


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow import keras

import cv2
import PIL
import os
import pathlib
import shutil
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.applications import vgg16 
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import nasnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

import gc
import skimage.io
# from livelossplot import PlotLossesKeras


# ## Load Data

# In[ ]:


train_dir = '../input/dog-breed-dataset-with-subdirectories-by-class/data/train'
test_dir = '../input/dog-breed-dataset-with-subdirectories-by-class/data/test'

train_labels = pd.read_csv('../input/dog-breed-identification/labels.csv', index_col = 'id')
submission=pd.read_csv('../input/dog-breed-identification/sample_submission.csv')


# In[ ]:


train_size = len(os.listdir(train_dir))
test_size = len(os.listdir(test_dir))

print(train_size,test_size)
print(train_labels.shape)
print(submission.shape)


# In[ ]:


target, dog_breeds = pd.factorize(train_labels['breed'], sort = True)
train_labels['target'] = target

print(dog_breeds)


# In[ ]:


display(train_labels.head())
display(submission.head())


# In[ ]:


train_labels['breed'].value_counts()


# In[ ]:


plt.figure(figsize=(15, 10))
train_labels['breed'].value_counts().plot(kind='bar')
plt.show()


# ## Global Variables

# In[ ]:


N_EPOCHS = 50
BATCH_SIZE = 128
IMG_HEIGHT = 331
IMG_WIDTH = 331


# ## Prepare the Datasets

# In[ ]:


train_ds = image_dataset_from_directory(
  directory = train_dir,
  labels = 'inferred',
  label_mode='int',
  batch_size=BATCH_SIZE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed=1234,
  validation_split=0.1,
  subset="training",
)


# In[ ]:


class_names = train_ds.class_names
print(len(class_names))
print(class_names)


# In[ ]:


val_ds = image_dataset_from_directory(
  directory = train_dir,
  labels = 'inferred',
  label_mode='int',
  batch_size=BATCH_SIZE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed=1234,
  validation_split=0.1,
  subset="validation",
)


# In[ ]:


test_ds = image_dataset_from_directory(
  directory = test_dir,
  label_mode= None,
  batch_size=BATCH_SIZE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  shuffle = False,
  seed=1234
)


# In[ ]:


del class_names


# ## Visualize the Data

# In[ ]:


plt.figure(figsize=(20, 20))

for images, labels in train_ds.take(1):
#   print(labels)
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(dog_breeds[labels[i]])
    plt.axis("off")


# In[ ]:


plt.figure(figsize=(20, 20))
for images, labels in val_ds.take(1):
#   print(labels)
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(dog_breeds[labels[i]])
    plt.axis("off")


# In[ ]:


plt.figure(figsize=(20, 20))
for images in test_ds.take(1):
#   print(labels)
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(dog_breeds[labels[i]])
    plt.axis("off")


# ## Configure the dataset for performance
# 
# Let's make sure to use buffered prefetching, so we can yield data from disk without having I/O become blocking. These are two important methods we should use when loading data.
# 
# .cache() keeps the images in memory after they're loaded off disk during the first epoch. 
# 
# .prefetch() overlaps data preprocessing and model execution while training.

# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE

# Without Caching
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# With Caching
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ## Data Augmentation layer

# In[ ]:


data_augmentation = Sequential(
  [
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
  ]
)


# Let's visualize what the first image of the first batch looks like after various random transformations:

# In[ ]:


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0), training=True
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.title(dog_breeds[labels[0]])
        plt.axis("off")


# ## Build a Model

# In[ ]:


base_model_1 = xception.Xception(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

base_model_2 = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

base_model_3 = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

# base_model_4 = resnet_v2.ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

base_model_5 = nasnet.NASNetLarge(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

# display(base_model_1.summary())
# display(base_model_2.summary())
# display(base_model_3.summary())
# display(base_model_4.summary())
# display(base_model_5.summary())

# train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Xception layers
base_model_1.trainable = False
base_model_2.trainable = False
base_model_3.trainable = False
# base_model_4.trainable = False
base_model_5.trainable = False

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
aug_inputs = data_augmentation(inputs)

## <-----  Xception   -----> ##
x1 = xception.preprocess_input(aug_inputs)
# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here by passing `training=False`.
x1 = base_model_1(x1, training=False)
x1 = GlobalAveragePooling2D()(x1)

## <-----  InceptionV3   -----> ##
x2 = inception_v3.preprocess_input(aug_inputs)
x2 = base_model_2(x2, training=False)
x2 = GlobalAveragePooling2D()(x2)

## <-----  InceptionResNetV2   -----> ##
x3 = inception_resnet_v2.preprocess_input(aug_inputs)
x3 = base_model_3(x3, training=False)
x3 = GlobalAveragePooling2D()(x3)

## <-----  ResNet152V2   -----> ##
# x4 = resnet_v2.preprocess_input(aug_inputs)
# x4 = base_model_4(x4, training=False)
# x4 = GlobalAveragePooling2D()(x4)

## <-----  NASNetLarge   -----> ##
x5 = nasnet.preprocess_input(aug_inputs)
x5 = base_model_5(x5, training=False)
x5 = GlobalAveragePooling2D()(x5)

## <-----  Concatenation  -----> ##
x = Concatenate()([x1, x2, x3, x5])
x = Dropout(.7)(x)
outputs = Dense(120, activation='softmax')(x)
model = Model(inputs, outputs)

display(model.summary())


# ## Training the Model

# In[ ]:


optimizer = Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)


# In[ ]:


EarlyStop_callback = EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

# # DECREASE LEARNING RATE EACH EPOCH
# annealer = LearningRateScheduler(lambda epoch: 1e-5 * 0.95 ** epoch, verbose=1)

# cb=[PlotLossesKeras(), annealer]


# In[ ]:


history = model.fit(
    train_ds,
    epochs=N_EPOCHS,
    validation_data=val_ds,
    callbacks=[EarlyStop_callback]
)


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.xticks(list(range(20)))
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xticks(list(range(20)))
plt.xlabel('Epoch')
plt.show()


# ## Analyse Wrong Predictions on Validation Set

# In[ ]:


wrong_pred_images = np.array([])
actual_labels = np.array([])
predicted_labels = np.array([])


batch = 1
for images, labels in val_ds:
    batch_predictions_probs = model.predict_on_batch(images)
    batch_predictions = np.argmax(batch_predictions_probs, axis=1)
    mask = (batch_predictions != labels.numpy())
    print("No of wrong predictions on batch {}: {}".format(batch, mask.sum()))
    
    wrong_pred_indices = np.arange(len(batch_predictions))[mask]
    print(wrong_pred_indices)
    
    if len(wrong_pred_images) == 0:
        wrong_pred_images = images.numpy()[wrong_pred_indices]
        actual_labels = labels.numpy()[wrong_pred_indices]
        predicted_labels = batch_predictions[wrong_pred_indices]
    else:
        wrong_pred_images = np.append(wrong_pred_images, images.numpy()[wrong_pred_indices], axis = 0)
        actual_labels = np.append(actual_labels, labels.numpy()[wrong_pred_indices], axis = 0)
        predicted_labels = np.append(predicted_labels, batch_predictions[wrong_pred_indices], axis = 0)
        
    batch = batch + 1
    
print(wrong_pred_images.shape)
print(actual_labels.shape)
print(predicted_labels.shape)


# In[ ]:


plt.figure(figsize=(20, 20))

for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(wrong_pred_images[i].astype("uint8"))
    plt.title("Actual : {}\n Predicted : {}".format(dog_breeds[actual_labels[i]], dog_breeds[predicted_labels[i]]))
    plt.axis("off")


# In[ ]:


plt.figure(figsize=(20, 20))

for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(wrong_pred_images[i+25].astype("uint8"))
    plt.title("Actual : {}\n Predicted : {}".format(dog_breeds[actual_labels[i+25]], dog_breeds[predicted_labels[i+25]]))
    plt.axis("off")


# ## Fine Tuning

# In[ ]:


# # Unfreeze the base_model. Note that it keeps running in inference mode
# # since we passed `training=False` when calling it. This means that
# # the batchnorm layers will not update their batch statistics.
# # This prevents the batchnorm layers from undoing all the training
# # we've done so far.
# base_model.trainable = True
# model.summary()


# In[ ]:


# print("Number of layers in the base model: ", len(base_model.layers))


# In[ ]:


# fine_tune_from = 100

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_from]:
#     layer.trainable =  False


# In[ ]:


# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=[keras.metrics.BinaryAccuracy()],
# )


# In[ ]:


# FINE_TUNE_EPOCHS = 10
# fine_tune_history = model.fit(train_ds,
#                          epochs = FINE_TUNE_EPOCHS + history.epoch[-1],
#                          initial_epoch = history.epoch[-1] + 1,
#                          validation_data = val_ds)


# In[ ]:


# accuracy += history_fine.history['accuracy']
# val_accuracy += history_fine.history['val_accuracy']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']


# In[ ]:


# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.8, 1])
# plt.plot([EPOCHS-1,EPOCHS-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([EPOCHS-1,EPOCHS-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()


# ## Predict on Test Dataset

# In[ ]:


# del train_ds, val_ds, train_labels, class_names


# In[ ]:


# file_paths = test_ds.file_paths
# print(file_paths)


# In[ ]:


predictions = model.predict(
    test_ds,
    batch_size = BATCH_SIZE,
    verbose=1         
)


# In[ ]:


print(predictions.shape)
print(predictions)


# ## Submission

# In[ ]:


submission.loc[:, dog_breeds] = predictions
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)
# submission.head()


# In[ ]:





# In[ ]:




