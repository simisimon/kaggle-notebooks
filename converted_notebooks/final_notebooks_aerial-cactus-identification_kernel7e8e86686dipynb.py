#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os


# Let's import the train dataset :

# In[ ]:


train = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv', dtype=str)
train.head()


# It's made of the name of each picture, we therefore need to extract the image from the .zip file :

# In[ ]:


import zipfile

with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/train.zip','r') as file:
    file.extractall('/kaggle/output/kaggle/working/train')

with zipfile.ZipFile('/kaggle/input/aerial-cactus-identification/test.zip','r') as file:
    file.extractall('/kaggle/output/kaggle/working/test')


# We need to check whether the dataset is imbalanced or not :

# In[ ]:


print("Proportion of picture with cactus : {} % ".format(100 * round(sum(train["has_cactus"].astype('int'))/train.shape[0], 4)))


# It is imbalanced, but not that much, so we will not try to rebalanced it.

# In[ ]:


train_path = "/kaggle/output/kaggle/working/train/train"
test_path = '/kaggle/output/kaggle/working/test/'
label_path = '/kaggle/input/aerial-cactus-identification/train.csv'
submission_path = "/kaggle/input/aerial-cactus-identification/sample_submission.csv"


# Let's look at the image of cactus we have : 

# In[ ]:


import cv2
from IPython.display import Image

Image(os.path.join(train_path,train["id"][0]),width=100,height=100)


# According to the dataset, the is some cactus in the picture, but it is obvious. We hope a neural network will perform better than us !

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# We don't have that much picture to train the neural network on, so we will use the *ImageDataGenerator* function from *keras* in order to get more picture by turning/zooming a bit the already existing pictures : 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(rescale=1./255,
                               zoom_range=0.1,
                               rotation_range=15,
                               horizontal_flip=True,
                               vertical_flip=True,
                               zca_whitening=True)


# We will now split with the train set into a training and validation set. We use the generator previously defined :

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid = train_test_split(train, test_size=0.2)

train_generator = generator.flow_from_dataframe(dataframe=X_train,
                                                directory=train_path,
                                                x_col="id", y_col="has_cactus", class_mode='binary',
                                                target_size=(32, 32),
                                                batch_size=64)


valid_generator = generator.flow_from_dataframe(dataframe=X_valid,
                                                       directory=train_path,
                                                       x_col="id", y_col="has_cactus", class_mode='binary',
                                                       target_size=(32, 32),
                                                       batch_size=64)


# # Model
# 
# We need to define the model. We tried a residual network and a conventionnal convolutionnal neural network. But after some test, it turns out that a separable convolutionnal neural network worked best for us.

# In[ ]:


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer="glorot_normal", input_shape=[32, 32, 3]),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=2),
    
    keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=2),
    
    keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=2),
    
    keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", kernel_initializer="glorot_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(256, kernel_initializer="glorot_normal", activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(256, kernel_initializer="glorot_normal", activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(learning_rate=0.001), metrics=["accuracy"])
model.summary()


# We can now train the model, we also make sure we have the GPU on, otherwise the model training will take a long time.

# In[ ]:


history=model.fit_generator(train_generator, epochs=100, validation_data=valid_generator, validation_steps=20)


# If we look at the learning curves we have :

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)


# # Prediction
# 
# We are now going to predict the test dataset :

# In[ ]:


test = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv', dtype=str)
test.head()


# We can't use the same generator as we already have used since we don't want to zoom in or rotate the picture.

# In[ ]:


soft_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = soft_generator.flow_from_directory(directory=test_path, target_size=(32, 32), batch_size=1, class_mode='binary', shuffle=False)


# In[ ]:


y_pred_proba = model.predict_generator(test_generator)

y_pred = [0 if value < 0.50 else 1 for value in y_pred_proba] 
y_pred = np.array(y_pred)
y_pred.reshape(4000,1)

submission = pd.DataFrame(data = {'id': test["id"], 'has_cactus': y_pred.reshape(-1).tolist()})
submission.to_csv('submission.csv', index=False)

