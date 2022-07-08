#!/usr/bin/env python
# coding: utf-8

# # ✨(05/13 Renewal)Simple CNN Model Using Keras✨

# **Update 05/13: The code & notebook has been renewed due to the inefficiency & limits of the previous code.**
# 
# This is a simple notebook that shows how to
# 
# - Read & Analyze, Preprocess the Dataset
# - Build a simple CNN model and train it
# - Predict & Analyze results
# 
# using Keras.

# # Import Packages & Dataset
# ## Packages

# In[ ]:


import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tqdm import tqdm


# ## Loading the Data

# In[ ]:


dataset = "/kaggle/input/devkor-image-classification/train"
        
images = []
labels = []
        
print("Loading {}".format(dataset))
        
# Iterate through each folder corresponding to a category
for folder in os.listdir(dataset):
    label = folder
            
    # Iterate through each image in our folder
    for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
        # Get the path name of the image
        img_path = os.path.join(os.path.join(dataset, folder), file)
                
        # Open the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grayscale
                
        # Append the image and its corresponding label
        images.append(image)
        labels.append(label)
                
images = np.array(images, dtype = 'float32')
labels = np.array(labels, dtype = 'int32')   


# In[ ]:


class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# # Data PreProcessing

# Normalize RGB Value From 0 ~ 256 to 0 ~ 1

# In[ ]:


images = images / 255.0 


# ## Check the data

# In[ ]:


print(images.shape)
print(labels.shape)


# Change images to 4-dimensions for keras model layer input

# In[ ]:


images = images.reshape(60000, 28, 28, 1)


# In[ ]:


#Display a random image and it's label
    
index = np.random.randint(images.shape[0])
plt.figure()
plt.imshow(images[index], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title(class_names[labels[index]])
plt.show()


# # Train-Test Split

# In[ ]:


train_images, val_images, train_labels, val_labels = train_test_split(images,labels , test_size=0.2, random_state=42)


# In[ ]:


print(train_images.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_labels.shape)


# Check the label distribution in the training set and validation set.

# In[ ]:


import pandas as pd

_, train_counts = np.unique(train_labels, return_counts=True)
_, val_counts = np.unique(val_labels, return_counts=True)
pd.DataFrame({'train': train_counts,
                    'test': val_counts}, 
             index=class_names
            ).plot.bar()
plt.show()


# # Build The Model

# ## CNN Model
# Implement a simple CNN Model, and compile it.

# In[ ]:


model = Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# ## Training

# In[ ]:


history = model.fit(
  x=train_images, y=train_labels,validation_data=(val_images, val_labels), batch_size=32, epochs=20
)


# ## Training Data Analysis

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20) # epochs:20 

plt.figure(figsize=(8, 8))
plt.tight_layout()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## Error Analysis

# Use the predictions on Validation Dataset to analyze errors of the classifier.

# In[ ]:


val_predictions =  np.argmax(model.predict(val_images), axis = 1)


# In[ ]:


#Print 25 examples of mislabeled images by the classifier
mislabeled_indices = np.where(val_labels != val_predictions)
mislabeled_images = val_images[mislabeled_indices]
mislabeled_labels = val_labels[mislabeled_indices]
mislabeled_predictions = val_predictions[mislabeled_indices]


fig = plt.figure(figsize=(20,20))
fig.tight_layout()
fig.suptitle("Some examples of mislabeled images by the classifier:", fontsize=25)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(mislabeled_images[i], cmap='gray')
    plt.xlabel("Original:{}\nPrediction:{}".format(class_names[mislabeled_labels[i]],class_names[mislabeled_predictions[i]]), fontsize=12)
plt.show()


# In[ ]:


CM = confusion_matrix(val_labels, val_predictions)
ax = plt.axes()
sn.set(rc = {'figure.figsize':(20,20)})
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()


# The result shows that the model is having models between distinguishing :
# - **T-shirt/top (0) vs Shirt (6), Pullover (2) vs Shirt (6), Shirt (6) vs Coat (4)**
# - Pullover (2) vs Coat (4)
# - Ankle Boot (9) vs Sneaker (7), Sneaker(7) vs Sandal(5)
# - Dress (3) vs Coat (4)

# # Improvement : Data Augmentation
# [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation) can be used to generate more data to make a better prediction result and prevent overfitting. Keras preprocessing layers (RandomFlip,RandomRotation, RandomZoom) can be used. 

# In[ ]:


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(28,
                                  28,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# Here is an example of generated images.

# In[ ]:


plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_images = data_augmentation(train_images[31]) # 31 is just a random number
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images, cmap='gray')
    plt.axis("off")


# ## Retrain the model

# In[ ]:


model = Sequential([
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(
  x=train_images, y=train_labels,validation_data=(val_images, val_labels), batch_size=32, epochs=20
)


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20) # epochs:20 

plt.figure(figsize=(8, 8))
plt.tight_layout()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## Prediction On Test Dataset & Submission

# In[ ]:


test_dataset = "/kaggle/input/devkor-image-classification/test"
        
test_images = []
        
print("Loading {}".format(test_dataset))
        
            
    # Iterate through each image in directory
for file in tqdm(sorted(os.listdir(test_dataset))): ##Important: os.listdir() does not give files in order, so sorting is needed when loading test dataset.
                
        # Get the path name of the image
    img_path = os.path.join(test_dataset, file)
                
        # Open the image
    test_image = cv2.imread(img_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                
        # Append the image
    test_images.append(test_image)
                
test_images = np.array(test_images, dtype = 'float32')


# In[ ]:


test_images = test_images/255.0
test_images = test_images.reshape(10000, 28, 28, 1)

test_predictions = np.argmax(model.predict(test_images), axis = 1)

submission = pd.read_csv("/kaggle/input/devkor-image-classification/sample_submission.csv")
submission.loc[:, "label"] = test_predictions
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()

