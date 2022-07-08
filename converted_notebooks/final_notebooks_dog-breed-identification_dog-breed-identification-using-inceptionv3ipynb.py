#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.preprocessing.image import load_img

from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Lambda, Input
from tensorflow.keras.models import Model
from keras import regularizers, optimizers
from keras.utils import to_categorical
import pandas as pd
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Step 1 - Data Preprocessing

# In[ ]:


traindf=pd.read_csv('../input/dog-breed-identification/labels.csv',dtype=str)
testdf=pd.read_csv("../input/dog-breed-identification/sample_submission.csv",dtype=str)


# In[ ]:


def append_ext(fn):
    return fn+".jpg"

traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)


# In[ ]:


traindf.head(5)


# In[ ]:


#Create list of alphabetically sorted labels.
classes = sorted(list(set(traindf['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))

#Map each label string to an integer label.
class_to_num = dict(zip(classes, range(n_classes)))
class_to_num


# In[ ]:


input_shape = (331,331,3)

# Transform image to matrice
def images_to_array(directory, label_dataframe, target_size = input_shape):
    
    image_labels = label_dataframe['breed']
    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]],dtype=np.uint8) #as we have huge data and limited ram memory. uint8 takes less memory
    y = np.zeros([len(label_dataframe),1],dtype = np.uint8)
    
    for ix, image_name in enumerate(tqdm(label_dataframe['id'].values)):
        img_dir = os.path.join(directory, image_name)
        img = load_img(img_dir, target_size = target_size)
        images[ix]=img
        del img
        
        dog_breed = image_labels[ix]
        y[ix] = class_to_num[dog_breed]
    
    y = to_categorical(y)
    
    return images,y

X,y = images_to_array('/kaggle/input/dog-breed-identification/train', traindf[:])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# In[ ]:


del X,y #to free up some ram memory
gc.collect()


# ## Step 2 - Transfer learning using InceptionV3

# In[ ]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers.experimental import preprocessing

model1 = InceptionV3(weights='imagenet', include_top=False)

for layer in model1.layers:
    layer.trainable = False
    
inputs = Input(shape=(331,331,3))

model_in = Lambda(preprocess_input)(inputs) # normalize image according to InceptionV3 preprocess input 

# Apply some modifications to images present in dataset
model_in = preprocessing.RandomFlip('horizontal')(model_in)
model_in = preprocessing.RandomContrast(0.5)(model_in)

model1_out = model1(model_in)
model1_out= GlobalAveragePooling2D()(model1_out)

x = Dropout(0.7)(model1_out)

x = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=inputs,outputs=x)


# In[ ]:


model.compile(optimizer = 'adam', loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


EarlyStop = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)

history = model.fit(X_train,y_train,
            validation_split=0.2,
            batch_size = 32,
            epochs=10,
            callbacks=[EarlyStop])


# In[ ]:


#Plot accuracy and loss performance

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ## Step 3 - Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test, y_pred)


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(cm, fmt="d", xticklabels = classes, yticklabels = classes, annot = True, cmap='Blues', cbar=False)
plt.show()


# ## Step 4 - Make predictions with custom images

# In[ ]:


#Custom input

from IPython.display import display, Image

def import_image(image_path, target_size = (331,331,3)):
    display(Image(image_path))
    
    #reading the image and converting it into an np array
    img_g = load_img(image_path,target_size = target_size)
    img_g = np.expand_dims(img_g, axis=0) # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format
    # img_g
    
    prediction = model.predict(img_g)

    print(f"Max value (probability of prediction): {np.max(prediction[0])}") # the max probability value predicted by the model
    print(f"Max index: {np.argmax(prediction[0])}") # the index of where the max value in predictions[0] occurs
    print(f"Predicted label: {classes[np.argmax(prediction[0])]}")


# In[ ]:


import_image('../input/dog-breed-identification/test/000621fb3cbb32d8935728e48679680e.jpg')

