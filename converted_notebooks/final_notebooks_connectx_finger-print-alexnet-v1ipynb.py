#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def extract_label(img_path, train=True):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    subject_id, etc = filename.split('__')
    
    if train:
        gender, lr, finger, _, _ = etc.split('_')
    else:
        gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr =0 if lr == 'Left' else 1
    
    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
     
    return np.array([finger], dtype=np.uint16)


# In[ ]:


IMG_SIZE = 96

def load_data(path, train):
    print("loading data from: ", path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            label = extract_label(os.path.join(path, img),train)
            data.append([label[0], img_resize ])
        except Exception as e:
            pass
    return data


# In[ ]:


Real_path = "../input/socofing/SOCOFing/Real"
Easy_path = "../input/socofing/SOCOFing/Altered/Altered-Easy"
Medium_path = "../input/socofing/SOCOFing/Altered/Altered-Medium"
Hard_path = "../input/socofing/SOCOFing/Altered/Altered-Hard"

easy_data = load_data(Easy_path, train = True)
medium_data = load_data(Medium_path, train = True)
hard_data = load_data(Hard_path, train = True)
test = load_data(Real_path, train = False)

data = np.concatenate([easy_data,medium_data,hard_data],axis=0)


# In[ ]:


import random
random.shuffle(data)
random.shuffle(test)


# In[ ]:


data[0]


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(data[0][1])


# In[ ]:


X, y = [],[]

for label, feature in data:
    y.append(label)
    X.append(feature)

 
X_train = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_train = X_train/255

y_train = np.array(y)


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


np.unique(y_train)


# In[ ]:


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        #dim_ordering='tf',
        fill_mode='nearest')


# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Dropout, Flatten
from tensorflow.keras import layers
from tensorflow.keras import optimizers


# In[ ]:


model = Sequential()
#1st convlayer 
model.add(Conv2D(input_shape=(96,96,1),filters=64,kernel_size=(11,11),padding='same', activation='relu'))

#first layer has 96 filters 
#model.add(BatchNormalization())
#max Pooling
model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

# 2nd conv layer 
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())

#max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd conv layer 
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
#model.add(BatchNormalization())

#4th conv layer 
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
#model.add(BatchNormalization())

#5th conv layer 
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
#max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(BatchNormalization())



model.add(Flatten())
#1st fully connected layer has 4096
model.add(Dense(4096, input_shape=(96,96,1)))
model.add(BatchNormalization())
#Add dropout 
model.add(Dropout(0.5))

#2nd Fully connected layer 
model.add(Dense(4096))
model.add(BatchNormalization())
#Add dropout 
model.add(Dropout(0.5))

#passing it to a fully connected layer 


#3rd Fully Connected Layer
model.add(Dense(1000))
model.add(BatchNormalization())


#Output layer

model.add (layers.Dense(units=5, activation='softmax'))


model.summary()

#Compile the model 
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])


# In[ ]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train, batch_size=128, epochs=10, validation_split=0.2)


# In[ ]:





# In[ ]:


plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title("Accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:


X_test,y_test = [],[]

for label, feature in test:
    y_test.append(label)
    X_test.append(feature)
    
X_test = np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_test = X_test/255

y_test = np.array(y_test)


# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


predictions = model.predict(X_test)
predictions[:5]


# In[ ]:


predicted = [np.argmax(i) for i in predictions]
predicted[:5]


# In[ ]:


import tensorflow as tf
cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted)


# In[ ]:


import seaborn as sn

sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predictions")
plt.ylabel("Truth")


# In[ ]:




