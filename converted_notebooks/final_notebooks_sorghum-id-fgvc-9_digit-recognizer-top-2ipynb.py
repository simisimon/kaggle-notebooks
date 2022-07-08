#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# select model
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


root = '/kaggle/input/digit-recognizer'


# # load data

# In[ ]:


train_data = pd.read_csv(os.path.join(root, 'train.csv'))
train_data.head()


# # Analyze data

# In[ ]:


train_data.isnull().sum()


# In[ ]:


Y = train_data['label']
images = train_data.drop(['label'], axis=1)
print(images.shape, Y.shape)
del train_data


# In[ ]:


X = images.to_numpy(dtype='float32').reshape((42000, 28, 28, 1)) / 255.0
X.shape


# # Show image

# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(X[10])


# # select model

# In[ ]:


num_classes = 10

Y = to_categorical(Y, num_classes=num_classes)


# In[ ]:


def summarize_diagnostics(history):
    plt.subplot(221)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.subplot(222)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    pass

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[ ]:


datagen = ImageDataGenerator(rotation_range=5,
                             zoom_range = 0.01,
                             width_shift_range=0.1, 
                             height_shift_range=0.1
                            )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')


# In[ ]:


steps = int(X.shape[0] // 64)
history = model.fit(datagen.flow(X, Y, batch_size=32),
                    steps_per_epoch=steps, epochs=40, batch_size=64, verbose=0)


# In[ ]:


(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(type(train_X), train_X.shape, test_X.shape)


# In[ ]:


X = np.concatenate((train_X, test_X), axis=0)
Y = np.concatenate((train_y, test_y), axis=0)
print(X.shape, Y.shape)


# In[ ]:


X = X.reshape(-1, 28, 28, 1)
X.shape


# In[ ]:


X = X / 255.0
Y = to_categorical(Y, num_classes=num_classes)
Y.shape


# In[ ]:


history = model.fit(x=X, y=Y, batch_size=64, epochs=100)


# In[ ]:


test = pd.read_csv(os.path.join(root, 'test.csv'))
test = test / 255.0
test = np.array(test).reshape(-1,28,28,1)


# In[ ]:


y_pred = model.predict(test)
y_pred = np.argmax(y_pred, axis=1)


# In[ ]:


y_pred


# In[ ]:


submission_df = pd.DataFrame()
image_id = [i for i in range(1, 28001)]
submission_df['ImageId'] = image_id
submission_df['Label'] = y_pred
submission_df


# # save submission as csv file

# In[ ]:


submission_df.to_csv('submission.csv', index = False)


# In[ ]:




