#!/usr/bin/env python
# coding: utf-8

#  <h1><center><font size="6">Fashion MNIST using CNN</font></center></h1>
# 
# 
# <center><img src="https://research.zalando.com/project/fashion_mnist/fashion_mnist/img/fashion-mnist-sprite.png" width="600"></img></center>

# In[ ]:


# import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


# import files
data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"
data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"

# read files csv
df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

# data as array in order to reshape them
train_data = np.array(df_train, dtype='float32')
test_data = np.array(df_test, dtype='float32')


# In[ ]:


# Normalize pixel values
X_train = train_data[:,1:]/255.0
y_train = train_data[:,0]
X_test = test_data[:,1:]/255.0
y_test = test_data[:,0]

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


# In[ ]:


# reshape
X_train_reshaped = X_train.reshape(X_train.shape[0],*(28,28,1))
X_test_reshaped = X_test.reshape(X_test.shape[0],*(28,28,1))


# In[ ]:


print('x_test shape: {}'.format(X_train_reshaped.shape))


# In[ ]:


plt.figure(figsize = (10,10))

for i in range(30):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_reshaped[i])
    plt.xlabel(y_train[i])
plt.show()


# In[ ]:


# adding the Input layer, Convolutional layer, Pooling layer, an second Convolutional Layer,
# Pooling Layer and 2 Dense layer. Must reshaped the input as (28,28,1) because the model didn't 
# accept the input_shape = (28,28).

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))


# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


# In[ ]:


model.summary() 


# In[ ]:


# Compile and train the model
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
history = model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test))


# In[ ]:


plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')


# In[ ]:


test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose = 2) 
print("Loss: ",test_loss)
print("Accuracy: ",test_acc)


# In[ ]:




