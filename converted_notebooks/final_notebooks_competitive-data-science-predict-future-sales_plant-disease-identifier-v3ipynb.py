#!/usr/bin/env python
# coding: utf-8

# ## Libraries & Modules

# In[ ]:


import os
import glob
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import tensorflow.keras as keras
import tensorflow as tf


# In[ ]:


print(keras.__version__)
print(tf.__version__)


# In[ ]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Data

# In[ ]:


train_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/"
test_dir = "../input/new-plant-diseases-dataset/test/test/"
valid_dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"


# ## Preprocessing

# In[ ]:


def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count


# In[ ]:


train_samples = get_files(train_dir)
num_classes_train = len(glob.glob(train_dir+"/*"))

# test_samples=get_files(test_dir)
# num_classes_test = len(glob.glob(test_dir+"/*"))

valid_samples=get_files(valid_dir)
num_classes_valid = len(glob.glob(valid_dir+"/*"))

print(num_classes_train,"Classes")
print(train_samples,"Train images")

# print(num_classes_test, "Classes")
# print(test_samples,"Test images")

print(num_classes_valid, "Classes")
print(valid_samples,"Valid images")


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


img_width,img_height =256,256

input_shape=(img_width,img_height,3)

batch_size =64

train_generator =train_datagen.flow_from_directory(train_dir, target_size=(img_width,img_height), batch_size=batch_size)

valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(img_width, img_height), batch_size=batch_size)

# test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True,target_size=(img_width,img_height), batch_size=batch_size)


# In[ ]:


train_generator.class_indices


# In[ ]:


valid_generator.class_indices


# In[ ]:


def printData(generator):
    print("Samples:",generator.samples)
    print("No of classes:",generator.num_classes)
    print("Batch size:", generator.batch_size)
    print("Data format:", generator.dtype)
    print("Color mode:",generator.color_mode)
    print("Image shape:", generator.image_shape)
    print("Allowed class modes:", generator.allowed_class_modes)
    print("Class Mode:", generator.class_mode)
    


# In[ ]:


printData(train_generator)


# In[ ]:


printData(valid_generator)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization


# ## First Model

# In[ ]:


model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128,activation='relu'))  

model.add(Dense(num_classes_train,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model_layers = [ layer.name for layer in model.layers]
print('layer name : ',model_layers)


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model_layers = [ layer.name for layer in model.layers]
print('layer name : ',model_layers)


# In[ ]:


from keras.preprocessing import image
import numpy as np

img_path = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/036662aa-5a39-4e1a-adc7-32017654d0ef___YLCV_GCREC 2463.JPG"

img1 = image.load_img(img_path)
plt.imshow(img1);
#preprocess image
img1 = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


# In[ ]:


from keras.models import Model

conv2d_0_output=Model(inputs=model.input,outputs=model.get_layer('conv2d').output)

max_pooling2d_0_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d').output)

conv2d_1_output = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)

max_pooling2d_1_output = Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_1').output)

conv2d_2_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_2').output)

max_pooling2d_2_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_2').output)

flatten_1_output=Model(inputs=model.input,outputs=model.get_layer('flatten').output)

conv2d_0_features = conv2d_0_output.predict(img)

max_pooling2d_0_features = max_pooling2d_0_output.predict(img)

conv2d_1_features = conv2d_1_output.predict(img)

max_pooling2d_1_features = max_pooling2d_1_output.predict(img)

conv2d_2_features = conv2d_2_output.predict(img)

max_pooling2d_2_features = max_pooling2d_2_output.predict(img)

flatten_1_features = flatten_1_output.predict(img)


# In[ ]:


import matplotlib.image as mpimg

# function to plot images after each layer

def plot_images(img_width, img_height, rows, columns, layer):
    fig=plt.figure(figsize=(img_width,img_height))
    columns = columns
    rows = rows
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title('filter'+str(i))
        plt.imshow(layer[0, :, :, i], cmap='viridis') # Visualizing in color mode.
    plt.show()


# In[ ]:


from keras.preprocessing import image
import numpy as np

img_path = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/036662aa-5a39-4e1a-adc7-32017654d0ef___YLCV_GCREC 2463.JPG"

img1 = image.load_img(img_path)
plt.imshow(img1);
#preprocess image
img1 = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)


# ### Plotting images after every layer for first model

# In[ ]:


## conv2d_0_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = conv2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_0_features

img_width = 14
img_height = 7

columns = 8
rows = 2

layer = max_pooling2d_0_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## conv2d_1_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = conv2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_1_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = max_pooling2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## conv2d_2_features

img_width = 16
img_height = 16

columns = 8
rows = 8

layer = conv2d_2_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_2_features

img_width = 14
img_height = 14

columns = 8
rows = 8

layer = max_pooling2d_2_features

plot_images(img_width, img_height, rows, columns, layer)


# ## Second Model

# In[ ]:


## Second model

model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.125))

model.add(Dense(128,activation='relu'))  

model.add(Dense(num_classes_train,activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from keras.models import Model

conv2d_0_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_3').output)

max_pooling2d_0_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_3').output)

conv2d_1_output = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)

max_pooling2d_1_output = Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_4').output)

conv2d_2_output=Model(inputs=model.input,outputs=model.get_layer('conv2d_5').output)

max_pooling2d_2_output=Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_5').output)

flatten_1_output=Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)

conv2d_0_features = conv2d_0_output.predict(img)

max_pooling2d_0_features = max_pooling2d_0_output.predict(img)

conv2d_1_features = conv2d_1_output.predict(img)

max_pooling2d_1_features = max_pooling2d_1_output.predict(img)

conv2d_2_features = conv2d_2_output.predict(img)

max_pooling2d_2_features = max_pooling2d_2_output.predict(img)

flatten_1_features = flatten_1_output.predict(img)


# ### Plotting images after every layer for second model/

# In[ ]:


## conv2d_0_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = conv2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_0_features

img_width = 14
img_height = 7

columns = 8
rows = 2

layer = max_pooling2d_0_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## conv2d_1_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = conv2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_1_features

img_width = 14
img_height = 7

columns = 8
rows = 4

layer = max_pooling2d_1_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## conv2d_2_features

img_width = 16
img_height = 16

columns = 8
rows = 8

layer = conv2d_2_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:


## max_pooling2d_2_features

img_width = 14
img_height = 14

columns = 8
rows = 8

layer = max_pooling2d_2_features

plot_images(img_width, img_height, rows, columns, layer)


# In[ ]:




