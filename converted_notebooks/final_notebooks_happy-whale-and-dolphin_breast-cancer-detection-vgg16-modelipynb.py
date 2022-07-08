#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, optimizers, losses, metrics, initializers
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV3Small, MobileNet, InceptionV3
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# **Gloabal Variables**

# In[ ]:


dir_path = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'
IMAGE_SHAPE = (224, 224)


# ## Build Functions
# Create methods to make the code more organized

# In[ ]:


# create prepare_image method
# used to preprocess the image for efficientNet model
def prepare_image(file):
    img = image.load_img(file, target_size=IMAGE_SHAPE)
    img_array = image.img_to_array(img)
    return tf.keras.applications.efficientnet.preprocess_input (img_array)


# ### Read the files from each dirctory
# Read all file from the three directories 'benign', 'malignant' and 'normal'

# In[ ]:


directories = os.listdir(dir_path) # read the folders

files = [] # save all images for each folder
labels = [] # set for each image the name of it

# read files for each directory
for folder in directories:
    
    fileList = glob.glob(dir_path + '/'+ folder + '/*')
    labels.extend([folder for l in fileList])
    files.extend(fileList)
    
len(files), len(labels)


# As we see, we have 1578 images in all dataset, BUT there are mask images which is not suitable for training with original images, so we will remove all the mask images from the dataset
# 
# ### Remove any mask image from files

# In[ ]:


# create two lists to hold only non-mask images and label for each one
selected_files = []
selected_labels = []

for file, label in zip(files, labels):
    if 'mask' not in file:
        selected_files.append(file)
        selected_labels.append(label)

    
len(selected_files), len(selected_labels)


# As we see, after removing the mask images the rest of dataset is 780 images which is not enough at all for training model from scratch, so we'll use **VGG16** model for Transfer Learning
# ### Prepare the images
# Prepare the images to be suitable as input for efficientnet model

# In[ ]:


# the dictionary holds list of images and for each one has its target/label
images = {
    'image': [], 
    'target': []
}

print('Preparing the image...')

for i, (file, label) in enumerate(zip(selected_files, selected_labels)):
    images['image'].append(prepare_image(file))
    images['target'].append(label)

print('Finished.')


# ### Prepare the target for splitting
# * Convert the images to numpy array for better computation
# * Encode the label to convert categorical names to numbers

# In[ ]:


# convert lists to arrays 
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])

# encode the target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

images['target'] = le.fit_transform(images['target'])

classes = le.classes_ # get the classes for each target
print(f'the target classes are: {classes}')


# **Split the data to train and test**

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images['image'], images['target'], test_size=.10)

x_train.shape, x_test.shape, y_train.shape, y_test.shape 


# ### Build the Model
# * Create VGG16 Model
# * Don't include_top layers to take only the features of the model
# * Freeze all layer weights of the model
# * Append my own layers for Transfer Learning
# * Compile my own model after Transfer Learning

# In[ ]:


from keras.applications.vgg16 import VGG16
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMAGE_SHAPE, 3),
    classes=3)

# Freeze the base_model
base_model.trainable = False

# append my own layers on the top of the model for Transfer Learning
x = base_model.output

# 1st conv block
x = layers.Conv2D(256, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D(keepdims = True)(x)

# 2nd conv block
x = layers.Conv2D(128, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D(keepdims = True)(x)

# 1st FC layer
x = layers.Flatten()(x) 
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# 2nd FC layer
x = layers.Dense(32, activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(.2)(x)

x = layers.Dense(3, 'softmax')(x)

incept_model = keras.models.Model(inputs = base_model.input, outputs = x)

# compile the model
incept_model.compile(optimizer=optimizers.RMSprop(.001), loss = losses.sparse_categorical_crossentropy, metrics= [metrics.SparseCategoricalAccuracy()])

# incept_model.summary()


# ### Train the model

# In[ ]:


earlyStop = keras.callbacks.EarlyStopping(patience=60) 
best_model = keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True) 

with tf.device('/gpu:0'):
    history = incept_model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=[earlyStop, best_model]) 


# In[ ]:


hist = history.history

plt.plot(hist['loss'], label=  'loss')
plt.plot(hist['val_loss'], label = 'val_loss')
plt.plot(hist['sparse_categorical_accuracy'], label='accuracy')
plt.plot(hist['val_sparse_categorical_accuracy'], label='val_accuracy')
plt.legend()


# ### Evaluate the model

# In[ ]:


incept_model.evaluate(x=x_test, y = y_test, batch_size=32, verbose=1)


# ### Make the last 100 layers as trainable

# In[ ]:


# open train the last 100 layers
for layer in incept_model.layers[720:]:
    layer.trainable = True
    
# compile the model with new optimizer and lr=.0001
incept_model.compile(optimizer=optimizers.RMSprop(.0001), loss = losses.sparse_categorical_crossentropy, metrics=[metrics.SparseCategoricalAccuracy()])

# incept_model.summary()


# ### Train the model again

# In[ ]:


earlyStop = keras.callbacks.EarlyStopping(patience=60) 
best_model = keras.callbacks.ModelCheckpoint(filepath='best_model_2.h5', save_best_only=True) 

# load the best weights
# incept_model.set_weights(best_weights)

with tf.device('/gpu:0'):
    history = incept_model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=[earlyStop, best_model]) 


# ### Evaluate the model

# In[ ]:


incept_model.evaluate(x=x_test, y = y_test, batch_size=32, verbose=1)


# There are some imporovement after fine-tuning the last 100 layers, so for more epochs and better learning rate maybe the accuracy gets better

# ### Predict the model

# In[ ]:


# used to predict the model and visualize the orignal image with title of true and pred values
def predict_image(img_path, label):
    img1 = prepare_image(img_path) # preprocess the image
    res = incept_model.predict(np.expand_dims(img1, axis = 0)) # predict the image
    pred = classes[np.argmax(res)]

    # Visualize the image
    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred}')


# In[ ]:


predict_image(dir_path + 'benign/benign (10).png', 'benign')


# In[ ]:


predict_image(dir_path + 'benign/benign (85).png', 'benign')


# In[ ]:


predict_image(dir_path + 'malignant/malignant (10).png', 'malignant')


# In[ ]:


predict_image(dir_path + 'normal/normal (10).png', 'normal')


# In[ ]:


incept_model.evaluate(np.array(x_test),np.array(y_test))


# In[ ]:


predicted = []
for item in incept_model.predict(x_test):
    predicted.append(np.argmax(item))


# In[ ]:


x_test.shape


# In[ ]:


for item in predicted:
    print(item,)


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
conf = confusion_matrix(y_test,predicted)
conf


# In[ ]:


#Heatmap
info = [
    'benign'   ,  # 0
    'normal'   ,  # 1
    'malignant',  # 2
]
plt.figure(figsize = (10,10))
ax = sns.heatmap(conf, cmap=plt.cm.Greens, annot=True, square=True, xticklabels = info, yticklabels = info)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)


# In[ ]:


import seaborn as sns
sns.heatmap(conf)


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(incept_model, to_file='aug_model_plot.png', show_shapes=True, show_layer_names=True)


# **Thanks for ur patience**, hope u **upvote** if your got anything from the notebook
# 
# if you have any suggestion to imporve it for others later, pls comment
