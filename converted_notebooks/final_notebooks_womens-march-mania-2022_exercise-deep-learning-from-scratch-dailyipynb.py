#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You've seen how to build a model from scratch to identify handwritten digits.  You'll now build a model to identify different types of clothing.  To make models that train quickly, we'll work with very small (low-resolution) images. 
# 
# As an example, your model will take an images like this and identify it as a shoe:
# 
# ![Imgur](https://i.imgur.com/GyXOnSB.png)

# # Data Preparation
# This code is supplied, and you don't need to change it. Just run the cell below.

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_7 import *
print("Setup Complete")


# # 1) Start the model
# Create a `Sequential` model called `fashion_model`. Don't add layers yet.

# In[ ]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Your Code Here
____
q_1.check()


# In[ ]:


#q_1.solution()


# # 2) Add the first layer
# 
# Add the first `Conv2D` layer to `fashion_model`. It should have 12 filters, a kernel_size of 3 and the `relu` activation function. The first layer always requires that you specify the `input_shape`.  We have saved the number of rows and columns to the variables `img_rows` and `img_cols` respectively, so the input shape in this case is `(img_rows, img_cols, 1)`.

# In[ ]:


# Your code here
____
q_2.check()


# In[ ]:


# q_2.hint()
#q_2.solution()


# # 3) Add the remaining layers
# 
# 1. Add 2 more convolutional (`Conv2D layers`) with 20 filters each, 'relu' activation, and a kernel size of 3. Follow that with a `Flatten` layer, and then a `Dense` layer with 100 neurons. 
# 2. Add your prediction layer to `fashion_model`.  This is a `Dense` layer.  We alrady have a variable called `num_classes`.  Use this variable when specifying the number of nodes in this layer. The activation should be `softmax` (or you will have problems later).

# In[ ]:


# Your code here
____

q_3.check()


# In[ ]:


# q_3.solution()


# # 4) Compile Your Model
# Compile fashion_model with the `compile` method.  Specify the following arguments:
# 1. `loss = "categorical_crossentropy"`
# 2. `optimizer = 'adam'`
# 3. `metrics = ['accuracy']`

# In[ ]:


# Your code to compile the model in this cell
____
q_4.check()


# In[ ]:


# q_4.solution()


# # 5) Fit The Model
# Run the command `fashion_model.fit`. The arguments you will use are
# 1. The data used to fit the model. First comes the data holding the images, and second is the data with the class labels to be predicted. Look at the first code cell (which was supplied to you) where we called `prep_data` to find the variable names for these.
# 2. `batch_size = 100`
# 3. `epochs = 4`
# 4. `validation_split = 0.2`
# 
# When you run this command, you can watch your model start improving.  You will see validation accuracies after each epoch.

# In[ ]:


# Your code to fit the model here
____
q_5.check()


# In[ ]:


#q_5.solution()


# # 6) Create A New Model
# 
# Create a new model called `second_fashion_model` in the cell below.  Make some changes so it is different than `fashion_model` that you've trained above. The change could be using a different number of layers, different number of convolutions in the layers, etc.
# 
# Define the model, compile it and fit it in the cell below.  See how it's validation score compares to that of the original model.

# In[ ]:


# Your code below
____

q_6.check()


# In[ ]:


#q_6.solution()


# # Keep Going
# You are ready to learn about **[strides and dropout](https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models)**, which become important as you start using bigger and more powerful models.
# 
