#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# # Introduction
# In this notebook we will disccus
# * How to load images and do data augmentation with Keras
# * How to use pretrained models and fine-tune them

# # Training Data

# In[ ]:


base_dir = '../input/state-farm-distracted-driver-detection'
train_dir = os.path.join(base_dir, "imgs/train")
test_dir = os.path.join(base_dir, "imgs")
data = pd.read_csv(os.path.join(base_dir, 'driver_imgs_list.csv'))
data.head()


# In[ ]:


data_samples = data.sample(12)

fig, axs = plt.subplots(3, 4, figsize=(17, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(plt.imread(os.path.join(train_dir, data_samples['classname'].iloc[i], data_samples['img'].iloc[i])))
    ax.axis(False)
    ax.set_title(data_samples['classname'].iloc[i])


# In[ ]:


args ={
    "kind":"bar",
    "title":"Classes Count",
    "figsize": (7,5)
}

classes_count = data['classname'].value_counts() 
fig = classes_count.plot(**args)


# In[ ]:


print("Average images per class = {:.3f} with std = {:.3f}".format(classes_count.mean(),
                                                          classes_count.std()))


# In[ ]:


# constats
val_ratio = 0.2
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ## ImageDataGenerator
# Tensorflow through keras API provides a very easy way to load images and do real-time data augmentation check the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) for more details
# 

# In[ ]:


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split = val_ratio
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator()


# In[ ]:


train_data = train_gen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    seed = 42,
    subset = "training"
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    seed = 42,
    subset = "validation"
)


# # Modeling
# [keras Applications](https://keras.io/api/applications/) API provides a set of deep learning models that can be used for prediction, feature extraction, and fine-tuning.
# In this notebook we will use the EfficientNet model which is devolped by Mingxing Tan and Quoc V. Le of Google Research, Brain team. they proposed a new scaling method that uniformly scales all dimensions of depth, width and resolution of a CNN network using a simple compound coefficient.
# 
# They developed a new baseline network using neural architecture search (NAS). the main building block for the network is the __MBConv__ which was introduced in MobileNetV2 architecture  
# 
# <img src="https://amaarora.github.io/images/mbconv.png" style = "display: block;
#   margin-left: auto; margin-right: auto; width: 300;"/>
#   
#  <div align="center"><i>MBConv Layer</i></div>
#  <br/>
#   
# #### The baseline network is called EfficientNet-B0 and the network is scalled to B7
# 
# <img src="https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png" style = "display: block;
#   margin-left: auto; margin-right: auto; width: 300;"/>
#   
#  <div align="center"><i>The Baseline Architecture</i></div>
#  <br/>
#  
#  <img src="https://1.bp.blogspot.com/-oNSfIOzO8ko/XO3BtHnUx0I/AAAAAAAAEKk/rJ2tHovGkzsyZnCbwVad-Q3ZBnwQmCFsgCEwYBhgL/s1600/image3.png" width="500" style = "display: block;
#   margin-left: auto; margin-right: auto; width: 500;"/>
#   
#  <div align="center"><i>EfficientNet Performance</i></div>
#  <br/>
#   
# **Additional Resources**
# 
# MBConv Block: https://paperswithcode.com/method/inverted-residual-block
# 
# Great Article About EfficientNet: https://amaarora.github.io/2020/08/13/efficientnet.html
# 
# EfficientNet Orginal Paper: [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)
# 
# Google AI Blog Post: https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

# In[ ]:


def build_model(num_class):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(units=num_class, activation=tf.keras.activations.softmax)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
    loss= tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy','Recall'],
    )
    return model


# In[ ]:


model = build_model(10)
model.summary()


# In[ ]:


history = model.fit(train_data,epochs=3,validation_data=val_data)


# In[ ]:


model.save_weights('my_checkpoint')


# # Predictions

# In[ ]:


test_data = test_gen.flow_from_directory(
    test_dir,
    shuffle = False,
    target_size = IMG_SIZE,
    classes = ['test'],
    batch_size = 32
)


# In[ ]:


preds = model.predict(test_data)


# In[ ]:


test_imgs = os.path.join(base_dir, "imgs/test")

test_ids = sorted(os.listdir(test_imgs))
pred_df = pd.DataFrame(columns = ['img','c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
for i in range(len(preds)):
    pred_df.loc[i, 'img'] = test_ids[i]
    pred_df.loc[i, 'c0':'c9'] = preds[i]
    
pred_df.to_csv('predictions2.csv', index=False)

