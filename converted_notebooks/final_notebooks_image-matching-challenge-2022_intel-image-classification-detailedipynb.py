#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic imports of tensorflow and keras 
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers


# Step 1 : Preparation of Data. As we know that we cannot pass the image data directly into the model without converting it into a readable array. We have image data with six classes and we need to put each image with a label. There are two ways in which we can prepare this data : 
# 
# 1. One way is that we can read the data using the opencv and then use it. 
# 2. Another way is using keras.preprocessing and directly accessing teh image_dataset_from_directory feature 
# 
# Before starting with these steps its important to check whether all the images are right or not. We should not feed corrupt images to our model for training. 

# In[ ]:


# Checking whether the images are corrupt or not. In this exercise we do not need this as the data is very clean 
# But in some cases we may encounter a situtaion in which data is not that clean and hence we need to ensure that
# we need to check the image data for whether its corrupt or not and delete the data. 

# We may not repeat this exercise in testing as it will help us to determine how the model performs if there are 
# some corrupt images feeded to the model 

import os

num_skipped = 0
for folder_name in ("buildings", "forest","glacier","mountain","sea","street"):
    folder_path = os.path.join("../input/intel-image-classification/seg_train/seg_train", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)


# In[ ]:


# Reading the data through open CV and using lables against it and then putting the entire thing in tensors 
# (image,labels)

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)

print(class_names_label)

def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """
    
    datasets = ['../input/intel-image-classification/seg_train/seg_train', '../input/intel-image-classification/seg_test/seg_test']
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            # Progress bar appears due to tqdm 
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))


# In[ ]:


# This method works best when you just have classified images but no train test split, One more way in which we can read the images as per fchollet notebook is this 
# as illustrated below. This works best when you have validation and train datasets togather and you need to make a test-train split. In this case we do not need validation
# split and hence we can keep the validation split as none

# Visit the documentation at https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

# This is an easy way of doing the same thing when dealing with an image classification problem. 

image_size = (150,150)
batch_size = 32 
train_ds = tf.keras.preprocessing.image_dataset_from_directory("../input/intel-image-classification/seg_train/seg_train",
                                                               labels='inferred',
                                                                label_mode='int',
                                                                class_names=None,
                                                                color_mode='rgb',
                                                                batch_size=batch_size,
                                                                image_size=image_size,
                                                                shuffle=True,
                                                                seed=None,
                                                                validation_split=None,
                                                                subset=None,
                                                                interpolation='bilinear',
                                                                follow_links=False,
                                                                crop_to_aspect_ratio=False
                                                              )
val_ds = tf.keras.preprocessing.image_dataset_from_directory("../input/intel-image-classification/seg_test/seg_test",
                                                            labels='inferred',
                                                                label_mode='int',
                                                                class_names=None,
                                                                color_mode='rgb',
                                                                batch_size=batch_size,
                                                                image_size=image_size,
                                                                shuffle=True,
                                                                seed=None,
                                                                validation_split=None,
                                                                subset=None,
                                                                interpolation='bilinear',
                                                                follow_links=False,
                                                                crop_to_aspect_ratio=False)


# After we have put the data in train and test splits and in a ready move tensors. We need to rescale our images. As we know that an image may have pixels ranging from 0-255 which makes it very difficult for our model to train as there are 255 different varieties, it has to deal with. Its very important to scale the images between 0-1 and hence we divide the pixel array of image with 255. 

# In[ ]:


# In case of normal reading we can perform like this 
train_images = train_images / 255.0 
test_images = test_images / 255.0


# Lets visualize the images now and the tensors that has been converted using labels and array. 

# In[ ]:


for images, labels in train_ds.take(1):
    print(images,labels)


# Now lets visualise our dataset after labelling. I have shown three ways of visualisation just to make you feel the difference between the various ways to visualise the image. 
# 
# 1. simple image array with numpy 
# 2. simple image array 
# 3. image array with numpy and uint8

# In[ ]:


# Visualising the data 
import matplotlib.pyplot as plt
import numpy

plt.figure(figsize = (10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy())
        plt.title(int(labels[i]))
        plt.axis("off")


# In[ ]:


# Visualising the data 
import matplotlib.pyplot as plt
import numpy

plt.figure(figsize = (10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(int(labels[i]))
        plt.axis("off")


# In[ ]:


# Visualising the data 
import matplotlib.pyplot as plt
import numpy

plt.figure(figsize = (10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        print(images[i].shape)
        plt.title(int(labels[i]))
        plt.axis("off")


# Now activities such as resizing, normalisation of pixels in 0-1 and also creating augmented data, in case data for training is less can be done using data augmentor or data augmentation techniques as well. In this case below i have made data augmentation a part of model training process where I have added sequential layers in Keras. This will augment the images while training itself. 

# In[ ]:


# Rescaling the RGB channel of [0-255] between [0-1] as [0-255] is nit suited for the neural network 
# We will create a data augmentation pre-processor for this purpose which can be used in augmenting the data as well 
# as with operations like rescaling and others 

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4)
])


# In[ ]:


# Lets visualise the augmented samples 
plt.figure(figsize = (10,10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


# In[ ]:


# Standardize the Data, Simultaneous augmetation and resizing of the images is a good phenomenon if you are training 
# your model on GPU. It will perform this operation simulteneously with the model training 
# Data Augmentation is inactive during test time and hence the samples will only be augmented during fit()
# not during evaluate() or predict()

input_shape = (150,150,3)
inputs = keras.Input(shape = input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)


# In[ ]:


# When you are training on CPU. This will happen asynchronously and will be buffered before going to the model
#augmented_train_ds = train_ds.map(lambda x,y : (data_augmentation(x,training = True),y))


# In[ ]:


# Configuring the Dataset for performance 
# buffered prefetching of the data so that we can yield data from disk 
# without I/O becoming blocking 

train_ds = train_ds.prefetch(buffer_size = 32)
val_ds = val_ds.prefetch(buffer_size = 32)


# Now we are ready to prepare a model with basics of neural netork. Lets start with a very basic model where we have two convolutional layers, 2 max pooling, two batch normalisation, one flatten and one dense. We can make this model as complicated as we want using the basic concepts on Artificial neural networks and CNN. 

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)


# We can fit the same model using train_ds and test_ds datasets as well. 

# In[ ]:


epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


# Predictions using the above models and metrics to observe in the above model. The prediction for all six classes came to be same and hence it is difficult to predict which class. Hence we should delve in some more complicated architectures and see how our model trains there. One can definitely increased epochs to increase the efficiency. 

# In[ ]:


img = keras.preprocessing.image.load_img(
    "../input/intel-image-classification/seg_pred/seg_pred/10004.jpg", target_size=IMAGE_SIZE
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(score)


# Now lets use Keras tuner to tune the hyperparameter in our model. KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search.One can follow the link : https://keras.io/keras_tuner/ for understanding of Keras Tuner. Hyper Parameter tuning can be of several types : 
# 
# 1. Randomised Search CV : It is faster than the grid search CV. Grid Saerch works well when we have a small number of hyperparameters and each hyper parameter has the same impact on the validation score of the model. Randomised search is batter when the magnitude of influence are inbalanced, which is more likely to happen as your number of parameter is growing. Randomised search will be randomly picking up some areas. It will provide borader ideas on Hyperparameter. It will narrow down our results and then one can apply Grid Search over this. 
# 
# 2. Grid Search CV :  It goes and check every possible combination fo parameters and every parameter list we provide. It may not necessarily be in one order. This can be cumbersome.  
# 
# 3. Byesian Optimisation : It builds a probability function for the objective function and use it to select most promising hyperparameter to evaluate true objective function. It in contrast to grid search keeps a track of past evaluation results which they use to form a probablistic model mapping. Read this amazing article on Towards Data Science : https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
# 
# 
# In keras.tuner we can simply employ these methods for hyper parameter optimisation. 
# 

# In[ ]:


# Installing Keras Tuner 
get_ipython().system('pip install keras-tuner')


# In[ ]:


# Lets define our model with some choices now so that we can tune the model using keras tuner and find the required hyper parameter 
# We have incorporated activation function as a Hyperparameter and dense units as Hyper parameter.
import keras_tuner as kt
def build_model_keras_tuner(hp):
    
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(hp.Choice("units",[32,16,8]), (3, 3), activation = hp.Choice("activation", ["relu", "tanh"]), input_shape = (150, 150, 3)), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = hp.Choice("activation", ["relu", "tanh"])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hp.Choice("units",[128,64,32,16,8]), activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
    ])
    
    #Do not forget to copile the model 
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    return model 


# In[ ]:


# Checking whether the model builds or not 
build_model_keras_tuner(kt.HyperParameters())


# In[ ]:


#Applying Random Search for tuning the hyperparameters 
tuner = kt.RandomSearch(
    hypermodel=build_model_keras_tuner,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


tuner.search(train_ds, epochs=3,validation_data=val_ds)


# In[ ]:


# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]


# In[ ]:


best_model.summary()


# Lets implement a more complicated model over this data and look at how this work. This has been picked up from fchollet notebook. https://keras.io/examples/vision/image_classification_from_scratch/

# Go checkout separable Conv2D here : https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

# In[ ]:


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


# Use sparse categorical entropy because binary and categorical crossentropy is more useful in case of when we 
# have one hot encoding

epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


# In[ ]:


#cancelled the run owing to many images

import pandas as pd
pre_directory_path = "../input/intel-image-classification/seg_pred/seg_pred"
df = pd.DataFrame(columns = ["Image_Name","Class","Score"])

for image in os.listdir(pre_directory_path):
    img = keras.preprocessing.image.load_img(
    os.path.join(pre_directory_path,image), target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    pred_dict = {"Image_Name": image, "Class": np.argmax(score),"Score":np.max(score) }
    df = df.append(pred_dict, ignore_index = True)
    break


