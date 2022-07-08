#!/usr/bin/env python
# coding: utf-8

# ## Xception
# ## Xception is a deep convolutional neural network architecture that involves Depthwise Separable Convolutions. This network was introduced Francois Chollet who works at Google, Inc.
# 
# https://iq.opengenus.org/xception-model/#:~:text=Xception%20is%20a%20deep%20convolutional,version%20of%20an%20Inception%20module.
# 
# * Xception is also known as “extreme” version of an Inception module.
# ![image.png](attachment:12dd3e4d-7c50-4f67-b9ef-c8bb2d585fe5.png) | ![image.png](attachment:8c4f2d90-6d54-4ab5-be21-7b73bc812d98.png)

# In[ ]:


# import the necessary packages
get_ipython().system('pip install imutils')
get_ipython().system('pip install wget')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dropout,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import wget
import cv2
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


# ## Downloading Dataset

# In[ ]:


_URL = 'http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz'
wget.download(_URL)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


zip_dir = tf.keras.utils.get_file('./logo', origin=_URL, untar=True,extract=True)


# In[ ]:


import tarfile

fname = './flickr_logos_27_dataset.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()


# In[ ]:


fname = './flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()


# In[ ]:


src_dir = "flickr_logos_27_dataset_images"
dest = "LOGOS"

if not os.path.exists(dest):
    os.makedirs(dest)


# ## Preprocessing

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)


# In[ ]:


df


# In[ ]:


X = df.iloc[:,0]
Y = df.iloc[:,1]


# In[ ]:


dtdir = './flickr_logos_27_dataset_images/'


# In[ ]:


im = df[0][0]


# In[ ]:


size = df.iloc[:,3:]


# In[ ]:


size


# In[ ]:


img = os.path.join(dtdir,im)


# In[ ]:


size = size.values.tolist()


# In[ ]:


size[0][0],size[0][1],size[0][2],size[0][3]


# In[ ]:


image = cv2.imread(img)
plt.imshow(image)
image.shape


# In[ ]:


image = cv2.imread(img)
image = image[size[0][1]:size[0][3],size[0][0]:size[0][2]]
plt.imshow(image)
image.shape


# In[ ]:


query = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt", sep='\s+',header=None)


# In[ ]:


query


# In[ ]:


img = os.path.join(dtdir,query[0][5])
image = cv2.imread(img)
plt.imshow(image)
image.shape


# In[ ]:


testdist = 'TEST'


# In[ ]:


if not os.path.exists(testdist):
    os.makedirs(testdist)


# In[ ]:


y = list(set(list(Y)))
y.sort()


# In[ ]:


for i in y:
    os.makedirs(os.path.join(testdist,i))


# In[ ]:


for i in y:
    os.makedirs(os.path.join(dest,i))


# In[ ]:


distractor = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_distractor_set_urls.txt", sep='\s+',header=None)


# In[ ]:


distractor


# In[ ]:


HEIGHT = 224
WIDTH =  224


# ## Removing Corrupt Images 

# In[ ]:


for i in range(len(X)):
    try:
        destrain = os.path.join(dest,Y[i])
        savepath = os.path.join(destrain,X[i])
        img  = os.path.join(dtdir,X[i])
        image = cv2.imread(img)
        image = image[size[i][1]:size[i][3],size[i][0]:size[i][2]]
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass


# In[ ]:


A = query.iloc[:,0]
B = query.iloc[:,1]


# In[ ]:


A


# In[ ]:


for i in range(len(A)):
    try:
        destrain = os.path.join(testdist,B[i])
        savepath = os.path.join(destrain,A[i])
        img  = os.path.join(dtdir,A[i])
        image = cv2.imread(img)
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass


# In[ ]:


imagePaths = list(paths.list_images(testdist))


# In[ ]:


img = imagePaths[40]
print(img)
image = cv2.imread(img)
plt.imshow(image)
image.shape


# ## Image Augmentation

# In[ ]:


train = ImageDataGenerator(
rescale = 1/255,
horizontal_flip=True,
vertical_flip=True,
shear_range=0.2,
zoom_range=0.2,
featurewise_center=True, # Set input mean to 0 over the dataset, feature-wise
featurewise_std_normalization=True, # Divide inputs by std of the dataset, feature-wise
rotation_range=40, # Degree range for random rotations
width_shift_range=0.2,
height_shift_range=0.2,
fill_mode='nearest',
validation_split = 0.2)


# In[ ]:


imagePaths = list(paths.list_images(dest))


# In[ ]:


os.makedirs('preview')


# In[ ]:


img = load_img(imagePaths[50])  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in train.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='yh', save_format='jpeg'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely


# ## Samples Visualization

# In[ ]:


image = cv2.imread(imagePaths[50])
plt.imshow(image)
image.shape


# In[ ]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

images = []
for img_path in glob.glob('preview/*.jpeg'):
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(int(len(images) / columns + 1), columns, i + 1)
    plt.imshow(image)


# In[ ]:


HEIGHT = 224
WIDTH = 224
INIT_LR = 1e-4
EPOCHS =  100
BS = 256


# ## Splitting into Train and Validation Set

# In[ ]:


trainset = train.flow_from_directory(dest,
target_size = (WIDTH,HEIGHT),
batch_size = BS,
shuffle=False,
seed=42,
color_mode='rgb',
subset = 'training',
class_mode='categorical')


# In[ ]:


validset = train.flow_from_directory(dest,
target_size = (WIDTH,HEIGHT),
batch_size = BS,
shuffle=False,
seed=42,
color_mode='rgb',
subset = 'validation',
class_mode='categorical')


# In[ ]:


imagePaths = list(paths.list_images(dest))


# In[ ]:


#trainset.filenames


# In[ ]:


trainset.class_indices


# ## Model Architecture

# In[ ]:


# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = Xception(weights="imagenet", include_top=False,input_tensor=Input(shape=(WIDTH, HEIGHT, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten(name="flatten")(headModel)

headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

headModel = Dense(trainset.num_classes, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
#sgd = SGD(lr=INIT_LR,momentum=0.9,nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer='Adam',metrics=["accuracy"])
# train the head of the network


# In[ ]:


totalTrain = len(list(paths.list_images(dest)))
totalTrain


# ## Train Results

# In[ ]:


H = model.fit_generator(
	trainset,
  validation_data = validset,
	epochs=EPOCHS)


# ## Results Plot

# In[ ]:


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss VS Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig('graph.png')


# In[ ]:


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title(" Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()


# ## F1-score Precision Recall

# In[ ]:


# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating after fine-tuning network...")
validset.reset()
predIdxs = model.predict(x=validset)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(validset.classes, predIdxs,
	target_names=validset.class_indices.keys()))
# serialize the model to disk
print("[INFO] serializing network...")
model.save('logo.model', save_format="h5")


# ## Some Prdictions from Test Set

# In[ ]:


testimage = list(paths.list_images('./flickr_logos_27_dataset_images'))


# In[ ]:


model.get_config


# In[ ]:


from PIL import Image
def predimage(path):
    image = Image.open(path)
    plt.imshow(image)
    test = load_img(path,target_size=(WIDTH,HEIGHT))
    test = img_to_array(test)
    test = np.expand_dims(test,axis=0)
    test /= 255 
    result = model.predict(test,batch_size = BS)
    y_class = result.argmax(axis=-1)
    result = (result*100)
    result = list(np.around(np.array(result),1))
    print(result)
    print(y[y_class[0]])


# In[ ]:


predimage(testimage[56])


# In[ ]:


predimage(imagePaths[2])


# In[ ]:


predimage(imagePaths[60])


# In[ ]:


predimage(testimage[30])


# In[ ]:


predimage(testimage[18])


# In[ ]:


predimage(testimage[25])

