#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, glob, random, matplotlib, cv2
import matplotlib.patches as mplpatches
from collections import Counter
import pandas as pd
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install xmltodict')


# In[ ]:


import xmltodict


# # Introduction <a class="anchor" id="introduction"></a>
# 
# With the face mask dataset provided I am going to define and implement a ViT architecture solution. First I am going to explore the data with EDA techinques, define the Transformer and Vision Transformer (ViT) architecture and then code it in python. Finally, I will be computing the metrics of the trained model. Let's begin!
# 
# <img src='https://storage.googleapis.com/kagglesdsdata/datasets/667889/1176415/images/maksssksksss116.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220521%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220521T202106Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=548b32d965de35a6ad797de828d6cb18e03727cd3701453ae5ea53bbe2ad6c74f5569e9de3bc975729cbe757d9d676b85585a2f69b8d298e7987920d170460adf430ef54eb474c642bbc90120fd9fea633fe00e7f828eaed65a2735c759cffb747cbe9a1fbf99edd6443617a68272c3dc816f7b8cbf41b2260c5425b119638e482ebac760c14eb69d7b4ce4454f6476d76bd662d315571ff0f4a7e8cca7e1dff76cfca3286f5921182227ecc40d03a97e8556b779efb943b413e435f9ebe91204286a3e6f262651fc3bd47b3f32bd17ff0bbe18341315d6e8a4244734390f798ea63470985537edc289c0de58fb91af80a987440a062e13ec400e05730d23f51' width="500" height="600" align="center" margin=50>
# 
# 
# ## Table of contents 
# 
# 1. [Introduction](#introduction)
# 2. [Load the data](#loaddata)
# 3. [Exploratory data analysis](#eda)
# 4. [Transformer](#transformer)<br>
#     4.1 [Transformer architecture](#transarch)<br>
#     4.2 [ViT architecture](#vitarch)<br>
#     4.3 [Implement ViT architecture](#impvitarch)<br>
# 5. [Prepare to train](#train)<br>
#     5.1 [Visualize ViT model](#visualize)<br>
#     5.2 [Fitting the model](#fitting)<br>
# 6. [Future work](#future)
#  

# # Load the data <a class="anchor" id="loaddata"></a>
# 
# We are going to get our data and have it splitted into train/test sets. Our train/test set will be distributed in 80/20 sets.

# In[ ]:


annotations_path = "/kaggle/input/face-mask-detection/annotations"
images_path = "/kaggle/input/face-mask-detection/images"

CHANNELS = 3
IMAGE_SIZE = 224
INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
TRAIN_SET_PRCNT = 0.85


# In[ ]:


def get_annotation(annotation_file):
    
    objects = {
            "xmin":[],
            "ymin":[],   
            "xmax":[],
            "ymax":[],
            "name":[],    
            "file":[],
            "width":[],
            "height":[],
           }

    tree = ET.parse(annotation_file)

    for el in tree.iter():
        if 'size' in el.tag:
            for attr in list(el):
                if 'width' in attr.tag: 
                    width = int(round(float(attr.text)))
                if 'height' in attr.tag:
                    height = int(round(float(attr.text)))    

        if 'object' in el.tag:
            for attr in list(el):

                if 'name' in attr.tag:
                    name = attr.text                 
                    objects['name']+=[name]
                    objects['width']+=[width]
                    objects['height']+=[height] 
                    objects['file']+=[annotation_file.split('/')[-1][0:-4]] 

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                            objects['xmin']+=[xmin]
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                            objects['ymin']+=[ymin]                                
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                            objects['xmax']+=[xmax]                                
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                            objects['ymax']+=[ymax]     
        
    return objects


# In[ ]:


# list of paths to images and annotations
image_files = [
    f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))
]
annot_files = [
    f for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))
]

image_files.sort()
annot_files.sort()

images, targets = [], []

# loop over the annotations and images, preprocess them and store in lists
for i in range(0, len(annot_files)):
    
    # Access bounding box coordinates
    annot = get_annotation(os.path.join(annotations_path, annot_files[i]))

    top_left_x, top_left_y = annot['xmax'], annot['ymax']
    bottom_right_x, bottom_right_y = annot['xmin'], annot['ymin']

    image = keras.utils.load_img(
        os.path.join(images_path, image_files[i]),
    )
    (w, h) = image.size[:2]

    # resize train set images
    #if i < int(len(annot_files) * TRAIN_SET_PRCNT):
        # resize image if it is for training dataset
    #    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # convert image to array and append to list
    #images.append(keras.utils.img_to_array(image))
    #targets.append((annot['xmax'], annot['ymax'],annot['xmin'], annot['ymin']))

    # apply relative scaling to bounding boxes as per given image and append to list
    for i in range(0, len(top_left_x)):
        # convert image to array and append to list
        images.append(keras.utils.img_to_array(image))
        targets.append(
            (
                float(top_left_x[i]) / w,
                float(top_left_y[i]) / h,
                float(bottom_right_x[i]) / w,
                float(bottom_right_y[i]) / h,
            )
        )
        

    
# Convert the list to numpy array, split to train and test dataset
(x_train), (y_train) = (
    np.asarray(images[: int(len(images) * TRAIN_SET_PRCNT)]),
    np.asarray(targets[: int(len(targets) * TRAIN_SET_PRCNT)]),
)

(x_test), (y_test) = (
    np.asarray(images[int(len(images) * TRAIN_SET_PRCNT) :]),
    np.asarray(targets[int(len(targets) * TRAIN_SET_PRCNT) :]),
)


# In[ ]:


input_image = x_train[0]
fig, ax1 = plt.subplots(1, figsize=(15, 15))
im = input_image
# Display the image
ax1.imshow(im.astype("uint8"))


input_image = cv2.resize(
    input_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
)
#input_image = np.expand_dims(input_image, axis=0)
#preds = vit_object_detector.predict(input_image)[0]
#
(h, w) = (im).shape[0:2]
#
for i in range(0,3):
    top_left_x, top_left_y = int(y_train[i][0] * 224), int(y_train[i][1] * 224)
    #
    bottom_right_x, bottom_right_y = int(y_train[i][2] * 224), int(y_train[i][3] * 224)
    #
    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    #
    ## Create the bounding box
    rect = mplpatches.Rectangle(
                            (top_left_x, top_left_y),
                            bottom_right_x - top_left_x,
                            bottom_right_y - top_left_y,
                            facecolor="none",
                            edgecolor="red",
                            linewidth=1,
                        )

    ax1.add_patch(rect)

plt.show()


# # Exploratory data analysis <a class="anchor" id="eda"></a>
# 
# <b>Exploratory data analysis</b> (EDA) is used to look at data before making any assumptions. It is important step in any Data Analysis or Data Science project as we can learn more about our target data, we can find outliers, relations among the variables, patterns and whatever it comes in your mind. In this case we only have three unique classes <i>'with_mask'</i>, <i>'mask_weared_incorrect'</i>, <i>'without_mask'</i>.
# 
# EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them.
# 
# We don't have much of a massive dataset with hundreds of columns with numerical and categorical variables so we won't be able to answer questions like what is the standard deviation in this column or check the categorical variables. We sure don't need that information as we are getting the bounding boxes coordinates (in 4 different columns), the name of the image file, the width and height of each image and the target variable. With this knowledge we will be able to see what is de distribution of our target classes and we will know if all the images have the same size. So let's dig in!

# In[ ]:


# Count how many files there are in annotation path. It should be the same number as in images path
count = 0
# Iterate directory
for path in os.listdir(annotations_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(annotations_path, path)):
        count += 1
print('File count:', count) 


# In[ ]:


# Let's find out how many people are wearing mask, not wearing it or wearing it wrong:
img_names = [] 
xml_names = [] 
for dirname, _, filenames in os.walk(images_path):
    for filename in filenames:
        img_names.append(filename)

img_annot_list = []
for i in img_names[:]:
    with open(os.path.join(annotations_path,i[:-4]+".xml")) as fd:
        doc = xmltodict.parse(fd.read())
    temp = doc["annotation"]["object"]
    if type(temp) == list:
        for i in range(len(temp)):
            img_annot_list.append(temp[i]["name"])
    else:
        img_annot_list.append(temp["name"])
        

classes = Counter(img_annot_list).keys()
target = Counter(img_annot_list).values()
print(list(zip(classes,target)))


# In[ ]:


matplotlib.pyplot.pie(target, 
                      labels=classes, 
                      colors=['green', 'yellow', 'red'],
                      pctdistance=0.8,
                      shadow=True, 
                      labeldistance=1.1, 
                      startangle=75, 
                      radius=2, 
                      counterclock=True)


# It is a pretty unbalanced dataset, huh. Let's be possitive and think the reason behind it means that most of people wear the mask correctly 😁.

# In[ ]:


def get_df_annotations():
    objects = {
            "xmin":[],
            "ymin":[],   
            "xmax":[],
            "ymax":[],
            "name":[],    
            "file":[],
            "width":[],
            "height":[],
           }

    for an_path in glob.glob(annotations_path+"/*.xml"):
        tree = ET.parse(an_path)

        for el in tree.iter():
            if 'size' in el.tag:
                for attr in list(el):
                    if 'width' in attr.tag: 
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))    

            if 'object' in el.tag:
                for attr in list(el):

                    if 'name' in attr.tag:
                        name = attr.text                 
                        objects['name']+=[name]
                        objects['width']+=[width]
                        objects['height']+=[height] 
                        objects['file']+=[an_path.split('/')[-1][0:-4]] 

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                objects['xmin']+=[xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                objects['ymin']+=[ymin]                                
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                objects['xmax']+=[xmax]                                
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                objects['ymax']+=[ymax]     
        
    return objects

objects = get_df_annotations()


# In[ ]:


df = pd.DataFrame(objects)
df.head(10)


# In[ ]:


file_name_str = 'maksssksksss737'
filterr = df[df['file'] == file_name_str]
# path 
path = fr'/kaggle/input/face-mask-detection/images/{file_name_str}.png'
   
# Reading an image in default mode
image = cv2.imread(path)
# Window name in which image is displayed
window_name = 'Image'

# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
    
# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
cv2.rectangle(image, (46, 71), (28, 55), color, thickness) #(xmax,ymax), (xmin,ymin)
cv2.rectangle(image, (111, 78), (98, 62), color, thickness)
cv2.rectangle(image, (193, 90), (159, 50), color, thickness)
cv2.rectangle(image, (313, 80), (293, 59), color, thickness)
cv2.rectangle(image, (372, 72), (352, 52), color, thickness)
cv2.rectangle(image, (241, 73), (228, 53), color, thickness)
  
# Displaying the image 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[ ]:


df.info()


# In[ ]:


result = list(zip(df.width, df.height))
cntr = dict(Counter(result).most_common(10))
df_aux = pd.DataFrame.from_dict(Counter(cntr), orient='index', columns=['Width_height'])
df_aux.plot.bar()
plt.ylabel('Counts per width-height pair')
plt.xlabel('Width-height pair')
plt.title('Are the images the same size?')


# We just checked that the images are not the same size. We will have this into account when buildint the Transformers model.

# # Transformer <a class="anchor" id="transformer"></a>
# 
# In this section we are going to learn about the "basic" Transformer architecture and how it changed into Vision Transformer architecture. This will help us understand better the code we are about to implement.
# 
# ## Define the Transformer achitecture <a class="anchor" id="transarch"></a>
# 
# This code is inpired in the paper [an image is worth 16X16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf). Please check it out if you want to learn further about the Vision Transformer architecture.
# 
# Before diving into Vision Transformer, I will explain a little the basic concepts of Transformer architecture and how it works.
# <br>
# <br>
# 
# <img src="attachment:c7ff71c2-ff17-4cbf-a670-65ce0071a270.png" width="400" height="600" alt="Vaswani, Ashish, et al. “Attention is all you need.”" align="left" title="Vaswani, Ashish, et al. “Attention is all you need.”">
# 
# The image on the left was extracted from the well known paper "[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)" by Vaswani, Ashish, et al. If you haven't read it yet, go ahead! 😄
# 
# As we can see in the image, Transformer architecture consists of two parts — the <b>encoder</b> and the <b>decoder</b>. The encoder is the part we are interested in as this part will be used in the Vision Transformer for the image classification task. he <b>encoder</b> is composed of a stack of 6 identical layers and each layer has a sublayer. The first is a <b>multi-head self-attention mechanism</b>, and the second is a simple, position wise fully connected feed forward network. The next step is to use a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $D{model}$= 512
# 
# That is the encoder, which is the one we are interested in in this notebook. I will explain how the decoder works just to have some notions about it. Mind that the decoder is very similar to the encoder.
# 
# The <b>decoder</b> is also composed of a stack of 6 identical layers plus two sub-layers in each encoder layer. The decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder,residual connections are used around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position <i>i</i> can depend only on the known outputs at positions less than <i>i</i>.
# 
# 
# 
# The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of the next image, respectively.

# ## ViT architecture <a class="anchor" id="vitarch"></a>
# <img src="attachment:2d024238-6b76-4067-8d56-31155b52d58f.png" width="600" alt="An image is worth 16X16 words: Transformers for image recognition at scale”" align="right" title="An image is worth 16X16 words: Transformers for image recognition at scale”">
# 
# As traditional Transformer used to handle 1D sequence of token embeddings and we are now using 2D images, we should reshape the 2D image into asequence of flattened 2D patches. This is a very elegant solution! The Transformer uses constant latent vector size D through all of its layers, so the patches are flattened and mapped to D dimensions with a trainable linear projection. The Transformer refers to the output of this projection as the patch embeddings. After this, the architecture betweet Transformer and ViT is almost the same.
# 
# With all that said, we are ready to implement in Python 🐍!

# # Implement ViT architecture <a class="anchor" id="impvitarch"></a>
# 
# 
# The ViT model has multiple Transformer blocks. The MultiHeadAttention layer is used for self-attention, applied to the sequence of image patches. 
# 
# <blockquote>💬"The basic idea of the Attention mechanism is to pay attention to specific input vectors of the input sequence based on the attention weights"</blockquote>
# 
# <br>
# Now that we are going to implement our knowledge in code we should have the steps we are going to follow very clear in our minds. This are the steps we are going to follow:
# 
# 1. Split the image into fixed-size patches.
# 
# 
# 2. Flatten the 2D image patches to 1D patch embedding and linearly embed them using a fully connected layer.
# 
# 3. Positional embeddings are added to the patch embeddings to retain positional information. (CLS) is added to positional encoding. The positional patch encoded vectors will be passed as an input to the Transformer Encoder.
# 
# 4. Transformer Encoder has alternating layers of multiheaded self-attention and MLP blocks. Layernorm (LN) is applied before the self-attention block and the MLP block. The residual connections are applied after every block.
# 
# 5. The last layer will consist on a Dense layer with 4 outputs which will be the bounding boxes coordinates.
# 

# In[ ]:


# Write here the variables so as to not drive ourselves crazy
# in case we want to change some variables in the code.
# This will make our code cleaner and more understandable
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 42
DROPOUT = 0.1
NUM_HEADS = 8
ACTIVATION = "relu" # Activation function
PADDING = "VALID"
EPSILON = 1e-6
EPOCHS = 30
PATCH_SIZE = 32 # Size of the patches to be extracted from the input images
NUM_LAYERS = 8
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
MLP_DIM = 128
PROJECTION_DIM = 64
LR = 1e-3


# In[ ]:


# Implement multilayer-perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# In[ ]:


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    #     Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding=PADDING,
        )
        # return patches
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])


# In[ ]:


### Display patches for an input image

random_num = random.randint(0, len(x_train)-1)

plt.figure(figsize=(4, 4))
plt.imshow(x_train[random_num]/255.)
plt.axis("off")

patches = Patches(PATCH_SIZE)(tf.convert_to_tensor([x_train[random_num]]))
print(f"Image size: {IMAGE_SIZE} X {IMAGE_SIZE}")
print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE}")
print(f"{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch")


n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(6, 6))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, CHANNELS))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")


# In[ ]:


class PatchEncoder(layers.Layer):
    # Implement the patch encoding layer

    # The PatchEncoder layer linearly transforms a patch by projecting it into a vector of size projection_dim. 
    # It also adds a learnable position embedding to the projected vector.
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# In[ ]:


def create_vit_object_detector(
                                input_shape,
                                patch_size,
                                num_patches,
                                projection_dim,
                                num_heads,
                                transformer_units,
                                transformer_layers,
                                mlp_head_units):
    
    
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(PATCH_SIZE)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=EPSILON)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=DROPOUT
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=EPSILON)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=DROPOUT)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=EPSILON)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(DROPOUT*3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=DROPOUT*3)

    bounding_box = layers.Dense(4)(
                                    features
                                  )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)


# # Prepare to train <a class="anchor" id="train"></a>
# 
# We are going to train with a GPU accelerator to make this training faster. To do so we should check a few things first:
# 1. The system has GPU.
# 2. You have installed the GPU compatible version of tensorflow.
# 3. Verify that the code is running on GPU.

# In[ ]:


from tensorflow.python.client import device_lib
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), "\n")

print(device_lib.list_local_devices())

tf.debugging.set_log_device_placement(True)
tf.random.set_seed(SEED)


# In[ ]:


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):

    optimizer = tfa.optimizers.AdamW(
                                        learning_rate=learning_rate, 
                                        weight_decay=weight_decay
                                    )

    # Compile model.
    model.compile(
                    optimizer=optimizer, 
                    loss=[keras.losses.MeanSquaredError(), tfa.losses.GIoULoss()],
                    #metrics=[tf.keras.metrics.MeanIoU(num_classes=len(classes)+1)]

                  )

    checkpoint_filepath = "logs/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
                            x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            validation_split=0.25,
                            callbacks=[
                                        checkpoint_callback,
                                        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
                                      ],
                         )

    return history


num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2


# Size of the transformer layers
transformer_units = [
                        PROJECTION_DIM * 2,
                        PROJECTION_DIM,
                    ]

transformer_layers = 6
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers


history = []
num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

vit_object_detector = create_vit_object_detector(
                                                    INPUT_SIZE,
                                                    PATCH_SIZE,
                                                    num_patches,
                                                    PROJECTION_DIM,
                                                    NUM_HEADS,
                                                    transformer_units,
                                                    transformer_layers,
                                                    mlp_head_units,
                                                )


# ## Visualize ViT model <a class="anchor" id="visualize"></a>

# In[ ]:


vit_object_detector.summary()


# In[ ]:


from tensorflow.keras.utils import plot_model
plot_model(vit_object_detector, to_file='model.png')


# ## Fitting the model <a class="anchor" id="fitting"></a>
# 
# Now that we have defined and implement our python code, it is time to train our model! Finally! Now we should be able to see how the loss decreases epoch after epoch. Have a cup of tea and enjoy the process.

# In[ ]:


# Train model
history = run_experiment(
                        vit_object_detector, LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS
                        )


# In[ ]:


# Saves the model in current path
# vit_object_detector.save("vit_object_detector.h5", save_format="h5")


# # Future work <a class="anchor" id="future"></a>
# 
# Next steps to this project would include:
# 1. Play around with hyperparameters (epoch, num of layers...)
# 2. Get more data (this is something a DS should always be looking forward to)
# 3. Apply Data Augmentation techniques so as to have a more balanced dataset.
# 4. Use a pre-trained model.

# In[ ]:




