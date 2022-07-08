#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from PIL import Image, ImageDraw
import collections
import glob 
from datetime import datetime as dt
import gc
import json

import matplotlib.image as mpimg

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/fungiclef2022/DF20-train_metadata.csv', delimiter=',')
pd.set_option('display.max_columns', None)
train.head()


# In[ ]:


train.dtypes


# In[ ]:


train["Habitat"].value_counts()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'red',
                      height =2000,
                      width = 2000
                     ).generate(str(train["Habitat"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Fungi Habitat")
plt.show()


# #Handle Missing Values to plot a Map. 

# In[ ]:


# Lets first handle numerical features with nan value
numerical_nan = [feature for feature in train.columns if train[feature].isna().sum()>1 and train[feature].dtypes!='O']
numerical_nan


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'blue',
                      height =2000,
                      width = 2000
                     ).generate(str(train["kingdom"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("FungiCLEF Kingdom")
plt.show()


# In[ ]:


train[numerical_nan].isna().sum()


# In[ ]:


## Replacing the numerical Missing Values

for feature in numerical_nan:
    ## We will replace by using median since there are outliers
    median_value=train[feature].median()
    
    train[feature].fillna(median_value,inplace=True)
    
train[numerical_nan].isnull().sum()


# #I don't know why infraspecificEpithet has still Missing Values.

# In[ ]:


#Code by Parul Pandey  https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python

# imputing with a constant

from sklearn.impute import SimpleImputer
train_constant = train.copy()
#setting strategy to 'constant' 
mean_imputer = SimpleImputer(strategy='constant') # imputing using constant value
train_constant.iloc[:,:] = mean_imputer.fit_transform(train_constant)
train_constant.isnull().sum()


# In[ ]:


# categorical features with missing values
categorical_nan = [feature for feature in train.columns if train[feature].isna().sum()>0 and train[feature].dtypes=='O']
print(categorical_nan)


# In[ ]:


# replacing missing values in categorical features
for feature in categorical_nan:
    train[feature] = train[feature].fillna('None')


# In[ ]:


train[categorical_nan].isna().sum()


# In[ ]:


import plotly.express as px
import folium

#Code by Varshini PJ  user: varshinipj

fig = px.scatter_mapbox(train,
# Here, plotly gets, (x,y) coordinates
lat="Latitude",
lon="Longitude",
text='level0Name',

                #Here, plotly detects color of series
                size="class_id",
                color = "phylum",
                labels="level0Name",

                zoom=14.5,
                center={"lat":56.975158, "lon":9.285525},
                height=600,
                width=800)
fig.update_layout(mapbox_style='stamen-toner')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="FungiCLEF 2022")
fig.show()


# In[ ]:


test = pd.read_csv('/kaggle/input/fungiclef2022/FungiCLEF2022_test_metadata.csv', delimiter=',')
test.head()


# In[ ]:


test["Substrate"].value_counts()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'green',
                      height =2000,
                      width = 2000
                     ).generate(str(test["Substrate"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Fungi substrate")
plt.show()


# In[ ]:


val = pd.read_csv('/kaggle/input/fungiclef2022/DF20-val_metadata.csv', delimiter=',')
val.head()


# In[ ]:


get_ipython().system('pip install kornia')


# In[ ]:


import cv2

import torch
import torchvision
import kornia as K


# In[ ]:


img_bgr: np.array = cv2.imread('../input/fungiclef2022/DF21-images-300/DF21_300/1-3343234404.JPG')  # HxWxC / np.uint8
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# In[ ]:


img_bgr: np.array = cv2.imread('../input/fungiclef2022/DF21-images-300/DF21_300/0-3008822344.JPG')  # HxWxC / np.uint8
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# In[ ]:


img_bgr: np.array = cv2.imread('../input/fungiclef2022/DF21-images-300/DF21_300/0-3008822345.JPG')  # HxWxC / np.uint8
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# #Load an image with Torchvision
# 
# It returns the images in a torch.Tensor in the shape (C,H,W)

# In[ ]:


x_rgb: torch.tensor = torchvision.io.read_image('../input/fungiclef2022/DF21-images-300/DF21_300/0-3008822387.JPG')  # CxHxW / torch.uint8
x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
print(x_rgb.shape);


# #Load an image with Kornia
# 
# "The utility is kornia.image_to_tensor which casts a numpy.ndarray to a torch.Tensor and permutes the channels to leave the image ready for being used with any other PyTorch or Kornia component. The image is casted into a 4D torch.Tensor with zero-copy."
# 
# https://kornia-tutorials.readthedocs.io/en/latest/hello_world_tutorial.html

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/hello_world_tutorial.html

x_bgr: torch.tensor = K.image_to_tensor(img_bgr)  # CxHxW / torch.uint8
x_bgr = x_bgr.unsqueeze(0)  # 1xCxHxW
print(f"convert from '{img_bgr.shape}' to '{x_bgr.shape}'")


# #Convert from BGR to RGB with a kornia.color component.

# In[ ]:


x_rgb: torch.tensor = K.color.bgr_to_rgb(x_bgr)  # 1xCxHxW / torch.uint8


# #Visualize an image with Matplotib

# In[ ]:


img_bgr: np.array = K.tensor_to_image(x_bgr)
img_rgb: np.array = K.tensor_to_image(x_rgb)


# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/hello_world_tutorial.html

fig, axs = plt.subplots(1, 2, figsize=(32, 16))
axs = axs.ravel()

axs[0].axis('off')
axs[0].imshow(img_rgb)

axs[1].axis('off')
axs[1].imshow(img_bgr)

plt.show()


# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html

from kornia import augmentation as K
from kornia.augmentation import AugmentationSequential
#from kornia.geometry import bbox_to_mask  ##Deprecated
from kornia.geometry.bbox import bbox_to_mask
from kornia.utils import image_to_tensor, tensor_to_image
from torchvision.transforms import transforms

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def plot_resulting_image(img, bbox, keypoints, mask):
    img = img * mask
    img_draw = cv2.polylines(np.array(to_pil(img)), bbox.numpy(), isClosed=True, color=(255, 0, 0))
    for k in keypoints[0]:
        img_draw = cv2.circle(img_draw, tuple(k.numpy()[:2]), radius=6, color=(255, 0, 0), thickness=-1)
    return img_draw

img = cv2.imread("../input/fungiclef2022/DF20-300px/DF20_300/2237851965-148421.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

img_tensor = image_to_tensor(img).float() / 255.
plt.imshow(img); plt.axis('off');


# #Since I got: IndexError: index 302 is out of bounds for dimension 1 with size 302
# 
# I changed evething to 200

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html

aug_list = AugmentationSequential(
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
    K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
    K.RandomPerspective(0.5, p=1.0),
    data_keys=["input", "bbox", "keypoints", "mask"],
    return_transform=False,
    same_on_batch=False,
)

bbox = torch.tensor([[[200,10],[230,10],[230,250],[200,250]]])
keypoints = torch.tensor([[[200, 115], [200, 116]]])
mask = bbox_to_mask(torch.tensor([[[155,0],[200,0],[200,200],[155,200]]]), w, h).float()##I had to reduce. Original was [155,0],[900,0],[900,400],[155,400]]

img_out = plot_resulting_image(img_tensor, bbox, keypoints, mask)
plt.imshow(img_out); plt.axis('off');


# #Forward Computations

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html

out_tensor = aug_list(img_tensor, bbox.float(), keypoints.float(), mask)
img_out = plot_resulting_image(
    out_tensor[0][0],
    out_tensor[1].int(),
    out_tensor[2].int(),
    out_tensor[3][0],
)
plt.imshow(img_out); plt.axis('off');


# #Inverse Transformations

# In[ ]:


out_tensor_inv = aug_list.inverse(*out_tensor)
img_out = plot_resulting_image(
    out_tensor_inv[0][0],
    out_tensor_inv[1].int(),
    out_tensor_inv[2].int(),
    out_tensor_inv[3][0],
)
plt.imshow(img_out); plt.axis('off');


# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_patch_sequential.html

from kornia.augmentation import PatchSequential, ImageSequential

pseq = PatchSequential(
    ImageSequential(
        K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        K.RandomPerspective(0.2, p=0.5),
        K.RandomSolarize(0.1, 0.1, p=0.5),
    ),
    K.RandomAffine(15, [0.1, 0.1], [0.7, 1.2], [0., 20.], p=0.5),
    K.RandomPerspective(0.2, p=0.5),
    ImageSequential(
        K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        K.RandomPerspective(0.2, p=0.5),
        K.RandomSolarize(0.1, 0.1, p=0.5),
    ),
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
    K.RandomAffine(15, [0.1, 0.1], [0.7, 1.2], [0., 20.], p=0.5),
    K.RandomPerspective(0.2, p=0.5),
    K.RandomSolarize(0.1, 0.1, p=0.5),
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
    K.RandomAffine(15, [0.1, 0.1], [0.7, 1.2], [0., 20.], p=0.5),
    ImageSequential(
        K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        K.RandomPerspective(0.2, p=0.5),
        K.RandomSolarize(0.1, 0.1, p=0.5),
    ),
    K.RandomSolarize(0.1, 0.1, p=0.5),
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
    K.RandomAffine(15, [0.1, 0.1], [0.7, 1.2], [0., 20.], p=0.5),
    K.RandomPerspective(0.2, p=0.5),
    K.RandomSolarize(0.1, 0.1, p=0.5),
    patchwise_apply=True,
    same_on_batch=True,
)
out_tensor = pseq(img_tensor[None].repeat(2, 1, 1, 1))
to_pil(torch.cat([out_tensor[0], out_tensor[1]], dim=2))


# #Patch Augmentation Sequential rocks!
# 
# Patch Augmentation Sequential with patchwise_apply=False
# 
# If patchwise_apply=False, all the args will be combined and applied as one pipeline for each patch.
# 
# https://kornia-tutorials.readthedocs.io/en/latest/data_patch_sequential.html

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_patch_sequential.html

pseq = PatchSequential(
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.75),
    K.RandomElasticTransform(alpha=(4., 4.)),
    patchwise_apply=False,
    same_on_batch=False
)
out_tensor = pseq(img_tensor[None].repeat(2, 1, 1, 1))
to_pil(torch.cat([out_tensor[0], out_tensor[1]], dim=2))


# In[ ]:


get_ipython().system('pip install kornia_moons')


# In[ ]:


#Code by Aziz Amindzhanov  https://www.kaggle.com/code/azizdzhon/kornia-moons-imc-2022

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
import torch
import kornia as K
from typing import List
import matplotlib.pyplot as plt

from kornia_moons.feature import *


# In[ ]:


#Code by Aziz Amindzhanov  https://www.kaggle.com/code/azizdzhon/kornia-moons-imc-2022

img = cv2.cvtColor(cv2.imread('../input/fungiclef2022/DF20-300px/DF20_300/2237851963-3.jpg'), cv2.COLOR_BGR2RGB)

det = cv2.ORB_create(500)
kps, descs = det.detectAndCompute(img, None)

out_img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img)


lafs = laf_from_opencv_ORB_kpts(kps)
visualize_LAF(K.image_to_tensor(img, False), lafs, 0)

kps_back = opencv_ORB_kpts_from_laf(lafs)
out_img2 = cv2.drawKeypoints(img, kps_back, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(out_img2)


# #Acknowledgements:
# 
# Kornia AI is on the mission to leverage and democratize the next generation of Computer Vision tools and Deep Learning libraries within the context of an Open Source community.
# 
# https://kornia.readthedocs.io/en/latest/
# 
# Aziz Amindzhanov https://www.kaggle.com/code/azizdzhon/kornia-moons-imc-2022
