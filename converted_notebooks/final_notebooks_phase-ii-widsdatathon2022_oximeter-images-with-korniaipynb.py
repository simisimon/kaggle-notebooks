#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](https://warehouse-camo.ingress.cmh1.psfhosted.org/15ca0d80ceb95dbdedc8ffa7a18f67e7f3e9c9c6/68747470733a2f2f6769746875622e636f6d2f6b6f726e69612f646174612f7261772f6d61696e2f6b6f726e69615f62616e6e65725f70697869652e706e67)pypi.org

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


get_ipython().system('pip install kornia')


# In[ ]:


import cv2

import torch
import torchvision
import kornia as K


# In[ ]:


img_bgr: np.array = cv2.imread('../input/oximeter-image-dataset/oximeter_sample_100/20210805_21_10_04_000_31cAu3oTkNUxb9F7CTb2OHB4JLp1_F_3264_2448.jpg')
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# In[ ]:


img_bgr: np.array = cv2.imread('../input/oximeter-image-dataset/oximeter_sample_100/20210809_17_46_49_000_2GHoe2v7KMZ7HJlQofmaGW9BKXw2_F_3000_4000.jpg')  
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# In[ ]:


img_bgr: np.array = cv2.imread('../input/oximeter-image-dataset/oximeter_sample_100/20210809_22_01_49_000_2GHoe2v7KMZ7HJlQofmaGW9BKXw2_F_3000_4000.jpg')  # HxWxC / np.uint8
img_rgb: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb); plt.axis('off');


# #Load an image with Torchvision
# 
# It returns the images in a torch.Tensor in the shape (C,H,W)

# In[ ]:


x_rgb: torch.tensor = torchvision.io.read_image('../input/oximeter-image-dataset/oximeter_sample_100/20210805_21_10_04_000_31cAu3oTkNUxb9F7CTb2OHB4JLp1_F_3264_2448.jpg')  # CxHxW / torch.uint8
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

img = cv2.imread("../input/oximeter-image-dataset/oximeter_sample_100/20210805_21_10_04_000_31cAu3oTkNUxb9F7CTb2OHB4JLp1_F_3264_2448.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

img_tensor = image_to_tensor(img).float() / 255.
plt.imshow(img); plt.axis('off');


# #Define Augmentation Sequential and Different Labels

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html?highlight=bbox

aug_list = AugmentationSequential(
    K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
    K.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
    K.RandomPerspective(0.5, p=1.0),
    data_keys=["input", "bbox", "keypoints", "mask"],
    return_transform=False,
    same_on_batch=False,
)

bbox = torch.tensor([[[355,10],[660,10],[660,250],[355,250]]])
keypoints = torch.tensor([[[465, 115], [545, 116]]])
mask = bbox_to_mask(torch.tensor([[[155,0],[900,0],[900,400],[155,400]]]), w, h).float()

img_out = plot_resulting_image(img_tensor, bbox, keypoints, mask)
plt.imshow(img_out); plt.axis('off');


# #That above was so small!

# #Forward Computations

# In[ ]:


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html?highlight=bbox

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


#Code by https://kornia-tutorials.readthedocs.io/en/latest/data_augmentation_sequential.html?highlight=bbox

out_tensor_inv = aug_list.inverse(*out_tensor)
img_out = plot_resulting_image(
    out_tensor_inv[0][0],
    out_tensor_inv[1].int(),
    out_tensor_inv[2].int(),
    out_tensor_inv[3][0],
)
plt.imshow(img_out); plt.axis('off');


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

img = cv2.imread("../input/oximeter-image-dataset/oximeter_sample_100/20210805_21_10_04_000_31cAu3oTkNUxb9F7CTb2OHB4JLp1_F_3264_2448.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

img_tensor = image_to_tensor(img).float() / 255.
plt.imshow(img); plt.axis('off');


# In[ ]:


#Code by https://github.com/kornia/kornia-examples/blob/master/aliased-and-not-aliased-patch-extraction.ipynb

img_original = cv2.cvtColor(cv2.imread('../input/oximeter-image-dataset/oximeter_sample_100/20210809_17_46_49_000_2GHoe2v7KMZ7HJlQofmaGW9BKXw2_F_3000_4000.jpg'), cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_original)
H,W,CH = img_original.shape

DOWNSAMPLE = 4
img_small = cv2.resize(img_original, (W//DOWNSAMPLE, H//DOWNSAMPLE), interpolation = cv2.INTER_AREA)
plt.figure()
plt.imshow(img_small);


# #Define a keypoint with a large support region.

# In[ ]:


#Code by https://github.com/kornia/kornia-examples/blob/master/aliased-and-not-aliased-patch-extraction.ipynb

import torch
import seaborn as sns
import kornia.feature as KF
import kornia as K

def show_lafs(img, lafs, idx=0, color='r', figsize = (10,7)):
    x,y = KF.laf.get_laf_pts_to_draw(lafs, idx)
    plt.figure(figsize=figsize)
    if (type(img) is torch.tensor):
        img_show = K.tensor_to_image(img)
    else:
        img_show = img
    plt.imshow(img_show)
    plt.plot(x, y, color)
    return

device = torch.device('cpu')

laf_orig  = torch.tensor([[150., 0, 180],
                     [0, 150, 280]]).float().view(1,1,2,3)
laf_small = laf_orig / float(DOWNSAMPLE)

show_lafs(img_original, laf_orig, figsize=(6,4))
show_lafs(img_small, laf_small, figsize=(6,4))


# #Compare how extracted patch would look like when extracted in a naive way and from scale pyramid.

# In[ ]:


#Code by https://github.com/kornia/kornia-examples/blob/master/aliased-and-not-aliased-patch-extraction.ipynb

PS = 32
with torch.no_grad():
    timg_original = K.image_to_tensor(img_original, False).float().to(device) / 255.
    patches_pyr_orig = KF.extract_patches_from_pyramid(timg_original,laf_orig.to(device), PS)
    patches_simple_orig = KF.extract_patches_simple(timg_original, laf_orig.to(device), PS)
    
    timg_small = K.image_to_tensor(img_small, False).float().to(device)/255.
    patches_pyr_small = KF.extract_patches_from_pyramid(timg_small, laf_small.to(device), PS)
    patches_simple_small = KF.extract_patches_simple(timg_small, laf_small.to(device), PS)
    
# Now we will glue all the patches together:

def vert_cat_with_margin(p1, p2, margin=3):
    b,n,ch,h,w = p1.size()
    return torch.cat([p1, torch.ones(b, n, ch, h, margin).to(device), p2], dim=4)

def horiz_cat_with_margin(p1, p2, margin=3):
    b,n,ch,h,w = p1.size()
    return torch.cat([p1, torch.ones(b, n, ch, margin, w).to(device), p2], dim=3)

patches_pyr = vert_cat_with_margin(patches_pyr_orig, patches_pyr_small)
patches_naive = vert_cat_with_margin(patches_simple_orig, patches_simple_small)

patches_all = horiz_cat_with_margin(patches_naive, patches_pyr)


# #Note how the patches extracted from the images of different sizes differ.
# 
# Bottom row is patches, which are extracted from images of different sizes using a scale pyramid. They are not yet exactly the same, but the difference is much smaller.
# 
# https://github.com/kornia/kornia-examples/blob/master/aliased-and-not-aliased-patch-extraction.ipynb

# In[ ]:


#Code by https://github.com/kornia/kornia-examples/blob/master/aliased-and-not-aliased-patch-extraction.ipynb

plt.figure(figsize=(10,10))
plt.imshow(K.tensor_to_image(patches_all[0,0]));


# #Check how much it influences local descriptor performance such as HardNet.

# In[ ]:


hardnet = KF.HardNet(True).eval()
all_patches = torch.cat([patches_pyr_orig,
                         patches_pyr_small,
                         patches_simple_orig,
                         patches_simple_small], dim=0).squeeze(1).mean(dim=1,keepdim=True)
with torch.no_grad():
    descs = hardnet(all_patches)
    distances = torch.cdist(descs, descs)
    print (distances.cpu().detach().numpy())


# In[ ]:


import torch
import kornia
import cv2
import numpy as np

import matplotlib.pyplot as plt

# read the image with OpenCV
img: np.ndarray = cv2.imread("../input/oximeter-image-dataset/oximeter_sample_100/20210805_21_10_04_000_31cAu3oTkNUxb9F7CTb2OHB4JLp1_F_3264_2448.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert to torch tensor
data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW

# create the operator
gauss = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

# blur the image
x_blur: torch.tensor = gauss(data.float())

# convert back to numpy
img_blur: np.ndarray = kornia.tensor_to_image(x_blur.byte())

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('image source')
axs[0].imshow(img)

axs[1].axis('off')
axs[1].set_title('image blurred')
axs[1].imshow(img_blur);


# #Patch Augmentation Sequential with patchwise_apply=True

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


# #Patch Augmentation Sequential with patchwise_apply=False

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

img = cv2.cvtColor(cv2.imread('../input/oximeter-image-dataset/oximeter_sample_100/20210807_12_44_08_000_1X9B9Lje7bMJUyiO8F29BvEjEbs1_F_4160_3120.jpg'), cv2.COLOR_BGR2RGB)

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
# @inproceedings{eriba2020kornia, author = {E. Riba, D. Mishkin, J. Shi, D. Ponsa, F. Moreno-Noguer and G. Bradski}, title = {A survey on Kornia: an Open Source Differentiable Computer Vision Library for PyTorch}, year = {2020}, }
# 
# @inproceedings{eriba2019kornia, author = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski}, title = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch}, booktitle = {Winter Conference on Applications of Computer Vision}, year = {2020}, url = {https://arxiv.org/pdf/1910.02190.pdf} }
# 
# @misc{Arraiy2018, author = {E. Riba, M. Fathollahi, W. Chaney, E. Rublee and G. Bradski}, title = {torchgeometry: when PyTorch meets geometry}, booktitle = {PyTorch Developer Conference}, year = {2018}, url = {https://drive.google.com/file/d/1xiao1Xj9WzjJ08YY_nYwsthE-wxfyfhG/view?usp=sharing} }
# 
# Aziz Amindzhanov  https://www.kaggle.com/code/azizdzhon/kornia-moons-imc-2022
