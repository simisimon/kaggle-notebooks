#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: black;"><b style="color:white;">In the future, everybody will be famous for fifteen minutes</b></h1></center>

# "Andy Warhol (; born Andrew Warhola Jr.; August 6, 1928 – February 22, 1987) was an American artist, film director, and producer who was a leading figure in the visual art movement known as pop art. His works explore the relationship between artistic expression, advertising, and celebrity culture that flourished by the 1960s, and span a variety of media, including painting, silkscreening, photography, film, and sculpture."
# 
# "Some of his best-known works include the silkscreen paintings Campbell's Soup Cans (1962) and Marilyn Diptych (1962), the experimental films Empire (1964) and Chelsea Girls (1966), and the multimedia events known as the Exploding Plastic Inevitable (1966–67)."
# 
# https://www.tate.org.uk/art/artists/andy-warhol-2121#:~:text=Andy%20Warhol%20(%3B%20born%20Andrew%20Warhola,movement%20known%20as%20pop%20art.
# 
# 
#  "In a Time magazine article in 1967, Andy Warhol predicted the future as a time "when everyone will be famous for fifteen minutes." The next year, he included a similar version of this quote in an art show, saying, "In the future, everybody will be world-famous for fifteen minutes."
#  
# https://www.shmoop.com/quotes/in-the-future-everyone-will-be-famous-for-15-minutes.html#:~:text=In%20a%20Time%20magazine%20article,%2Dfamous%20for%20fifteen%20minutes.%22 

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfQGANfDTGt7nLi5BRM8J-egHMgmJkJ1W2QA&usqp=CAU)boldomatic.com

# ![](https://image.slidesharecdn.com/stateoftheartimageaugmentationswithalbumentations-201208095937/85/eugene-khvedchenya-state-of-the-art-image-augmentations-with-albumentations-7-320.jpg?cb=1608633519)slideshare.com

# <h1><span class="label label-default" style="background-color:black;border-radius:100px 100px; font-weight: bold; font-family:Garamond; font-size:20px; color:white; padding:10px">Albumentations</span></h1><br>

# "Image augmentation is a machine learning technique that "boomed" in recent years along with the large deep learning systems. In this article, we present a visualization of spatial-level augmentation techniques available in the albumentations."
# 
# "The provided descriptions mostly come the official project documentation available at https://albumentations.ai/"
# 
# https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/17/Spatial-level-transforms-using-albumentations-package.html

# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)


# In[ ]:


image = cv2.imread('../input/cusersmarilonedriveimagensandypng/andy.png')


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

def augment_and_show(aug, image):
    image = aug(image=image)['image']
    plt.figure(figsize=(5,5))
    plt.imshow(image)


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

AUG=[HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, 
     Blur, OpticalDistortion, GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, 
     MotionBlur, MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, 
     Flip, ]
AUGSTR=['HorizontalFlip', 'IAAPerspective', 'ShiftScaleRotate', 'CLAHE', 'RandomRotate90', 'Transpose', 
        'ShiftScaleRotate', 'Blur', 'OpticalDistortion', 'GridDistortion', 'HueSaturationValue', 
        'IAAAdditiveGaussianNoise', 'GaussNoise', 'MotionBlur', 'MedianBlur', 'IAAPiecewiseAffine', 
        'IAASharpen', 'IAAEmboss', 'RandomContrast', 'RandomBrightness', 'Flip', ]


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

n=len(AUGSTR)
N=random.sample(range(n),k=9)
print(N)


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

fig,axs = plt.subplots(3,3,figsize=(15,15))
for i in range(9):
    image = image
    r=i//3
    c=i%3
    axs[r][c].set_xticks([])
    axs[r][c].set_yticks([])
    axs[r][c].set_title(AUGSTR[N[i]])
    aug=AUG[N[i]](p=1)
    image = aug(image=image)['image']
    ax=axs[r][c].imshow(image)
plt.show()


# #RandomBrightness

# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

#RandomBrightness

aug = RandomBrightness(p=1)
augment_and_show(aug, image)
#plt.figure(figsize=(8,8));


# #HorizontalFlip

# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

#HorizontalFlip
aug = HorizontalFlip(p=1)
augment_and_show(aug, image)


# #IAAPerspective
# 
# "Perform a random four point perspective transform of the input."
# 
# https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/17/Spatial-level-transforms-using-albumentations-package.html#IAAPerspective

# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

#IAAPerspective
aug = IAAPerspective(scale=0.2, p=1)
augment_and_show(aug, image)


# #IAAPiecewiseAffine
# 
# Perform a random four point perspective transform of the input.
# 
# 

# In[ ]:


#Code by https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/17/Spatial-level-transforms-using-albumentations-package.html#IAAPerspective

from albumentations.imgaug.transforms import IAAPerspective 

scale=(0.3, 0.1)
keep_size=True

transform = IAAPerspective(scale, keep_size, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #IAASharpen
# 
# Sharpen the input image and overlays the result with the original image.

# In[ ]:


from albumentations.imgaug.transforms import IAAPerspective 

alpha=(0.1, 0.3)
lightness=(0.1, 1.0)

transform = IAAPerspective(alpha, lightness, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #ShiftScaleRotate

# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

#ShiftScaleRotate
aug = ShiftScaleRotate(p=1)
augment_and_show(aug, image)


# In[ ]:


#Code by stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations

#Combination1
def augment_flips_color(p=.5):
    return Compose([
        CLAHE(),
        RandomRotate90(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        Blur(blur_limit=3),
        OpticalDistortion(),
        GridDistortion(),
        HueSaturationValue()
    ], p=p)


# In[ ]:


aug = augment_flips_color(p=1)
augment_and_show(aug, image)


# #IAASuperpixels

# In[ ]:


#Code by https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/17/Spatial-level-transforms-using-albumentations-package.html#IAAPerspective

from albumentations.imgaug.transforms import IAASuperpixels 

p_replace=0.5
n_segments=10

transform = IAASuperpixels(p_replace, n_segments, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# CoarseDropout
# 
# CoarseDropout of the rectangular regions in the image.
# 
# https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/17/Spatial-level-transforms-using-albumentations-package.html#IAAPerspective

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import CoarseDropout


max_holes=10
max_height=50
max_width=8
min_holes=1
min_height=None
min_width=None
fill_value=255
mask_fill_value=None

transform =CoarseDropout(max_holes, max_height, max_width, min_holes, min_height, min_width,
                         fill_value, mask_fill_value, 
                         p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# In[ ]:


get_ipython().system('pip install -U albumentations')


# In[ ]:


get_ipython().system('pip install -U git+https://github.com/albumentations-team/albumentations')


# #GridDistortion
# 
# Add distorition to the image, not really well documanted, hyperparameters are not clear for me
# 
# by Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import GridDistortion

num_steps=50
distort_limit=0.5
interpolation=1
border_mode=4
value=None
mask_value=None

transform = GridDistortion(num_steps, distort_limit, interpolation, border_mode, value, mask_value, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #GridDropout
# 
# GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
# 
# Inspired by Chen et al. paper from 2020, avalible here: https://arxiv.org/pdf/2001.04086.pdf
# 
# 

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import GridDropout

ratio=0.4
unit_size_min=2
unit_size_max=19
holes_number_x= 10
holes_number_y= 10
shift_x=100 
shift_y=0
random_offset=False

transform = GridDropout(ratio, unit_size_min, unit_size_max, holes_number_x,
                        holes_number_y, shift_x, shift_y, random_offset, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #IAAAffine
# 
# Place a regular grid of points on the input and randomly move the neighbourhood of these point around via affine transformations.
# 
# by Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.imgaug.transforms import IAAAffine

scale=0.1
translate_percent=10
translate_px=None
rotate=0.5
shear=0.5
order=1
cval=0
mode='reflect'

transform = IAAAffine(scale, translate_percent, translate_px, rotate, shear, order, cval, mode, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #IAAAdditiveGaussianNoise
# 
# Add gaussian noise to the input image.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.imgaug.transforms import IAAAdditiveGaussianNoise

loc=50
scale=(5, 12.75)
per_channel=True

transform = IAAAdditiveGaussianNoise(loc, scale, per_channel, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #IAAEmboss
# 
# Emboss the input image and overlays the result with the original image.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.imgaug.transforms import IAAEmboss

alpha=(0.2, 0.5)
strength=(0.2, 0.7)

transform = IAAEmboss(alpha, strength, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #IAAPerspective
# 
# Perform a random four point perspective transform of the input.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog
# 
# I think it's Flip with another name.

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import Flip

transform = Flip(p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #OpticalDistortion
# 
# No description, hyperparameters not clear
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import OpticalDistortion

distort_limit=2.5
shift_limit=0.9
interpolation=1
border_mode=4
value=None
mask_value=None

transform = OpticalDistortion(distort_limit, shift_limit, interpolation, border_mode, value, mask_value, p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #RandomGridShuffle
# 
# Random shuffle grid's cells on image.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import RandomGridShuffle

transform = RandomGridShuffle(grid=(4, 3), p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #Transpose
# 
# Transpose the input by swapping rows and columns.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import Transpose

transform = Transpose(p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# #VerticalFlip
# 
# Flip the input vertically around the x-axis.
# 
# By Piotr Mazurek https://tugot17.github.io/data-science-blog

# In[ ]:


#Code by Piotr Mazurek https://tugot17.github.io/data-science-blog

from albumentations.augmentations.transforms import VerticalFlip

transform = VerticalFlip(p=1.0)

augmented_image = transform(image=image)['image']
Image.fromarray(augmented_image)


# In[ ]:


get_ipython().system('pip install -U albumentations[imgaug]')


# In[ ]:


org_img = cv2.cvtColor(cv2.imread('../input/cusersmarilonedriveimagensandypng/andy.png'), cv2.COLOR_BGR2RGB)


# In[ ]:


# for Albumentations
def Alb_Augmentations(org_img = None, label = None, suptitle = 'Albumentation', albs = None):
    img_list = [org_img]
    label_list = ['orginal']
    for _ in range(3):
        aug_img = albs(image = org_img)['image']
        img_list.append(aug_img)
        label_list.append(label)
    fig, ax = plt.subplots(figsize = (22, 5), nrows = 1, ncols = 4)
    for i in range(4):
        ax[i].imshow(img_list[i])
        ax[i].set_title(label_list[i])
    
    plt.suptitle(f'{suptitle}')


# #Crop
# 
# Only albumentations has Crop (Albumentations 만 Crop 기능을 가지고 있습니다.)

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

import albumentations as alb

# Albumentation
albs = alb.Crop(x_min = 50, y_min = 50, x_max = 150, y_max = 150, p = 0.8)
Alb_Augmentations(org_img=org_img, label = 'Crop', albs = albs) #I reduced the original was 100/100/300/300

albs = alb.Compose([
    alb.Crop(x_min = 50, y_min = 50, x_max = 150, y_max = 150, p = 0.8),
    alb.Resize(621, 934)
])
Alb_Augmentations(org_img=org_img, label = 'Crop', albs = albs)


# #RandomCrop, Resize

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.Compose([
    alb.CenterCrop(height = 50, width = 150, p = 0.8),#Original was 100/300 I reduced them
    alb.Resize(621, 934)
])
Alb_Augmentations(org_img=org_img, label = 'CenterCrop', albs = albs)
# Albumentation
albs = alb.Compose([
    alb.RandomCrop(height = 50, width = 150, p = 0.8),
    alb.Resize(621, 934)
])
Alb_Augmentations(org_img=org_img, label = 'RandomCrop', albs = albs)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator
def Idg_Augmentations(image, data_generator, label = None, suptitle = 'ImageDataGenerator'):
    img = np.expand_dims(image, axis = 0)
    data_generator.fit(img)
    img_gen = data_generator.flow(img)
    
    fig, ax = plt.subplots(figsize = (22, 5), nrows = 1, ncols = 4)
    for i in range(4):
        if i == 0:
            ax[i].imshow(image)
            ax[i].set_title('Original')
        else:
            img_batch = next(img_gen)
            img_batch = np.squeeze(img_batch).astype('int')
            ax[i].imshow(img_batch)
            ax[i].set_title(label)

    plt.suptitle(f'{suptitle}')


# #RandomBrightnessContrast

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4), p=0.8)
Alb_Augmentations(org_img=org_img, label = 'RandomBrightnessContrast', albs = albs)

# ImageDataGenerator
data_generator = ImageDataGenerator(brightness_range=(0.1, 0.8))
Idg_Augmentations(image=org_img, data_generator=data_generator, label = 'Bright')


# #ChannelShuffle

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.ChannelShuffle(p=0.8)
Alb_Augmentations(org_img=org_img, label = 'ChannelShuffle', albs = albs)

# ImageDataGenerator
data_generator = ImageDataGenerator(channel_shift_range = 60 )
Idg_Augmentations(image=org_img, data_generator=data_generator, label = 'channel_shift_range')


# #HueSaturationValue

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.HueSaturationValue(p=0.8)
Alb_Augmentations(org_img=org_img, label = 'HueSaturationValue', albs = albs)


# #RGBShift
# 
# r_shift_limit : Set a Scope(red shift range) in tuple. (튜플 내에 변환할 범위를 정해야 합니다.)

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.RGBShift(p=0.8)
Alb_Augmentations(org_img=org_img, label = 'RGBShift', albs = albs)


# #CoarseDropout

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.CoarseDropout(p=0.8, max_holes = 50)
Alb_Augmentations(org_img=org_img, label = 'CoarseDropout', albs = albs)


# #Oneof

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.OneOf([
    alb.Rotate(limit=(50, 150), p = 1, border_mode= cv2.BORDER_CONSTANT),
    alb.HorizontalFlip(p = 1),
    alb.ChannelShuffle(p = 1),
    alb.RGBShift(p = 1)
])
Alb_Augmentations(org_img=org_img, label = 'Oneof', albs = albs)


# #Compose
# 
# it's similar ImageDataGenerator (ImageDataGenerator 와 비슷하다고 보면 됩니다.)

# In[ ]:


#Code by WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# Albumentation
albs = alb.Compose([
    alb.Rotate(limit=(50, 150), p = 1, border_mode= cv2.BORDER_CONSTANT),
    alb.HorizontalFlip(p = 1),
    alb.ChannelShuffle(p = 1),
    alb.RGBShift(p = 1),
    alb.CLAHE(p = 1),
    alb.GaussNoise(p = 1, var_limit = (50, 100)),
    alb.CoarseDropout(p = 1, max_holes = 50)
])
Alb_Augmentations(org_img=org_img, label = 'Compose', albs = albs)

# ImageDataGenerator
data_generator = ImageDataGenerator(vertical_flip=True,
                                    horizontal_flip=True,
                                    shear_range=True,
                                    rotation_range= 150,
                                    zoom_range = 0.7
                                   )
Idg_Augmentations(image=org_img, data_generator=data_generator, label = 'ImageDataGenerator')


# #I think Warhol and Basquiat would enjoy my piece of art.
# 
# ![](https://occ-0-246-1380.1.nflxso.net/dnm/api/v6/9pS1daC2n6UGc3dUogvWIPMR_OU/AAAABefzm8ZGPidA1PF3iCoxwD6P3w08LiEajMPZdYOYbtpIaHkx-hbpaZXuIlU5_3yh_4ECC1X7rMFuE6weGJVb96SrYzUkjmzVPvgwaMEb37UZJgiDpGqoxGVp.jpg?r=ae5)https://www.netflix.com/br/title/81026142

# #Acknowledgements:
# 
# Piotr Mazurek https://tugot17.github.io/data-science-blog
# 
# stpete_ishii  https://www.kaggle.com/stpeteishii/image-alubmentations
# 
# WONGI PARK https://www.kaggle.com/kalelpark/compare-albumentations-and-imagedatagenerator

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPfyCTpTu_GSl1YPIvyMTxf1WSPBpHXg3sgQ&usqp=CAU)quickmeme.com
