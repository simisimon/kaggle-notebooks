#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is a learning and researching purpose kernel. In this journey, we will explore other kernels and works around the internet while will be trying to apply newly learned materials. Stay tuned. 

# ## Problem Statement: 
# Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface.
# 
# But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings. More alarmingly, it leads to potentially dangerous situations for oil and gas company drillers.
# 
# To create the most accurate seismic images and 3D renderings, TGS (the world’s leading geoscience data company) is hoping Kaggle’s machine learning community will be able to build an algorithm that automatically and accurately identifies if a subsurface target is salt or not.

# ## Submission Guideline
# Submission File
# 
# In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
# 
# The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
# 
# The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.
# 
# The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.
# 
# id,                     rle_mask
# 
# 3e06571ef3,   1 1
# 
# a51b08d882,  1 1
# 
# c32590b06f,  1 1
# 
# etc.

# ## Exploring Image Data
# We will explore the image data in an ameture way following [this kernel](https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies), [this kernel](http://https://www.kaggle.com/skainkaryam/basic-data-visualization-using-pytorch-dataset). Special thanks to them. 
# 

# ## Import Libraries

# In[ ]:


import os
import pathlib
import imageio
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# ## Directory List

# In[ ]:


print('Parent Directory: ', os.listdir("../input"))
print('train Directory:  ', os.listdir("../input/train"))
print('test Directory:   ', os.listdir("../input/train"))


# In[ ]:


depths_df = pd.read_csv('../input/depths.csv')
train_df = pd.read_csv("../input/train.csv")
sample_submission_df = pd.read_csv("../input/sample_submission.csv")


# ## Read Data

# In[ ]:


train_df.head()


# In[ ]:


depths_df.head()


# In[ ]:


sample_submission_df.head()


# ## Load & View Data using Pytorch Dataset Class
# Pytorch Data Loading is neat. The official tutorial is [here](http://https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). The below work is taken from [this excellent kernel](http://https://www.kaggle.com/skainkaryam/basic-data-visualization-using-pytorch-dataset)

# In[ ]:


class TGSSaltDataset(Dataset):
    
    def __init__(self, root_path, file_list):
        self.root_path = root_path
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        image = np.array(imageio.imread(image_path), dtype=np.uint8)
        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        
        return image, mask


# In[ ]:


depths_df = pd.read_csv('../input/train.csv')

train_path = "../input/train/"
file_list = list(depths_df['id'].values)


# In[ ]:


dataset = TGSSaltDataset(train_path, file_list)


# In[ ]:


def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')


# In[ ]:


for i in range(5):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)

