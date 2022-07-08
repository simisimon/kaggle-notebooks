#!/usr/bin/env python
# coding: utf-8

# Notebook using which the metadata was generated: https://www.kaggle.com/code/defcodeking/metadata-of-records-for-shuffling-val-split.

# In[ ]:


import pandas as pd
import numpy as np
import bson
from sklearn.model_selection import train_test_split
import os
import io
from PIL import Image
import torch
import albumentations as A
from sklearn import preprocessing


# # Config

# In[ ]:


DATA_DIR = "../input/cdiscount-image-classification-challenge"
METADATA_DIR = "../input/cdiscount-metadata-for-shuffling-val-split"


# # Preprocessing

# An optional but good step before using the dataset is to move away from a per-product row system to a per-image row system. That is, instead of having a row for each product, we should have a row for each image. This can be easily achieved by repeating all rows where `n_imgs` is more than 1 and then adding an additional column which goes from 0 to `n_imgs - 1` for each product. Then, we can simply index the `imgs` array that is part of each record to obtain the required image.
# 
# After this preprocessing step, we are essentially dealing with a dataset with one row per image.

# In[ ]:


filepath = os.path.join(METADATA_DIR, "train_metadata.csv")
train_md = pd.read_csv(filepath)
train_md.head()


# In[ ]:


filepath = os.path.join(METADATA_DIR, "test_metadata.csv")
test_md = pd.read_csv(filepath)
test_md.head()


# ## Repeating Rows

# In[ ]:


len(train_md), len(test_md)


# In[ ]:


train_md = train_md.loc[train_md.index.repeat(train_md["n_imgs"])]
test_md = test_md.loc[test_md.index.repeat(test_md["n_imgs"])]
len(train_md), len(test_md)


# In[ ]:


train_md.head()


# In[ ]:


test_md.head()


# ## Adding Column

# In[ ]:


grouped_train_md = train_md.groupby("pid")
train_md["img_idx"] = grouped_train_md.cumcount()
train_md.head()


# In[ ]:


grouped_test_md = test_md.groupby("pid")
test_md["img_idx"] = grouped_test_md.cumcount()
test_md.head()


# # Reading a Random Image

# We define a function `read_image()` which takes a file object, a start position, a length and an index. It seeks the file pointer to the start position, reads `length`-number of bytes, decodes the record and then returns the image at the correct index as PIL image.

# In[ ]:


def read_image(f, start, length, idx):
    f.seek(start)
    record = bson.decode(f.read(length))
    img = record["imgs"][idx]["picture"]
    return Image.open(io.BytesIO(img))


# In[ ]:


# Training data
# Run cell multiple times to see more images
n_imgs = len(train_md)
idx = np.random.choice(range(n_imgs))
md = train_md.iloc[idx]

filepath = os.path.join(DATA_DIR, "train.bson")
with open(filepath, "rb") as f:
    img = read_image(f, md["start"], md["length"], md["img_idx"])
    
img


# In[ ]:


# Test data
# Run cell multiple times to see more images
n_imgs = len(test_md)
idx = np.random.choice(range(n_imgs))
md = test_md.iloc[idx]

filepath = os.path.join(DATA_DIR, "test.bson")
with open(filepath, "rb") as f:
    img = read_image(f, md["start"], md["length"], md["img_idx"])
    
img


# # Shuffling

# Since we are now dealing with a dataset with one row per image, we can simply shuffle the dataframe to shuffle the dataset.

# In[ ]:


train_md = train_md.sample(frac=1)
train_md.head()


# In[ ]:


test_md = test_md.sample(frac=1)
test_md.head()


# # Validation Split

# Similar to shuffling, we can simply split the dataframe to split the dataset.

# In[ ]:


# Encode labels before splitting
encoder = preprocessing.LabelEncoder()
train_md["category_id"] = train_md["category_id"].astype(str)
train_md["labels"] = encoder.fit_transform(train_md["category_id"])


# In[ ]:


train_md, val_md = train_test_split(train_md, test_size=0.1, shuffle=True, random_state=42)


# In[ ]:


train_md.head()


# In[ ]:


val_md.head()


# In[ ]:


len(train_md), len(val_md)


# # Example PyTorch Dataset

# In[ ]:


class CDiscountDataset(torch.utils.data.Dataset):
    def __init__(self, ds_filepath, metadata_df, has_labels=True, transforms=None):
        self.f = open(ds_filepath, "rb")
        self.md_df = metadata_df
        self.transforms = transforms
        self.has_labels = has_labels
        
    def read_image(self, record_metadata):
        start = record_metadata["start"]
        length = record_metadata["length"]
        idx = record_metadata["img_idx"]
        
        self.f.seek(start)
        record = bson.decode(self.f.read(length))
        
        img = record["imgs"][idx]["picture"]
        return Image.open(io.BytesIO(img))
        
    def __len__(self):
        return len(self.md_df)
    
    def __getitem__(self, idx):
        metadata = self.md_df.iloc[idx]
        img = self.read_image(metadata)
        
        img = np.array(img, dtype=np.float32)
        
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
            
        img = np.swapaxes(img, -1, 0)
            
        sample = {"imgs": torch.tensor(img)}
            
        if self.has_labels is True:
            sample["labels"] = metadata["labels"]
            
        return sample


# In[ ]:


filepath = os.path.join(DATA_DIR, "train.bson")
train_ds = CDiscountDataset(ds_filepath=filepath, metadata_df=train_md)
val_ds = CDiscountDataset(ds_filepath=filepath, metadata_df=val_md)


# In[ ]:


filepath = os.path.join(DATA_DIR, "test.bson")
test_ds = CDiscountDataset(ds_filepath=filepath, metadata_df=test_md, has_labels=False)


# In[ ]:


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_ds,
    batch_size=16,
    shuffle=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_ds,
    batch_size=16,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_ds,
    batch_size=16,
    shuffle=True,
)


# In[ ]:


itrain_dl = iter(train_dataloader)
ival_dl = iter(val_dataloader)
itest_dl = iter(test_dataloader)


# In[ ]:


# Run cell again to get next batch
train_sample = next(itrain_dl)
val_sample = next(ival_dl)
test_sample = next(itest_dl)

train_sample["imgs"].shape, val_sample["imgs"].shape, test_sample["imgs"].shape, 

