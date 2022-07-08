#!/usr/bin/env python
# coding: utf-8

# # DeepLabV3Plus Simple Trained on Multiple Losses for almost 20 Epochs. (0.837)

# # Training Notebook [https://www.kaggle.com/code/salmanahmedtamu/deeplab-v3-pytorch-training-notebook-0-837](https://www.kaggle.com/code/salmanahmedtamu/deeplab-v3-pytorch-training-notebook-0-837)

# In[ ]:


import sys
sys.path.append("../input/segmentation-models-pytorch/segmentation_models.pytorch-master")
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append("../input/pretrainedmodels/pretrainedmodels-0.7.4")
sys.path.append("../input/efficientnet-pytorch/EfficientNet-PyTorch-master")


# In[ ]:


import pandas as pd
import numpy as np
import os
from glob import glob
import gc
gc.enable()
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import albumentations as A
import time
import os
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
from torch.cuda import amp
scaler = amp.GradScaler()


# In[ ]:


os.listdir('../input')


# In[ ]:


df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG == True:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop('segmentation')
    df['predicted'] = ""


# In[ ]:


df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
df["case_id"] = df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

# 2. Get Day as a column
df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

# 3. Get Slice Identifier as a column
df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

if DEBUG:
    TRAIN_DIR = '../input/uw-madison-gi-tract-image-segmentation/train'
else:
    TRAIN_DIR = '../input/uw-madison-gi-tract-image-segmentation/test'
# Get all training images
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

# 4. Get full file paths for the representative scans
# df["_partial_ident"] = (+ "\\" + +"\\"+ 
#                        + "\\scans\\"+df["slice_id"]) 

p = []
x = all_train_images[0].rsplit("/", 4)[0]
for i in range(0, df.shape[0]):
    p.append(os.path.join(x, df["case_id_str"].values[i], df["case_id_str"].values[i]+"_"+df["day_num_str"].values[i], "scans", df["slice_id"].values[i]))
df["_partial_ident"] = p

p = []
for i in range(0, len(all_train_images)):
    p.append(str(all_train_images[i].rsplit("_",4)[0]))
    
_tmp_merge_df = pd.DataFrame()
_tmp_merge_df['_partial_ident'] = p
_tmp_merge_df['f_path'] = all_train_images


df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

# 5. Get slice dimensions from filepath (int in pixels)
df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

# 6. Pixel spacing from filepath (float in mm)
df["px_spacing_h"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
df["px_spacing_w"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))

df1 = df[df.index % 3 == 0]
df2 = df[df.index % 3 == 1]
df3 = df[df.index % 3 == 2]
df = df1.copy()
df.pop('class')
gc.collect()

del x, df1, df2, df3, _tmp_merge_df, all_train_images
gc.collect()
df = df.reset_index(drop=True)


# In[ ]:


print (df.shape)
if DEBUG:
    df = df.sample(frac=0.1).reset_index(drop=True)
print (df.shape)


# In[ ]:


class TractDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.sh = df["slice_h"]
        self.sw = df["slice_w"]
        
        self.image_path = df['f_path']
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        _img = self.open_gray16(self.image_path[idx])
        _img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
        img = (_img * 255.0).astype('uint8')
        
        if self.transform is not None:
            aug = self.transform(image=img)
            img = aug['image']
            
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        return torch.tensor(img, dtype=torch.float), self.sh[idx], self.sw[idx]
    
    def open_gray16(self, _path, normalize=True, to_rgb=True):
        """ Helper to open files """
        if normalize:
            if to_rgb:
                return np.tile(np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH)/65535., axis=-1), 3)
            else:
                return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)/65535.
        else:
            if to_rgb:
                return np.tile(np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH), axis=-1), 3)
            else:
                return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)


# In[ ]:


def test_epoch(loader):
    for each in models:
        each.eval()
        
    lbs = []
    sbs = []
    sts = []
    LOGITS = []
    
    with torch.no_grad():
        
        for (data, sh, sw) in tqdm(loader):
            
            data = data.to(CFG['device'])
            outputs = []
            for each in models:
                output = each(data)
                output = torch.sigmoid(output.cpu())
                outputs.append(output)
            
            output = (outputs[0] + outputs[1] ) / 2.0
            output = torch.round(output).numpy()
            
            
            for idx in range(0, len(sh)):
                root_shape = (int(sh[idx]), int(sw[idx]))
                pred_arr = np.round(cv2.resize(output[idx, 0, :, :].astype('uint8'), root_shape, interpolation=cv2.INTER_NEAREST)).astype('uint8')
                lbs.append(rle_encode(pred_arr))
                pred_arr = np.round(cv2.resize(output[idx, 1, :, :].astype('uint8'), root_shape, interpolation=cv2.INTER_NEAREST)).astype('uint8')
                sbs.append(rle_encode(pred_arr))
                pred_arr = np.round(cv2.resize(output[idx, 2, :, :].astype('uint8'), root_shape, interpolation=cv2.INTER_NEAREST)).astype('uint8')
                sts.append(rle_encode(pred_arr))

    gc.collect()
    return lbs, sbs, sts


# In[ ]:


def rle_encode(img):
    """ TBD
    
    Args:
        img (np.array): 
            - 1 indicating mask
            - 0 indicating background
    
    Returns: 
        run length as string formated
    """
    
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


CFG = {
    'fold' : 0, 
    'batch_size' : 8,
    'image_size' : 256,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'init_lr' : 1e-3,
    'warmup_factor' : 10,
    'warmup_epo' : 4,
    'n_epochs' : 20,
    'num_workers' : 4,
}


# In[ ]:


models = []
for fold in range(0, 2):
    model = smp.DeepLabV3Plus('efficientnet-b1', encoder_weights=None, classes=3, activation=None)
    model = model.to(CFG['device'])
    model = nn.DataParallel(model)
    model_file = f'../input/../input/deeplab-20-epochs/deep_lab_final_fold_{fold}.pth'
    model.load_state_dict(torch.load(model_file))
    models.append(model)


# In[ ]:


transform_test = A.Compose([A.Resize(CFG['image_size'], CFG['image_size'], interpolation=cv2.INTER_NEAREST)])
test_set = TractDataset(df.reset_index(drop=True), transform_test)
test_loader = DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False) 
lbs, sbs, sts = test_epoch(test_loader)

del test_set, test_loader, transform_test
gc.collect()

df = df[['id', 'slice_h', 'slice_w']]
gc.collect()


# In[ ]:


del models
gc.collect()


# In[ ]:


df = df[['id']]
gc.collect()


# In[ ]:


ids = []
classes = []
rles = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    ids.extend([row['id']] * 3)
    classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    rles.extend([lbs[index], sbs[index], sts[index]])


# In[ ]:


del lbs, sbs, sts
gc.collect()


# In[ ]:


df = pd.DataFrame()
df['id'] = ids
df['class'] = classes
df['predicted'] = rles
df.to_csv("submission.csv", index=False)


# In[ ]:


df.head()


# In[ ]:





# In[ ]:




