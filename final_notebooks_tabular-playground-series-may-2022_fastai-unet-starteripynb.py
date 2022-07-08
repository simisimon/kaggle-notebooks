#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


get_ipython().system(' pip install segmentation-models-pytorch')


# In[ ]:


import fastai
from fastai.vision.all import *
from pathlib import Path
import os
import cv2
from sklearn.model_selection import GroupKFold, StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import segmentation_models_pytorch as smp


# ## Dataset Creation 

# In[ ]:


train_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
display(train_df.head())


# In[ ]:


## reference https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda#helper_functions
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)


print("\n... UPDATING DATAFRAMES WITH ACCESSIBLE INFORMATION STARTED ...\n\n")

# 1. Get Case-ID as a column (str and int)
train_df["case_id_str"] = train_df["id"].apply(lambda x: x.split("_", 2)[0])
train_df["case_id"] = train_df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

# 2. Get Day as a column
train_df["day_num_str"] = train_df["id"].apply(lambda x: x.split("_", 2)[1])
train_df["day_num"] = train_df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

# 3. Get Slice Identifier as a column
train_df["slice_id"] = train_df["id"].apply(lambda x: x.split("_", 2)[2])

# 4. Get full file paths for the representative scans
train_df["_partial_ident"] = (TRAIN_DIR+"/"+ # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
                             train_df["case_id_str"]+"/"+ # .../case###/
                             train_df["case_id_str"]+"_"+train_df["day_num_str"]+ # .../case###_day##/
                             "/scans/"+train_df["slice_id"]) # .../slice_#### 
_tmp_merge_df = pd.DataFrame({"_partial_ident":[x.rsplit("_",4)[0] for x in all_train_images], "f_path":all_train_images})
train_df = train_df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

# Minor cleanup of our temporary workaround
del _tmp_merge_df; 

# 5. Get slice dimensions from filepath (int in pixels)
train_df["slice_h"] = train_df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
train_df["slice_w"] = train_df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

# 6. Pixel spacing from filepath (float in mm)
train_df["px_spacing_h"] = train_df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
train_df["px_spacing_w"] = train_df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))

# 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
l_bowel_train_df = train_df[train_df["class"]=="large_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"lb_seg_rle"})
s_bowel_train_df = train_df[train_df["class"]=="small_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"sb_seg_rle"})
stomach_train_df = train_df[train_df["class"]=="stomach"][["id", "segmentation"]].rename(columns={"segmentation":"st_seg_rle"})
train_df = train_df.merge(l_bowel_train_df, on="id", how="left")
train_df = train_df.merge(s_bowel_train_df, on="id", how="left")
train_df = train_df.merge(stomach_train_df, on="id", how="left")
train_df = train_df.drop_duplicates(subset=["id",]).reset_index(drop=True)
train_df["lb_seg_flag"] = train_df["lb_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["sb_seg_flag"] = train_df["sb_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["st_seg_flag"] = train_df["st_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["n_segs"] = train_df["lb_seg_flag"].astype(int)+train_df["sb_seg_flag"].astype(int)+train_df["st_seg_flag"].astype(int)

# 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
train_df = train_df[["id", "f_path", "n_segs",
                     "lb_seg_rle", "lb_seg_flag",
                     "sb_seg_rle", "sb_seg_flag", 
                     "st_seg_rle", "st_seg_flag",
                     "slice_h", "slice_w", "px_spacing_h", 
                     "px_spacing_w", "case_id_str", "case_id", 
                     "day_num_str", "day_num", "slice_id",]]

# 9. Display update dataframe
print("\n... UPDATED TRAINING DATAFRAME... \n")
display(train_df)


# In[ ]:


N_FOLDS = 8
gkf = GroupKFold(n_splits = N_FOLDS)
train_df = train_df[train_df.n_segs > 0].reset_index(drop = True)

print (f"Dropped all the cases with zero segmentation. Size of the dataframe {len(train_df)}")

train_df["which_segs"] = train_df["lb_seg_flag"].astype(int).astype(str) + train_df["sb_seg_flag"].astype(int).astype(str) + train_df["st_seg_flag"].astype(int).astype(str)

for train_idxs, val_idxs in gkf.split(train_df["id"], train_df["which_segs"], train_df["case_id"]):
    sub_train_df = train_df.iloc[train_idxs]
    N_TRAIN = len(sub_train_df)
    sub_train_df = sub_train_df.sample(N_TRAIN).reset_index(drop = True)
    
    sub_val_df = train_df.iloc[val_idxs]
    N_VAL = len(sub_val_df)
    sub_val_df = sub_val_df.sample(N_VAL).reset_index(drop = True)
    
    break
    
sub_train_df.lb_seg_rle.fillna("", inplace = True)
sub_train_df.sb_seg_rle.fillna("", inplace = True)
sub_train_df.st_seg_rle.fillna("", inplace = True)

sub_val_df.lb_seg_rle.fillna("", inplace = True)
sub_val_df.sb_seg_rle.fillna("", inplace = True)
sub_val_df.st_seg_rle.fillna("", inplace = True)


print ("FOLD 1 TRAIN DF:")
display(sub_train_df.head())

print ("FOLD 1 VAlID DF:")
display(sub_val_df.head())


# ## Helper Functions

# In[ ]:


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_decode_top_to_bot_first(mask_rle, shape):
    """ TBD
    
    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return 
    
    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype = int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0]*shape[1]), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape((shape[1], shape[0]), order = 'F').T

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        #plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Stomach", "Large Bowel", "Small Bowel"]
        plt.legend(handles,labels)
    plt.axis('off')
    
    

class CFG :
    exp_name = "fastai-baseline"
    model_name = "unet"
    backbone = "efficientnet-b1"
    img_size = (256, 256)
    epochs = 5
    wd = 1e-6
    num_classes = 3
    train_bs = 20
    valid_bs = 20
    
def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).cpu().numpy()
        msk = msks[idx,].permute((1, 2, 0)).cpu().numpy()
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


# In[ ]:


device = torch.device("cuda:0")

class masked_ds(Dataset):
    
    def __init__(self, df, subset = "train", transforms = None):
        self.df = df
        self.subset = subset
        self.transforms = transforms
        
        
    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) :
        masks = np.zeros((CFG.img_size[0], CFG.img_size[1], 3), dtype = np.float32)
        img_path = self.df["f_path"].iloc[idx]
        w = self.df["slice_w"].iloc[idx]
        h = self.df["slice_h"].iloc[idx]
        img = self.__load_img(img_path)
        if self.subset == "train" :
            for k,j in zip([0,1,2], ["lb_seg_rle", "sb_seg_rle", "st_seg_rle"]):
                rles = self.df[j].iloc[idx]
                mask = rle_decode(rles, shape = (h,w,1))
                mask = cv2.resize(mask, CFG.img_size)
                masks[:, :, k] = mask
        masks = masks.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)
        if self.subset == "train" : return torch.tensor(img), torch.tensor(masks)
        else : return torch.tensor(img)
    
    def __load_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = (img - img.min())/(img.max() - img.min())*255.0 
        img = cv2.resize(img, CFG.img_size)
        img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
        img = img.astype(np.float32) /255.
        return img
    


img_size = (256, 256)    
data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
                ], p=0.2),

        ], p=1.0),
    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
}


# In[ ]:


train_ds = masked_ds(sub_train_df, transforms = data_transforms["train"])
valid_ds = masked_ds(sub_val_df, transforms = data_transforms["valid"])


# In[ ]:


x,y = train_ds.__getitem__(1)


# In[ ]:


x.shape, y.shape


# In[ ]:


train_dl = DataLoader(train_ds, batch_size = CFG.train_bs, shuffle = True, num_workers = 4,  pin_memory = True)
valid_dl = DataLoader(valid_ds, batch_size = CFG.valid_bs, shuffle = False, num_workers = 4, pin_memory = True)


# In[ ]:


for b, (X_train, y_train) in enumerate(train_dl):
    print (X_train.shape, y_train.shape)
    break


# In[ ]:


imgs, msks = next(iter(valid_dl))
imgs.shape, msks.shape


# In[ ]:


plot_batch(imgs, msks)


# ## Converting pytorch dataloader to FastAI dataloaders

# In[ ]:


dls = DataLoaders(train_dl, valid_dl)


# ## Model Training

# In[ ]:


def build_model():
    model = smp.Unet(encoder_name = CFG.backbone,
                     encoder_weights = "imagenet",
                     in_channels = 3,
                     classes = CFG.num_classes,
                     activation = None)
    
    model.to(device = "cuda")
    return model


# In[ ]:


model = build_model()


# In[ ]:


JaccardLoss = smp.losses.JaccardLoss(mode = "multilabel")
DiceLoss = smp.losses.DiceLoss(mode = "multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

def dice_coef(y_pred, y_true, thr = 0.5, dim = (2,3), epsilon = 0.001):
    y_pred = nn.Sigmoid()(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim = dim)
    den = y_true.sum(dim = dim) + y_pred.sum(dim = dim)
    dice = ((2*inter+epsilon)/(den + epsilon)).mean(dim = (1, 0))
    return dice

def iou_coef(y_pred, y_true, thr = 0.5, dim = (2,3), epsilon = 0.001):
    y_pred = nn.Sigmoid()(y_pred)
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim = dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim = dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim = (1,0))
    return iou

LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)


# In[ ]:


learn = Learner(dls, model, loss_func = criterion, metrics = [dice_coef, iou_coef]) 


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.lr_find()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'learn.fit_one_cycle(CFG.epochs, 1e-3, wd = CFG.wd)\n')


# In[ ]:




