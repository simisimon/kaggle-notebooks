#!/usr/bin/env python
# coding: utf-8

# # Import party!!

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from glob import glob
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import pytorch_lightning as pl
from torch.utils.data import Dataset , DataLoader
from sklearn.model_selection import train_test_split


# # Simple and basic file visualisation

# In[ ]:


def read_img(path):
    img= torchvision.io.read_image(path)
    return img


def plot(df,r=8,c=8,figsize=(20,20),train_files=True):
    
    _,axs=plt.subplots(r,c,figsize=figsize)
    axs=axs.flatten()
    
    for n, ax in enumerate(axs):
        
        img= read_img(df.image[n])
        
        if train_files:
            label=df.species[n] 
            ax.set_title(label)
        
        ax.imshow(F.to_pil_image(img))
        ax.axis('off')
        
    plt.show()
    plt.tight_layout()
    
    
def display_(path,base_folder_path):
    df=pd.read_csv(path)
    df.image=base_folder_path+'/'+df.image
    display(df)
    print(df.info())
    for col in df.columns:
        print(col ,'         ' , df[col].nunique())
    
        
    return df

    
    
def main():
    train_folder_path='../input/happy-whale-and-dolphin/train_images'
    test_folder_path='../input/happy-whale-and-dolphin/test_images'

    train_df_path='../input/happy-whale-and-dolphin/train.csv'
    submission_df_path='../input/happy-whale-and-dolphin/sample_submission.csv'
    
    print('training files ')
    train_df=display_(train_df_path,train_folder_path)
    plot(train_df)
    
    print('test files ')
    submission_df=display_(submission_df_path,test_folder_path)
    plot(submission_df,train_files=False)
    
    
    
main()


# # Pipeline 

# In[ ]:


class pipeline_basic(Dataset):
    
    def __init__(self,df):
        self.df=df
        
    def __len__(self):
        return len(self.df)
    
    def read_img(self,path):
        
        img= torchvision.io.read_image(path)
        img=img/255.0
        
        return img
        
    def __getitem__(self,idx):
        
        x=self.read_img(self.df.image[idx])
        y=self.df.species[idx]
        
        return x,y

    
class PL_dataset(pl.LightningDataModule):
    
    def __init__(
                    self,
                    df,
                    test_df,
                    Dataset,
                    batch_size=32,
                    num_workers=1
                ):
        self.Dataset=Dataset
        self.df=df
        self.test_df=test_df
        self.batch_size=batch_size
        self.num_workers=num_workers
        
    def setup(self):
        
        self.train_df,self.val_df=train_test_split(self.df)    
        
    def training_dataloader(self):
        
        return Dataloader(Dataset(self.train_df),batch_size=self.batch_size)
    
    def validation_dataloader(self):
        
        return Dataloader(Dataset(self.val_df),batch_size=self.batch_size)
    
    def test_dataloader(self):
        
        return Dataloader(Dataset(self.test_df),batch_size=self.batch_size)

        
def display_(path,base_folder_path):
    df=pd.read_csv(path)
    df.image=base_folder_path+'/'+df.image
    return df


def plot_pipeline(dataset,r=8,c=8,figsize=(20,20)):
    
    _,axs=plt.subplots(r,c,figsize=figsize)
    axs=axs.flatten()
    
    for n, ax in enumerate(axs):
        
        img,label= dataset[n]
        ax.set_title(label)
        ax.imshow(F.to_pil_image(img))
        ax.axis('off')
        
    plt.show()
    plt.tight_layout()

def main():
    
    train_folder_path='../input/happy-whale-and-dolphin/train_images'
    test_folder_path='../input/happy-whale-and-dolphin/test_images'

    train_df_path='../input/happy-whale-and-dolphin/train.csv'
    submission_df_path='../input/happy-whale-and-dolphin/sample_submission.csv'
    
    train_df=display_(train_df_path,train_folder_path)
    
    submission_df=display_(submission_df_path,test_folder_path)
    
    print('testing basic pipeline')
    Da=pipeline_basic(train_df)
    plot_pipeline(Da)
    

    
main()    


# # Model

# In[ ]:


# good scope to improve with different type of heads, backbones , 2 optimizer method etc.
# basic model
class Model(pl.LightningModule):
    def __init__(
                 self,
                 model,
#                  img_size,
    ):
        super().__init__()
        self.base_model=model()
        self.linear_head=torch.nn.Linear(in_features=1000,out_features=30)
        self.relu=torch.nn.Softmax(dim=1)
        
        
    def forward(self,x):
        out=self.base_model(x)
        out=self.linear_head(out)
        out=self.relu(out)
        
        return out
    

# testing if model works. Easy pisy unit test
def main():
    
    train_folder_path='../input/happy-whale-and-dolphin/train_images'
    test_folder_path='../input/happy-whale-and-dolphin/test_images'

    train_df_path='../input/happy-whale-and-dolphin/train.csv'
    submission_df_path='../input/happy-whale-and-dolphin/sample_submission.csv'
    
    train_df=display_(train_df_path,train_folder_path)
    
    submission_df=display_(submission_df_path,test_folder_path)
    
    Da=pipeline_basic(train_df)
    model=Model(torchvision.models.resnet18)
    img,label=Da[3]
    preds=model(img[None,:])                          # model demands 4 D tensor (B,C,H,W)
    print(len(preds[0]),preds)
    

    
main()    
        
        


# In[ ]:




