#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as dataloader
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm
import os

get_ipython().system('pip install bs4')
from glob import glob
from bs4 import BeautifulSoup
import cv2


# In[ ]:


imgs_dir = list(sorted(glob('../input/face-mask-detection/images/*.png')))
labels_dir = list(sorted(glob("../input/face-mask-detection/annotations/*.xml")))


# In[ ]:


class dataset(Dataset) :
    def __init__(self, imgs, labels) :
        self.imgs = imgs
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        
    def __len__(self) :
        return len(self.imgs)
    
    def __getitem__(self, index) :
        x = cv2.imread(self.imgs[index])
        x = self.transform(x).to(self.device)
        
        y = dict()
        with open(self.labels[index]) as f :
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            data = soup.find_all('object')
            
            box = []
            label = []
            for obj in data :
                xmin = int(obj.find('xmin').text)
                ymin = int(obj.find('ymin').text)
                xmax = int(obj.find('xmax').text)
                ymax = int(obj.find('ymax').text)
                
                label_ = 0
                if obj.find('name').text == 'with_mask' :
                    label_ = 1
                elif obj.find('name').text == 'mask_weared_incorrect' :
                    label_ = 2
                
                box.append([xmin, ymin, xmax, ymax])
                label.append(label_)
                
            box = torch.FloatTensor(box)
            label = torch.IntTensor(label)
            
            y['image_id'] = torch.FloatTensor([index]).to(device)
            y["boxes"] = box.to(device)
            y["labels"] = torch.as_tensor(label, dtype=torch.int64)
            
        return x, y
    
def collate_fn(batch) : return tuple(zip(*batch))


# In[ ]:


mydataset = dataset(imgs_dir, labels_dir)

train_size=int(len(mydataset)*0.7)
test_size=len(mydataset)-train_size
print('Length of dataset is', len(mydataset), '\nLength of training set is :',train_size,'\nLength of test set is :', test_size)

trainset,testset=torch.utils.data.random_split(mydataset,[train_size,test_size])

train_data = DataLoader(trainset, batch_size = 3, shuffle = True,
                       collate_fn = collate_fn)
test_data = DataLoader(testset, batch_size = 4, shuffle = True,
                       collate_fn = collate_fn)


# In[ ]:


#load pre-trained model
def get_model(output_shape) :
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, output_shape)
    return model

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = get_model(3).to(device)
print(model)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = 1e-5, weight_decay = 0.001)

num_epoch = 60


# In[ ]:


def train(model, device, loader, optimizer):
    
    #initialise counters
    epoch_loss = 0
    
    #Set Network in train mode
    model.train()
    
    for imgs, annotations in tqdm(loader):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        predict = model(imgs, annotations)
        losses = sum(loss for loss in predict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        epoch_loss += losses
        
    epoch_loss /= len(loader)

    print(epoch+1, '/', num_epoch, ' : {:.5f}'.format(epoch_loss))

    #return the average loss from the epoch as well as the logger array       
    return epoch_loss


# In[ ]:


#Log the training losses
training_loss_logger = []

#This cell implements our training loop

#Record the start time
Start_time = time.time()

for epoch in range(num_epoch):

    #call the training function and pass training dataloader etc
    train_loss = train(model, device, train_data, optimizer)

    training_loss_logger.append(train_loss)           

End_time = time.time()

print("Training time %.2f seconds" %(End_time - Start_time))


# In[ ]:


def plot_img(img, predict, annotation) :
    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    img = img.cpu().data
    
    ax[0].imshow(img.permute(1, 2, 0)) #rgb, w, h => w, h, rgb
    ax[1].imshow(img.permute(1, 2, 0))
    ax[0].set_title("real")
    ax[1].set_title("predict")
    
    for box in annotation["boxes"] :
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(rect)
        
    for box in predict["boxes"] :
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        ax[1].add_patch(rect)
    plt.show()


# In[ ]:


model.eval()

with torch.no_grad() :
    for imgs, annotations in test_data:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        preds = model(imgs)
        
        for i in range(3) :
            plot_img(imgs[i], preds[i], annotations[i])
        
        break

