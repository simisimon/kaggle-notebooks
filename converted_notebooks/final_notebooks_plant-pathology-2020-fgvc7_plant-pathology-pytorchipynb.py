#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from PIL import Image
import matplotlib.pyplot as plt
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets, models
from tqdm import tqdm
import torchvision
from pytorch_lightning.metrics.functional import accuracy
from collections import defaultdict
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import time
import os
import copy
import cv2
import glob
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn as cudnn


# In[ ]:


#read Data
PATH='../input/plant-pathology-2020-fgvc7/'
train_data=pd.read_csv(PATH+'train.csv')
test_data=pd.read_csv(PATH+'test.csv')
train_data.head()


# In[ ]:


# image path  as column 
def image_(dat):
    return PATH+'images/'+dat+'.jpg'

train_data['image_path']=train_data[["image_id"]].apply(image_)
test_data['image_path']=test_data[["image_id"]].apply(image_)


# In[ ]:


#apply albumenation for transformation
train_transform =  A.Compose([
 A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=128, width=128),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
Val_transform=transforms.Compose([
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


# In[ ]:


#splitting data as train and test 
validation_split = .3
random_seed= 42
shuffle_dataset = True

# tr, val = train_test_split(data.label, stratify=data.label, test_size=0.1)
dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# In[ ]:


from torch.utils.data.sampler import SubsetRandomSampler

# Creating  data samplers:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


# In[ ]:


names=train_data.iloc[:,1:-1].columns


# In[ ]:


class plant_data(Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image=cv2.imread(self.data.loc[index,'image_path'])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label=torch.tensor(self.data.loc[index,names])
        if self.transform is not None:
              image = self.transform(image=image)["image"]
    

        return image,label


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)


# In[ ]:


tran_dataset = plant_data(train_data,transform=train_transform)


# In[ ]:


train_loader = torch.utils.data.DataLoader(tran_dataset, batch_size=4,  sampler=train_sampler)
valid_loader= torch.utils.data.DataLoader(tran_dataset, batch_size=4,  sampler=valid_sampler)


# In[ ]:


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()  


# In[ ]:


random.seed(42)
visualize_augmentations(tran_dataset)


# In[ ]:


"""def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=1), (output.size(0)))"""


# In[ ]:


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# In[ ]:


class DenseCrossEntropy(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,logits,labels):
        logits = logits.float()
        labels = labels.float()
        
        logprobs = F.log_softmax(logits,dim=1)
        
        loss =-labels*logprobs
        loss = loss.sum(-1)
        
        return loss.mean()


# In[ ]:


"""params = {
    "model": "resnet50",
    "device": "cuda",
    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 4,
    "epochs": 10,
}"""


# In[ ]:


def train(train_loader, model, criterion, optimizer, epoch):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target.type_as(output))
        #acc = calculate_accuracy(output, target.type_as(output))
        score=f1_score(target.data.to('cpu'), output.data.to('cpu') > 0.5, average="samples")
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("F1 score", score.item())
        #metric_monitor.update("Accuracy", acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train: {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor,))
        


# In[ ]:


def validate(val_loader, model, criterion, epoch):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            
            loss = criterion(output, target.type_as(output))
            #acc = calculate_accuracy(output, target.type_as(output))
            score=f1_score(target.data.to('cpu'), output.data.to('cpu') > 0.5, average="samples")
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("F1 score", score.item())
            #metric_monitor.update("Accuracy", acc)
            stream.set_description(
            "Epoch: {epoch}. Val: {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor,))


# In[ ]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,512),
                        nn.ReLU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(512,4))

model_ft = model_ft.to(device)

criterion =  nn.MultiLabelSoftMarginLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


for epoch in range(1, 30 + 1):
    train(train_loader, model_ft, criterion, optimizer_ft, epoch)
    validate(valid_loader, model_ft, criterion, epoch)


# In[ ]:


class plant_test_data(Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        image=cv2.imread(self.data.loc[index,'image_path'])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
              image =self.transform(image=image)["image"]
    

        return image
    
test_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160),
        A.CenterCrop(height=128, width=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


# In[ ]:


test_dataset = plant_test_data(test_data,test_transform)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False, pin_memory=True)     


# In[ ]:


model = model_ft.eval()
predicted_labels = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        #predictions = (torch.sigmoid(output) >= 0.5)[:, 0].cpu().numpy()
        predicted_labels.append(torch.sigmoid(output).cpu().numpy())


# In[ ]:


samp=pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")


# In[ ]:


final=pd.concat([samp,pd.DataFrame(np.array(predicted_labels).reshape(-1,4),columns=names)],axis=1)
final.reset_index(drop=True, inplace=True)


# In[ ]:


final.to_csv('sample_submission.csv',index=False)


# In[ ]:




