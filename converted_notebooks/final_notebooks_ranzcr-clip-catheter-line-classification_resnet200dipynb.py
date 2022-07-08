#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import timm


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
path = '../input/ranzcr-clip-catheter-line-classification/'
train_labels = pd.read_csv(path+'train.csv')
import timm
import cv2
from tqdm.notebook import tqdm
IMAGE_SIZE = 640
BATCH_SIZE = 128
TEST_PATH = '../input/ranzcr-clip-catheter-line-classification/test'
#export CUDA_VISIBLE_DEVICES=""
from torch.cuda.amp import autocast, GradScaler


# In[ ]:


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


# In[ ]:


if torch.cuda.is_available():
    device= 'cuda'
else:
    device='cpu'
print(device)


# In[ ]:


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


# In[ ]:


def get_transforms():
        return Compose([
            Resize(IMAGE_SIZE, IMAGE_SIZE),
            Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225],
     ),
    ToTensorV2()
        ])


# In[ ]:


class ResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, 11)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output


# In[ ]:


def inference(models, test_loader, device):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for model in models:
            with torch.no_grad():
                y_preds1 = model(images)
                y_preds2 = model(images.flip(-1))
            y_preds = (y_preds1.sigmoid().to('cpu').numpy() + y_preds2.sigmoid().to('cpu').numpy()) / 2
            avg_preds.append(y_preds)
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


# In[ ]:


MODEL_PATH = '../input/resnet200d-baseline-benchmark-public/resnet200d_fold4_cv954.pth'

MODEL_PATH957 = '../input/resnet200d-baseline-benchmark-public/resnet200d_fold3_cv957.pth'


# In[ ]:


model = ResNet200D() 
#model = nn.DataParallel(model)
model.load_state_dict(torch.load(MODEL_PATH,map_location=map_location),strict=False)
model.eval()
models = [model.to(device)]
model957 = ResNet200D() 
#model = nn.DataParallel(model)
model957.load_state_dict(torch.load(MODEL_PATH957,map_location=map_location),strict=False)
model957.eval()
models957 = [model957.to(device)]


# In[ ]:


test = pd.read_csv('../input/ranzcr-clip-catheter-line-classification/sample_submission.csv')
test_dataset = TestDataset(test, transform=get_transforms())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4 , pin_memory=True)
predictions = inference(models, test_loader, device)
predictions957 = inference(models957, test_loader, device)


# In[ ]:


#predictions = .50 *predictions + .50 * predictions
target_cols = test.iloc[:, 1:12].columns.tolist()
test[target_cols] =(predictions  + predictions957)/2
test[['StudyInstanceUID'] + target_cols].to_csv('submission.csv', index=False)
test.head()

