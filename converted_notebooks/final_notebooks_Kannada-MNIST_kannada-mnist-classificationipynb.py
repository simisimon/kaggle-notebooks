#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dependencies
import PIL
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import math
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

import os
import zipfile
from copy import deepcopy

# progress bar
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import pandas as pd


# In[ ]:


train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')
print(len(train_df))
train_df


# In[ ]:


sns.countplot(x='label', data=train_df)


# # Display image from csv
# 
# Colormap: https://matplotlib.org/stable/gallery/color/colormap_reference.html
# 
# Just see images. any of transforming be not yet.

# In[ ]:


def array_to_image_tensor(array):
    return np.array(array).reshape(28, 28, 1).astype(np.int32)


# In[ ]:


fig, ax = plt.subplots(3, 10, figsize=(10, 4))
fig.suptitle('Labels / Images')
ax = ax.ravel()
for i in range(30):
    raw = np.array(train_df.iloc[i])
    label, img = raw[0], array_to_image_tensor(raw[1:])
    ax[i].imshow(img, cmap='gist_gray')
    ax[i].axis("off")
    ax[i].set_title(str(label))
plt.subplots_adjust(hspace=0.2)


# In[ ]:


mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


# # Put into Dataset and see it

# Make custom Dataset for load binary pixels from dataframe, and convert it to RGB pixels with size (3, 28, 28)
# 
# Because of memory, we can not have all pixels after load from dataframe. So, just do transform only when get item

# In[ ]:


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.targets = np.array(df['label']).reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(self.df.iloc[idx][1:]).reshape(28, 28) # (28, 28)
        image = image.astype(np.float32)
        label = self.df.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def gettargets(self):
        return self.targets


# In dataset, the size of image is `(28, 28)`. so we needs to transform it to resize to `(224, 224)` for VGG and also RGB channels.
# 
# `bin_to_rgb(t)` makes binary image to RGB image with `(3, 28, 28)`. now it can be PIL Image.

# In[ ]:


def bin_to_rgb(t):
    h, w = t.shape
    img = np.array([np.array(t).reshape(1, h, w)] * 3).reshape(3, h, w).astype(np.float32)
    return torch.tensor(img)

train_set = DigitDataset(train_df, transform=transforms.Compose([
    bin_to_rgb,
    normalize,
]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)

images, labels = next(iter(train_loader))
to_pil_image(images[0]).resize((240, 240), PIL.Image.NEAREST)


# In[ ]:


rows = 4
cols = 8
images, labels = next(iter(train_loader))

fig, ax = plt.subplots(rows, cols, figsize=(12, 6))
fig.suptitle('from dataset')
ax = ax.ravel()
for i in range(rows * cols):
    image = images[i]
    label = labels[i].numpy().item()
    ax[i].imshow(np.asarray(to_pil_image(image)))
    ax[i].axis("off")
    ax[i].set_title(label)
plt.subplots_adjust(hspace=0.5)


# In[ ]:


# CPU/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# In[ ]:


class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),         # (128, 14, 14)
            
            nn.Conv2d(128, 256, 3, padding=1), # (128, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), # (256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (256, 7, 7)
            
            nn.Conv2d(256, 512, 3),            # (512, 5, 5)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),            # (512, 3, 3)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),         # (512, 2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = MyConvNet().to(device)

print("Params to learn:")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=0.00002)

print(model)


# In[ ]:


from datetime import datetime, timedelta
start_time = datetime.now()
print('start at:', start_time)


# In[ ]:


EPOCHES = 50
fold_count = 3
batch_size = 64

kfold = StratifiedKFold(n_splits=fold_count, shuffle=True)

print(f'Epoches = {EPOCHES}, Fold = {fold_count}, batch size = {batch_size}')

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

best = {
    'epoch': 0,
    'train_loss': 1e9,
    'valid_loss': 1e9,
    'state': {},
    'cm': confusion_matrix([], [], labels=range(10)),
}

def train_valid_diff(epoch):
    if epoch < 0: return math.inf
    return abs(train_losses[epoch] - valid_losses[epoch])

model.to(device)

# Run apoch
epoch = 0
early_stop = None
while epoch < EPOCHES:
    # K-Fold cross validation
    splited_folds = kfold.split(train_set, train_set.targets)
    for fold, (train_idx, valid_idx) in enumerate(splited_folds):
        pbar = tqdm_nb(total=len(train_idx)+len(valid_idx)/2, desc=f'{epoch+1}/{EPOCHES} epoch')
        # Split dataset and loader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)
        # Use as train/valid set from train data set by k-fold
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_subsampler)

        running_loss = 0.0
        running_acc = []

        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs.data, dim=1).to(device)

            running_loss += loss.item()
            running_acc.append(torch.sum(pred == labels).to('cpu') / len(labels))

            pbar.update(len(labels))

        # train loss (average)
        train_loss = running_loss / len(train_loader)
        train_acc = np.array(running_acc).mean()
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model_state = deepcopy(model.state_dict())

        # valid loss (just for check)
        model.eval()
        with torch.no_grad():
            valid_loss_sum = 0
            valid_acc = []
            valid_cm = confusion_matrix([], [], labels=range(10))
            v_cnt = len(valid_loader) * 0.5 # partial valid set
            for i, (images, labels) in enumerate(valid_loader):
                if i >= v_cnt: break
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)
                pred = torch.argmax(outputs.data, dim=1)
                pred = pred.to(device)
                last_valid_result = (images, labels, pred)
                valid_loss_sum += criterion(outputs, labels).item()
                valid_acc.append(torch.sum(pred == labels).to('cpu') / len(labels))
                cm = confusion_matrix(labels.to('cpu'), pred.to('cpu'), labels=range(10))
                valid_cm = np.add(valid_cm, cm)
                pbar.update(len(labels))
            valid_loss = valid_loss_sum / v_cnt
            valid_acc_avg = np.array(valid_acc).mean()
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc_avg)

        print('[%d] fold=%d, train loss: %.6f (%.3f %%), valid loss: %.6f (%.3f %%)' % \
              (epoch + 1, fold, train_loss, train_acc * 100, valid_loss, valid_acc_avg * 100))

        # get best
        if valid_loss < best['valid_loss']:
            best = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'state': model_state,
                'cm': valid_cm,
            }

        pbar.close()

        epoch += 1
        if epoch >= EPOCHES: break
        if epoch < 3: continue

        tl = np.array(train_losses[-3:]).mean()
        vl = np.array(valid_losses[-3:]).mean()
        minl, maxl = min(vl, tl), max(vl, tl)

        # Early stop conditions
        if minl > 0.5 and 2 * minl < maxl and 2 * train_valid_diff(epoch-1) < train_valid_diff(epoch):
            early_stop = 'Train and valid have distance by overfitting'
            break
        if np.array([tl, vl]).mean() < 0.0005:
            early_stop = 'Trained well enough'
            break
        if start_time + timedelta(hours=2) < datetime.now():
            early_stop = 'Too much time'
            break
    if early_stop != None:
        print('Early Stop -', early_stop)
        break


# In[ ]:


# save model
SAVE_BEST_PATH = './best_parameters.pth'
torch.save(best['state'], SAVE_BEST_PATH)
best_ = deepcopy(best)
best_.pop('state')
best_.pop('cm')
print(best_)


# In[ ]:


# Draw chart
plt.plot(train_losses, label="train loss")
plt.plot(valid_losses, label="valid loss")
plt.title("Loss")
plt.xlabel("epoch")
plt.axvline(best['epoch'], color='red', linestyle=':')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()
plt.show()


# In[ ]:


# Draw chart
plt.plot(train_accs, label="train acc")
plt.plot(valid_accs, label="valid acc")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.axvline(best['epoch'], color='red', linestyle=':')
plt.axhline(1.0, color='gray', linestyle=':')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="train loss")
plt.title("Train Loss")
plt.xlabel("epoch")
plt.axvline(best['epoch'], color='red', linestyle=':')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(valid_losses, label="valid loss")
plt.title("Valid Loss")
plt.xlabel("epoch")
plt.axvline(best['epoch'], color='red', linestyle=':')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()

plt.show()


# In[ ]:


images, labels, preds = last_valid_result
plt.figure(figsize=(15, 10))
for index in range(5 * 5):
    plt.subplot(5, 5, index + 1)
    # image = unnormalize(images[index])
    image = images[index]
    image = np.asarray(to_pil_image(image.to('cpu')))
    label = labels[index].to('cpu').numpy().item()
    guess = preds[index].to('cpu').numpy().item()
    plt.title('{} [L:{}]'.format(guess, label))
    plt.imshow(image)
    plt.axis("off")
plt.show()


# In[ ]:


sns.heatmap(best['cm'], annot=True, cmap='Greens')


# # Now, prepare to submit
# 
# According document of competition, submission file format is:
# ```
# ImageId,Label
# 1,0
# 2,0
# 3,0
# ...
# ```
# 
# so, predict result by trained model and write it down.

# In[ ]:


class SubmitDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(self.df.iloc[idx][1:]).reshape(28, 28) # (28, 28)
        if self.transform:
            image = self.transform(image)
        return image, idx


# In[ ]:


test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')
test_set = SubmitDataset(test_df, transform=transforms.Compose([
    bin_to_rgb,
    normalize,
]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
test_df


# In[ ]:


rows = 4
cols = 8
images, indices = next(iter(test_loader))

fig, ax = plt.subplots(rows, cols, figsize=(12, 6))
fig.suptitle('test set')
ax = ax.ravel()
for i in range(rows * cols):
    image = images[i]
    index = indices[i]
    ax[i].imshow(np.asarray(to_pil_image(image)))
    ax[i].set_title(f'id={index}')
    ax[i].axis("off")
plt.subplots_adjust(hspace=0.5)


# Load model which getting the best during train according by accuracy.

# In[ ]:


best_model = MyConvNet().to(device)
best_model.load_state_dict(torch.load(SAVE_BEST_PATH))


# In[ ]:


result_df = pd.DataFrame(columns=['id', 'label'])

best_model.eval()
with torch.no_grad():
    for images, indices in tqdm_nb(test_loader):
        images, indices = images.to(device), indices.to(device)
        outputs = best_model(images).to(device)
        pred = torch.argmax(outputs.data, dim=1)
        last_test_result = (images, pred.to(device))
        
        indices = indices.to('cpu').numpy()
        pred = pred.to('cpu').numpy()
        pred_df = pd.DataFrame({'id': indices, 'label': pred})
        result_df = result_df.append(pred_df)

result_df.to_csv('submission.csv', index=False)

result_df


# In[ ]:


images, preds = last_test_result

rows = 2
cols = len(images) // rows
fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
ax = ax.ravel()
for i in range(len(images)):
    # image = unnormalize(images[i])
    image = images[i]
    image = np.asarray(to_pil_image(image.to('cpu')))
    guess = preds[i].to('cpu').numpy().item()
    ax[i].imshow(image)
    ax[i].axis("off")
    ax[i].set_title(guess)
plt.subplots_adjust(hspace=0.2)
plt.show()

