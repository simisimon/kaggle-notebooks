#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


get_ipython().system('pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')


# In[ ]:


import os
import pickle
import numpy as np
import torch
import torchvision
from skimage import io
import time
import copy


from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import datasets, models, transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.io import read_image

from matplotlib import colors, pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# в sklearn не все гладко, чтобы в colab удобно выводить картинки 
# мы будем игнорировать warnings
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# # Global variables

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
SEED = 42


# # Data preparation

# ## Reading data

# Let's create two lists (for training and test data), in which we will store the paths to the pictures

# In[ ]:


TRAIN_DIR = Path('../input/journey-springfield/train')
TEST_DIR = Path('../input/journey-springfield/testset')
train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
len(train_val_files), len(test_files)


# We can access the class labels for the training data using the following method

# In[ ]:


train_val_files[0].parent.name


# ## Splitting data

# In[ ]:


from sklearn.model_selection import train_test_split

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.20, stratify=train_val_labels, random_state=SEED)
train_labels_raw, val_labels_raw = [path.parent.name for path in train_files], [path.parent.name for path in val_files]


# Immediately encode the class labels

# In[ ]:


label_encoder = LabelEncoder()
label_encoder.fit(train_labels_raw)

train_labels = label_encoder.transform(train_labels_raw)
val_labels = label_encoder.transform(val_labels_raw)
test_labels = None

assert len(train_labels) == len(train_files)

class2label = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label2class = {v:k for k,v in class2label.items()}


# ## Transformations

# In[ ]:


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
}


# ## Hyperparameters

# In[ ]:


batch_size = 64
epochs = 10


# ## Dataset

# It would be convenient to use the `datasets.ImageFolder` class, but we have data for training and validation in the same folder and it is not separated in advance, so we have to make it ourselves

# In[ ]:


class SimpsonsDataset(Dataset):
    def __init__(self, file_names, img_labels, mode, transform=None, target_transform=None):
        self.file_names = file_names
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        img_path = str(self.file_names[idx])
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'test':
            return image
        else: 
            label = self.img_labels[idx]
        
            if self.target_transform:
                label = self.target_transform(label)
            
            return image, label


# In[ ]:


train_dataset = SimpsonsDataset(train_files, train_labels, "train", data_transforms["train"])
val_dataset = SimpsonsDataset(val_files, val_labels, "val", data_transforms["val"])
test_dataset = SimpsonsDataset(test_files, test_labels, "test", data_transforms["test"])
image_datasets = {"train":train_dataset, "val":val_dataset}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# ## Dataloaders

# In[ ]:


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


dataloaders = {}
dataloaders["train"] = train_dataloader
dataloaders["val"] = val_dataloader


# Let's look at some examples from the training dataloader

# In[ ]:


def imshow(inp, title=None, plt_ax=plt):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    
# Get a batch of training data
inputs, classes = next(iter(train_dataloader))
titles = [label2class[x] for x in classes.detach().numpy()]

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 8), sharey=True)

fig.tight_layout(pad=4.0)

i = 0
for ax in axs.flatten():
    imshow(inputs[i].cpu(), title=titles[i], plt_ax=ax)
    i += 1


# # Training the model

# In[ ]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm_notebook(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step(running_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# We'll be using pretrained convnext_base model, for which we will finetune only the layer for classification

# In[ ]:


from functools import partial
from torch.nn import functional as F

model_conv = torchvision.models.vgg16_bn(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_classes = len(class2label)
model_conv.classifier = nn.Linear(512 * 7 * 7, num_classes)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = torch.optim.AdamW(model_conv.parameters(), lr=1e-3, amsgrad=True)

exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', patience=3)


# In[ ]:


model_ft = train_model(model_conv, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)


# In[ ]:


def predict(model, test_loader):
    with torch.no_grad():
        logits = []
    
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


# In[ ]:


probs = predict(model_ft, test_dataloader)

preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))


# In[ ]:


test_filenames = [path.name for path in test_dataset.file_names]


# In[ ]:


import pandas as pd
submission = pd.read_csv('/kaggle/input/journey-springfield/sample_submission.csv')
submission = pd.DataFrame({'Id': test_filenames, 'Expected': preds}).sort_values('Id')
submission.to_csv('./submission.csv', index=False)


# In[ ]:




