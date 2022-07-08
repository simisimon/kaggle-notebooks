#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[ ]:


from PIL import Image

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split


# ## Build DataLoader

# In[ ]:


transform=transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor(),
])


# In[ ]:


dataset = ImageFolder(root="/kaggle/input/devkor-image-classification/train",
                      transform=transform)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)


# ## Build NN

# In[ ]:


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        # x -> [128, 1, 28, 28] -> [128, 784]
        out = self.layer(x.flatten(1))
        # out -> [128, 10]
        return out


# ## Train

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[+] Train with {device}")

model = NN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


print("[+] Train Start")
total_epochs = 1

for epoch in range(total_epochs):
    train_l = []
    val_l = []
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_l.append(loss.detach().cpu())
        
    total = 0
    correct = 0
    for x, y in val_dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        correct += (y_pred.argmax(dim=1) == y).sum().detach().cpu().item()
        total += len(y)
        
        val_l.append(loss.detach().cpu())
    print(f"Epoch: {epoch}, Train Loss: {np.array(train_l).mean():.3f}, Val Loss: {np.array(val_l).mean():.3f}, Val Accuracy: {correct / total:.3f}")


# ## Inference

# In[ ]:


BASE_DIR = "/kaggle/input/devkor-image-classification/test"

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.file_list = [os.path.join(BASE_DIR, f"{str(i).zfill(4)}.png") for i in range(10000)]
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        return transform(img)

test_dataset = TestDataset()
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# In[ ]:


pred = []
for x in test_dataloader:
    pred += model(x.to(device)).detach().cpu().argmax(dim=1).tolist()
print(len(pred))


# ## Submission

# In[ ]:


submission = pd.read_csv("/kaggle/input/devkor-image-classification/sample_submission.csv")
submission.head()


# In[ ]:


submission.loc[:, "label"] = pred
submission.to_csv("result.csv", index=False)
submission.head()

