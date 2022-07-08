#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


input_dir = '../input/forest-cover-type-prediction/'
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from copy import deepcopy as dp
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

from torch.nn.modules.loss import _WeightedLoss
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                  pos_weight = pos_weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# In[ ]:


get_ipython().system('pip install fast_ml')
def convert_to_one_hot(y):
    seen = False
    for i in range(len(y)):
        l = [0 for i in range(7)]
        l[y[i] - 1] = 1
        if not seen:
            y_o = np.array(l.copy())
            seen = True
        else:
            y_o = np.vstack([y_o, l.copy()])
    return y_o
    
train = pd.read_csv(input_dir + 'train.csv')
test = pd.read_csv(input_dir + 'test.csv')
train = train.drop("Id", axis = 1)
# x_train = train.drop("Cover_Type", axis = 1)
# y_train = train[train.columns[-1]]
from fast_ml.model_development import train_valid_test_split

x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(train, target = 'Cover_Type', 
                                                                            train_size=0.8, valid_size=0.1, test_size = 0.1)
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
y_train = convert_to_one_hot(y_train)


x_valid = x_valid.to_numpy()
y_valid = y_valid.to_numpy()
y_valid = convert_to_one_hot(y_valid)

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
y_test = convert_to_one_hot(y_test)

print(x_train.shape)
print(y_train.shape)

print(x_valid.shape)
print(y_valid.shape)

print(x_test.shape)
print(y_test.shape)


print(train.shape)
y = train[train.columns[-1]]
y = y.to_numpy()
y = convert_to_one_hot(y)
target_cols = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
train[target_cols] = y
train = train.drop("Cover_Type", axis = 1)
print(train.shape)



# In[ ]:


# HyperParameters
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

target_cols = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
#target_cols = [i for i in range(7)]
num_features=x_train.shape[1]
num_targets=len(target_cols)
hidden_size=4096

tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
tar_weight0_min = dp(np.min(tar_weight0))
tar_weight = tar_weight0_min/tar_weight0
pos_weight = torch.tensor(tar_weight).to(DEVICE)

train_dataset = TrainDataset(x_train, y_train)
valid_dataset = TrainDataset(x_valid, y_valid)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )

model.to(DEVICE)

model.dense3 = nn.utils.weight_norm(nn.Linear(model.cha_po_2, num_targets))
model.to(DEVICE)
    
    
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                          max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

loss_tr = SmoothBCEwLogits(smoothing = 0.001)
loss_va = nn.BCEWithLogitsLoss()    

early_stopping_steps = EARLY_STOPPING_STEPS
early_step = 0

oof = np.zeros((len(train), num_targets))
best_loss = np.inf

mod_name = f"1D_CNN.pth"

for epoch in range(EPOCHS):

    train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
    valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
    print(f"EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

    if valid_loss < best_loss:

        best_loss = valid_loss
        oof = valid_preds
        torch.save(model.state_dict(), mod_name)

    elif(EARLY_STOP == True):

        early_step += 1
        if (early_step >= early_stopping_steps):
            break

#--------------------- PREDICTION---------------------
testdataset = TestDataset(x_test)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

model = Model(
    num_features=num_features,
    num_targets=num_targets,
    hidden_size=hidden_size,
)

model.load_state_dict(torch.load(mod_name))
model.to(DEVICE)

predictions = np.zeros((len(test), len(target_cols)))
predictions = inference_fn(model, testloader, DEVICE)


# In[ ]:


sum = 0
for i in range(len(oof)):
    x = np.argmax(oof[i])
    y = np.argmax(y_valid[i])
    sum += (x == y)
print(sum * 100/len(oof))


# In[ ]:


from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# In[ ]:


from hyperopt import fmin, type, hp
best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print(best)


# In[ ]:


def f(lr, bs):
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 25
    BATCH_SIZE = bs
    LEARNING_RATE = lr
    WEIGHT_DECAY = 1e-5
    NFOLDS = 5
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False

    target_cols = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
    #target_cols = [i for i in range(7)]
    num_features=x_train.shape[1]
    num_targets=len(target_cols)
    hidden_size=4096

    tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
    tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
    tar_weight0_min = dp(np.min(tar_weight0))
    tar_weight = tar_weight0_min/tar_weight0
    pos_weight = torch.tensor(tar_weight).to(DEVICE)

    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
                num_features=num_features,
                num_targets=num_targets,
                hidden_size=hidden_size,
            )

    model.to(DEVICE)

    model.dense3 = nn.utils.weight_norm(nn.Linear(model.cha_po_2, num_targets))
    model.to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = SmoothBCEwLogits(smoothing = 0.001)
    loss_va = nn.BCEWithLogitsLoss()    

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(train), num_targets))
    best_loss = np.inf

    mod_name = f"1D_CNN.pth"

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        print(f"EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:

            best_loss = valid_loss
            oof = valid_preds
            torch.save(model.state_dict(), mod_name)

        elif(EARLY_STOP == True):

            early_step += 1
            if (early_step >= early_stopping_steps):
                break

    return model, train_loss


# In[ ]:


def obj(space):
    model, train_loss = f(space["lr"], space["bs"])
    return train_loss


# In[ ]:


import math
trials = Trials()
space = {"lr" : hp.loguniform("lr", math.exp(-4), 0.5), "bs" : hp.choice("bs", [1024, 2048, 4096, 8192, 512])}
best = fmin(fn = obj, space = space, algo = tpe.suggest, max_evals = 10, trials = trials)


# In[ ]:


best


# In[ ]:




