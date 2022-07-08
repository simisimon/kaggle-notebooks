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


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim

import matplotlib.pyplot as plt

get_ipython().system('pip install -q torchviz')
from torchviz import make_dot

import random
import scipy


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col='id')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv',index_col='id')


# # Introduction
# A simple realization of DNN with PyTorch and GPU acceleration. 
# * V3: add some plots to visualize the model and the learning curve.
# * V4: use all training to finalize models and ensemble them. Add regularization.
# * V5: tune regularization.
# * V6: use k-fold training to finalize models.
# * V7: remove rankdata

# # Feature Engineering
# See AMBROSM's [notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense).

# In[ ]:


for df in [train, test]:
    # Extract the 10 letters of f_27 into individual features
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    # unique_characters feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
    # Feature interactions: create three ternary features
    # Every ternary feature can have the values -1, 0 and +1
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

y = train.target.values
train = train.drop(['f_27', 'target'], axis = 1)
test = test.drop('f_27', axis = 1)
test.head()


# In[ ]:


# Scaling test and train
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)


# # Define DNN
# The DNN structure is inspired by AMBROSM's [notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras).

# In[ ]:


# Definition of a DNN Model class
class DNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(DNN, self).__init__()
        
        
        # Fully Connected Layer
        self.layers = nn.Sequential(nn.Linear(input_size, 64),
                                nn.SiLU(),
                                nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, 16),
                                nn.SiLU(),
                                nn.Linear(16, output_size),
                                nn.Sigmoid()
                               )
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x


# In[ ]:


model_dnn_plot = DNN(train.shape[1])
x_plot = torch.randn(1, train.shape[1])
y_plot = model_dnn_plot(x_plot)
make_dot(y_plot.mean(), params=dict(model_dnn_plot.named_parameters()))


# # Define Functions

# In[ ]:


# Validation function
def validation(model, loader, criterion, device="cpu"):
    model.eval()
    loss = 0
    preds_all = torch.LongTensor()
    labels_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x, labels in loader:
            labels_all = torch.cat((labels_all, labels), dim=0)
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            output = model.forward(batch_x)
            loss += criterion(output,labels).item()
            preds_all = torch.cat((preds_all, output.to("cpu")), dim=0)
    total_loss = loss/len(loader)
    auc_score = roc_auc_score(labels_all, preds_all)
    return total_loss, auc_score


# In[ ]:


# Training function
def train_model(model, trainloader, validloader, criterion, optimizer, 
                scheduler, epochs=20, device="cpu", print_every=1):
    model.to(device)
    best_auc = 0
    best_epoch = 0
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the learning rates in each eporch
    learning_rates = []
    
    for e in range(epochs):
        model.train()
        
        for batch_x, labels in trainloader:
            batch_x, labels = batch_x.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            
            # Training 
            optimizer.zero_grad()
            output = model.forward(batch_x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # at the end of each epoch calculate loss and auc score:
        model.eval()
        train_loss, train_auc = validation(model, trainloader, criterion, device)
        valid_loss, valid_auc = validation(model, validloader, criterion, device)
        
        #### record loss and learning rate
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        learning_rates.append(scheduler._last_lr)
        
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = e
            torch.save(model.state_dict(), "best-state.pt")
        if e % print_every == 0:
            to_print = "Epoch: "+str(e+1)+" of "+str(epochs)
            to_print += ".. Train Loss: {:.4f}".format(train_loss)
            to_print += ".. Valid Loss: {:.4f}".format(valid_loss)
            to_print += ".. Valid AUC: {:.3f}".format(valid_auc)
            print(to_print)
    # After Training:
    model.load_state_dict(torch.load("best-state.pt"))
    to_print = "\nTraining completed. Best state dict is loaded.\n"
    to_print += "Best Valid AUC is: {:.4f} after {} epochs".format(best_auc,best_epoch+1)
    print(to_print)
    return train_losses, valid_losses, learning_rates


# In[ ]:


# Prediction function
def prediction(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    preds_all = torch.LongTensor()
    
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            
            output = model.forward(batch_x).to("cpu")
            preds_all = torch.cat((preds_all, output), dim=0)
    return preds_all


# # Random Ensembling

# In[ ]:


# Checking if GPU is available
if torch.cuda.is_available():
    my_device = "cuda"
    print("GPU is enabled")
else:
    my_device = "cpu"
    print("No GPU :(")


# In[ ]:


input_size = train.shape[1]
test_tensor = torch.tensor(test).float()
test_DL = DataLoader(test_tensor, batch_size=2048)


# In[ ]:


# %%time
# max_learning_rate = 0.01
# epochs = 30
# pred_list = []


# # Prepare Data
# ## Converting train and validation labels into tensors
# X_train_tensor = torch.tensor(train).float()
# y_train_tensor = torch.tensor(y)
# ## Creating train and validation tensors
# train_DS = TensorDataset(X_train_tensor, y_train_tensor)
# ## Defining the dataloaders
# train_DL = DataLoader(train_DS, batch_size=2048, shuffle=True)


# for seed in range(10):
#     print(f"** Seed: {seed+1} ** ........training ...... \n")
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
    
#     # Initialize Model
#     model_dnn = DNN(input_size)
#     # criterion, optimizer, scheduler
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model_dnn.parameters(), lr=max_learning_rate, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
#                                               max_lr = max_learning_rate,
#                                               epochs = epochs,
#                                               steps_per_epoch = len(train_DL),
#                                               pct_start = 0.01,
#                                               anneal_strategy = "cos")
    
#     # Training
#     train_losses, valid_losses, learning_rates = train_model(model = model_dnn,
#                                                              trainloader = train_DL,
#                                                              validloader = train_DL,
#                                                              criterion = criterion,
#                                                              optimizer = optimizer,
#                                                              scheduler = scheduler,
#                                                              epochs = epochs,
#                                                              device = my_device,
#                                                              print_every = round(epochs/2)-1)
    
#     model_dnn.load_state_dict(torch.load("best-state.pt"))
#     pred_test = prediction(model_dnn, test_DL, device=my_device)
#     pred_test_rank = scipy.stats.rankdata(pred_test.tolist())
#     pred_list.append(pred_test_rank)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# K-Fold Cross Validation\nmax_learning_rate = 0.01\nepochs = 30\npred_list = []\n\nkf = KFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train)):\n    print(f"** fold: {fold+1} ** ........training ...... \\n")\n    \n    # Initialize Model\n    model_dnn = DNN(input_size)\n    \n    # Prepare Data\n    X_train, X_valid = train[idx_tr], train[idx_va]\n    y_train, y_valid = y[idx_tr], y[idx_va]\n    \n    ## Converting train and validation labels into tensors\n    X_train_tensor = torch.tensor(X_train).float()\n    X_valid_tensor = torch.tensor(X_valid).float()\n    y_train_tensor = torch.tensor(y_train)\n    y_valid_tensor = torch.tensor(y_valid)\n\n    ## Creating train and validation tensors\n    train_DS = TensorDataset(X_train_tensor, y_train_tensor)\n    valid_DS = TensorDataset(X_valid_tensor, y_valid_tensor)\n\n    ## Defining the dataloaders\n    train_DL = DataLoader(train_DS, batch_size=2048, shuffle=True)\n    valid_DL = DataLoader(valid_DS, batch_size=2048)\n    \n    # criterion, optimizer, scheduler\n    criterion = nn.BCELoss()\n    optimizer = optim.Adam(model_dnn.parameters(), lr=max_learning_rate, weight_decay=1e-4)\n    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,\n                                              max_lr = max_learning_rate,\n                                              epochs = epochs,\n                                              steps_per_epoch = len(train_DL),\n                                              pct_start = 0.01,\n                                              anneal_strategy = "cos")\n    \n    # Training\n    train_losses, valid_losses, learning_rates = train_model(model = model_dnn,\n                                                 trainloader = train_DL,\n                                                 validloader = valid_DL,\n                                                 criterion = criterion,\n                                                 optimizer = optimizer,\n                                                 scheduler = scheduler,\n                                                 epochs = epochs,\n                                                 device = my_device,\n                                                 print_every = round(epochs/2)-1)\n#     break # test\n    model_dnn.load_state_dict(torch.load("best-state.pt"))\n    pred_test = prediction(model_dnn, test_DL, device=my_device)\n    pred_list.append(pred_test.tolist())\n#     pred_test_rank = scipy.stats.rankdata(pred_test.tolist())\n#     pred_list.append(pred_test_rank)\n    \n')


# In[ ]:


fig, ax1 = plt.subplots()
ax1.plot(range(epochs), train_losses, label='Train Loss')
ax1.plot(range(epochs), valid_losses, label='Valid Loss')
ax1.set_title('Learning Curve')
ax1.set_xlabel("Number of Epochs")
ax1.set_ylabel("Loss")
ax2 = ax1.twinx()
ax2.set_ylabel("Learning Rate")
ax2.plot(range(epochs), learning_rates, label='Learning Rate', color='g')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.show()


# # Submission

# In[ ]:


submission = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv")
submission['target'] = np.array(pred_list).mean(axis=0)
submission.to_csv('submission.csv', index=False)
submission

