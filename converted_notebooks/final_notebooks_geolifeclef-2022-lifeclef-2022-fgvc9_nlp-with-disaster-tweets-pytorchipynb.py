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


import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import *
import wandb
import nltk
from nltk.stem.porter import *
PROJECT_NAME = "Natural-Language-Processing-with-Disaster-Tweets"
np.random.seed(55)
stemmer = PorterStemmer()
device = 'cpu'


# In[ ]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())


# In[ ]:


tokenize("Testing this tokenize function")


# In[ ]:


def stem(word):
    return stemmer.stem(word.lower())


# In[ ]:


stem('organic')


# In[ ]:


def words_to_int(words,all_words):
    new_words = []
    for word in words:
        new_words.append(stem(word))
    list_of_os = np.zeros(len(all_words))
    for i in range(len(all_words)):
        if all_words[i] in new_words:
            list_of_os[i] = 1.0
    return list_of_os


# In[ ]:


words_to_int(["test"],["testing","I","test","grswgre"])


# In[ ]:


test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data = data.sample(frac=1)
data = data.sample(frac=1)
data = data.sample(frac=1)


# In[ ]:


X = data['text']
y = data['target']


# In[ ]:


all_words = []
tags = []


# In[ ]:


from tqdm import tqdm


# In[ ]:


for x_iter,y_iter in tqdm(zip(X,y)):
    x_iter = tokenize(x_iter)
    new_x_iter = []
    for x_iter_i in x_iter:
        new_x_iter.append(stem(x_iter_i))
    all_words.extend(new_x_iter)
    tags.append(y_iter)


# In[ ]:


np.random.shuffle(all_words)
np.random.shuffle(all_words)


# In[ ]:


all_words = sorted(set(all_words))


# In[ ]:


tags = sorted(set(tags))


# In[ ]:


new_X = []
new_y = []


# In[ ]:


for X_iter,y_iter in tqdm(zip(X,y)):
    new_X.append(words_to_int(X_iter,all_words))
    new_y.append(tags.index(y_iter))


# In[ ]:


X = np.array(new_X)
y = np.array(new_y)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.0625,shuffle=True)


# In[ ]:


X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)


# In[ ]:


class Model(Module):
    def __init__(self,activation=ReLU,neurons=512):
        super().__init__()
        self.activation = activation()
        self.output_activation = Sigmoid()
        self.linear1 = Linear(len(all_words),neurons)
        self.linear2 = Linear(neurons,neurons*2)
        self.linearbn = BatchNorm1d(neurons*2)
        self.linear3 = Linear(neurons*2,neurons*2)
        self.linear4 = Linear(neurons*2,neurons*3)
        self.linear5 = Linear(neurons*3,neurons*2)
        self.output = Linear(neurons*2,1)
    
    def forward(self,X):
        preds = self.activation(self.linear1(X))
        preds = self.activation(self.linear2(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linearbn(preds))
        preds = self.activation(self.linear3(preds))
        preds = self.activation(self.linear4(preds))
        preds = self.activation(self.linear5(preds))
        preds = self.output_activation(self.output(preds))
        return preds


# In[ ]:


model = Model(activation=LeakyReLU,neurons=256).to(device)
criterion = MSELoss()
optimizer = Adam(model.parameters(),lr=0.001)
epochs = 100
batch_size = 32


# In[ ]:


def accuracy(model,X,y):
    correct = 0
    total = 0
    preds = model(X.float())
    for pred,y_batch in zip(preds,y):
        pred = int(torch.round(pred))
        y_batch = int(y_batch)
        if pred == y_batch:
            correct += 1
        total += 1
    acc = round(correct/total,3)*100
    return acc


# In[ ]:


def g_loss(model,X,y):
    preds = model(X.float())
    loss = criterion(preds.view(-1),y.float().view(-1))
    return loss.item()


# In[ ]:


# wandb.init(project=PROJECT_NAME,name='BaseLine')
# wandb.watch(model)
iter_epochs = tqdm(range(epochs))
for _ in iter_epochs:
    for i in range(0,len(X_train),batch_size):
        try:
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            preds = model(X_batch.float())
            loss = criterion(preds.float().view(-1),y_batch.float().view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass
    model.eval()
    print(f"Val Accuracy = {accuracy(model,X_test,y_test)} | Accuracy = {accuracy(model,X_train,y_train)} | Val Loss = {g_loss(model,X_test,y_test)} | Loss = {g_loss(model,X_train,y_train)}")
    model.train()
#     wandb.watch(model)
# wandb.watch(model)
# wandb.finish()


# In[ ]:


model.eval()
preds = model(X_test.float())


# In[ ]:


new_test = []


# In[ ]:


for X_iter in tqdm(test['text']):
    new_test.append(words_to_int(X_iter,all_words))


# In[ ]:


new_test = torch.from_numpy(np.array(new_test)).to(device)


# In[ ]:


preds = model(new_test.float())


# In[ ]:


ids = test['id']


# In[ ]:


submission = {
    "id":[],
    "target":[]
}


# In[ ]:


for pred,id in tqdm(zip(preds,ids)):
    submission['id'].append(id)
    submission['target'].append(int(torch.round(pred)))


# In[ ]:


submission = pd.DataFrame(submission)


# In[ ]:


submission.to_csv("./base.csv",index=False)


# In[ ]:





# In[ ]:




