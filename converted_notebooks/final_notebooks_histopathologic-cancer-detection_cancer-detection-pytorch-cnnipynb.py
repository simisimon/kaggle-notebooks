#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset


# # Dataの読み込み

# In[ ]:


path='../input/histopathologic-cancer-detection/train/'
annotation_file='../input/histopathologic-cancer-detection/train_labels.csv'
test_path='../input/histopathologic-cancer-detection/test/'


# In[ ]:


train_data =pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
sub_df = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')
train_data.head()


# In[ ]:


print(train_data.shape)


# # Dataの内訳

# In[ ]:


#　円グラフにてデータの内訳を視覚化
plt.pie(train_data.label.value_counts(), labels=['No Cancer', 'Cancer'], 
        colors=['#1f18ed', '#ed1818'], autopct='%1.1f', startangle=90)
plt.show()


# In[ ]:


train_data['label'].value_counts()


# # 画像の視覚化
# ランダムで画像を２０枚抽出しラベル１とラベル０を貼り付ける

# In[ ]:


# サイズの指定
fig = plt.figure(figsize=(25, 25))
# 20枚画像を表示
train_imgs = os.listdir(path)
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])
    im = Image.open(path + img)
    plt.imshow(im)
    lab = train_data.loc[train_data['id'] == img.split('.')[0], 'label'].values[0]
    ax.set_title(f'Label: {lab}')


# # Dataを学習用とテスト用に分ける

# In[ ]:


train, val = train_test_split(train_data, stratify=train_data.label, test_size=0.1)
len(train), len(val)


# # DadaSetの作成

# In[ ]:


class Dataset(Dataset):
    def __init__(self, data_df, data_dir = './', transform=None):
        super().__init__()
        self.df = data_df.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name + '.tif')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[ ]:


batch_size = 128

valid_size = 0.1


# # 画像の処理

# In[ ]:


trans_train = transforms.Compose([transforms.ToPILImage(),                      #compose=複数のTransformを連続して行う
                                  transforms.Pad(64, padding_mode='reflect'),   #テンソルまたは ndarray を PIL Image オブジェクトに変換する
                                  transforms.RandomHorizontalFlip(),            #ランダムに左右反転を行う
                                  transforms.RandomVerticalFlip(),              #ランダムに上下反転を行う
                                  transforms.RandomRotation(20),                #ランダムに回転を行う
                                  transforms.ToTensor(),                        #PIL Image をテンソルに変換する
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])      #正規化を行う()

trans_valid = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = Dataset(data_df=train, data_dir=path, transform=trans_train)
dataset_valid = Dataset(data_df=val, data_dir=path, transform=trans_valid)

train_loader = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)


# # モデルの定義

# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1, 2)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)
        return x


# # モデルの入出力の確認

# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
print(model)


# # 最適化と損失関数
# 

# In[ ]:


criterion = nn.CrossEntropyLoss()

learning_rate = 0.002
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# # 学習モデル

# In[ ]:


num_epochs = 10
total_step = len(train_loader)
for epoch in range(num_epochs):
    
    train_loss = 0.0
    valid_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
        train_loss = train_loss/len(train_loader.sampler)
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# # テストモデル

# In[ ]:


model.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
          
    print('Test Accuracy of the model on the 22003 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


# In[ ]:


dataset_valid = Dataset(data_df=sub_df, data_dir=test_path, transform=trans_valid)
loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)


# In[ ]:


model.eval()

preds = []
for batch_i, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    output = model(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)


# # 提出

# In[ ]:


sub_df.shape, len(preds)
sub_df['label'] = preds
sub_df.to_csv("./submission.csv", index=False)


# In[ ]:




