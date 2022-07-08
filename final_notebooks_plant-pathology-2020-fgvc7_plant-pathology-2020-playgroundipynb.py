#!/usr/bin/env python
# coding: utf-8

# このNotebookは以下のCodeを参考にしています
# 
# https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
# 
# https://www.kaggle.com/miyashitaryuma/avilen-003
# 
# https://www.kaggle.com/piantic/pytorch-tpu
# 
# https://www.kaggle.com/samarthsarin/resnet-pytorch

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))



# In[ ]:


import albumentations as A
import cv2
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tqdm import tqdm #完了までのバーを表示してくれる
tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision.io import read_image

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
# import ToTensor
from albumentations.pytorch import ToTensorV2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_path = '../input/plant-pathology-2020-fgvc7/'
image_path = data_path + 'images/'

test_df = pd.read_csv(data_path + 'test.csv')
train_df = pd.read_csv(data_path + 'train.csv')
submission_df = pd.read_csv(data_path + '/sample_submission.csv')


# データの簡単なチェック

# In[ ]:


test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


SAMPLE_LEN = 100 #適当に画像を選別する際など


# In[ ]:


## 画像読み込みの関数
def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(image_path + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 100枚をランダムにimagesフォルダから取得
train_images = train_df["image_id"][:SAMPLE_LEN].progress_apply(load_image)


# In[ ]:


type(train_images)


# In[ ]:


train_images.head()


# In[ ]:


leaf_fig = px.imshow(cv2.resize(train_images[42], (200, 150))) #サイズを調整した上でcv2を使って読み込み
leaf_fig.show()


# In[ ]:


leaf_fig = px.imshow(train_images[42]) # リサイズをしない場合、1300×2000ぐらいのよう
leaf_fig.show()


# In[ ]:


for i in range(10):
    print(f"height: {train_images[i].shape[0]}, weight: {train_images[i].shape[1]}")


# 画像は全て1365×2048のよう

# # EDA

# In[ ]:





# # 学習準備

# In[ ]:


# ハイパーパラメータの指定
EPOCHS = 20 #エポック数
N_FOLDS = 5 # CV数
SEED = 12345 # シード値
BATCH_SIZE = 64


# In[ ]:


class PlantPathologyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_transform=None):
        # annotations_file(このコンペだとtrain.csv)を読込
        # →画像のファイル名とラベルを取得
        self.df = df
        # 画像のディレクトリ
        self.img_dir = img_dir
        # transformの指定
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        # データセット(tran.csv)のサンプル数
        return len(self.df)
        
    def __getitem__(self, idx):
        # 画像のディレクトリ+画像名を結合して画像のパスを作成
        img_path = os.path.join(self.img_dir, (self.df.iloc[idx, 0]+'.jpg'))
        # 画像の読み込み
        image = read_image(img_path)
        # 画像のラベルをannotations_fileから読み込んだ情報から取得
        # 後述するtrain_loopでモデルの出力を確率で出力するので、ラベルは(N,)で定義する
        label = self.df.loc[idx, ['healthy','multiple_diseases','rust','scab']].values
        label = torch.from_numpy(label.astype(np.int8))
        label = label.argmax()
        
        # transform
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # 読み込んだidxに対応するサンプル
        sample = {"image": image, "label": label}
        return sample


# In[ ]:


tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# In[ ]:


dataset_test = PlantPathologyDataset(submission_df, image_path, transform=tf)
test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)


# In[ ]:


# submissions = None # UnboundLocalError: local variable 'submissions' referenced before assignmentがここだとでてしまう
# train_results = []

train_labels = train_df.iloc[:, 1:].values
# Need for the StratifiedKFold split
train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3
oof_preds = np.zeros((train_df.shape[0], 4))


# In[ ]:


folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# In[ ]:


def train_one_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid):
    
    train_fold_results = []

    for epoch in range(EPOCHS):

        # print('  Epoch {}/{}'.format(epoch + 1, N_EPOCHS))
        # print('  ' + ('-' * 20))
        os.system(f'echo \"  Epoch {epoch}\"')

        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):

            images = batch['image']
            labels = batch['label']

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)                
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch['image']
            labels = batch['label']

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels), dim=0)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                if val_preds is None:
                    val_preds = outputs.cpu()
                else:
                    val_preds = torch.cat((val_preds, outputs.cpu()), dim=0)


        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(dataloader_train),
            'valid_loss': val_loss / len(dataloader_valid),
#             'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),
        })

    return val_preds, train_fold_results


# In[ ]:


def train_kfold(model, criterion, optimizer, transforms_train, transforms_valid, dataloader_test):
    train_results = []
    submissions = None
    
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)    

        dataset_train = PlantPathologyDataset(train, image_path, transforms_train)
        dataset_valid = PlantPathologyDataset(valid, image_path, transforms_valid)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

        val_preds, train_fold_results = train_one_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid)
        oof_preds[valid_idx, :] = val_preds.numpy()
        
        train_results = train_results + train_fold_results
#         print(f"{train_fold_results}")

        model.eval()
        test_preds = None

        for step, batch in enumerate(dataloader_test):

            images = batch['image']
            images = images.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(images)

                if test_preds is None:
                    test_preds = outputs.cpu()
                else:
                    test_preds = torch.cat((test_preds, outputs.cpu()), dim=0)


#         print(test_preds.shape)
#         print(test_preds)
#         print(submission_df.shape)
#         display(submission_df.head())
        
        # Save predictions per fold
        submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_preds
        submission_df.to_csv('submission_fold_{}.csv'.format(i_fold), index=False)

        # logits avg
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS

    print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))
    
    return pd.DataFrame(train_results)


# ### モデル定義

# In[ ]:


from torchvision import models

class vgg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
#         self.conv1 = nn.Conv2d(1,3,1)
        self.vgg = models.vgg16(pretrained=True)
#         self.fc = nn.Linear(1000, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
#         x = self.conv1(x)
        x = self.vgg(x)
#         x = self.fc(x)
        x = self.softmax(x)
        return x


# In[ ]:


# GPU/CPUの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


model_vgg = vgg(num_classes=4).to(device)
model_vgg.vgg.classifier[6] = nn.Linear(in_features=4096, out_features=4)
model_vgg = model_vgg.to(device)


# In[ ]:


print(model_vgg)


# ### 学習の実施

# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer_fn = torch.optim.Adam(model_vgg.parameters(), lr=0.001)


# In[ ]:


train_result1 = train_kfold(model_vgg, loss_fn, optimizer_fn, tf, tf, test_dataloader)


# In[ ]:


train_result1.head(10)


# In[ ]:


def plot_train_result(train_results):
    fig = make_subplots(rows=2, cols=1)

    colors = [
        ('#d32f2f', '#ef5350'),
        ('#303f9f', '#5c6bc0'),
        ('#00796b', '#26a69a'),
        ('#fbc02d', '#ffeb3b'),
        ('#5d4037', '#8d6e63'),
    ]

    for i in range(N_FOLDS):
        data = train_results[train_results['fold'] == i]

        fig.add_trace(go.Scatter(x=data['epoch'].values,
                                 y=data['train_loss'].values,
                                 mode='lines',
                                 visible='legendonly' if i > 0 else True,
                                 line=dict(color=colors[i][0], width=2),
                                 name='Train loss - Fold #{}'.format(i)),
                     row=1, col=1)

        fig.add_trace(go.Scatter(x=data['epoch'],
                                 y=data['valid_loss'].values,
                                 mode='lines+markers',
                                 visible='legendonly' if i > 0 else True,
                                 line=dict(color=colors[i][1], width=2),
                                 name='Valid loss - Fold #{}'.format(i)),
                     row=1, col=1)

    #     fig.add_trace(go.Scatter(x=data['epoch'].values,
    #                              y=data['valid_score'].values,
    #                              mode='lines+markers',
    #                              line=dict(color=colors[i][0], width=2),
    #                              name='Valid score - Fold #{}'.format(i),
    #                              showlegend=False),
    #                  row=2, col=1)

    fig.update_layout({
      "annotations": [
        {
          "x": 0.225, 
          "y": 1.0, 
          "font": {"size": 16}, 
          "text": "Train / valid losses", 
          "xref": "paper", 
          "yref": "paper", 
          "xanchor": "center", 
          "yanchor": "bottom", 
          "showarrow": False
        }, 
        {
          "x": 0.775, 
          "y": 1.0, 
          "font": {"size": 16}, 
          "text": "Validation scores", 
          "xref": "paper", 
          "yref": "paper", 
          "xanchor": "center", 
          "yanchor": "bottom", 
          "showarrow": False
        }, 
      ]
    })

    fig.show()


# In[ ]:


plot_train_result(train_result1)

