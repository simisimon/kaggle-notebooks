#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install wtfml==0.0.2')


# In[ ]:


get_ipython().system('pip install efficientnet_pytorch')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image

from sklearn import model_selection
from sklearn import metrics

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import efficientnet_pytorch

import albumentations as A

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:





# In[ ]:


import pandas as pd
from sklearn import model_selection

# Training data is in a csv file called train.csv
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# fetch targets
y = df.target.values
# initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)
# fill the new kfold column

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
# save the new csv with kfold column
df.to_csv("train_folds.csv", index=False)



# In[ ]:


df_train_folds =pd.read_csv("train_folds.csv")
df_train_folds


# In[ ]:


train_dir = '../input/siim-isic-melanoma-classification/jpeg/train'
test_dir = '../input/siim-isic-melanoma-classification/jpeg/test'
t = os.listdir(train_dir)

t1 = os.listdir(test_dir)
print(len(t),len(t1),len(t)+len(t1))


# In[ ]:


'''
img2np = np.array(img)
img2np.shape

ten = torch.from_numpy(img2np)
ten.shape
'''


# In[ ]:


def train(fold):
    train_dir = train_dir
    df = pd.read_csv('./train_folds.csv')
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Normalize the images
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    


# In[ ]:


#train
train_dir :train_dir

train_aug = A.Compose(
        [
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            A.Flip(p=0.5)  
        ]          
    )

train_images = df_train.image_name.values.tolist()
train_images = [os.path.join(train_dir,i +'.png') for i in train_images]
train_targets = df_train.targets.values

train_dataset = ClassificationLoader(
    image_path = train_images,
    targets = train_targets,
    resize = 220*220,
    augmentation = train_aug
)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)


# In[ ]:


# Valid
valid_images = df_valid.image_name.values.tolist()
valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
valid_targets = df_valid.target.values

valid_aug = A.Compose([
    A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
])

valid_dataset = ClassificationLoader(
    image_paths=valid_images,
    targets=valid_targets,
    resize=None,
    augmentations=valid_aug,
)

valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4)


# In[ ]:


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b4'
        )
        self.base_model._fc = nn.Linear(
            in_features=1792, 
            out_features=1, 
            bias=True
        )
        
    def forward(self, image, targets):
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss


# In[ ]:


model = EfficientNet()
model.to(device)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

es = EarlyStopping(patience=5, mode="max")


# In[ ]:


epochs = 50
for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


# In[ ]:


# torch.save
torch.save(model.state_dict(), './modetor.pt')


# In[ ]:


#Test
def predict(fold):
    print(f"Generating Predictions for saved model, fold = {fold+1}")
    test_data_path = "/kaggle/input/siic-isic-224x224-images/test"
    df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
    df_test.loc[:,'target'] = 0
    
    #model_path = "f'/kaggle/working/model_fold{fold}'"
    #model_path = '/kaggle/working/model_fold0_epoch0.bin'
    model_path = './model_fold_0.bin'
    
    device = 'cuda'
    
    test_bs = 16
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    test_aug = A.Compose(
        [
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    test_images_list = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path,i + '.png') for i in test_images_list]
    test_targets = df_test.target.values
    
    test_dataset = ClassificationLoader(
        image_paths = test_images,
        targets= test_targets,
        resize = None,
        augmentations = test_aug
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = test_bs,
        shuffle = False,
        num_workers=4
    )
    #Earlier defined class for model
    model = EfficientNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    predictions_op = Engine.predict(
        test_loader,
        model,
        device
    )
    return np.vstack((predictions_op)).ravel()


# In[ ]:


# prediction
pred = predict(0)


# In[ ]:


predictions = pred
sample = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample.loc[:, "target"] = predictions
sample.to_csv("submission.csv", index=False)

