#!/usr/bin/env python
# coding: utf-8

# **GPU болон Pytorch ашиглан загвар сургахтай холбоотой өгөгдлийг бэлтгэх нөүтбүүк**

# # Сангуудыг дуудах

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import KFold
import os
import zipfile
import gc


# **Зургийн хэмжээг 2 дахин жижигрүүлж 512х512 хэмжээтэй болгож хадгалсан. **

# In[ ]:


sz = 512   #the size of tiles
reduce = 2 #reduce the original images by 2 times 
MASKS = '../input/mongolian-ger-detection/train/labels/'
DATA = '../input/mongolian-ger-detection/train/images/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'


# # Dataframe үүсгэх

# **Зургуудыг дуудаж, зураг бүрийн нэрийг листэд хадгалах**

# In[ ]:


train_imgs = glob(f'{DATA}*.png')
img_ids = [m[-9:-4] for m in train_imgs]


# **Тект өгөгдлөөс зураг бүр дээрх гэр бүрийн боксүүдийг dataframe-рүү хадгалах.**

# In[ ]:


train_df = {'img_name':[], 'img_class':[], 
         'x':[], 'y':[], 'width':[],
          'height':[]}

for img_id in tqdm(img_ids):
    with open(f'../input/mongolian-ger-detection/train/labels/{img_id}.txt') as f:
        label = f.readlines()
        for l in label:
            train_df['img_name'].append(img_id)
            train_df['img_class'].append(l.strip('\n').split(" ")[0])
            train_df['x'].append(l.strip('\n').split(" ")[1])
            train_df['y'].append(l.strip('\n').split(" ")[2])
            train_df['width'].append(l.strip('\n').split(" ")[3])
            train_df['height'].append(l.strip('\n').split(" ")[4])


# In[ ]:


train_df = pd.DataFrame.from_dict(train_df)
train_df.head()


# **String төрлийг float төрөлрүү хөрвүүлэх**

# In[ ]:


train_df['img_class'] = train_df['img_class'].astype('int8')
train_df['x'] = train_df['x'].astype('float32')
train_df['y'] = train_df['y'].astype('float32')
train_df['width'] = train_df['width'].astype('float32')
train_df['height'] = train_df['height'].astype('float32')
train_df.head()


# **Нормчилсан өгөгдлийг зурагны хэмжээнд хөрвүүлэх**

# In[ ]:


train_df[['x', 'y', 'width', 'height']] = 1024*train_df[['x', 'y', 'width', 'height']].values
train_df.head()


# **Зургийг дүрсэлж харуулахын тулд Float төрлийг Integer төрөлрүү хөрвүүлэх.**

# In[ ]:


train_df['x'] = train_df['x'].astype('int32')
train_df['y'] = train_df['y'].astype('int32')
train_df['width'] = train_df['width'].astype('int32')
train_df['height'] = train_df['height'].astype('int32')


# # Зургийн хэмжээг бууруулж, хадгалах

# In[ ]:


H, W = 1024, 1024
x_tot,x2_tot = [],[]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
 zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
        for img_name in tqdm(train_df['img_name'].unique()):
            img = cv2.imread(f'{DATA}{img_name}.png')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            msk = np.zeros((H, W), dtype=np.uint8)
            boxes = train_df.loc[train_df['img_name']==img_name]

            for x, y, width, height in zip(boxes['x'], boxes['y'], boxes['width'], boxes['height']):
                msk[y-int(height/2):y+int(height/2), x-int(width/2):x+int(width/2)] = 1
            
            # Зураг болон маскны хэмжээг бууруулах
            img = cv2.resize(img,(H//reduce, W//reduce), interpolation = cv2.INTER_AREA)
            msk = cv2.resize(msk,(H//reduce,W//reduce), interpolation = cv2.INTER_NEAREST)
            
            
            #Зургийн дундаж болон стандарт хазайлтыг хадгалж, дараагаар зургийг нормчилоход ашиглах
            x_tot.append((img/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0))
            
            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{img_name}.png', img)
            msk = cv2.imencode('.png',msk)[1]
            mask_out.writestr(f'{img_name}.png', msk)
                        

img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)


# # Түүвэр зураг дүрсэлж харуулах

# In[ ]:


from PIL import Image

columns, rows = 4,4
idx0 = 20
fig=plt.figure(figsize=(columns*4, rows*4))
with zipfile.ZipFile(OUT_TRAIN, 'r') as img_arch, \
     zipfile.ZipFile(OUT_MASKS, 'r') as msk_arch:
    fnames = sorted(img_arch.namelist())[8:]
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            img = cv2.imdecode(np.frombuffer(img_arch.read(fnames[idx0+idx]), 
                                             np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = cv2.imdecode(np.frombuffer(msk_arch.read(fnames[idx0+idx]), 
                                              np.uint8), cv2.IMREAD_GRAYSCALE)
    
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow(Image.fromarray(img))
            plt.imshow(Image.fromarray(mask), alpha=0.2)
plt.show()

