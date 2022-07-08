#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('cd', '../')
get_ipython().system('mkdir /kaggle/tmp')
get_ipython().system('mkdir /kaggle/tmp/mask')
get_ipython().system('mkdir /kaggle/tmp/train')
get_ipython().system('mkdir /kaggle/tmp/test')
get_ipython().run_line_magic('cd', 'tmp')


# In[ ]:


get_ipython().system('tar -zxf /kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/mask.tar.gz -C /kaggle/tmp/mask')
get_ipython().system('tar -zxf /kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/train.tar.gz -C /kaggle/tmp/train')
get_ipython().system('tar -zxf /kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/test.tar.gz -C /kaggle/tmp/test')


# In[ ]:


# Download YOLOv5
get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().run_line_magic('cd', 'yolov5')
# Install dependencies
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install dependencies')


# In[ ]:


get_ipython().run_line_magic('cd', '../')


# In[ ]:


## Install W&B 
#!pip install -q --upgrade wandb
## Login 
#import wandb
#wandb.login()


# In[ ]:


import torch
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt

import csv


# In[ ]:


study_df = pd.read_csv("/kaggle/input/siim-covid19-detection/train_study_level.csv")
image_df = pd.read_csv("/kaggle/input/siim-covid19-detection/train_image_level.csv")
meta_df = pd.read_csv("/kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/meta.csv")
fold_df = pd.read_csv("/kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/updated_iamge_level.csv")


# In[ ]:


_duplicateList_path = '/kaggle/input/siiim-covid-stratified-k-fold-and-create-mask/dublicate.txt'


# In[ ]:


_duplicateList = []
with open(_duplicateList_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        _duplicateList += row

_duplicateList[:5]


# In[ ]:


TRAIN_PATH = '/kaggle/tmp/train/'
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 16


# In[ ]:


# Modify values in the id column
df = fold_df.copy()

df['id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df['path'] = df.apply(lambda row: TRAIN_PATH+row.id+'.png', axis=1)
# Get image level labels
df['image_level'] = df.apply(lambda row: row.label.split(' ')[0], axis=1)

df.head(5)


# In[ ]:


meta_df.head()


# In[ ]:


df = df.merge(meta_df, left_on='id', right_on="image_id")
df.head(2)


# In[ ]:


print("before drop duplicate", len(df))
df = df[~df['id'].isin(_duplicateList)]
print("after drop duplicate", len(df))


# In[ ]:


os.makedirs('tmp/covid/images/train', exist_ok=True)
os.makedirs('tmp/covid/images/valid', exist_ok=True)

os.makedirs('tmp/covid/labels/train', exist_ok=True)
os.makedirs('tmp/covid/labels/valid', exist_ok=True)


# In[ ]:


df.head()


# In[ ]:


# Move the images to relevant split folder.
# 5 fold
for _fold in range(5):
    os.makedirs(f'/kaggle/tmp/covid/images/train/fold{_fold}', exist_ok=True)
    os.makedirs(f'/kaggle/tmp/covid/images/valid/fold{_fold}', exist_ok=True)

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        if row.fold != _fold:
            copyfile(row.path, f'/kaggle/tmp/covid/images/train/fold{_fold}/{row.id}.png')
        else:
            copyfile(row.path, f'/kaggle/tmp/covid/images/valid/fold{_fold}/{row.id}.png')


# In[ ]:


get_ipython().system('ls /kaggle/tmp/covid/images/train/fold0/ | wc -l')


# In[ ]:


# Get the raw bounding box by parsing the row value of the label column.
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []  
            
    return bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE/row.dim1
    scale_y = IMG_SIZE/row.dim0
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0] # xmax - xmin
        h = bbox[3] - bbox[1] # ymax - ymin
        xc = bbox[0] + int(np.round(w/2)) # xmin + width/2
        yc = bbox[1] + int(np.round(h/2)) # ymin + height/2
        
        yolo_boxes.append([xc/img_w, yc/img_h, w/img_w, h/img_h]) # x_center y_center width height
    
    return yolo_boxes


# In[ ]:


# Prepare the txt files for bounding box
for _fold in range(5):
    os.makedirs(f'/kaggle/tmp/covid/labels/train/fold{_fold}', exist_ok=True)
    os.makedirs(f'/kaggle/tmp/covid/labels/valid/fold{_fold}', exist_ok=True)
    
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        # Get image id
        img_id = row.id
        # Get split
        split = row.split
        # Get image-level label
        label = row.image_level

        if row.fold != _fold:
            file_name = f'/kaggle/tmp/covid/labels/train/fold{_fold}/{row.id}.txt'
        else:
            file_name = f'/kaggle/tmp/covid/labels/valid/fold{_fold}/{row.id}.txt'
        
        if label=='opacity':
            # Get bboxes
            bboxes = get_bbox(row)
            # Scale bounding boxes
            scale_bboxes = scale_bbox(row, bboxes)
            # Format for YOLOv5
            yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)

            with open(file_name, 'w') as f:
                for bbox in yolo_bboxes:
                    bbox = [1]+bbox
                    bbox = [str(i) for i in bbox]
                    bbox = ' '.join(bbox)
                    f.write(bbox)
                    f.write('\n')


# In[ ]:


get_ipython().system('cat /kaggle/tmp/covid/labels/valid/fold0/0012ff7358bc.txt')


# In[ ]:


get_ipython().system('ls /kaggle/tmp/covid/labels/valid/fold1 | wc -l')


# In[ ]:


get_ipython().run_line_magic('cd', 'yolov5')


# In[ ]:


# Create .yaml file 
import yaml
for _fold in range(5):
    data_yaml = dict(
        train = f'../covid/images/train/fold{_fold}',
        val = f'../covid/images/valid/fold{_fold}',
        nc = 2,
        names = ['none', 'opacity']
    )

    # Note that I am creating the file in the yolov5/data/ directory.
    with open(f'data/data-fold{_fold}.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)
    
get_ipython().run_line_magic('cat', 'data/data-fold0.yaml')
get_ipython().run_line_magic('cat', 'data/data-fold1.yaml')
get_ipython().run_line_magic('cat', 'data/data-fold2.yaml')
get_ipython().run_line_magic('cat', 'data/data-fold3.yaml')
get_ipython().run_line_magic('cat', 'data/data-fold4.yaml')


# In[ ]:


get_ipython().system('ls /kaggle/tmp/covid/labels/valid/fold0 | wc -l')


# In[ ]:


get_ipython().system("WANDB_MODE='dryrun' python train.py --img {IMG_SIZE}                   --batch {BATCH_SIZE}                   --epochs {EPOCHS}                   --data data-fold0.yaml                   --weights yolov5s.pt                   --save_period 1                  --project /kaggle/working/kaggle-siim-covid")


# In[ ]:


get_ipython().system('ls /kaggle/working/kaggle-siim-covid')


# In[ ]:


#%cp /content/yolov5/weights/


# In[ ]:


get_ipython().system("WANDB_MODE='dryrun' python train.py --img {IMG_SIZE}                   --batch {BATCH_SIZE}                   --epochs {EPOCHS}                   --data data-fold1.yaml                   --weights yolov5s.pt                   --save_period 1                  --project /kaggle/working/kaggle-siim-covid")


# In[ ]:


get_ipython().system("WANDB_MODE='dryrun' python train.py --img {IMG_SIZE}                   --batch {BATCH_SIZE}                   --epochs {EPOCHS}                   --data data-fold2.yaml                   --weights yolov5s.pt                   --save_period 1                  --project /kaggle/working/kaggle-siim-covid")


# In[ ]:


get_ipython().system("WANDB_MODE='dryrun' python train.py --img {IMG_SIZE}                   --batch {BATCH_SIZE}                   --epochs {EPOCHS}                   --data data-fold3.yaml                   --weights yolov5s.pt                   --save_period 1                  --project /kaggle/working/kaggle-siim-covid")


# In[ ]:


get_ipython().system("WANDB_MODE='dryrun' python train.py --img {IMG_SIZE}                   --batch {BATCH_SIZE}                   --epochs {EPOCHS}                   --data data-fold4.yaml                   --weights yolov5s.pt                   --save_period 1                  --project /kaggle/working/kaggle-siim-covid")

