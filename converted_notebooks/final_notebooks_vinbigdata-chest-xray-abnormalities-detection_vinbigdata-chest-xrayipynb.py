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


# # Setup

# In[ ]:


get_ipython().system('pip install pycocotools')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
import glob

import time
import tqdm
import random
import torchvision.transforms as T
import math
import random

import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


# In[ ]:


df_train = pd.read_csv("../input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
df_train


# In[ ]:


df_train.loc[df_train['image_id'] == '08d0eab34ea5bab14e7e56dabafcdb7f']


# In[ ]:


import numpy as np
import json

save_json_path = 'traincoco.json'

images = []
categories = []
annotations = []

# category = {}
# category['supercategory'] = 'None'
# category['id'] = 0
# category['name'] = 'None'
# categories.append(category)

# df_train.fillna(0, inplace=True)
df_train = df_train[df_train['class_id'] != 14].reset_index(drop=True)

df_train['file_id'] = df_train['image_id'].astype('category').cat.codes
df_train['category_id'] = pd.Categorical(df_train['class_name'], ordered= True).codes + 1
df_train['annid'] = df_train.index
df_train


# In[ ]:


# IMG_SIZE = 512
# df_train['x_min'] = (df_train['x_min']/df_train['width'])*IMG_SIZE
# df_train['y_min'] = (df_train['y_min']/df_train['height'])*IMG_SIZE
# df_train['x_max'] = (df_train['x_max']/df_train['width'])*IMG_SIZE
# df_train['y_max'] = (df_train['y_max']/df_train['height'])*IMG_SIZE


# In[ ]:


len(df_train.class_name.unique())


# ## Export train.csv to coco json

# In[ ]:


def image(row):
    image = {}
    image["height"] = abs(row.y_max - row.y_min)
    image["weight"] = abs(row.x_max - row.x_min)
    image["id"] = row.file_id
    image["file_name"] = row.image_id + '.dicom'
    return image

def category(row):
    category = {}
    category['supercategory'] = 'None'
    category['id'] = row.category_id
    category['name'] = row.class_name
    return category

def annotation(row):
    annotation = {}
    area = (row.x_max - row.x_min) * (row.y_max - row.y_min) 

    annotation['segmentation'] = []
    annotation['iscrowd'] = 0
    annotation['area'] = area
    annotation['image_id'] = row.file_id
    
    annotation['bbox'] = [row.x_min, row.y_min, row.x_max - row.x_min, row.y_max - row.y_min]
   
    annotation['category_id'] = row.category_id
    annotation['id'] = row.annid
    return annotation

for row in df_train.itertuples():
    annotations.append(annotation(row))
    
imagedf = df_train.drop_duplicates(subset=['file_id']).sort_values(by='file_id')
for row in imagedf.itertuples():
    images.append(image(row))
    
catdf = df_train.drop_duplicates(subset=['category_id']).sort_values(by='category_id')
for row in catdf.itertuples():
    categories.append(category(row))
    
data_coco = {}
data_coco['images'] = images
data_coco['categories'] = categories
data_coco['annotations'] = annotations
json.dump(data_coco, open(save_json_path, 'w'), indent=4)

print("Create COCO json finished!")


# ## Create DataLoader custom

# In[ ]:


# class VBDDataset(Dataset):
#     def __init__(self, df, image_dir, mode='train', transforms=None):
#         super().__init__()
        
#         self.image_ids = df['image_id'].unique()
#         self.df = df
#         self.image_dir = image_dir
#         self.mode = mode
#         self.transforms = transforms
    
#     def __getitem__(self, idx):
#         image_dir = "../input/vinbigdata-chest-xray-abnormalities-detection/" + self.mode
#         image_id = self.image_ids[idx]
#         records = self.df[self.df['image_id'] == image_id]

#         image_file = pydicom.dcmread(os.path.join(image_dir, f'{image_id}.dicom'))
#         image = np.array(image_file.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
#         image = torch.from_numpy(image)
# #         image = cv2.imread(f'{self.image_dir}/{image_id}.png', cv2.IMREAD_COLOR)
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
# #         image /= 255.0

#         boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         area = torch.as_tensor(area, dtype=torch.float32)
#         # all the labels are shifted by 1 to accomodate background
#         labels = torch.squeeze(torch.as_tensor((records.class_id.values+1,), dtype=torch.int64))
        
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
#         target = {}
#         target['boxes'] = boxes
#         target['labels'] = labels
#         # target['masks'] = None
#         target['image_id'] = torch.tensor([idx])
#         target['area'] = area
#         target['iscrowd'] = iscrowd
#         if self.transforms:
#             sample = {
#                 'image': image,
#                 'bboxes': target['boxes'],
#                 'labels': labels
#             }
#             sample = self.transforms(**sample)
#             image = sample['image']
            
#             target['boxes'] = torch.as_tensor(sample['bboxes'])

#         return image, target, image_id

#     def __len__(self):
#         return self.image_ids.shape[0]
        


# In[ ]:


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    
    dicom = pydicom.dcmread(path)
    
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    return data


# In[ ]:


class VBDDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
#         img_arr = dicom2array(os.path.join(self.root, path))
#         img = Image.fromarray(img_arr)
        image_file = pydicom.dcmread(os.path.join(self.root, path))
        img = np.array(image_file.pixel_array, dtype=np.float32)[np.newaxis]  # Add channel dimension
        img = torch.from_numpy(img)
#         img = Image.open(os.path.join(self.root, path))
        
        # number of objects in the image
        num_objs = len(coco_annotation)
        
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Labels
        labels = torch.squeeze(torch.as_tensor((num_objs,), dtype=torch.int64))
        
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation['boxes'] = boxes
        my_annotation['labels'] = labels
        my_annotation['image_id'] = img_id
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd
        
#         if self.transforms:
#             sample = {
#                 'image': img,
#                 'bboxes': my_annotation['boxes'],
#                 'labels': labels
#             }
#             sample = self.transforms(**sample)
#             img = sample['image']
            
#             my_annotation['boxes'] = torch.as_tensor(sample['bboxes'])
        
#         if self.transforms:
#             img = self.transforms(img)
        
        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)
        
        


# In[ ]:


TRAIN_DIR = "../input/vinbigdata-chest-xray-abnormalities-detection/train"
TEST_DIR = "../input/vinbigdata-chest-xray-abnormalities-detection/test"

TRAIN_COCO = "./traincoco.json"


# In[ ]:


from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# In[ ]:


def get_transform():
    return A.Compose([
#         A.Flip(0.5),
#         A.ShiftScaleRotate(rotate_limit=10, p=0.5),
#         A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=1.0),
#         A.Resize(height=512, width=512, always_apply=True),
#         ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# def get_transform():
#     custom_transforms = []
#     custom_transforms.append(torchvision.transforms.ToPILImage())
#     custom_transforms.append(torchvision.transforms.ToTensor())
#     return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = VBDDataset(root=TRAIN_DIR, annotation=TRAIN_COCO)

batch_size = 16

data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)


# # EDA

# In[ ]:


from pydicom.pixel_data_handlers.util import apply_voi_lut
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def get_bb_info(df, img_id):
    bounding_boxes_info = df.loc[df["image_id"]==img_id, ['x_min', 'y_min', 'x_max', 'y_max', "class_id"]]

    bboxes = []
    for _, row in bounding_boxes_info.astype(np.int16).iterrows():
        bboxes.append(list(row))
    
    return bboxes

label2color = { 0:("Aortic enlargement","#2a52be"),
                1:("Atelectasis","#ffa812"),
                2:("Calcification","#ff8243"),
                3:("Cardiomegaly","#4682b4"),
                4:("Consolidation","#ddadaf"),
                5:("ILD","#a3c1ad"),
                6:("Infiltration","#008000"),
                7:("Lung Opacity","#004953"),
                8:("Nodule/Mass","#e3a857"),
                9:("Other lesion","#dda0dd"),
               10:("Pleural effusion","#e6e8fa"),
               11:("Pleural thickening","#800020"),
               12:("Pneumothorax","#918151"),
               13:("Pulmonary fibrosis","#e75480")}

def bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0,0,1,1])
    
    # plot the image
    plt.imshow(img_as_arr, cmap="gray")
    plt.title(img_id)

    # add the bounding boxes
    for row in bounding_boxes_info:
        # each row contains 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
        xmin = row[0]
        xmax = row[2]
        ymin = row[1]
        ymax = row[3]

        width = xmax - xmin
        height = ymax - ymin
        
        edgecolor = label2color[row[4]][1]
        ax.annotate(label2color[row[4]][0], xy=(xmax - 40, ymin + 20))

        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)

    plt.show()



# In[ ]:


train_images = glob.glob(TRAIN_DIR+"/**.dicom")
test_images = glob.glob(TEST_DIR+"/**.dicom")

img_ids = df_train['image_id'].unique()
shortlisted_img_ids = img_ids[:10]
og_imgs = [dicom2array(f'{TRAIN_DIR}/{path}.dicom') for path in shortlisted_img_ids]


for img_as_arr, img_id in zip(og_imgs,shortlisted_img_ids):    
    bounding_boxes_info = get_bb_info(df_train, img_id)
    bounding_box_plotter(img_as_arr, img_id, bounding_boxes_info)


# # Model

# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device


# In[ ]:


# # DataLoader is iterable over Dataset
# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)


# In[ ]:


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

print('Done')


# In[ ]:


# def train(train_data_loader, model, optimizer, idx):
#     epoch_loss = []
#     start_time = time.time()
    
#     model.train()
    
#     for imgs, annotations in data_loader:
#         start_loop = time.time()
#         imgs = list(img.to(device) for img in imgs)
#         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
#         # Reseting Gradients
#         optimizer.zero_grad()
        
#         # Calculating Loss
#         loss_dict = model(imgs, annotations)
#         losses = sum(loss for loss in loss_dict.values())
#         epoch_loss.append(loss)
        
#         # Backward
#         losses.backward()
#         optimizer.step()
#         end_loop = time.time()
#         total_time_loop = end_loop - start_loop
        
#         print(f'Iteration: {idx}/{len_dataloader}, Loss: {losses}, Time: {total_time_loop}')
#     # Overall Epoch Results
#     end_time = time.time()
#     total_time = end_time - start_time
    
#     # Loss
#     epoch_loss = np.mean(epoch_loss)
    
#     return epoch_loss, total_time

def train(model, dataloader, device, epochs, optimizer):    
    best_loss = 1e4
    itr = 1
    all_losses = []
    model.train()
    train_loss = 0

    for epoch in range(epochs):
    
        for images, targets in dataloader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            all_losses.append(loss_value)
            
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1


        print(f"Epoch #{epoch} loss: {all_loss[itr - 1]}\n")
        
    return all_losses

print('Done')


# In[ ]:


num_classes = len(df_train['class_name'].unique())
num_epochs = 10

model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)
    
# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# len_dataloader = len(data_loader)
idx = 0

# for epoch in range(num_epochs):
#     train(train_data_loader = data_loader, model = model, optimizer = optimizer, idx = idx)
#     idx += 1

len_dataloader = len(data_loader)

train(model, data_loader, device, num_epochs, optimizer)


# In[ ]:




