#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install imutils')
get_ipython().system('pip install wget')
get_ipython().system('pip install split-folders')


# In[ ]:


from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import splitfolders
from torch import nn
import numpy as np
import os
import wget
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


# ## Downloading Dataset

# In[ ]:


_URL = 'http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz'
wget.download(_URL)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


zip_dir = tf.keras.utils.get_file('./logo', origin=_URL, untar=True,extract=True)


# In[ ]:


import tarfile

fname = 'flickr_logos_27_dataset.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()


# In[ ]:


fname = 'flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz'

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()


# In[ ]:


src_dir = "flickr_logos_27_dataset_images"
dest = "LOGOS"

if not os.path.exists(dest):
    os.makedirs(dest)


# ## Preprocessing

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt", sep='\s+',header=None)


# In[ ]:


df


# In[ ]:


X = df.iloc[:,0]
Y = df.iloc[:,1]


# In[ ]:


dtdir = './flickr_logos_27_dataset_images/'


# In[ ]:


im = df[0][0]


# In[ ]:


size = df.iloc[:,3:]


# In[ ]:


size


# In[ ]:


img = os.path.join(dtdir,im)


# In[ ]:


size = size.values.tolist()


# In[ ]:


size[0][0],size[0][1],size[0][2],size[0][3]


# In[ ]:


image = cv2.imread(img)
plt.imshow(image)
image.shape


# In[ ]:


image = cv2.imread(img)
image = image[size[0][1]:size[0][3],size[0][0]:size[0][2]]
plt.imshow(image)
image.shape


# In[ ]:


query = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt", sep='\s+',header=None)


# In[ ]:


query


# In[ ]:


img = os.path.join(dtdir,query[0][5])
image = cv2.imread(img)
plt.imshow(image)
image.shape


# In[ ]:


y = list(set(list(Y)))
y.sort()


# In[ ]:


for i in y:
    os.makedirs(os.path.join(dest,i))


# In[ ]:


distractor = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_distractor_set_urls.txt", sep='\s+',header=None)


# In[ ]:


distractor


# In[ ]:


HEIGHT = 224
WIDTH =  224
BS = 256


# ## Removing Corrupt Images 

# In[ ]:


for i in range(len(X)):
    try:
        destrain = os.path.join(dest,Y[i])
        savepath = os.path.join(destrain,X[i])
        img  = os.path.join(dtdir,X[i])
        image = cv2.imread(img)
        image = image[size[i][1]:size[i][3],size[i][0]:size[i][2]]
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass


# In[ ]:


A = query.iloc[:,0]
B = query.iloc[:,1]


# In[ ]:


A


# In[ ]:


for i in range(len(A)):
    try:
        destrain = os.path.join(dest,B[i])
        savepath = os.path.join(destrain,A[i])
        img  = os.path.join(dtdir,A[i])
        image = cv2.imread(img)
        image = cv2.resize(image,(WIDTH,HEIGHT))
        cv2.imwrite(savepath,image)
    except:
        print('error')
        pass


# In[ ]:


imagePaths = list(paths.list_images(dest))


# In[ ]:


img = imagePaths[40]
print(img)
image = cv2.imread(img)
plt.imshow(image)
image.shape


# ## Train Val Split

# In[ ]:


path = 'LOGOS'


# In[ ]:


splitfolders.ratio(path, output="data", seed=42, ratio=(0.8,0.2))


# ## Image Augmentation

# In[ ]:


# initialize our data augmentation functions
resize = transforms.Resize(size=(WIDTH,HEIGHT))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)
coljtr = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
raf = transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0)
rrsc = transforms.RandomResizedCrop(size=WIDTH, scale=(0.8, 1.0))
ccp  = transforms.CenterCrop(size=WIDTH)  # Image net standards
nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards


# In[ ]:


# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize,hFlip,vFlip,rotate,raf,rrsc,ccp,coljtr,transforms.ToTensor(),nrml])
valTransforms = transforms.Compose([resize,hFlip,vFlip,rotate,raf,rrsc,ccp,coljtr,transforms.ToTensor(),nrml])


# In[ ]:


# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root='./data/train',transform=trainTransforms)
valDataset = ImageFolder(root='./data/val', transform=valTransforms)
print("[INFO] training dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(len(valDataset)))


# In[ ]:


# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, batch_size=BS, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=BS,shuffle=True)


# In[ ]:


examples = iter(valDataLoader)
example_data, example_targets = examples.next()
for i in range(9):
    plt.subplot(3,3,i+1)
    img =example_data[i].cpu().numpy().T
    plt.imshow(img)
    plt.axis("off")
plt.show() 


# In[ ]:


import torchvision.models as models


# In[ ]:


print(trainDataset.class_to_idx)


# ## Model Architecture - InceptionV3

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


model = models.mobilenet_v2(pretrained=True)
model.aux_logits=False

# Freeze training for all layers
for param in model.parameters():
    param.requires_grad = False

# append a new classification top to our feature extractor and pop it
# on to the current device
num_feat = model.classifier[1].in_features

features = list(model.classifier.children())[:-1] # Remove last layer

features.extend([nn.Linear(num_feat, 256),
                 nn.Dropout(0.5),
                 nn.ReLU(inplace=True), 
                 nn.Linear(256, len(trainDataset.classes)),                   
                 nn.LogSoftmax(dim=1)]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

model = model.to(device)


# ## Loss Function and Optimizer

# In[ ]:


loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())


# ## Train Function

# In[ ]:


def train(epoch):
  model.train()
  net_loss = 0
  correct = 0
  for batch_idx, (data, target) in enumerate(trainDataLoader):
    (data, target) = (data.to(device), target.to(device))
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, target)
    output = torch.exp(output)
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
    loss.backward()
    optimizer.step()
    
    net_loss = net_loss + loss.item()
  acc = correct / len(trainDataLoader.dataset)
  return net_loss,acc


# ## Test Function

# In[ ]:


def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in valDataLoader:
      (data, target) = (data.to(device), target.to(device))
      output = model(data)
      test_loss += loss_func(output, target).item()
      output = torch.exp(output)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(valDataLoader.dataset)
  acc = correct / len(valDataLoader.dataset)
  return test_loss,acc


# ## Driver Code

# In[ ]:


LOSSTR = []
ACCTE = []
LOSSTE = []
ACCTR = []
import time
n_epochs = 100
for epoch in range(1, n_epochs + 1):
  start = time.time()
  print("--- Epoch {} ---".format(epoch))
  epoch_loss,tracc = train(epoch)
  LOSSTR.append(epoch_loss)
  ACCTR.append(tracc)
  print("\tTrain Accuracy = {} || Train Loss  = {} ".format(tracc,epoch_loss))
  tloss,tacc =  test()
  print("\tTest Accuracy =  {} || Test Loss = {} ".format(tacc,tloss))
  ACCTE.append(tacc)
  LOSSTE.append(tloss)
  stop = time.time()
  print("\tTraining time = ", (stop - start))


# ## Accuracy - Loss Plot

# In[ ]:


xx = np.arange(n_epochs)
plt.style.use("ggplot")

acctr = torch.Tensor(ACCTR).detach().cpu().numpy()
lsstr = torch.Tensor(LOSSTR).detach().cpu().numpy()
accte = torch.Tensor(ACCTE).detach().cpu().numpy()
lsste = torch.Tensor(LOSSTE).detach().cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6))
fig.suptitle('ACC vs LOSS')
ax1.plot(xx, acctr,label='Train')
ax1.plot(xx,accte,label='Val')
ax1.legend(loc="best")
ax2.plot(xx, lsstr,label='Train')
ax2.plot(xx, lsste,label='Val')
ax2.legend(loc="best")
plt.show()


# ## Prediction On Test Images

# In[ ]:


testimage = list(paths.list_images('./flickr_logos_27_dataset_images'))


# ![image.png](attachment:dbacbc64-b039-4beb-877c-5061780c5dc3.png)

# In[ ]:


def predimg(path):
    from PIL import Image
    image = Image.open(path)
    plt.imshow(image)
    plt.axis("off")
    plt.show() 
    model.eval()
    with torch.no_grad():
      img =  load_img(path)
      mean = [0.485, 0.456, 0.406] 
      std = [0.229, 0.224, 0.225]
      transform_norm = transforms.Compose([transforms.ToTensor(), 
      transforms.Resize((224,224)),transforms.Normalize(mean, std)])
      img_normalized = transform_norm(img).float()
      img_normalized = img_normalized.unsqueeze_(0)
      img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
      img = img_normalized.to(device)
      output = model(img)
      output = torch.exp(output)
      #print(output)
      index = output.data.cpu().numpy().argmax()
      result = list(np.around(output.data.cpu().numpy()*100,1))
      print(result)
      print("PREDICTED CLASS = ",trainDataset.classes[index])


# In[ ]:


predimg(testimage[8])


# In[ ]:


predimg(testimage[1])


# In[ ]:


predimg(testimage[16])


# In[ ]:


predimg(testimage[20])


# In[ ]:


predimg(testimage[30])

