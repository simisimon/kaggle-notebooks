#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


# In[ ]:


# import module we'll need to import our custom module
from shutil import copyfile

copyfile(src = "../input/imetfgvc/ml_stratifiers.py", dst = "../working/ml_stratifiers.py")

# import all our functions

from ml_stratifiers import MultilabelStratifiedKFold


# In[ ]:


import numpy as np
import pandas as pd
import os
import copy
import sys
from PIL import Image
import time 
from tqdm.autonotebook import tqdm
import random
import gc
import cv2
import scipy
import math
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler


# In[ ]:


import scipy.special

SEED = 1996
base_dir = '../input/'
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


# In[ ]:


train_df = pd.read_csv('../input/imet-2019-fgvc6/train.csv')
labels_df = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')
test_df = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(train_df.attribute_ids)

def get_label(attribute_ids):
    attribute_ids = attribute_ids.split()
    for _,ids in enumerate(attribute_ids):
        attribute_ids[_] = int(ids)
    one_hot = torch.zeros(1103).scatter_(0, torch.LongTensor(attribute_ids), 1)
    return one_hot

train_df['attribute_ids_encoded'] = train_df['attribute_ids'].apply(get_label)

img_class_dict = {k:v for k, v in zip(train_df.id, train_df.attribute_ids_encoded)}


# In[ ]:


class iMetDataset(Dataset):
    def __init__(self, datafolder, datatype='train', idx=[], transform = transforms.Compose([transforms.RandomResizedCrop(128),transforms.ToTensor()]), \
                labels_dict={}):

        self.datafolder = datafolder
        self.datatype = datatype
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.image_files_list = [self.image_files_list[i] for i in idx]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        img = Image.open(img_name)

        image = self.transform(img)

        #image = random_erase(image)

        img_name_short = self.image_files_list[idx].split('.')[0]

        if self.datatype == 'train':
            #label = get_label(self.labels_dict[img_name_short])
            label = self.labels_dict[img_name_short]
        else:
            label = torch.zeros(1103)
        return image, label


# In[ ]:


data_transforms_crop = transforms.Compose([
    transforms.RandomApply([
        lambda im: transforms.RandomCrop(min(im.size[0],im.size[1]))(im),
    ], p=0.35),
    transforms.RandomResizedCrop(352, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

data_transforms_crop_test = transforms.Compose([
    transforms.RandomApply([
        lambda im: transforms.RandomCrop(min(im.size[0],im.size[1]))(im),
    ], p=0.35),
    transforms.RandomResizedCrop(352 ,scale=(0.6, 1.0), ratio=(0.9, 1.11)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# In[ ]:


################################################################################################### Define Cycle LR
# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


# In[ ]:


import torch.nn.utils.weight_norm as weightNorm
import torch.nn.init as init

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


# In[ ]:


################################################################################################################### Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class FbetaLoss(nn.Module):
    def __init__(self, beta=1):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss


# In[ ]:


from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), grad)
    # print('elu!!!!!!')
    loss = torch.dot(F.elu(errors_sorted)+1, grad)
    # loss = torch.dot(F.leaky_relu(errors_sorted)+1, grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# In[ ]:


class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=2)
        self.focal_loss = FocalLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, labels):
        loss_fbeta_loss = self.fbeta_loss(logits, labels)
        loss_focal = self.focal_loss(logits, labels)
        loss_lovasz = lovasz_hinge(logits, labels)
        return 1.2*loss_lovasz + loss_focal


# In[ ]:


get_ipython().system('pip install pytorchcv')


# In[ ]:


from pytorchcv.model_provider import get_model as ptcv_get_model


# In[ ]:


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmoid(x)

model_conv = ptcv_get_model("ibn_resnext101_32x4d", pretrained=True)
model_conv.features.final_pool = nn.AdaptiveAvgPool2d(1)
model_conv.output = nn.Linear(in_features=2048, out_features=1103, bias=True)


# In[ ]:


################################################################################################# Define training
from torch.nn.parallel.data_parallel import data_parallel

num_classes = 1103

model_conv.cuda()


# In[ ]:


criterion = CombineLoss()


# In[ ]:


batch_size = 32
val_batch_size = 128
num_workers = 4
num_epoch = 8


# In[ ]:


n_splts = 7
mskf = MultilabelStratifiedKFold(n_splits=n_splts, random_state=SEED)
splits = mskf.split(train_df['id'], labels_encoded)


# In[ ]:


######################################### Define find threshold function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold_search(y_pred, y_true):
    score = []
    candidates = np.arange(0.2, 0.5, 0.01)
    for th in candidates:
        yp = (y_pred > th*np.ones_like(y_pred)).astype(int)
        #print(yp)
        #print(y_true)
        score.append(fbeta_score(y_pred=yp, y_true=y_true, beta=2, average="samples"))
    score = np.array(score)
    pm = score.argmax()
    best_th, best_score = candidates[pm], score[pm]


    return best_th, best_score


# In[ ]:


def PairwiseConfusion(features, target):
    features = nn.Sigmoid()(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        batch_left = features[:int(0.5*(batch_size-1))]
        batch_right = features[int(0.5*(batch_size-1)):batch_size-1]
        target_left = target[:int(0.5*(batch_size-1))]
        target_right = target[int(0.5*(batch_size-1)):batch_size-1]

    else:
        batch_left = features[:int(0.5*batch_size)]
        batch_right = features[int(0.5*batch_size):]
        target_left = target[:int(0.5*batch_size)]
        target_right = target[int(0.5*batch_size):]

    #target_mask_t = torch.eq(target_left, target_right) # get (batchsize/2, target dim) all 1 tensor is equal 
    #target_mask_tensor_s = torch.sum(1 - target_mask_t, 1) # get (batchsize/2,) non 0 is non-equal, 0 is equal
    #target_mask_tensor_n = torch.eq(torch.zeros_like(target_mask_tensor_s), target_mask_tensor_s) # get (batchsize/2,) 0 is non-equal, 1 is equal
    #target_mask_tensor = 1 - target_mask_tensor_n # get (batchsize/2,) 1 is non-equal, 0 is equal

    #target_mask_tensor = target_mask_tensor.type(torch.cuda.FloatTensor)

    #number = target_mask_tensor.sum()

    loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)
    #loss = torch.norm((batch_left - batch_right).abs(),2, 1)*target_mask_tensor / float(number)

    return loss

def EntropicConfusion(features):
    features = nn.Sigmoid()(features)
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)


# In[ ]:


fix_fold = 0 #just run this fold
for fold_, (tr, val) in enumerate(splits):
    if(fold_<fix_fold):
        continue
    if(fold_>fix_fold):
        break
        
    ################################### split data

    dataset = iMetDataset(datafolder='../input/imet-2019-fgvc6/train/', datatype='train', idx=tr, \
            transform=data_transforms_crop, labels_dict=img_class_dict)
    val_set = iMetDataset(datafolder='../input/imet-2019-fgvc6/train/', datatype='train', idx=val,\
        transform=data_transforms_crop_test, labels_dict=img_class_dict)
    
    #model_conv.load_state_dict(torch.load("../input/pytorch-model-zoo/densenet121-fbdb23505.pth"), strict=False)
    
    ##################################
    print(time.ctime(), 'Fold:', fold_+1)
    
    valid_f2_max = -np.Inf
    # current number of tests, where validation f2 didn't increase
    p_max = 0
    lr_p_max = 0
    patience = 5
    lr_patience = 1
    # whether training should be stopped
    stop = False
    start = 0
    step_size = 8000
    
    lr = 1.5e-2
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()),\
                                   lr, weight_decay=0.0002, momentum=0.9)

    base_lr, max_lr = lr/6, lr 
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size=step_size,
                mode='exp_range', gamma=0.99994)
    
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, num_workers=num_workers)
    ####################################### Training
    for epoch in range(start, num_epoch): 

        if(epoch+1==6):
            lr /= 4
            print("lr changing from ", lr*4, " to ", lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            base_lr, max_lr = lr/6, lr 
            scheduler.base_lrs = [base_lr]
            scheduler.max_lrs = [max_lr]
        
        print("current lr is: ", lr)
        if(stop):
            break
        
        seed_everything(SEED+epoch)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        print(time.ctime(), 'Epoch:', epoch+1)
            
        train_loss = []
        train_f2 = []

        model_conv.train()
            
        for tr_batch_i, (data, target) in enumerate(train_loader):     

            scheduler.batch_step()

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            results = data_parallel(model_conv, data)

            loss = criterion(results, target.float())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model_conv.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()

            train_loss.append(loss.item()) 

            a = target.data.detach().cpu().numpy()
            b = results.detach().cpu().numpy()

            eval_step = len(train_loader)   

            if (tr_batch_i+1)%eval_step == 0:  

                model_conv.eval()
                val_loss = []
                val_f2 = []
                y_pred_val = np.zeros((len(val), 1103))
                y_true_val = np.zeros((len(val), 1103))

                with torch.no_grad():
                    for val_batch_i, (data, target) in enumerate(valid_loader):
                        data, target = data.cuda(), target.cuda()

                        results = data_parallel(model_conv, data)


                        loss = criterion(results, target.float())
                        val_loss.append(loss.item()) 


                        a = target.data.detach().cpu().numpy()
                        b = results.detach().cpu().numpy()

                        y_pred_val[val_batch_i*val_batch_size:val_batch_i*val_batch_size+b.shape[0]] = sigmoid(b)
                        y_true_val[val_batch_i*val_batch_size:val_batch_i*val_batch_size+b.shape[0]] = a

                best_threshold_val, best_score_val = threshold_search(y_pred_val, y_true_val)
                val_f2.append(best_score_val)


                print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.'%(epoch+1, tr_batch_i, np.mean(train_loss), np.mean(val_loss)))

                print('Epoch %d, batches:%d, valid f2: %.4f.'%(epoch+1, tr_batch_i, np.mean(val_f2)))

                valid_f2 = np.mean(val_f2)
                if valid_f2 > valid_f2_max:

                    print('Validation f2 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_f2_max, valid_f2))

                    torch.save(model_conv.state_dict(), "model_"+str(fold_))
                    valid_f2_max = valid_f2
                    p_max = 0
                    lr_p_max = 0

                else:
                    print("Validatione f2 doesn't increase")
                    p_max += 1
                    lr_p_max += 1
                
                if p_max > patience:
                    stop = True
                    break 

                if(lr_p_max>=lr_patience):
                    print("lr change from: ", lr, " to ", lr/4)
                    lr /= 4
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    base_lr, max_lr = lr/6, lr 
                    scheduler.base_lrs = [base_lr]
                    scheduler.max_lrs = [max_lr]

                    lr_p_max = 0
                torch.cuda.empty_cache() 
                model_conv.train()

