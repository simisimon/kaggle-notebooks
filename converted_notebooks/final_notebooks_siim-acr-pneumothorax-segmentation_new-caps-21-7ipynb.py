#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('../input/siim-acr-pneumothorax-segmentation/stage_2_images')
from mask_functions import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
get_ipython().system('pip install torch-summary')
import numpy as np
import pandas as pd
import random
import cv2 as cv
import albumentations as A
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
from torch.optim.swa_utils import AveragedModel, SWALR
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydicom
from glob import glob
import os
torch.autograd.set_detect_anomaly(True)


# In[ ]:


def squash(x, dim = -1):
    square_norm = torch.sum(x ** 2, dim = dim, keepdim = True)
    return square_norm / (0.5 + square_norm) * x / (torch.sqrt(square_norm + 1e-6) + 1e-6)
def power_squash(x, dim = -1, n = 3):
    x = squash(x, dim = dim)
    square_norm = torch.sum(x ** 2, dim = dim, keepdim = True) + 1e-6
    return square_norm ** (n / 2) * x / (torch.sqrt(square_norm) + 1e-6)


# In[ ]:


class XrayDataset(Dataset):
    def __init__(self, paths, df, target_shape = (512, 512), mode = 'train', has_upsample = True, props = (3, 2), transforms = None):
        super().__init__()
        self.paths = paths
        self.df = df
        self.target_shape = target_shape
        self.mode = mode
        self.transforms = transforms
        self.data = self.prepare_data()
        if mode == "train" and has_upsample:
            self.upsample_positive_data(props)            
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        info = self.data[idx]
        path = info['path']
        dcmdata = pydicom.dcmread(path)
        image = dcmdata.pixel_array
        if self.target_shape is not None:
            image = cv.resize(image, self.target_shape[::-1], interpolation = cv.INTER_CUBIC)
        if self.mode == "test":
            image = np.expand_dims(image, 0) / 255.
            image = image.astype(np.float32)
            return torch.from_numpy(image)
        encoded_masks = info['masks']
        mask = np.zeros(self.target_shape)
        is_mask = False
        for encoded_mask in encoded_masks:
            if encoded_mask != "-1":
                _mask = rle2mask(encoded_mask, 1024, 1024).T
                _mask = cv.resize(_mask, self.target_shape[::-1], interpolation = cv.INTER_CUBIC)
                mask[_mask > 127] = 255
                is_mask = True
        if self.transforms:
            aug = self.transforms(image = image, mask = mask)
            image = aug['image']
            mask = aug['mask']        
        image = np.expand_dims(image, 0) / 255.
        mask = np.expand_dims(mask, 0) / 255.
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        image = torch.Tensor(image).float()
        mask = torch.Tensor(mask).float()
        return image, mask, image * mask, torch.Tensor([int(is_mask)])

    def prepare_data(self):
        data = []
        image_ids = self.df['ImageId'].unique()
        for image_id in tqdm(image_ids):
            index = list(filter(lambda x: image_id in self.paths[x], range(len(self.paths))))
            if len(index) == 0:
                continue
            index = index[0]
            path = self.paths[index]
            all_chests = self.df[self.df["ImageId"] == image_id]
            encode_rois = []
            for _, row in all_chests.iterrows():
                encode_rois.append(row[" EncodedPixels"])
            data.append({
                'image_id': image_id,
                'path': path,
                'masks': encode_rois
            })
        return data
    def upsample_positive_data(self, props = (1, 1)):
        positive_data = list(filter(lambda x: x["masks"] != ["-1"], self.data))
        n_positive = len(positive_data)
        n_negative = len(self.data) - n_positive
        n_new_samples = props[0] * n_negative // props[1] - n_positive
        if n_new_samples <= 0: return
        self.data.extend(random.choices(positive_data, k = n_new_samples))


# In[ ]:


class SpatialCapsuleAttn(nn.Module):
    def __init__(self, in_capsules):
        super().__init__()
        self.in_capsules = in_capsules
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, in_capsules, 1, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, in_capsules, 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.view(b * c, d, h , w)
        avg = x * self.avg_pool(x)
        avg = torch.sum(avg, dim = 1, keepdim = True)
        t = avg.view(b * c, h * w)
        mean = t.mean(dim = 1, keepdim = True)
        std = t.std(dim = 1, keepdim = True) + 1e-6
        t = (t - mean) / std
        t = t.view(b, c, 1, h, w)
        t = t * self.weight + self.bias
        x = x.view(b, c, d, h, w) * self.sigmoid(t)
        return x
class CapXLayer(nn.Module):
    def __init__(self, in_capsules, in_cap_dim, middle_cap_dim, out_capsules, out_cap_dim, iterations = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.in_capsules = in_capsules
        self.in_cap_dim = in_cap_dim
        self.middle_cap_dim = middle_cap_dim
        self.out_capsules = out_capsules
        self.out_cap_dim = out_cap_dim
        self.iterations = iterations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_residual = (self.in_capsules * self.in_cap_dim) == (self.out_capsules * self.out_cap_dim)
        self.spatial_attn = SpatialCapsuleAttn(out_capsules)
        for _ in range(in_capsules):
            self.blocks.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_cap_dim, middle_cap_dim, kernel_size = 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(middle_cap_dim, middle_cap_dim, kernel_size = 3, padding = 1),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(middle_cap_dim, out_capsules * out_cap_dim, kernel_size = 1)
                )
            )

    def forward(self, x):
        assert x.shape[1] == self.in_capsules * self.in_cap_dim
        x_temp = x.view(x.shape[0], self.in_capsules, self.in_cap_dim, x.shape[-2], x.shape[-1])
        outputs = []
        for i, block in enumerate(self.blocks):
            inp = x_temp[:, i]
            out = block(inp)
            out = out.reshape(out.shape[0], self.out_capsules, self.out_cap_dim, out.shape[-2], out.shape[-1])
            outputs.append(out)
        u_hat = torch.stack(outputs, dim = 1)
        if self.iterations > 1:
            u_hat_squash = squash(u_hat, dim = 3)
        b = torch.zeros(out.shape[0], self.in_capsules, self.out_capsules, u_hat.shape[-2], u_hat.shape[-1]).to(self.device)
        for _ in range(self.iterations - 1):
            c = torch.sigmoid(b)
            s = torch.sum(u_hat_squash * c.unsqueeze(3), dim = 1)
            v = squash(s, dim = 2)
            b = b + torch.sum(u_hat_squash * v.unsqueeze(1), dim = 3)
        c = torch.sigmoid(b)
        s = torch.sum(u_hat * c.unsqueeze(3), dim = 1)
        s = self.spatial_attn(s)
        out = s.reshape(s.shape[0], -1, *s.shape[-2:])
        out = out + x
        return out


# In[ ]:


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.bn(self.conv(x))
class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.act = nn.ReLU(inplace = True)
    def forward(self, x):
        return self.act(self.conv(x))
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace = True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class DeConv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv = False):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.block = nn.Sequential(
                ConvBnRelu(in_channels, middle_channels),
                nn.Conv2d(middle_channels, out_channels, kernel_size = 3, padding = 1)
            )
    def forward(self, x1, x2):
        out_x1 = self.up_sample(x1)
        out = torch.cat((out_x1, x2), dim = 1)
        out = self.block(out)
        return out


# In[ ]:


class XNet(nn.Module):
    def __init__(self, iterations = 3):
        super().__init__()
        self.conv1 = ConvRelu(1, 64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1),
            CapXLayer(8, 8, 4, 8, 8, iterations = iterations),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            CapXLayer(16, 8, 4, 16, 8, iterations = iterations),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            CapXLayer(16, 16, 8, 16, 16, iterations = iterations),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )

        self.upconv3 = DeConv(256 + 128, 256, 128)
        self.upconv3_residual = nn.Sequential(
            CapXLayer(16, 8, 4, 16, 8, iterations = iterations),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.upconv2 = DeConv(128 + 64, 128, 64)
        self.upconv2_residual = nn.Sequential(
            CapXLayer(8, 8, 4, 8, 8, iterations = iterations),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.upconv1 = DeConv(128, 96, 64)
        self.upconv1_residual = nn.Sequential(
            CapXLayer(8, 8, 4, 8, 8, iterations = iterations),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.re_conv = nn.Sequential(
            ConvRelu(64, 32),
            ConvRelu(32, 32),
            nn.Conv2d(32, 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            ConvRelu(64, 32),
            ConvRelu(32, 32),
            nn.Conv2d(32, 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace = True)
        self.prob_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, input, mask = None):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_up3   = self.upconv3(out_conv4, out_conv3)
        out_up3_r = self.upconv3_residual(out_up3)
        out_up2   = self.upconv2(out_up3_r, out_conv2)
        out_up2_r = self.upconv2_residual(out_up2)
        out_up1   = self.upconv1(out_up2_r, out_conv1)
        out_up1_r = self.upconv1_residual(out_up1)
        out_prob = self.prob_head(out_conv4)
        out = self.decoder(out_up1_r)
        if mask is not None:
            out_re = self.re_conv(out_up1_r * mask)
        else:
            out_re = None
        return out_prob, out, out_re


# In[ ]:


# summary(XNet(), (1, 128, 128))


# In[ ]:


def get_margin_loss(m_pos, m_neg, scale = .5):
    def margin_loss(inputs, targets):
        diff = torch.where(targets == 1, m_pos - inputs, inputs - m_neg)
        diff = F.relu(diff)
        diff = diff ** 2
        diff = torch.where(targets == 1, diff, scale * diff)
        return diff.mean()
    return margin_loss
def get_acc_score(input, target):
    pred = (input > 0.5).to(torch.float32)
    return torch.mean((pred == target).to(torch.float32))
def dice_score(inputs, targets, smooth = 1e-6):
    num = (2 * (inputs * targets).sum(dim = [1, 2, 3]) + smooth) 
    de = (inputs.sum(dim = [1, 2, 3]) + targets.sum(dim = [1, 2, 3]) + smooth)
    return torch.mean(num / de)
def val_dice_score(inputs, targets, threshold = 0.5, smooth = 1e-6):
    inputs = torch.where(inputs > threshold, 1., 0.)
    num = (2 * (inputs * targets).sum(dim = [1, 2, 3]) + smooth) 
    de = (inputs.sum(dim = [1, 2, 3]) + targets.sum(dim = [1, 2, 3]) + smooth)
    return torch.mean(num / de)
def dice_loss(inputs, targets, smooth = 1e-6):
    return 1 - dice_score(inputs, targets, smooth)
def focal_loss(inputs, targets, alpha = 0.8, gamma = 2, reduction = 'mean'):
    assert reduction in ['none', 'mean', 'sum']
    loss = F.binary_cross_entropy(inputs, targets, reduction = 'none')
    coeff = (1 - inputs) ** gamma
    loss = coeff * loss
    focal_loss = torch.where(targets == 1, loss, loss * alpha)
    if reduction == "none": return focal_loss
    elif reduction == "mean": return focal_loss.mean()
    elif reduction == "sum": return focal_loss.sum()
    else:
        raise Exception()
class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth = 1e-6):
        
        intersection = (inputs * targets).sum(dim = [1, 2, 3])
        total = (inputs + targets).sum(dim = [1, 2, 3])
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU.mean()


# In[ ]:


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou = IoULoss()
        self.mse = nn.MSELoss()
    def forward(self, input1, target1, input2, target2, input3 = None, target3 = None, weight_factors = [2, 1, 2, 2]):
        return self.bce(input1, target1) * weight_factors[0] + \
                self.iou(input1, target1) * weight_factors[1] + \
                focal_loss(input1, target1) * weight_factors[2] + \
                self.bce(input2, target2) * weight_factors[3] + (self.mse(input3, target3) * 0.005 if input3 is not None else 0)
                    


# In[ ]:


paths = glob(os.path.join('..', 'input', 'siim-train-test', 'dicom-images-train', '*', '*', '*.dcm'))
df = pd.read_csv("../input/siim-train-test/train-rle.csv")


# In[ ]:


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT_LR = 1e-5
BATCH_SIZE = 12
TRAIN_RATE = 0.7
EPOCHS = 8


# In[ ]:


train_paths, test_paths = train_test_split(paths, train_size = TRAIN_RATE, random_state = 22)
np.random.seed(22)
np.random.shuffle(test_paths)
test_size = len(test_paths) // 2
val_paths = test_paths[:test_size]
test_paths = test_paths[test_size:]


# In[ ]:


transforms = A.Compose([
    A.HorizontalFlip(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        ], p = 0.4),
    A.OneOf([
        A.Blur(blur_limit = 5), 
        A.MedianBlur(blur_limit = 5)
    ], p = 0.4),
    A.ShiftScaleRotate(shift_limit = 0.01, rotate_limit = 15, border_mode = cv.BORDER_CONSTANT)
])


# In[ ]:


train_dataset = XrayDataset(train_paths, df, target_shape = (256, 256), props = (11, 9), transforms = transforms)
val_dataset = XrayDataset(val_paths, df, target_shape = (256, 256), mode = 'val')


# In[ ]:


train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)


# In[ ]:


model = XNet(1).to(DEVICE)
# swa_model = AveragedModel(model)
optimizer = torch.optim.Adam(model.parameters(), lr = INIT_LR)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scheduler, verbose = True)
# swa_scheduler = SWALR(optimizer, anneal_epochs = 5, swa_lr = 1e-3)
# criterion = ComboLoss().to(DEVICE)


# In[ ]:


criterion = ComboLoss().to(DEVICE)


# In[ ]:


model.load_state_dict(torch.load('../input/newcaps2107/checkpoint_epoch_r1_29_07.tar', map_location = DEVICE)['model'])


# In[ ]:


ITERS_PER_EPOCH = len(train_dataset) // BATCH_SIZE


# In[ ]:


# get_weight_prob_decay = lambda x: max(0.4, -1.6 * x / (10 * ITERS_PER_EPOCH) + 2)
# iter_count = 0
for epoch in range(EPOCHS):
    model.train()
    bar = tqdm(enumerate(train_loader), total = ITERS_PER_EPOCH)
    for i, (images, masks, rois, lbs) in bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        rois = rois.to(DEVICE)
        lbs = lbs.to(DEVICE)
        optimizer.zero_grad()
        pred_probs, pred_masks, pred_rois = model(images)
        loss = criterion(pred_masks, masks, pred_probs, lbs, pred_rois, rois, weight_factors = [3, 2, 3, 1])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_dice = val_dice_score(pred_masks, masks).cpu().item()
            acc = get_acc_score(pred_probs, lbs).cpu().item()
            dice = dice_score(pred_masks, masks).cpu().item()
#         iter_count += 1
        bar.set_description(f'epoch {epoch + 1} iter {i + 1} loss {loss.cpu().detach().item(): .6f} dice {dice:.4f} acc {acc: .4f} val_dice {val_dice :.4f}')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': i
        }, f'checkpoint_epoch_{epoch + 1}.tar')
    model.eval()
    with torch.no_grad():
        acc_loss = 0
        acc_dice = 0
        count = 0
        acc_acc = 0
        for images, masks, rois, lbs in tqdm(val_loader):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
#             rois = rois.to(DEVICE)
            lbs = lbs.to(DEVICE)
            pred_probs, pred_masks, _ = model(images)
            loss = criterion(pred_masks, masks, pred_probs, lbs)
            acc_loss += loss.cpu().item()         
            acc_dice += val_dice_score(pred_masks, masks).cpu().item()
            acc_acc  += get_acc_score(pred_probs, lbs).cpu().item()
            count += 1
        acc_loss /= count
        acc_dice /= count
        acc_acc  /= count
        with open('log.txt', 'a') as f:
            f.write(f'[VAL] epoch {epoch + 1} loss {acc_loss: .4f} dice {acc_dice: .4f} acc {acc_acc: .4f}\n')
        print(f'[VAL] epoch {epoch + 1} loss {acc_loss: .4f} dice {acc_dice: .4f} acc {acc_acc: .4f}')


# In[ ]:


# pred_probs


# In[ ]:


# lbs


# In[ ]:


# idx = 1
# plt.imshow(pred_masks.detach().cpu().numpy()[idx, 0], 'gray')


# In[ ]:


# pred_masks.detach().cpu().numpy()[idx, 0].max()


# In[ ]:


# plt.imshow(masks.cpu().numpy()[idx, 0], 'gray')


# In[ ]:


# torch.save({
#         'epoch': epoch,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'iter': i
#         }, f'checkpoint_epoch_r1_29_07.tar')


# In[ ]:


# torch.cuda.empty_cache()


# In[ ]:


# with torch.no_grad():
#         acc_loss = 0
#         acc_dice = 0
#         count = 0
#         acc_acc = 0
#         for images, masks, rois, lbs in tqdm(val_loader):
#             images = images.to(DEVICE)
#             masks  = masks.to(DEVICE)
# #             rois = rois.to(DEVICE)
#             lbs = lbs.to(DEVICE)
#             pred_probs, pred_masks, _ = model(images)
#             loss = criterion(pred_masks, masks, pred_probs, lbs)
#             acc_loss += loss.cpu().item()         
#             acc_dice += val_dice_score(pred_masks, masks).cpu().item()
#             acc_acc  += get_acc_score(pred_probs, lbs).cpu().item()
#             count += 1
#         acc_loss /= count
#         acc_dice /= count
#         acc_acc  /= count
#         with open('log.txt', 'a') as f:
#             f.write(f'[VAL] epoch {epoch + 1} loss {acc_loss: .4f} dice {acc_dice: .4f} acc {acc_acc: .4f}\n')
#         print(f'[VAL] epoch {epoch + 1} loss {acc_loss: .4f} dice {acc_dice: .4f} acc {acc_acc: .4f}')


# In[ ]:




