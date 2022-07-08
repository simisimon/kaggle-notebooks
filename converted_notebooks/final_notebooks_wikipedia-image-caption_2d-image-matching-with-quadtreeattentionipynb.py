#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# 
# In this notebook I'm presenting a simplified version of this competition: how to match an image with a subcrop of itself.
# 
# You can find explanations about my motivations here : https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/323403. My final goal is obviously much more complex but I first need to solve the basics.
# 
# # QuadTreeAttention
# 
# I'm showing here how to use a pretrained version of QuadTreeAttention : https://arxiv.org/abs/2201.02767
# 
# Eventhough the current training pipeline does not seem to work I'll update it as soon as I find the solution. Any help would be appreciated.
# 
# I think it might be worse to give it a shot in this competition as is.
# 
# # Weird behaviors
# 
# - The pretrained model is actually pretty good at finding crops of different sizes inside the orginal image. However, as soon as you perform a simple 90° rotation nothing seems to be working anymore. How can SOTA model not be robust to 90° rotations? Is this common in image matching?
# 
# - As soon as I try to fine tune the model (with the hope in the future to be able to make it robust to rotations and many more augmentations) everything collapses. I'd be really grateful to anyone helping me finding where I got things wrong.

# In[ ]:


get_ipython().system('pip -q install -U kornia')
get_ipython().system('pip -q install kornia-moons')
get_ipython().system('pip3 -q install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html')
get_ipython().system('pip -q install ninja')
get_ipython().system('pip -q install loguru')
get_ipython().system('pip -q install einops')
get_ipython().system('pip -q install timm')


# In[ ]:


get_ipython().system('cp -r ../input/quadtreeattention/ ../working/ # input folder is read only')
get_ipython().system(' cd ../working/quadtreeattention/QuadTreeAttention-master/QuadTreeAttention/ && pip install .')


# In[ ]:


import cv2
import kornia as K
import kornia.feature as KF
from kornia.feature.loftr import LoFTR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import glob
import random

import torchvision
import kornia_moons.feature as KMF
from PIL import Image

import sys
sys.path.append('../working/quadtreeattention/QuadTreeAttention-master/')
sys.path.append('../working/quadtreeattention/QuadTreeAttention-master/FeatureMatching/')
sys.path.append('../working/quadtreeattention/QuadTreeAttention-master/QuadTreeAttention/')


# # Utilities functions

# In[ ]:


from FeatureMatching.src.utils.plotting import make_matching_figure
from pathlib import Path
import matplotlib.cm as cm

def get_images_path(path):
    path_to_imgs = [str(p) for p in Path(path).rglob("**/images/*.jpg")]
    # remove macros
    return path_to_imgs

def load_torch_image(fname, device="cuda"):
    """
        Load an image
    """
    img = K.image_to_tensor(cv2.imread(fname), False).float().to(device) /255.
    img = K.color.bgr_to_rgb(img)
    return img.squeeze()

def match_and_draw_dataset(matcher, dataset, conf_thresh=0, max_img=20, device="cuda", rotate=False):
    """
    Match and draw from all elements in a dataset
    for a specific model

    Parameters
    ----------
    matcher (torch nn module): a matcher model (LOFTR, QUADTREE)
    dataset (torch dataset): a dataset with matching pairs
    conf_thresh (float): between 0 and 1, confidence of shown matches
    max_img (int) : max images to plot
    device (str) : device to make inference
    """
    matcher.eval()
    matcher.to(device)

    for idx in range(min(len(dataset), max_img)):
        batch = dataset[idx]
        batch["image0"] = batch['image0'].unsqueeze(0).to(device)
        batch["image1"] = batch['image1'].unsqueeze(0).to(device)

        img0_raw = np.tile(batch["image0"].squeeze(0).cpu().numpy().transpose(1, 2, 0), 3)
        img1_raw = np.tile(batch["image1"].squeeze(0).cpu().numpy().transpose(1, 2, 0), 3)

        with torch.no_grad():
            matcher.eval()
            matcher.to(device)
            matcher(batch)
            mconf = batch['mconf'].cpu().numpy()
            mask_conf = mconf > conf_thresh
            mconf = mconf[mask_conf]
            mkpts0 = batch['mkpts0_f'].cpu().numpy()[mask_conf]
            mkpts1 = batch['mkpts1_f'].cpu().numpy()[mask_conf]
            
            color = cm.jet(mconf)

        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
        
        if rotate:
            # Look at transposition
            batch["image0"] = batch['image0']
            batch["image1"] = torch.transpose(batch['image1'], 2, 3)
            img0_raw = np.tile(batch["image0"].squeeze(0).cpu().numpy().transpose(1, 2, 0), 3)
            img1_raw = np.tile(batch["image1"].squeeze(0).cpu().numpy().transpose(1, 2, 0), 3)

            with torch.no_grad():
                matcher(batch)
                mconf = batch['mconf'].cpu().numpy()
                mask_conf = mconf > conf_thresh
                mconf = mconf[mask_conf]
                mkpts0 = batch['mkpts0_f'].cpu().numpy()[mask_conf]
                mkpts1 = batch['mkpts1_f'].cpu().numpy()[mask_conf]
                
                color = cm.jet(mconf)

            text = [
                'LoFTR transpose',
                'Init Matches: {}'.format(len(mask_conf)),
                'Thresh Matches: {}'.format(len(mkpts0)),
            ]
            fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
    return


# # Synthetic Dataset
# 
# The goal here is simply to pick an image as input, sample a smaller crop inside and try to match the crop with the image.
# 
# 
# There must be something wrong here as the training breaks everything.
# 
# 

# In[ ]:


from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self,
                 path_to_imgs,
                 img_size=(320, 320), # (64*6, 48*6) (640, 480)
                 crop_ratio=2,
                 augment_fn=None):
        """
        Creates artificial dataset.
        
        Args:
            - path_to_imgs (list): iterable of path to images
            - img_resize (int, int): Final size of image shown to model (should be divisible by 64?)
            - augment_fn (callable, optional): augments images with pre-defined visual effects.
            - crop_ratio (int): ratio between original image and image crop
        """
        super().__init__()
        self.path_to_imgs = path_to_imgs
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.path_to_imgs)
    
    def __getitem__(self, idx):
        img_path = self.path_to_imgs[idx]
        img_name = img_path.split('/')[-1]
        
        image0 = self.load_torch_image(img_path) #(h, w)
        h_init, w_init = image0.shape[1:]
        
        # resize imgs
        image0 = torchvision.transforms.Resize(self.img_size)(image0)
        
        
        scale0 = image0.shape[1:] #(h, w)
        h0, w0 = scale0
        depth0 = torch.ones(scale0) # everything to 1?
        # intrinsinc matrix
        K_0 = torch.tensor([[h0/h_init, 0, h0/2], # h0/h_init
                            [0, w0/w_init, w0/2], # w0/w_init
                            [0, 0, 1]
                           ]) # 1 to 1 pixel, centered position
        # rotation matrix
        R0 = np.diag([1, 1, 1]) # no rotation
        # translation vector
        Tv0 = np.array([[h_init/2, w_init/2, 1]])
        T0 = np.concatenate((R0, Tv0.T), axis=1)
        T0 = np.concatenate((T0, np.asarray([[0, 0, 0, 1]])), axis=0)
        
        # get a random crop
        image1, crop_pos = self.basic_crop(image0)
        hc, wc = image1.shape[1:]
        image1 = torchvision.transforms.Resize(self.img_size)(image1)
        
        depth1 = torch.ones(scale0) # everything to 1?
        K_1 = torch.tensor([[h0/hc, 0, h0/2], # h0/hc
                            [0, w0/wc, w0/2], # w0/wc
                            [0, 0, 1]
                           ]) # 1 to 1 pixel, centered position
      
        R1 = np.diag([1, 1, 1]) # no rotation
        Tv1 = np.array([[np.mean(crop_pos[0]), np.mean(crop_pos[1]), 1]])
        T1 = np.concatenate((R1, Tv1.T), axis=1)
        T1 = np.concatenate((T1, np.asarray([[0, 0, 0, 1]])), axis=0)
        
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()
        
        if self.augment_fn is not None:
            image0 = image0.numpy().transpose(1, 2, 0).astype(np.uint8)
            image1 = image1.numpy().transpose(1, 2, 0).astype(np.uint8)

            image0 = self.augment_fn()(image=image0)["image"]
            image1 = self.augment_fn()(image=image1)["image"]

            image0 = torch.tensor(image0.transpose(2, 0, 1))
            image1 = torch.tensor(image1.transpose(2, 0, 1))
            
        data = {
            'image0': image0.float().mean(axis=0, keepdim=True) / 255,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1.float().mean(axis=0, keepdim=True) / 255,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
#             'scale0': scale0,  # [scale_w, scale_h]
#             'scale1': scale1,
            'dataset_name': "scannet", #'SyntheticDataset',
            'scene_id': idx,
            'pair_id': idx,
            'pair_names': (img_name, "macro_"+img_name),
        }
        return data
    
    def load_torch_image(self, fname):
        img = K.image_to_tensor(cv2.imread(fname), False)#.float()
        img = K.color.bgr_to_rgb(img).squeeze()
        c, h, w = img.shape

        return img
    
    def basic_crop(self, img):
        """
        This simply takes a crop without changing aspect ratio
        but following crop ratio for the size of the crop
        """
        c, h, w = img.shape
        crop_size = h // self.crop_ratio, w // self.crop_ratio
        
        max_x = max(1, h - crop_size[0])
        max_y = max(1, w - crop_size[0])

        rand_x = np.random.randint(max_x)
        rand_y = np.random.randint(max_y)

        end_x = rand_x + crop_size[0]
        end_y = rand_y + crop_size[1]

        return img[:, rand_x:end_x, rand_y:end_y], ((rand_x,end_x), (rand_y,end_y))


# In[ ]:


# data module

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BasicDataModule(pl.LightningDataModule):
    "Dummy module for dataloaders where train, validation, test are the same"
    def __init__(self, path_to_imgs, transforms=None,
                 batch_size: int = 4, crop_ratio=2, num_workers = 4):
        super().__init__()
        self.batch_size = batch_size
        self.path_to_imgs = path_to_imgs
        self.crop_ratio = crop_ratio
        self.transforms = transforms
        self.num_workers = num_workers

    def setup(self, stage):
        return

    def train_dataloader(self):
        return DataLoader(SyntheticDataset(self.path_to_imgs,
                                                 augment_fn=self.transforms,
                                                 crop_ratio=self.crop_ratio
                                                ),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(SyntheticDataset(self.path_to_imgs,
                                           augment_fn=self.transforms),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(SyntheticDataset(self.path_to_imgs, augment_fn=self.transforms),
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(SyntheticDataset(self.path_to_imgs),
                          batch_size=self.batch_size, num_workers=self.num_workers)


# # Config for pretrained Quadtree

# In[ ]:


from FeatureMatching.src.config.default import get_cfg_defaults
config = get_cfg_defaults()
CROP_RATIO = 3 # 2
TRANSFORMS = None
NB_EPOCHS = 5


# INDOOT lofrt_ds_quadtree config
config.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
config.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
config.LOFTR.RESNETFPN.INITIAL_DIM = 128
config.LOFTR.RESNETFPN.BLOCK_DIMS=[128, 196, 256]
config.LOFTR.COARSE.D_MODEL = 256
config.LOFTR.COARSE.BLOCK_TYPE = 'quadtree'
config.LOFTR.COARSE.ATTN_TYPE = 'B'
config.LOFTR.COARSE.TOPKS=[32, 16, 16]
config.LOFTR.FINE.D_MODEL = 128
config.TRAINER.WORLD_SIZE = 1 # 8
config.TRAINER.CANONICAL_BS = 32
config.TRAINER.TRUE_BATCH_SIZE = 1
_scaling = 1
config.TRAINER.ENABLE_PLOTTING = False
config.TRAINER.SCALING = _scaling
config.TRAINER.TRUE_LR = 1e-3 # 1e-4 config.TRAINER.CANONICAL_LR * _scaling
config.TRAINER.WARMUP_STEP = 0 #math.floor(config.TRAINER.WARMUP_STEP / _scaling)


# In[ ]:


# arguments 

import argparse
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default="../input/kornia-loftr/outdoor_ds.ckpt",
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    parser = pl.Trainer.add_argparse_args(parser)
    nb_epochs = NB_EPOCHS # 20
    return parser.parse_args(f'../input/loftrutils/LoFTR-master/LoFTR-master/configs/data/megadepth_trainval_640.py ../input/loftrutils/LoFTR-master/LoFTR-master/configs/loftr/outdoor/loftr_ds_dense.py --exp_name test --gpus 0 --num_nodes 0 --accelerator gpu --batch_size 1 --check_val_every_n_epoch 1 --log_every_n_steps 1 --flush_logs_every_n_steps 1 --limit_val_batches 1 --num_sanity_val_steps 10 --benchmark True --max_epochs {nb_epochs}'.split())

from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
import pprint
args = parse_args()
rank_zero_only(pprint.pprint)(vars(args))


# In[ ]:


train_images = get_images_path('../input/image-matching-challenge-2022/train/brandenburg_gate/')[:5]


# # Define Trainer

# In[ ]:


from FeatureMatching.src.utils.profiler import build_profiler
from FeatureMatching.src.lightning.lightning_loftr import PL_LoFTR
from loguru import logger as loguru_logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


# lightning module
disable_ckpt = True
profiler_name = None # help='options: [inference, pytorch], or leave it unset
profiler = build_profiler(profiler_name)
model = PL_LoFTR(config,
                 pretrained_ckpt= "../input/quadtreecheckpoints/outdoor_quadtree.ckpt", # args.ckpt_path, from scratch atm
                 profiler=profiler)
loguru_logger.info(f"LoFTR LightningModule initialized!")

# lightning data
data_module = BasicDataModule(train_images, transforms=TRANSFORMS, crop_ratio=CROP_RATIO)
loguru_logger.info(f"LoFTR DataModule initialized!")

# TensorBoard Logger
logger = TensorBoardLogger(save_dir="../working/logs",
                           name="test_kaggle",
                           default_hp_metric=False)
ckpt_dir = Path(logger.log_dir) / 'checkpoints'

# Callbacks
# TODO: update ModelCheckpoint to monitor multiple metrics
ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
                                save_last=True,
                                dirpath=str(ckpt_dir),
                                filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [lr_monitor]
if not disable_ckpt:
    callbacks.append(ckpt_callback)

# Lightning Trainer
trainer = pl.Trainer.from_argparse_args(
                    args=args,
#                     plugins=DDPPlugin(find_unused_parameters=False,
#                                       num_nodes=num_nodes,
#                                       sync_batchnorm=False, #config.TRAINER.WORLD_SIZE > 0
#                                      ),
                    gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
                    callbacks=callbacks,
                    logger=logger,
#                     sync_batchnorm=False, #config.TRAINER.WORLD_SIZE > 0,
                    replace_sampler_ddp=False,  # use custom sampler
#                     reload_dataloaders_every_epoch=False,  # avoid repeated samples!
                    weights_summary='full',
                    profiler=profiler)


# # BEFORE TRAINING (rely on pretraining weights)
# 
# As you can see, the pretrained model is pretty good with basic crops.
# 
# However, even a basic 90° rotation make things wrong.

# In[ ]:


CONF_THRESH = 0.
MAX_IMG = 5


# In[ ]:


# Have a look at training images
dataset = SyntheticDataset(train_images, augment_fn=None, crop_ratio=CROP_RATIO)
match_and_draw_dataset(matcher=model.matcher,
                       dataset=dataset,
                       conf_thresh=CONF_THRESH,
                       max_img=MAX_IMG,
                       rotate=True
                      )


# In[ ]:


# Have a look at validaiton images

path_to_imgs = get_images_path("../input/image-matching-challenge-2022/train/notre_dame_front_facade/")
dataset = SyntheticDataset(path_to_imgs, augment_fn=None, crop_ratio=CROP_RATIO)
match_and_draw_dataset(matcher=model.matcher,
                       dataset=dataset,
                       conf_thresh=CONF_THRESH,
                       max_img=MAX_IMG,
                       rotate=True
                      )


# # Now let's train for only a few epochs

# In[ ]:


trainer.running_sanity_check = False
loguru_logger.info(f"Trainer initialized!")
loguru_logger.info(f"Start training!")
trainer.fit(model, datamodule=data_module)


# In[ ]:


# Have a look at training images
dataset = SyntheticDataset(train_images, augment_fn=None, crop_ratio=CROP_RATIO)
match_and_draw_dataset(matcher=model.matcher,
                       dataset=dataset,
                       conf_thresh=CONF_THRESH,
                       max_img=MAX_IMG,
                       rotate=True
                      )


# In[ ]:




