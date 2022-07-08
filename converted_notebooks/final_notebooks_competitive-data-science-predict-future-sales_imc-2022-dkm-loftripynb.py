#!/usr/bin/env python
# coding: utf-8

# # LoFTR: Detector-Free Local Feature Matching with TransformersLoFTR 

# In[ ]:


dry_run = False
get_ipython().system('pip install ../input/kornia-loftr/kornia-0.6.4-py2.py3-none-any.whl')
get_ipython().system('pip install ../input/kornia-loftr/kornia_moons-0.1.9-py3-none-any.whl')


# In[ ]:


import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc


# In[ ]:


device = torch.device('cuda')
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("../input/kornia-loftr/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()


# In[ ]:


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)


# In[ ]:


F_dict_L = {}
import time
for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row
    # Load the images.
    st = time.time()
    image_1 = load_torch_image(f'{src}/test_images/{batch_id}/{image_1_id}.png', device)
    image_2 = load_torch_image(f'{src}/test_images/{batch_id}/{image_2_id}.png', device)
    print(image_1.shape)
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1), 
              "image1": K.color.rgb_to_grayscale(image_2)}

    with torch.no_grad():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    if len(mkpts0) > 7:
        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.1900, 0.9999, 200000)
        inliers = inliers > 0
        assert F.shape == (3, 3), 'Malformed F?'
        F_dict_L[sample_id] = F
    else:
        F_dict_L[sample_id] = np.zeros((3, 3))
        continue
    gc.collect()
    nd = time.time()    


# # DKM: Deep Kernelized Dense Geometric Matching

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, csv
from PIL import Image
import cv2, gc
import matplotlib.pyplot as plt
import torch
#sys.path.append('/kaggle/input/imc2022-dependencies/DKM/')
sys.path.append('/kaggle/input/dkm-models/DKM/')

dry_run = False


# In[ ]:


get_ipython().system('mkdir -p pretrained/checkpoints')
get_ipython().system('cp /kaggle/input/dkm-models/pretrained/dkm.pth pretrained/checkpoints/dkm_base_v11.pth')

get_ipython().system('pip install -f /kaggle/input/dkm-models/wheels --no-index einops')
get_ipython().system('cp -r /kaggle/input/dkm-models/DKM/ /kaggle/working/DKM/')
get_ipython().system('cd /kaggle/working/DKM/; pip install -f /kaggle/input/dkm-models/wheels -e .')


# In[ ]:


import torch
if not torch.cuda.is_available():
    print('You may want to enable the GPU switch?')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.hub.set_dir('/kaggle/working/pretrained/')
from dkm import dkm_base
model = dkm_base(pretrained=True, version="v11").to(device).eval()
# model.load_state_dict(torch.load(WEIGHTS))


# In[ ]:


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


# In[ ]:


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]

if dry_run:
    for sample in test_samples:
        print(sample)


# In[ ]:


F_dict_D = {}
for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row

    img1 = cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.png') 
    img2 = cv2.imread(f'{src}/test_images/{batch_id}/{image_2_id}.png')
        
    img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    dense_matches, dense_certainty = model.match(img1PIL, img2PIL)
    dense_certainty = dense_certainty.sqrt()
    sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, 1999)
    
    mkps1 = sparse_matches[:, :2]
    mkps2 = sparse_matches[:, 2:]
    
    h, w, c = img1.shape
    mkps1[:, 0] = ((mkps1[:, 0] + 1)/2) * w
    mkps1[:, 1] = ((mkps1[:, 1] + 1)/2) * h

    h, w, c = img2.shape
    mkps2[:, 0] = ((mkps2[:, 0] + 1)/2) * w
    mkps2[:, 1] = ((mkps2[:, 1] + 1)/2) * h

    F, mask = cv2.findFundamentalMat(mkps1, mkps2, cv2.USAC_MAGSAC, 0.1999, 0.8873, 250_000)

    
    good = F is not None and F.shape == (3,3)
    
    if good:
        F_dict_D[sample_id] = F
    else:
        F_dict_D[sample_id] = np.zeros((3, 3))
        continue

    gc.collect()  


# # OUTPUT

# In[ ]:


F_dict_L 


# In[ ]:


F_dict_D


# In[ ]:


_F_dict = {}
for k in F_dict_L.keys():
    _F_dict[k] = (F_dict_L[k] + F_dict_D[k])/F_dict_L[k]


# In[ ]:


with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in _F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

if dry_run:
    get_ipython().system('cat submission.csv')


# In[ ]:


pd.read_csv('submission.csv')

