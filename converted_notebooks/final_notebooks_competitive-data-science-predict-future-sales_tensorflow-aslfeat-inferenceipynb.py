#!/usr/bin/env python
# coding: utf-8

# In this new notebook I'am using the Tensorflow implementation of 'ASLFeat: Learning Local Features of Accurate Shape and Localization' as described in the paper [https://arxiv.org/abs/2003.10071](https://arxiv.org/abs/2003.10071) to predict the fundamental matrix for this competition.
# 
# The original implementation in [github](https://github.com/lzx551402/ASLFeat) is made in Tensorflow 1.X. I've used the available tools to upgrade the code to Tensorflow 2.X and made some slight modifications to be able to run it. More improvements could be made to the code.
# 
# Another change compared to the original code is that I'am using the cv2.FLANNBasedMatcher instead of the cv2.BFMatcher.
# 
# I hope you like the code .. and if you do please give it an upvote ;-)

# In[ ]:


# Import Modules
import numpy as np 
import pandas as pd
import csv
import cv2
import gc
import tensorflow as tf
import sys
import yaml
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Import ASLFeat
sys.path.append('../input/aslfeat')
from models import get_model

# Disable Eager Execution
tf.compat.v1.disable_eager_execution()


# ## Test Data

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


# ## Support Functions

# In[ ]:


def FlattenMatrix(M, num_digits = 8):
    '''Convenience function to write CSV files.'''    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])

def get_fundamental_matrix(kpts1, kpts2, err_thld):    
    if len(kpts1) > 7:
        F, inliers = cv2.findFundamentalMat(kpts1, 
                                            kpts2, 
                                            cv2.USAC_MAGSAC, 
                                            ransacReprojThreshold = err_thld, 
                                            confidence = 0.99999, 
                                            maxIters = 100000) # Lower maxIters to increase speed / lower accuracy
        return F, inliers
    else:
        return np.random.rand(3, 3), None


# ## Setup ASLFeat

# In[ ]:


# Import Config
with open('../input/aslfeat/configs/matching_eval2.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# There are 2 checkpoints in the pretrained folder. This one should be the best...
aslfeat_model_path = '../input/aslfeat/pretrained/aslfeatv2/model.ckpt-60000' 
config['model_path'] = aslfeat_model_path
config['net']['config']['kpt_n'] = 8000 # Sames as original config ... just for convenience ;-)

# Summary config
print(config)


# In[ ]:


# Create Model
model = get_model('feat_model')(aslfeat_model_path, **config['net'])


# In[ ]:


# ASLFeat Functions    
def load_imgs(img_paths):
    rgb_list = []
    gray_list = []
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
        
    return rgb_list, gray_list

def extract_local_features(gray_list):    
    descs = []
    kpts = []
    
    for gray_img in gray_list:
        desc, kpt = [], []
        desc, kpt, _ = model.run_test_data(gray_img)
        descs.append(desc)
        kpts.append(kpt)
        
    return descs, kpts

class MatcherWrapper(object):
    """OpenCV matcher wrapper."""

    def __init__(self):
        # Swapped BFMatcher to FlannBasedMatcher
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
        search_params = dict(checks = 125)   # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def get_matches(self, feat1, feat2, cv_kpts1, cv_kpts2, ratio = 0.8, cross_check = True, err_thld = 0.5):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """
        
        init_matches1 = self.matcher.knnMatch(feat1, feat2, k = 2)
        init_matches2 = self.matcher.knnMatch(feat2, feat1, k = 2)

        good_matches = []

        for i in range(len(init_matches1)):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio is not None and ratio < 1:
                cond2 = init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance
                cond *= cond2
            if cond:
                good_matches.append(init_matches1[i][0])

        if type(cv_kpts1) is list and type(cv_kpts2) is list:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in good_matches])
        elif type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            good_kpts1 = np.array([cv_kpts1[m.queryIdx] for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx] for m in good_matches])
        else:
            good_kpts1 = np.empty(0)
            good_kpts2 = np.empty(0)
            
        # Calculate Fundamental Mask and inliers
        F, mask = get_fundamental_matrix(good_kpts1, good_kpts2, err_thld)
        return F, good_matches, mask
            
    def draw_matches(self, img1, cv_kpts1, img2, cv_kpts2, good_matches, mask, match_color = (0, 255, 0), pt_color = (0, 0, 255)):
        """Draw matches."""
        if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
            cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1) for i in range(cv_kpts1.shape[0])]
            cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1) for i in range(cv_kpts2.shape[0])]
            
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor = match_color,
                                  singlePointColor = pt_color,
                                  matchesMask = mask.ravel().tolist(), flags=4)
        return display


# In[ ]:


def get_aslfeat_fmatrix(batch_id, img_id1, img_id2, plot = False):
    image_fpath_1 = f'{src}/test_images/{batch_id}/{img_id1}.png'
    image_fpath_2 = f'{src}/test_images/{batch_id}/{img_id2}.png'
    
    # Load Test Image Pair
    rgb_list, gray_list = load_imgs([image_fpath_1, image_fpath_2])    

    # Extract Local Features
    descs, kpts = extract_local_features(gray_list)
        
    # feature matching and draw matches.
    matcher = MatcherWrapper()
    fundamental_matrix, match, mask = matcher.get_matches(descs[0], descs[1], kpts[0], kpts[1],
                                                          ratio = None, 
                                                          cross_check = True, # I'am only using the Cross Check...not the ratio test.
                                                          err_thld = 0.20)

    if plot:
        # Draw Matches
        disp = matcher.draw_matches(rgb_list[0], kpts[0], rgb_list[1], kpts[1], match, mask)

        # Save Plot Image
        output_name = f'{batch_id}.jpg'
        plt.imsave(output_name, disp) 
        
        # Show Plot Image
        plt.figure(figsize = (18, 18))
        plt.imshow(disp)
        plt.show() 
        
        # Info
        print(f'n_putative: {len(match)}  n_inlier: {np.count_nonzero(mask)}')

    return fundamental_matrix


# ## Predictions

# In[ ]:


f_matrix_dict = {}
for i, row in tqdm(enumerate(test_samples)):
    sample_id, batch_id, img_id1, img_id2 = row

    # Set Plot
    plot = False
    if i < 3: plot = True
        
    # Get Fundamental matrix with ASLFeat And FLANNBasedMatcher
    f_matrix_dict[sample_id] = get_aslfeat_fmatrix(batch_id, img_id1, img_id2, plot)
        
    # Mem Cleanup
    gc.collect()


# ## Create Submission

# In[ ]:


# Write Submission File   
with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in f_matrix_dict.items():                
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')


# In[ ]:


# Summary
sub = pd.read_csv('submission.csv')
sub.head()

