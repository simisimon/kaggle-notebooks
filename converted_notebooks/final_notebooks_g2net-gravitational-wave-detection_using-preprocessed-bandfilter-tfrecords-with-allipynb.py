#!/usr/bin/env python
# coding: utf-8

# * Hi. In this notebook, you can conveniently using Preprocessed Bandfiltered tfrecords.
# * In case of TPU or others, Sometimes applying BandPass Filter on the fly is hard.
# * So i made bandfiltered tfrecords.
# * if it is helped please give me a Upvote.
# * Thanks!
# * 
# * fs: 2048
# * order: 7
# * hz: 20-1000 hz
# # (modified 20-350hz to 20-1000hz)

# # **Below is Train TFRecords !**

# In[ ]:


import os
import math
import random
import re
import warnings
import numpy as np
import tensorflow as tf
from kaggle_datasets import KaggleDatasets


def get_datapath():
    gcs_paths = []    
    for i, j in [(0, 4), (5, 9), (10, 14), (15, 19)]:            
        GCS_path = KaggleDatasets().get_gcs_path(f"bftfrec{i}{j}")
        gcs_paths.append(GCS_path)
        print(GCS_path)

    all_files = []
    for path in gcs_paths:
        all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/bf_train*.tfrecords"))))
           
    print("train_files: ", len(all_files))
    return all_files

all_files = get_datapath()


# # **Below is Test TFRecords !**

# In[ ]:


gcs_paths = []
for i, j in [(0, 4), (5, 9)]:    
    GCS_path = KaggleDatasets().get_gcs_path(f"bftfrectest{i}{j}")
    gcs_paths.append(GCS_path)
    print(GCS_path)

all_files = []
for path in gcs_paths:   
    all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/bf_test*.tfrecords"))))    

print("test_files: ", len(all_files))  


# # Enjoy it!
