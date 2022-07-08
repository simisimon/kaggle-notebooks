#!/usr/bin/env python
# coding: utf-8

# #  [BMS-MT] EDA + Simple Model
# 
# **Approach:**
#  - Target processing:
#      - Find the X most common target strings 
#          - {where X is some yet to be determined number}
#      - Compare each training target to top X values and pick the one that has the lowest levenshtein distance
#  - Downsample
#      - This is for faster processing, and allows us to run modeling in-memory
#  - Feature Processing:
#      - Create training sample of (N x 16k) using Image Hashing features 
#          - {where N is some yet to be determined number}
#  - Modeling:
#      - Tree Based model using Image Hash features to predict the best top bucketed responses that produces the lowest levenshtein distance
#  
#  
#  **Resources:**
#  - https://www.kaggle.com/yeayates21/panda-densenet-keras-starter-gpu
#  - https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
#  - https://www.kaggle.com/yeayates21/image-captioning-with-tensorflow
#  - https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-94.php
#  - https://www.kaggle.com/yasufuminakama/molecular-translation-naive-baseline
#  - https://www.kaggle.com/ghaiyur/baseline-starter

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from tqdm.notebook import tqdm
from Levenshtein import distance as levenshtein_distance
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
import imagehash
from PIL import Image
from datetime import date, datetime


# # Settings

# In[ ]:


# time the notebook and stop it from running past the Kaggle allowed compute time
startTime = datetime.now()


# # Review Labels

# In[ ]:


df = pd.read_csv("../input/bms-molecular-translation/train_labels.csv")
df.head()


# # EDA
# 
# ### Review Top 10 Labels

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df.groupby([\'InChI\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# ### Results:
# 
#  - All target values are unique, which makes sense given the problem.
#  
# ### Next Steps:
#  - We'll need to break the target down and try to find some common patterns.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf['InChI_list'] = df['InChI'].apply(lambda x: x.split('/'))\ndf['InChI_length'] = df['InChI_list'].apply(len)\nInChI_df = df['InChI_list'].apply(pd.Series)\ndf = pd.concat([df, InChI_df.add_prefix('InChI_')], axis=1)\ndf.head()\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df[[\'image_id\',\'InChI_length\']].groupby([\'InChI_length\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# ### Results:
# 
#  - The most common number of subsections divided by '/' is 4.
#  
# ### Next Steps:
#  - Let's find the 3 top patters for each subsection, thus giving us 3**4=81 top categories.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df[[\'image_id\',\'InChI_0\']].groupby([\'InChI_0\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# ### Results:
# 
#  - The only value for InChI subsection 0 is 'InChI=1S'.
#  
# ### Next Steps:
#  - Let's find the 5 top patterns for each remaining subsection, thus giving us 5x3=125 top categories.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df[[\'image_id\',\'InChI_1\']].groupby([\'InChI_1\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df[[\'image_id\',\'InChI_2\']].groupby([\'InChI_2\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# #### The values in InChI_2 are hard to read, so let's print them out

# In[ ]:


print("Top values for InChI_2:")
for index, row in dfg.head().iterrows():
    print(row['InChI_2'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfg = df[[\'image_id\',\'InChI_3\']].groupby([\'InChI_3\']).count()\ndfg.rename(columns={"image_id": "Count"}, inplace=True)\ndfg = dfg.reset_index(drop=False)\ndfg.sort_values(by=\'Count\', ascending=False, inplace=True)\ndfg.head()\n')


# ### Results:
# 
#  - Great!  We have our top values!.
#  
# ### Next Steps:
#  - Let's compile a list of our new top 15 categories.

# In[ ]:


c0 = 'InChI=1S'
ch1 = ['C15H22N2O2','C16H24N2O2','C14H20N2O2','C17H26N2O2','C14H22N2O2']
ch2 = ['c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18(19)20',
       'c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20(21)22',
       'c1-13(22)17-6-7-18-16-5-4-14-12-15(23)8-10-20(14,2)19(16)9-11-21(17,18)3',
       'c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22(23)24',
       'c1-18-9-7-13(20)11-12(18)3-4-14-15-5-6-17(21)19(15,2)10-8-16(14)18']
ch3 = ['h2-8H,1H3','h2-7H,1H3','h3-8H,1-2H3','h2-9H,1H3','h2-6H,1H3']
top15 = []
for c1 in ch1:
    for c2 in ch2:
        for c3 in ch3:
            top15.append(c0 + '/' + c1 + '/' + c2 + '/' + c3)
            
print("Total values: ", len(top15))
print("First 5 of our manufactured top 125 labels:")
print(top15[:5])


# ### Notes:
#  - We're going to remove the last 25 because they're going to be an issue later on since they're less frequent and will be selected less frequently by our levenshtein_distance

# In[ ]:


top15 = top15[:120]
print("Total values: ", len(top15))


# # Downsample

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndfs = df[[\'image_id\',\'InChI\']].sample(n=37500, random_state=0)\nprint("Shape of downsampled training data: ", dfs.values.shape)\ndfs.head()\n')


# # Create Target
# 
#  - Compare each training target to our top label values and pick the one that has the lowest levenshtein distance

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# get target values\ndef find_best_target_value(x):\n    return np.argmin([levenshtein_distance(x,v) for v in top15])\ndfs[\'Target\'] = dfs[\'InChI\'].apply(find_best_target_value)\n\n# get image paths\ntraining_image_folder = "../input/bms-molecular-translation/train/"\ndef get_image_path(x):\n    return training_image_folder + x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".png"\ndfs["image_path"] = dfs["image_id"].apply(lambda x: get_image_path(x))\n\ndfs.head()\n')


# In[ ]:


# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-94.php
unique_elements, counts_elements = np.unique(dfs['Target'].values, return_counts=True)
print("Frequency of unique values of the Target array:")
print(np.asarray((unique_elements, counts_elements)))


# # Feature Processing
# 
# - Create training sample of (10k x ~2k) using ResNet features

# ### Model

# In[ ]:


# show what image hash looks like
hashcode = imagehash.average_hash(Image.open('../input/bms-molecular-translation/train/0/0/0/0000a5af84ef.png'))
print(hashcode)


# In[ ]:


# create vocab dictionary to convert hash to array
vals = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l',
        'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
vocab = {}
for i, v in enumerate(vals):
    vocab[v] = i


# In[ ]:


# example of converting hash to numpy array
nphsh = np.array([vocab[x] for x in str(hashcode)])
print(nphsh.shape)
print(nphsh)


# ### Create in-memory training set

# In[ ]:


def img_preprocessing(image_path):
    hashcode = imagehash.average_hash(Image.open(image_path))
    nphsh = np.array([vocab[x] for x in str(hashcode)])
    return nphsh

# data generator, intended to be used in a call to model.fit()
def data_generator(df):
    # loop for ever over images
    while 1:
        for index, row in df.iterrows():
            yield img_preprocessing(row['image_path']), row['Target']


# In[ ]:


# define the generator
generator = data_generator(dfs) 
# get the number of training images from the target\id dataset
N = dfs.shape[0]
# create an empty matrix for storing the image features and target values
x_train = np.empty((N, 16), dtype=np.float)
y_train = np.empty((N, 1), dtype=np.int)
# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, image_id in enumerate(tqdm(dfs['image_id'])):
    x, y = next(generator)
    x_train[i, :] = np.array(x)
    y_train[i, :] = np.array(y)

print(x_train.shape)
print(y_train.shape)


# # Modeling:
# 
#      - XGB model using ResNet features to predict the best top-10 value that produces the lowest levenshtein distance

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npipe = XGBClassifier()\ndistributions = {'n_estimators': randint(5,150),\n                 'max_depth': randint(1,4)}\nclf = RandomizedSearchCV(pipe, distributions, random_state=87, n_iter=2, cv=2, \n                         scoring='roc_auc_ovr', n_jobs=-1, verbose=2, error_score='raise')\nsearch = clf.fit(x_train, y_train.flatten()) # best model is search.best_estimator_\n")


# In[ ]:


print("Best model scored avg AUC {} using the following hyperparameters: {}".format(search.best_score_, search.best_params_))


# In[ ]:


pickle.dump(search.best_estimator_, open("bmsmt_tp125_rf_model2.pkl", "wb"))


# # Inference

# In[ ]:


# load test data ids
dft = pd.read_csv("../input/bms-molecular-translation/sample_submission.csv")

# change default value
# https://www.kaggle.com/ghaiyur/baseline-starter
dft["InChI"] = ['InChI=1S/C12H23NO3/c1-1-1-2-12-14-13(12-18-12)10-1-2-1-1(-13-6-1-12/h8-9,1-H,11H2,1-2H3,(H,1,18)'] * len(dft)

# get image paths
test_image_folder = "../input/bms-molecular-translation/test/"
def get_image_path(x):
    return test_image_folder + x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".png"
dft["image_path"] = dft["image_id"].apply(lambda x: get_image_path(x))

# data generator, intended to be used in a call to model.fit()
def data_generator(df):
    # loop for ever over images
    while 1:
        for index, row in df.iterrows():
            yield img_preprocessing(row['image_path'])

# define the generator
generator = data_generator(dft) 
# get the number of training images from the target\id dataset
N = dft.shape[0]
# create an empty matrix for storing the image features and target values
x_train = np.empty((1, 16), dtype=np.float)
# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, image_id in enumerate(tqdm(dft['image_id'])):
    x = next(generator)
    yhat = search.best_estimator_.predict(x.reshape(1,-1))
    dft.loc[i,'InChI'] = top15[yhat[0]]
    # stop if we run too long (Kaggle allows runtime of 9 hrs)
    td = datetime.now()-startTime
    hours = td.seconds // 3600
    if hours > 4:
        break


# In[ ]:


dft[['image_id','InChI']].to_csv("submission.csv", index=False)


# In[ ]:




