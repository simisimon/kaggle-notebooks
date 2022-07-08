#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook

data = pd.read_csv('/kaggle/input//histopathologic-cancer-detection/train_labels.csv')
train_path = '/kaggle/input/histopathologic-cancer-detection/train/'
test_path = '/kaggle/input/histopathologic-cancer-detection/test/'
# quick look at the label stats
data['label'].value_counts()


# In[ ]:


cancer_data = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_path = '/kaggle/input/histopathologic-cancer-detection/train/'
test_path = '/kaggle/input/histopathologic-cancer-detection/test/'
# quick look at the label stats
cancer_data['label'].value_counts()


# We will be looking at what kind of labels there are in the dataset.
# 
# In our case, the training data contains 130908 negative samples, and 89117 positive samples.  This is roughly 60:40 split in our training set. It means that we might be more biased to negative samples as we are training. We probably want to balance it out during our splits.

# In[ ]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Negative', 'Positive'
sizes = [130908, 89117]
explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


plt.imshow(cv2.imread(train_path + "00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif"))


# In[ ]:


def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img


# Here we show the image q

# In[ ]:


plt.imshow(readImage(train_path + "00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif"))


# # Looking at the raw data
# Now that we have looked at the label distribution for training, let's take a look at the raw data. This image is in the train/directory and contains images in tif format. We will use open cv's library cv2 to read an image and show it.
# 
# We added a function called readImage becasue cv2 reads the image in BGR order or 'blue' 'green' 'red' order and we want to plot the image using matplotlib imshow which takes images in RGB order. readImage does this reordering

# 

# 

# In[ ]:


sd = shuffle(data)


# In[ ]:


sd[sd['label'] == 0].head(4)


# In[ ]:


fig,ax = plt.subplots(1,3)
for i in range(3):
    id_ = sd[sd['label'] == 0].iloc[i]['id']
    image_path_ = train_path + id_ + ".tif"
    ax[i].imshow(readImage(image_path_))
fig.suptitle('Negative samples')


# In[ ]:


fig,ax = plt.subplots(1,3)
for i in range(3):
    id_ = sd[sd['label'] == 1].iloc[i]['id']
    image_path_ = train_path + id_ + ".tif"
    ax[i].imshow(readImage(image_path_))
fig.suptitle('Positive samples')


# In[ ]:


plt.imshow(readImage(train_path + "00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif"))


# # Data Augmentation
# 
# We can improve our results of our machine learning pipeline by having more data.  Omne techniqure for adding more data is to use the existing set of data and apply transformations to it.
# 
# In this case we will be applying the following transformations:
# * rotation
# * cropping
# * lighting
# * flip
# 
# 

# ## Image rotation

# In[ ]:


sample_ = '../input/histopathologic-cancer-detection/test/00006537328c33e284c973d7b39d340809f7271b.tif'
img_ = cv2.imread(sample_)
print(img_.shape)


#apply rotation
width_ = img_.shape[0]
center_ = (width_//2, width_//2)
dims_ = (width_, width_)
rotation_ = 10
M = cv2.getRotationMatrix2D(center_,rotation_,1)   # the center point is the rotation anchor
img_rot_ = cv2.warpAffine(img_,M,dims_,flags=cv2.INTER_CUBIC)
# cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC (alow) & cv2.INTER_LINEAR for zooming
#plt.imshow(img_rot_)

#showing results
fig,ax = plt.subplots(1,2)

ax[0].imshow(img_)
ax[0].set_title('original')
ax[1].imshow(img_rot_)
ax[1].set_title('rotated')


# ## Cell Images
# 
# 

# # Preparing train and test split
# 
# We will be using `train_test_split` from `sklearn.model_selection` to split the dataset into a training set and a validation set. 
# 
# The idea behind the train_test_split below is to use the 'id' column which is the column of filenames, as the thing we want to split. We are using the `data.index` as the `y` value which is typically the label for `x`. This is just a little trick to get the row number that corresponds to the split.
# 
# The `train_test_split` function will then break up the `x` and `y` into two sets, a training set and a validation set. What we are interested in here are the indices of the rows that `sklearn` uses for the split.
# 
# The reason for using this function is the stratify part which allows one to make sure there is an equal distribution of labels in both the train and validation parts. Here we use a 0.1 split, i.e. a 9:1 split of training and validation.

# In[ ]:


from sklearn.model_selection import train_test_split

train_df = cancer_data.set_index('id')
train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

#Should be x_train, x_val, y_train, y_val
tr_n, val_n, tr_idx, val_idx = train_test_split(
    train_names, range(len(train_names)),
    test_size=0.1, stratify=train_labels,
    random_state=123)

tr_n1,val_n1, tr_idx1, val_idx1 = train_test_split(
    cancer_data['id'].values, cancer_data.index,
    test_size=0.1, stratify=cancer_data['label'],
    random_state=123)

print(tr_n.shape, val_n.shape,22003/198022)
len(val_n), len(val_idx)
print(tr_n[0], tr_idx[0], tr_n1[0], tr_idx1[0])
print(cancer_data[cancer_data['id'] == tr_n[0]].index, tr_idx[0])
#print(all(tr_idx == tr_idx1), all(tr_n == tr_n1))


#all(val_n1 == val_n)
#all(tr_n1 == tr_n)
#data[data['id'] == tr_n[0]], val
#data['id'].values
#data['label'].values
#train_n, traind_idx, validation_n, validation_idx = train_test_split(data['id'].values, )


# `stratify` allos us to have the same proportion of positive and negative samples in the train set and the validation sets. We can check that is the case (see cell below). We find that dtratify keeps a 0.68 ratio of positive to negative samples in both the training and in the validation sets. And this ratio also matches that of the original distribution in our labeled data.

# In[ ]:


tr_tmp = cancer_data.iloc[tr_idx, :]
vl_tmp = cancer_data.iloc[val_idx,:]
print(
    tr_tmp[tr_tmp['label'] == 0].count(),
    tr_tmp[tr_tmp['label'] == 1].count(),
    vl_tmp[vl_tmp['label'] == 0].count(),
    vl_tmp[vl_tmp['label'] == 1].count(),
)
print('Ratios:')
print('Training:', 80205/117817)
print('Validation:', 8912/13091)
print('Original:', 40.5//59.5 )
#40.5


# ## Libraries to use for classification
# We will be using `fastai.vision` and `torchvision` models.

# In[ ]:


from fastai import *
from fastai.vision import *
from torchvision.models import *


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


arch = densenet169
batch_size = 128
sz = 90 # imagesize is 96
model_path = str(arch).split()[1]


# In[ ]:


df = cancer_data.copy()
df.columns = ['name', 'label']
df['name'] = train_path + df['name'] + '.tif'

df_test = pd.DataFrame([test_path + f for f in os.listdir(test_path)],
             columns=['name'])


# ## Convert filenames to images
# 
# References:
# https://docs.fast.ai/tutorial.datablock.html
# https://docs.fast.ai/vision.data.html
# https://docs.fast.ai/tutorial.vision
# https://docs.fast.ai/data.core.html#DataLoaders
# 
# We will be using `fast.ai` datablocks to specify the image filenames. `DataBlocl` is not specific to images, but by specifying the `ImageBlock` and `CategoryBlock` as the conversion between filenames to labels, filenames in our dataframe `df` can be converted later to actual image data, and the labels in `df` will be used as categories.
# 
# `DataBlock` is really a specification, as it doesn't take in our dataframe `df` right away. This will happen when we call `datasets` on the `DataBlock`. At this point it will generate `DataSets` given the input dataframe and the specification in `DataBlock`.
# 
# Since the input to `datasets` is a generic dataframe, `DataBlock` must also be given hints as to what part of the dataframe to use for the data and what part to use for the label. This is accomplished by two auxiliary functions `get_items` and `get_y`. In our case we pass two functions, one to gt the `name` and the other to get the `label` for a particular ro in the dataframe, i.e. these functions expect a row of data and we can extract different parts of the row.

# In[ ]:


from fastai.data.all import *
from fastai.vision.all import *


# In[ ]:


fastai.vision.all.get_image_files
    


# In[ ]:


dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_x = lambda r: r['name'],
                   get_y     = lambda r: r['label'])


# In[ ]:


#dblock = DataBlock()
dsets = dblock.datasets(df)
len(dsets.train),len(dsets.valid)


# In[ ]:


dsets.train[0]


# In[ ]:


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x = lambda r: r['name'], 
    get_y = lambda r: r['label'])
dsets = dblock.datasets(df)
dsets.train[0]


# # Batches for training
# Images are not directly used until they are needed. We can use the `dataloaders` function which works in a similar way as `datasets` above in which it can take a dataframe as an input and uses the specification given in the `DataBlock`.
# 
# This allows the images to be read in on the fly. For example, we show a batch of data below, the title on top of each image corresponds to whether it is positive (1) or negative (0.

# In[ ]:


dls = dblock.dataloaders(df)
dls.show_batch()


# # Specifying the splitter
# Note we have already shown previously how to use `train_test_split` from `sklearn`. We are going to use it here on conjunction with `DataBlock`. We use this functionality becasue `DataBlock` doesn't have a native option to stratify the data as we have done previously.
# 
# To add a specification of train and validation sets we incorporate the indices obtained from `train_test_split` as outpit to the `splitter` function below which will be passed to the specification in `DataBlock` through the `splitter` option.

# In[ ]:


def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid


# In[ ]:


def splitter(df):
    tr_n, tr_idx, val_n, val_idx = train_test_split(
        df['name'].values, df.index,
        test_size=0.1, stratify=df['label'],
        random_state=123)
    return val_n.tolist(), val_idx.tolist()


# In[ ]:


#splitter(df)


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=splitter,
                   get_x = lambda r: r['name'], 
                   get_y = lambda r: r['label'])
dls = dblock.dataloaders(df)


# In[ ]:


dls.show_batch()


# # Training the data with convolutional neural nets

# In[ ]:


learn = cnn_learner(dls, resnet18)


# In[ ]:


x,y = to_cpu(dls.train.one_batch())
activs = learn.model(x)
activs.shape


# In[ ]:


activs[0]


# In[ ]:


arch


# In[ ]:


# Next, we create a convnet learner object
# ps = dropout percentage (0-1) in the final layer
#def getLearner():
#    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

#ps = dropout percentage (0-1) in the final layer
learner = cnn_learner(dls, 
                     arch, 
                     pretrained=True, 
                     path='.', 
                     metrics=accuracy, 
                     ps=0.5) 
                     #callback_fns=ShowGraph)


# In[ ]:


learner.lr_find(start_lr=1e-03, end_lr=3e-01, num_it=100)


# In[ ]:


lr_max = 0.01302280556410551
wd = 1e-2
learner.fit_one_cycle(1, lr_max=lr_max)
learner.save('_stage')


# In[ ]:


learner.recorder.plot_sched()


# Next, we plot the losses
# 
# https://docs.fast.ai/learner.html#Recorder.plot.loss

# In[ ]:


# before we continue, lets save the model at this stage
learner.save('_stage1')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


learner.recorder.plot_loss()


# We would also like to plot the confusion matrix. Fast.ai provides an interpretation class:
# 
# - https://docs.fast.ai/interpret.html

# In[ ]:


# predict the validation set with our model
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(title='Confusion matrix')


# To get a number for the accuracy of our predictions, we can use the accuracy function, which takes the predictions, and the labeled data to give us a percentage. 
# 
# $
# \rm{accuracy} = \frac{number of correct predictions}{Total number of predictions}
# $

# In[ ]:


preds,y, loss = learner.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))


# In[ ]:


from sklearn.metrics import roc_curve, auc
# probs from log preds
probs = np.exp(preds[:,1])
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

# Compute ROC area
roc_auc = auc(fpr, tpr)
print('ROC area is {0}'.format(roc_auc))


# # Test set
# 
# In this section we will run our model prediction against the test set.  There is a separate directory in the competition labeled `test`.
# 
# This directory just contains a set of images. We need to load this set of images into the `learner`. Fortunately there is a function in the learner that creates a data loader for our purposes. `leader.dls.test_dl` accepts a list of files and returns a DataLoaders object. And then we can use the model's `get_preds` to directly get prediction just on the files from the test set by passing the corresponding `DataLoaders` object to it.
# 
# 
# 4. results (tables, figures etc) and analysis (reasoning of why or why not something worked well, also troubleshooting and hyperparameter optimization procedure summary)

# In[ ]:


test_files = fastai.vision.all.get_image_files(test_path)


# In[ ]:


test_dl = learner.dls.test_dl(test_files)
test_preds = learner.get_preds(dl=test_dl)


# Now that we have predictions, `test_preds`, we need to preapre them for submission. The competition gives a `sample_submission.csv` from which we need to use the correct column headings. In this case the columns should be called `id` and `label`. We also note that in the `id` column all the id's are not filenames but just the 'stem' of the file. We use `pathlib`'s stem function to return just the stem of the path.
# 
# `get_preds` returns a tuple, and we just want to operate on the first element of the tuple which is a $n$ by 2 tensor, where $n$ is the number of elements in the tedt set.  We need to convert the predicted probability into a classification into one of the labels. We can do that by taking the max probability from these two predictions and assign the class to the `argmax` of these two predictions. Note in the code below `axis-1` corresponds to the second dimension in the tensor, which is 2 in our case, that is, for every element in the test set we will take the argmax for the prediction possibilities.
# 
# Then we create a pandas DataFrame with columns labeled as 'id' and 'label', with the stem of the filenames in the first column and the classes from the argmax in the second column. And we save the DataFrame to a file called 'submission.csv' for Kaggle.

# In[ ]:


test_stem = [x.stem for x in test_files]


# In[ ]:


test_classes = np.argmax(test_preds[0], axis=1)


# In[ ]:


my_submission = pd.DataFrame({'id': test_stem, 'label' : test_classes})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('ls')


# ## 
