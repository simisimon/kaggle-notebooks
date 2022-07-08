#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# * Part 1: [Train --> uwmgtis Keras train_03](https://www.kaggle.com/code/benoitdacosta/uwmgtis-keras-train-03) 
# * Part 2: [Inference --> uwmgtis Keras infer_03](https://www.kaggle.com/code/benoitdacosta/uwmgtis-keras-infer-03)

# # Libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import os 
from glob import glob
from joblib import parallel_backend, Parallel, delayed , dump , load
from tqdm import tqdm
from tqdm import tqdm
from tqdm.notebook import tqdm
import gc

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, Input , Dropout

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import Rectangle
import cv2

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

pd.set_option('display.max_columns',200)
pd.set_option('display.max_colwidth', 200)


# # Data & Configuration

# In[ ]:


repertory ='/kaggle/input/'

DIR = repertory + 'uw-madison-gi-tract-image-segmentation/' 
TRAIN_DIR = DIR + 'train'
TEST_DIR = DIR + 'test'
train_csv = DIR +'train.csv' 
sample_sub = DIR + 'sample_submission.csv'

df_train = pd.read_csv(train_csv)
df_train.head(10)


# In[ ]:


class CFG:
    BATCH_SIZE    = 16
    img_size      = (256, 256, 3)
    n_fold        = 5
    fold_selected = 1
    epochs        = 100
    seed          = 42
    nb_cpu        = mp.cpu_count()
    steps_per_epoch_train = None
    steps_per_epoch_val = None

AUTOTUNE = tf.data.AUTOTUNE


# # PREPROCESSING

# In[ ]:


# Metadata
def preprocessing(df, subset = 'train'):
    #--------------------------------------------------------------------------
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    #--------------------------------------------------------------------------
    if subset == 'train':
        all_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
        x = all_images[0].rsplit('/' , 4)[0] ## ../uw-madison-gi-tract-image-segmentation/train
    else: 
        all_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)
        x = all_images[0].rsplit('/' , 4)[0] ## ../uw-madison-gi-tract-image-segmentation/test

    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(os.path.join(x,
                              "case"+str(df["case"].values[i]),
                              "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                              "scans",
                              "slice_"+str(df["slice"].values[i])))
    df["path_partial"] = path_partial_list
    #--------------------------------------------------------------------------
    path_partial_list = []
    for i in range(0, len(all_images)):
        path_partial_list.append(str(all_images[i].rsplit("_",4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df['path_partial'] = path_partial_list
    tmp_df['path'] = all_images
    #--------------------------------------------------------------------------
    df = pd.merge(df,tmp_df, on="path_partial").drop(columns=["path_partial"])
    #--------------------------------------------------------------------------
    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))
    #--------------------------------------------------------------------------
    del x, path_partial_list, tmp_df
    #--------------------------------------------------------------------------
    gc.collect()
    return df 


# In[ ]:


train_df = preprocessing(df_train)
train_df.head()


# # Function
# 

# In[ ]:


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


def id2mask(id_):
    itrain_df = train_df[train_df['id']==id_]
    wh = itrain_df[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        ctrain_df = itrain_df[itrain_df['class']==class_]
        rle = ctrain_df.segmentation.squeeze()
        if len(ctrain_df) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0,0),(0,0),(1,0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask

def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)


# In[ ]:


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') # original is uint16
    img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
    img = img.astype('uint8')
    return img

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


# In[ ]:


# row=1; col=4
# plt.figure(figsize=(5*col,5*row))
# for i, id_ in enumerate(train_df[train_df.segmentation.notna()].sample(frac=1.0)['id'].unique()[:row*col]):
#     img = load_img(train_df[train_df['id']==id_].path.iloc[0])
#     mask = id2mask(id_)*255
#     plt.subplot(row, col, i+1)
#     i+=1
#     show_img(img, mask=mask)
#     plt.tight_layout()


# In[ ]:


# Restructure
def restructure(df, subset="train"):
    # RESTRUCTURE  DATAFRAME
    df_out = pd.DataFrame({'id': df['id'][::3]})

    if subset=="train":
        df_out['large_bowel'] = df['segmentation'][::3].values
        df_out['small_bowel'] = df['segmentation'][1::3].values
        df_out['stomach'] = df['segmentation'][2::3].values

    df_out['path'] = df['path'][::3].values
    df_out['case'] = df['case'][::3].values
    df_out['day'] = df['day'][::3].values
    df_out['slice'] = df['slice'][::3].values
    df_out['width'] = df['width'][::3].values
    df_out['height'] = df['height'][::3].values

    df_out=df_out.reset_index(drop=True)
    df_out=df_out.fillna('')
    if subset=="train":
        df_out['count'] = np.sum(df_out.iloc[:,1:4]!='',axis=1).values
        
    display(df_out.sample(5))
    return df_out


# In[ ]:


DF_train= restructure(train_df, subset="train")


# # DATASET

# In[ ]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = CFG.BATCH_SIZE, subset="train", shuffle=False , img_shape = CFG.img_size , aug_dat=False):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.aug_dat = aug_dat
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        X = np.empty((self.batch_size,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        y = np.empty((self.batch_size,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        id_, heights, widths, classes = [] , [] ,[], [] 
        
        for i, img_path in enumerate(self.df['path'].iloc[indexes]):
            if self.subset != 'train':
                id_.append(self.df['id'].iloc[indexes[i]])
                heights.append(self.df['height'].iloc[indexes[i]])
                widths.append(self.df['width'].iloc[indexes[i]])
                classes.append(self.df['class'].iloc[indexes[i]])
            
            w=self.df['width'].iloc[indexes[i]]
            h=self.df['height'].iloc[indexes[i]]
            
            img = self.__load_grayscale(img_path)  
#             X[i,] = img   
            X[i,] = np.concatenate((img , img , img), axis = -1)
            
            if self.subset == 'train':
                for k,j in enumerate(["large_bowel","small_bowel","stomach"]):
                    rles = self.df[j].iloc[indexes[i]]
                    mask = rle_decode(rles, shape=(h, w, 1))
                    mask = cv2.resize(mask, self.img_shape[0:2] )
                    y[i,:,:,k] = mask
                  
        if self.aug_dat : 
            imag_ , mask_ = self.__aug_img(X,y)
#             imag_ , mask_ = aug_img(X,y , self.batch_size ,self.img_shape)
            X = np.concatenate((X , imag_ ), axis = 0)
            y = np.concatenate((y , mask_ ), axis = 0)

        if self.subset == 'train':
            return tf.convert_to_tensor(X,dtype=tf.float32), tf.convert_to_tensor(y,dtype=tf.float32)
        else: 
            return X , id_ , widths , heights , classes

        
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    #         img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        dsize = self.img_shape[0:2]
        img = cv2.resize(img, dsize)
    #         img = img.astype(np.int8) / 255.
        img = img.astype('float32') # original is uint16
        img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
        img = img.astype('uint8')/255
        img = np.expand_dims(img, axis=-1)
        return img
    
    def __aug_img(self, img , mask) :
        im = np.empty((self.batch_size,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        mk = np.empty((self.batch_size,self.img_shape[0],self.img_shape[1],self.img_shape[2])) 
        for i in range(self.batch_size):
            img_ = img[i,]
            mask_ = mask[i,] 
            while len(np.unique( img[i,] - img_)) == 1 : 
                Nb_random =np.random.randint(low = 0, high=100, size=2)
                seed = (Nb_random[0],Nb_random[1])
                if seed[0]/100 <0.3 : 
                    img_ = tf.image.stateless_random_flip_up_down(img_, seed=seed )
                    mask_ = tf.image.stateless_random_flip_up_down(mask_, seed=seed )
                elif seed[0]/100 >0.7 : 
                    img_ = tf.image.stateless_random_flip_left_right(img_ , seed=seed )
                    mask_ =  tf.image.stateless_random_flip_left_right(mask_ , seed=seed )
                else : 
                    img_ = tf.image.stateless_random_flip_left_right(img_ , seed=seed )
                    mask_ =  tf.image.stateless_random_flip_left_right(mask_ , seed=seed )
                    img_ = tf.image.stateless_random_flip_up_down(img_, seed=seed )
                    mask_ = tf.image.stateless_random_flip_up_down(mask_, seed=seed )
            im[i,] = img_
            mk[i,] = mask_
        return im, mk   

    
# def aug_img( img , mask, batch_size ,img_shape  ) :
#     im = np.empty((batch_size,img_shape[0],img_shape[1],img_shape[2]))
#     mk = np.empty((batch_size,img_shape[0],img_shape[1],img_shape[2])) 
#     for i in range(batch_size):
#         img_ = img[i,]
#         mask_ = mask[i,] 
#         while len(np.unique( img[i,] - img_)) == 1 : 
#             Nb_random =np.random.randint(low = 0, high=100, size=2)
#             seed = (Nb_random[0],Nb_random[1])
#             if seed[0]/100 <0.3 : 
#                 img_ = tf.image.stateless_random_flip_up_down(img_, seed=seed )
#                 mask_ = tf.image.stateless_random_flip_up_down(mask_, seed=seed )
#             elif seed[0]/100 >0.7 : 
#                 img_ = tf.image.stateless_random_flip_left_right(img_ , seed=seed )
#                 mask_ =  tf.image.stateless_random_flip_left_right(mask_ , seed=seed )
#             else : 
#                 img_ = tf.image.stateless_random_flip_left_right(img_ , seed=seed )
#                 mask_ =  tf.image.stateless_random_flip_left_right(mask_ , seed=seed )
#                 img_ = tf.image.stateless_random_flip_up_down(img_, seed=seed )
#                 mask_ = tf.image.stateless_random_flip_up_down(mask_, seed=seed )
#         im[i,] = img_
#         mk[i,] = mask_
#     return im, mk    


# In[ ]:


Train_masks = list(DF_train[DF_train['large_bowel']!=''].index)
Train_masks += list(DF_train[DF_train['small_bowel']!=''].index)
Train_masks += list(DF_train[DF_train['stomach']!=''].index)

DF_training = DF_train[DF_train.index.isin(Train_masks)]
DF_training.reset_index(inplace=True, drop = True)
print(DF_training.shape)


# # FOLDS

# In[ ]:


skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(X=DF_training, y=DF_training['count'],groups =DF_training['case']), 1):
    DF_training.loc[val_idx, 'fold'] = fold
    
# DF_training['fold'] = DF_training['fold'].astype(np.uint8)

train_idx = DF_training[DF_training["fold"]!=CFG.fold_selected].index
valid_idx = DF_training[DF_training["fold"]==CFG.fold_selected].index

# CFG.steps_per_epoch_train = len(train_idx) // CFG.BATCH_SIZE 
# if len(train_idx) % CFG.BATCH_SIZE !=0: 
#     CFG.steps_per_epoch_train +=1
    
# CFG.steps_per_epoch_val = len(valid_idx) //CFG.BATCH_SIZE*2
# if len(valid_idx) //CFG.BATCH_SIZE*2 !=0: 
#     CFG.steps_per_epoch_val +=1

train_generator = DataGenerator(DF_training[DF_training.index.isin(train_idx)], batch_size = 11, subset="train", 
                                shuffle=True , img_shape = CFG.img_size , aug_dat=True)
val_generator = DataGenerator(DF_training[DF_training.index.isin(valid_idx)], batch_size = 29, subset="train", 
                                shuffle=False , img_shape = CFG.img_size , aug_dat=False )
# display(DF_training.groupby('fold').size())
# display(DF_training.groupby(['fold','count'])['id'].count())


# In[ ]:


# # Visualizing
# fig = plt.figure(figsize=(10, 25))
# gs = gridspec.GridSpec(nrows=6, ncols=2)
# colors = ['red','green','blue']
# labels = ["Large Bowel", "Small Bowel", "Stomach"]
# patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

# cmap1 = mpl.colors.ListedColormap(colors[0])
# cmap2 = mpl.colors.ListedColormap(colors[1])
# cmap3 = mpl.colors.ListedColormap(colors[2])

# for i in range(6):
#     images, mask = train_generator[i]
#     sample_img=images[0,:,:,0]
#     mask1=mask[0,:,:,0]
#     mask2=mask[0,:,:,1]
#     mask3=mask[0,:,:,2]
    
#     ax0 = fig.add_subplot(gs[i, 0])
#     im = ax0.imshow(sample_img, cmap='bone')

#     ax1 = fig.add_subplot(gs[i, 1])
#     if i==0:
#         ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)
#         ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)
#         plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 14,title='Mask Labels', title_fontsize=14, edgecolor="black",  facecolor='#c5c6c7')

#     l0 = ax1.imshow(sample_img, cmap='bone')
#     l1 = ax1.imshow(np.ma.masked_where(mask1== False,  mask1),cmap=cmap1, alpha=1)
#     l2 = ax1.imshow(np.ma.masked_where(mask2== False,  mask2),cmap=cmap2, alpha=1)
#     l3 = ax1.imshow(np.ma.masked_where(mask3== False,  mask3),cmap=cmap3, alpha=1)
#     _ = [ax.set_axis_off() for ax in [ax0,ax1]]

#     colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]


# # Metrics

# In[ ]:


#Metrics
from tensorflow.keras.metrics import binary_crossentropy as BCE

# def replacenan(t):            
#     return tf.where(tf.math.is_nan(t), tf.ones_like(t), t)

# def replacenan(t):
#     t = tf.clip_by_value(t, clip_value_min=0.0, clip_value_max=1.0)
#     return tf.math.multiply_no_nan(t, tf.ones_like(t))

# def dice_coef(y_true, y_pred, smooth=1.0):
#     y_true_f = replacenan(y_true)
#     y_pred_f = replacenan(y_pred)
    
#     y_true_f = tf.keras.backend.flatten(y_true_f)
#     y_pred_f = tf.keras.backend.flatten(y_pred_f)
    
#     intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
#     sum_ = tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f)
    
#     dice = tf.math.divide_no_nan((2.0 * intersection + smooth) , ( sum_ + smooth))
#     return dice
    
# def iou_coef(y_true, y_pred, smooth=1.0):   
#     y_true_f = replacenan(y_true)
#     y_pred_f = replacenan(y_pred)
    
#     intersection = tf.math.reduce_sum(y_true_f * y_pred_f, axis=[1,2,3])
#     union = tf.math.reduce_sum(y_true_f,[1,2,3])+tf.math.reduce_sum(y_pred_f,[1,2,3])-intersection
    
#     iou = tf.math.reduce_mean( tf.math.divide_no_nan((intersection + smooth),(union + smooth)) , axis=0)
#     return iou

# def dice_loss(y_true, y_pred):
#     loss = 1 - dice_coef(y_true, y_pred)
#     return loss

# def bce_dice_loss(y_true, y_pred):   
#     y_true_f = replacenan(y_true)
#     y_pred_f = replacenan(y_pred)
#     bce = BCE(y_true_f, y_pred_f)
#     bce = tf.math.multiply_no_nan(bce, tf.ones_like(bce))
#     return bce + dice_loss(y_true, y_pred)

def dice_coef(y_true,y_pred):
    y_true_f=tf.reshape(tf.dtypes.cast(y_true, tf.float32),[-1])
    y_pred_f=tf.reshape(tf.dtypes.cast(y_pred, tf.float32),[-1])
    intersection=tf.reduce_sum(y_true_f*y_pred_f)
    sum_ = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + 1.) / (sum_ + 1.)

def iou_coef(y_true, y_pred):   
    y_true_f=tf.reshape(tf.dtypes.cast(y_true, tf.float32),[-1])
    y_pred_f=tf.reshape(tf.dtypes.cast(y_pred, tf.float32),[-1])
    intersection=tf.reduce_sum(y_true_f*y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou = (intersection+1.) / (union+1.)
    return iou

def bce_dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return BCE(y_true, y_pred) + (1-dice_coef(y_true, y_pred))


# # U-NET Models

# In[ ]:


def conv_block(inputs, num_filters, batchnorm ):
    x = Conv2D(num_filters, kernel_size = (3,3), padding="same",)(inputs)
    x = Activation("relu")(x)
    if batchnorm : 
        x = BatchNormalization()(x)
    x = Conv2D(num_filters, kernel_size = (3,3), padding="same")(x)
    x = Activation("relu")(x)
    if batchnorm : 
        x = BatchNormalization()(x)
    return x

def encoder_block(inputs, num_filters, dropout = False ,  batchnorm =True):
    x = conv_block(inputs, num_filters , batchnorm)
    p = MaxPool2D((2, 2))(x)
    if dropout : 
        p = Dropout(0.3)(p)
    return x, p

def decoder_block(inputs, skip_features, num_filters, dropout = False, batchnorm = True):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    if dropout : 
        x = Dropout(0.3)(x)
    x = conv_block(x, num_filters, batchnorm)
    return x

def build_unet(input_shape, dropout = False , batchnorm = True , activation = 'sigmoid'):

    inputs = Input(shape=input_shape)
    
    s0, p0 = encoder_block(inputs, 32 , dropout , batchnorm )
    s1, p1 = encoder_block(p0, 64 , dropout , batchnorm )
    s2, p2 = encoder_block(p1, 128, dropout , batchnorm  )
    s3, p3 = encoder_block(p2, 256, dropout , batchnorm )
    s4, p4 = encoder_block(p3, 512, dropout , batchnorm )
    s5, p5 = encoder_block(p4, 1024, dropout , batchnorm )

    b1 = conv_block(p5, 2048, batchnorm)

    d0 = decoder_block(b1, s5, 1024, dropout , batchnorm )
    d1 = decoder_block(d0, s4, 512, dropout , batchnorm )
    d2 = decoder_block(d1, s3, 256, dropout , batchnorm )
    d3 = decoder_block(d2, s2, 128, dropout , batchnorm )
    d4 = decoder_block(d3, s1, 64, dropout , batchnorm )
    d5 = decoder_block(d4, s0, 32, dropout , batchnorm )

    outputs = Conv2D(3, 1, padding="same", activation =activation )(d5)

    model = Model(inputs, outputs, name="U-Net")

    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=bce_dice_loss,
                  metrics=[iou_coef , dice_coef])
    return model

# PLOT TRAINING
def plot_train(model):
    losses = model if isinstance(model, pd.DataFrame) else pd.DataFrame(model.history.history) 
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(losses['loss'].index,losses['loss'],label='Train_Loss')
    plt.plot(losses['val_loss'].index,losses['val_loss'],label='Val_loss')
    plt.title('LOSS'); plt.xlabel('Epoch'); plt.ylabel('loss');plt.legend();

    plt.subplot(1,3,2)
    plt.plot(losses['dice_coef'].index,losses['dice_coef'],label='Train_dice_coef')
    plt.plot(losses['val_dice_coef'].index,losses['val_dice_coef'],label='Val_dice_coef')
    plt.title('DICE'); plt.xlabel('Epoch'); plt.ylabel('dice_coef');plt.legend(); 

    plt.subplot(1,3,3)
    plt.plot(losses['iou_coef'].index,losses['iou_coef'],label='Train_iou_coef')
    plt.plot(losses['val_iou_coef'].index,losses['val_iou_coef'],label='Val_iou_coef')
    plt.title('IOU'); plt.xlabel('Epoch'); plt.ylabel('iou_coef');plt.legend();
    plt.show()

def fit_model(model ,  model_name , train_dataset , validation_dataset , workers = None):
    if os.path.isfile(models_path+str(model_name)+'.h5'): 
        model = load_model(models_path+str(model_name)+'.h5', custom_objects={'bce_dice_loss': bce_dice_loss ,'iou_coef':iou_coef ,'dice_coef':dice_coef  })
        results = load(results_path+'score_'+str(model_name)+'.joblib')
        # model.summary()
        plot_train(results)
    else :
        # model.summary()
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        if workers : 
            model.fit(x = train_dataset, epochs=CFG.epochs,validation_data=validation_dataset, callbacks=[early_stop],
                      workers = CFG.nb_cpu)
   
        else : 
            model.fit(x = train_dataset, epochs=CFG.epochs, validation_data=validation_dataset, callbacks=[early_stop])
 
        results = pd.DataFrame(model.history.history)
        plot_train(model)
        
        dump(results,results_path+'score_'+str(model_name)+'.joblib',compress = True)
        model.save(models_path+str(model_name)+'.h5') 
    
    return model , results


# In[ ]:


models_path = '/kaggle/working/'
results_path = '/kaggle/working/'


# In[ ]:


input_shape = CFG.img_size
gc.collect()
model = build_unet(input_shape, dropout = True , batchnorm = True) 


# In[ ]:


model , results = fit_model(model , 'U-net',train_generator , val_generator, workers = None)


# # Prediction

# In[ ]:


# sub_df = pd.read_csv(sample_sub)
# if not len(sub_df):
#     debug = True
#     sub_df = pd.read_csv(train_csv)
#     test_df = preprocessing(df_train,  subset = 'train')[0:3000]
# else : 
#     debug = False
#     test_df = preprocessing(sub_df , subset = 'test')
    
# test_df.head(5)


# In[ ]:


# def infer(DF , model , batch_size = CFG.BATCH_SIZE) : 
#     pred_rle = []; pred_ids = []; pred_classes = [];
    
#     DF_batch = DataGenerator(DF, batch_size =batch_size, subset="test", shuffle=False)
#     for idx , (img , id_, widths , heights , classes) in enumerate(tqdm(DF_batch)):
# #         msk = np.empty((batch_size,CFG.img_size[0],CFG.img_size[1],CFG.img_size[2]))
                                       
#         preds = model.predict(img,verbose=0)
        
#         # Rle encode 
#         for j in range(batch_size):
#             k = 0 if classes[j]=='large_bowel' else 1 if classes[j]=='small_bowel' else 2

#             pred_img = cv2.resize(preds[j,:,:,k], ( widths[j] , heights[j]),
#                                   interpolation=cv2.INTER_NEAREST) # resize probabilities to original shape
#             pred_img = (pred_img>0.5).astype(dtype='uint8')    # classify

#             pred_ids.append(id_[j])
#             pred_classes.append(classes[j])
#             pred_rle.append(rle_encode(pred_img))
    
#     return pred_rle, pred_ids , pred_classes


# In[ ]:


# pred_rle, pred_ids , pred_classes = infer(test_df, model)


# In[ ]:


# submission = pd.DataFrame({
#     "id":pred_ids,
#     "class":pred_classes,
#     "predicted":pred_rle
# })

# if debug :
#     sub_df = pd.read_csv(train_csv)
#     del sub_df['segmentation']
# else:
#     sub_df = pd.read_csv(sample_sub)
#     del sub_df['predicted']

# sub_df = sub_df.merge(submission, on=['id','class'])
# sub_df.to_csv('submission.csv',index=False)

# submission.sample(10)


# In[ ]:





# In[ ]:




