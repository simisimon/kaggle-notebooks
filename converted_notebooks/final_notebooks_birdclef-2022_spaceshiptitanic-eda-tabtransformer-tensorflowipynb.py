#!/usr/bin/env python
# coding: utf-8

# ---
# # [Spaceship Titanic][1]
# 
# - We are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# 
# ---
# #### **The aim of this notebook is to**
# - **1. Conduct exploratory data analysis (EDA).**
# - **2. Converting numerical features into categorical features by binning.**
# - **3. Conduct feature engineering on 'Cabin' feature.**
# - **4. Build and train a TabTransformer model.**
# 
# ---
# **References:** Thanks to previous great codes and notebooks.
# - [🔥🔥[TensorFlow]TabTransformer🔥🔥][2]
# - [Structured data learning with TabTransformer][3]
# - [Sachin's Blog Tensorflow Learning Rate Finder][4]
# 
# ---
# ### **If you find this notebook useful, please do give me an upvote. It helps me keep up my motivation.**
# #### **Also, I would appreciate it if you find any mistakes and help me correct them.**
# 
# ---
# [1]: https://www.kaggle.com/competitions/spaceship-titanic/overview
# [2]: https://www.kaggle.com/code/usharengaraju/tensorflow-tabtransformer
# [3]: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/tabtransformer.ipynb
# [4]: https://sachinruk.github.io/blog/tensorflow/learning%20rate/2021/02/15/Tensorflow-Learning-Rate-Finder.html

# <h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>0. TABLE OF CONTENTS</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Settings</a></li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Data Loading</a></li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. Exploratory Data Analysis</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#3.1" class="list-group-item list-group-item-action">3.1 Feature Engineering</a></li>
#             <li><a href="#3.2" class="list-group-item list-group-item-action">3.2 Target Distribution</a></li>
#             <li><a href="#3.3" class="list-group-item list-group-item-action">3.3 Numerical Features</a>
#                 <ul class="list-group" style="list-style-type:none;">
#                     <li><a href="#3.3.1" class="list-group-item list-group-item-action">3.3.1 Statistics of Numerical Features</a></li>
#                     <li><a href="#3.3.2" class="list-group-item list-group-item-action">3.3.2 Binning for Numerical Features</a></li>
#                 </ul>
#             </li>
#             <li><a href="#3.4" class="list-group-item list-group-item-action">3.4 Categorical Feature</a></li>
#             <li><a href="#3.5" class="list-group-item list-group-item-action">3.5  Data Processing Complete</a></li>
#             <li><a href="#3.6" class="list-group-item list-group-item-action">4.6 Validation Split</a></li>
#         </ul>
#     </li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. Model Building</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 Dataset</a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 Preprocessing Model</a></li>
#             <li><a href="#4.3" class="list-group-item list-group-item-action">4.3 Training Model</a></li>
#         </ul>
#     </li>
#     <li><a href="#5" class="list-group-item list-group-item-action">5. Model Training</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#5.1" class="list-group-item list-group-item-action">5.1 Learning Rate Finder</a></li>
#             <li><a href="#5.2" class="list-group-item list-group-item-action">5.2 Model Training</a></li>
#         </ul>
#     </li>
#     <li><a href="#6" class="list-group-item list-group-item-action">6. Inference</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#6.1" class="list-group-item list-group-item-action">6.1 Finalize Model</a></li>
#             <li><a href="#6.2" class="list-group-item list-group-item-action">6.2 Test Inference</a></li>
#         </ul>
#     </li>
#     <li><a href="#7" class="list-group-item list-group-item-action">7. Cross Validation and Ensebmling</a></li>
# </ul>
# 

# <a id ="1"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>1. Settings</center></h1>

# In[ ]:


## Import dependencies 
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import pathlib
import gc
import sys
import re
import math 
import random
import time 
import datetime as dt
from tqdm import tqdm 

import sklearn
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import warnings
warnings.filterwarnings('ignore')

print('import done!')


# In[ ]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


## Limit GPU Memory in TensorFlow
## Because TensorFlow, by default, allocates the full amount of available GPU memory when it is launched. 
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


# In[ ]:


## Parameters
data_config = {'train_csv_path': '../input/spaceship-titanic/train.csv',
               'test_csv_path': '../input/spaceship-titanic/test.csv',
               'sample_submission_path': '../input/spaceship-titanic/sample_submission.csv',
              }

exp_config = {'n_bins': 10,
              'n_splits': 5,
              'batch_size': 512,
              'num_columns': 13,
              'learning_rate': 2e-4,
              'weight_decay': 0.0001,
              'train_epochs': 60,
              'checkpoint_filepath': './tmp/model/exp.ckpt',
              'finalize': True,
              'cross_validation': True,
             }

model_config = {'cat_embedding_dim': 12,
                'num_transformer_blocks': 4,
                'num_heads': 3,
                'tf_dropout_rates': [0., 0., 0., 0.,],
                'ff_dropout_rates': [0., 0., 0., 0.,],
                'mlp_dropout_rates': [0.2, 0.1],
                'mlp_hidden_units_factors': [2, 1],
                'label_smoothing': 0.01,
               }

print('Parameters setted!')


# <a id ="2"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>2. Data Loading</center></h1>

# ---
# ### [File and Data Field Descriptions](https://www.kaggle.com/competitions/spaceship-titanic/data)
# 
# - **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
#  - `PassengerId` - A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
#  - `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
#  - `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
#  - `Cabin` - The cabin number where the passenger is staying. Takes the form `deck/num/side`, where `side` can be either `P` for *Port* or `S` for *Starboard*.
#  - `Destination` - The planet the passenger will be debarking to.
#  - `Age` - The age of the passenger.
#  - `VIP` - Whether the passenger has paid for special VIP service during the voyage.
#  - `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the *Spaceship Titanic*'s many luxury amenities.
#  - `Name` - The first and last names of the passenger.
#  - `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# 
# 
# - **test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of `Transported` for the passengers in this set.
# 
# 
# - **sample_submission.csv** - A submission file in the correct format.
#  - `PassengerId` - Id for each passenger in the test set.
#  - `Transported` - The target. For each passenger, predict either *True* or *False*.
# 
# ---
# ### [Submission & Evaluation](https://www.kaggle.com/competitions/spaceship-titanic/overview/evaluation)
# 
# - Submissions are evaluated based on their classification accuracy, the percentage of predicted labels that are correct.
# 
# ---

# In[ ]:


## Data Loading
train_df = pd.read_csv(data_config['train_csv_path'])
test_df = pd.read_csv(data_config['test_csv_path'])
submission_df = pd.read_csv(data_config['sample_submission_path'])

print(f'train_length: {len(train_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


# In[ ]:


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')
print('test_df.info()'); print(test_df.info(), '\n')

## train_df Check
train_df.head()


# <a id ="3"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>3. Exploratory Data Analysis</center></h1>

# <a id ="3.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.1 Feature Engineering</center></h2>

# In[ ]:


## Feature Selection
numerical_columns = ['Age', 'RoomService', 'FoodCourt',
                     'ShoppingMall', 'Spa', 'VRDeck']
categorical_columns = ['PassengerId', 'HomePlanet', 'CryoSleep',
                       'Cabin', 'Destination', 'VIP', 'Name']
target = 'Transported'

## Number of unique values in each categorical features.
categorical_n_unique = {cc: train_df[cc].nunique() for cc in categorical_columns}
categorical_n_unique


# In[ ]:


def preprocess_df(dataframe):
    df = dataframe.copy()
    
    ## Drop 'Name'
    df = df.drop(['Name'], axis=1)
    
    ## Transform 'Transported' column to 0 or 1.
    if 'Transported' in df.columns:
        df.loc[df['Transported']==True, 'Transported'] = 1.
        df.loc[df['Transported']==False, 'Transported'] = 0.
        df['Transported'] = df['Transported'].astype('int64')
    
    ## Transform True-False features (CryoSleep and VIP) to 'Yes' or 'No'.
    df.loc[df['CryoSleep']==True, 'CryoSleep'] = 'Yes'
    df.loc[df['CryoSleep']==False, 'CryoSleep'] = 'No'
    df['CryoSleep'] = df['CryoSleep'].astype(str)
    
    df.loc[df['VIP']==True, 'VIP'] = 'Yes'
    df.loc[df['VIP']==False, 'VIP'] = 'No'
    df['VIP'] = df['VIP'].astype(str)
    
    ## Transform the dtypes of HomePlanet and Destination to str
    df['HomePlanet'] = df['HomePlanet'].astype(str)
    df['Destination'] = df['Destination'].astype(str)
    
    return df

train = preprocess_df(train_df)
train.head()


# **Caution: After `astype(str)`, null values (np.nan) are replaced by the string 'nan'.**

# In[ ]:


## Handle 'Cabin' feature
def cabin_split(dataframe):
    df = dataframe.copy()
    
    df['Cabin'] = df['Cabin'].astype(str)
    cabins = df['Cabin'].str.split('/', expand=True)
    cabins.columns = ['Cabin_0', 'Cabin_1', 'Cabin_2']
    
    df = pd.concat([df, cabins], axis=1)
    df = df.drop(['Cabin'], axis=1)
    df['Cabin_0'].astype(str)
    df['Cabin_1'] = pd.to_numeric(df['Cabin_1'], errors='coerce')
    df['Cabin_2'].astype(str)
    df['Cabin_2'] = df['Cabin_2'].map(lambda x: 'nan' if x is None else x)
    
    return df

train = cabin_split(train)
train.head()


# In[ ]:


categorical_columns = ['HomePlanet', 'CryoSleep',
                       'Destination', 'VIP']

train_pos = train.query('Transported==1').reset_index(drop=True)
train_neg = train.query('Transported==0').reset_index(drop=True)
print(f'positive samples: {len(train_pos)}, negative samples: {len(train_neg)}')


# <a id ="3.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.2 Target Distribution</center></h2>

# In[ ]:


## Target Distribution
target_count = train.groupby(['Transported'])['PassengerId'].count()
target_percent = target_count / target_count.sum()

## Make Figure object
fig = go.Figure()

## Make trace (graph object)
data = go.Bar(x=target_count.index.astype(str).values, 
              y=target_count.values)

## Add the trace to the Figure
fig.add_trace(data)

## Setting layouts
fig.update_layout(title = dict(text="Target distribution"),
                  xaxis = dict(title="Transported' values"),
                  yaxis = dict(title='counts'))

## Show the Figure
fig.show()


# <a id ="3.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.3 Numerical Features</center></h2>

# <a id ="3.3.1"></a><h2 style="background:#D4F1F4; border:0; border-radius: 12px; color:black"><center>3.3.1 Statistics of Numerical Features </center></h2>

# In[ ]:


train.describe().T.style.bar(subset=['mean'],)\
                        .background_gradient(subset=['std'], cmap='coolwarm')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:


train.groupby('Transported').describe().T


# In[ ]:


quantiles = [0, 0.9, 0.95, 0.98, 0.99, 1]
train_quantile_values = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].quantile(quantiles)
train_quantile_values


# ---
# #### There seems to be outliers...
# 
# ---

# In[ ]:


## Clipping outliers on 99% quantile
def clipping_quantile(dataframe, quantile_values=None, quantile=0.99):
    df = dataframe.copy()
    if quantile_values is None:
        quantile_values = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].quantile(quantile)
    
    for num_column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        num_values = df[num_column].values
        threshold = quantile_values[num_column]
        num_values = np.where(num_values > threshold, threshold, num_values)
        df[num_column] = num_values    
    return df

train = clipping_quantile(train, quantile_values=None, quantile=0.99)

train.describe().T.style.bar(subset=['mean'],)\
                        .background_gradient(subset=['std'], cmap='coolwarm')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')


# In[ ]:


## After clipping outliers on 99% quantile
train.groupby('Transported').describe().T


# In[ ]:


## After clipping outliers on 99% quantile
n_cols = 2
n_rows = int(np.ceil(len(numerical_columns) / n_cols))

fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(20,15))

bins = 50
for i, column in enumerate(numerical_columns):
    q, mod = divmod(i, n_cols)
    sns.histplot(x=column, data=train, hue='Transported', ax=axes[q][mod], bins=bins, stat="percent", legend=True)
    axes[q][mod].set_title(f'Distribution of {numerical_columns[i]}',size=15)
    
fig.suptitle('Blue: Transported=0, Red: Transported=1', fontsize=20)
fig.tight_layout()
plt.show()


# In[ ]:


## Heat map of Correlation Matrix
fig = px.imshow(train.corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0, 
                aspect='auto')
fig.update_layout(height=500, 
                  width=500,
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


# <a id ="3.3.2"></a><h2 style="background:#D4F1F4; border:0; border-radius: 12px; color:black"><center>3.3.2 Binning for Numerical Features </center></h2>

# ### Binning Method
# - `Age`: 0 to 100 at intervals of 5.
# 
# - `other numerical features`: Split into 10 bins.
#  - 1. Value=0 is the first bin ( get by (-1, 0] ).
#  - 2. Get quantiles at [ 0, 0.9, 0.95, 0.99, 1 ].
#  - 3. Split between quantiles_0 and quantiles_0.9 into 6 bins.
#  - 4. Use quantiles_0.95, _0.99, _1 for the rest boundary.

# In[ ]:


def bin_split(dataframe, column, n_bins, thresholds=None):
    if thresholds is None:
        if column == 'Age':
            bins = np.array([i*5 for i in range(21)])
        else:
            bins = np.array([-1, ])
            x = dataframe[column]
            x_quantiles = x.quantile([0, 0.9, 0.95, 0.99, 1])
            bins = np.append(bins, [i * ((x_quantiles.iloc[1] - x_quantiles.iloc[0]) / (n_bins-4)) for i in range(n_bins-4)])
            bins = np.append(bins, [x_quantiles.iloc[1], x_quantiles.iloc[2], x_quantiles.iloc[3], x_quantiles.iloc[4]+1])
    else:
        bins = thresholds[column]
        
    splits = pd.cut(dataframe[column], bins=bins, labels=False, right=True)
    return splits, bins

def binning(dataframe, numerical_columns, n_bins, thresholds=None):
    df = dataframe.copy()
    df_split_bins = {}
    for num_column in numerical_columns:
        splits, bins = bin_split(df, num_column, n_bins, thresholds)
        df[num_column] = splits
        df_split_bins[num_column] = bins    
    return df, df_split_bins

n_bins = exp_config['n_bins']
train, train_split_bins = binning(train, numerical_columns, n_bins, thresholds=None)

for key in train_split_bins:
    print(f'{key} bins: \n{train_split_bins[key]}\n\n')


# In[ ]:


## After Binning
n_cols = 2
n_rows = int(np.ceil(len(numerical_columns) / n_cols))

fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(20,15))

bins = 50
for i, column in enumerate(numerical_columns):
    q, mod = divmod(i, n_cols)
    sns.histplot(x=column, data=train, hue='Transported', ax=axes[q][mod], bins=bins, stat="percent", legend=True)
    axes[q][mod].set_title(f'Distribution of {numerical_columns[i]}',size=15)
    
fig.suptitle('Blue: Transported=0, Red: Transported=1', fontsize=20)
fig.tight_layout()
plt.show()


# <a id ="3.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.4 Categorical Features</center></h2>

# In[ ]:


## Make Figure object
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=categorical_columns,
                    shared_yaxes='all')

for i in range(2):
    for j in range(2):
        n = i*2 + j
        ## Make trace (graph object)
        data0 = go.Histogram(x=train_neg[categorical_columns[n]],
                             marker = dict(color='#0000FF'), ## Blue
                             name='Transporetd=0')
        data1 = go.Histogram(x=train_pos[categorical_columns[n]],
                             marker = dict(color='#FF0000'), ## Red
                             name='Transported=1')
        
        ## Add the trace to the Figure
        fig.add_trace(data0, row=i+1, col=j+1)
        fig.add_trace(data1, row=i+1, col=j+1)
        
        fig.update_traces(opacity=0.75, histnorm='probability')
        #fig.update_layout(barmode='overlay')

## Setting layouts
fig.update_layout(title = dict(text='Blue: Transported=0, Red: Transported=1'),
                  showlegend=False,)
fig.update_yaxes(title='probability', row=1, col=1)
fig.update_yaxes(title='probability', row=2, col=1)

## Show the Figure
fig.show()


# ### Cabin Features

# In[ ]:


## 'Cabin_0'
sns.countplot(x='Cabin_0', data=train, hue='Transported')


# In[ ]:


## 'Cabin_1'
sns.histplot(x='Cabin_1', data=train, hue='Transported', kde=True)


# In[ ]:


## 'Cabin_2'
sns.countplot(x='Cabin_2', data=train, hue='Transported')


# ### Binning 'Cabin_1'

# In[ ]:


## Histogram of 'Cabin_1' by Plotly (interactive)
fig = go.Figure()

data0 = go.Histogram(x=train_neg['Cabin_1'],
                             marker = dict(color='#0000FF'), # Blue
                             opacity=0.6,
                             name='Transporetd=0')
data1 = go.Histogram(x=train_pos['Cabin_1'],
                             marker = dict(color='#FF0000'), # Red
                             opacity=0.6,
                             name='Transported=1')

fig.add_trace(data0)
fig.add_trace(data1)

fig.update_layout(xaxis = dict(title='Cabin_1'),
                  yaxis = dict(title='Count'))
fig.update_layout(barmode='overlay')

fig.show()


# In[ ]:


## Binning 'Cabin_1' based on the above graph
cabin_1_bins = np.array([0, 300, 600, 1150, 1500, 1700, 2000])
train['Cabin_1'] = pd.cut(train['Cabin_1'], bins=cabin_1_bins, labels=False, right=False)


# In[ ]:


## 'Cabin_1' after binning
sns.countplot(x='Cabin_1', data=train, hue='Transported')


# <a id ="3.5"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.5 Data Processing Complete </center></h2>

# In[ ]:


numerical_columns_0 = ['Age', 'RoomService', 'FoodCourt',
                     'ShoppingMall', 'Spa', 'VRDeck']
numerical_columns_1 = ['Age', 'RoomService', 'FoodCourt',
                     'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_1']
categorical_columns_0 = ['PassengerId', 'HomePlanet', 'CryoSleep',
                       'Cabin', 'Destination', 'VIP', 'Name']
categorical_columns_1 = ['PassengerId', 'HomePlanet', 'CryoSleep',
                       'Cabin', 'Destination', 'VIP', 'Name',
                       'Cabin_0', 'Cabin_2']


# In[ ]:


## Before filling null values,　making the string 'nan' (transformed by astype(str) in preprocess_df() function) back to np.nan.
for column in ['CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Cabin_0', 'Cabin_2']:
    train[column] = train[column].map(lambda x: np.nan if x=='nan' else x)


# In[ ]:


## Filling null values with mode
train = train.fillna(train.mode().iloc[0])

for numerical in numerical_columns_1:
    train[numerical] = train[numerical].astype('int64')

train.info()


# In[ ]:


## Test Data Processing
test = preprocess_df(test_df)
test = cabin_split(test)

test = clipping_quantile(test, quantile_values=train_quantile_values.loc[0.99])
test, _ = binning(test, numerical_columns_0, n_bins, thresholds=train_split_bins)
test['Cabin_1'] = pd.cut(test['Cabin_1'], bins=cabin_1_bins, labels=False, right=False)

for column in ['CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Cabin_0', 'Cabin_2']:
    test[column] = test[column].map(lambda x: np.nan if x=='nan' else x)

test = test.fillna(train.mode().iloc[0])

for numerical in numerical_columns_1:
    test[numerical] = test[numerical].astype('int64')

test.info()


# <a id ="3.6"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.6 Validation Split </center></h2>

# In[ ]:


## Split train samples for cross-validation
n_splits = exp_config['n_splits']
skf = StratifiedKFold(n_splits=n_splits)
train['k_folds'] = -1
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train,
                                                        y=train['Transported'])):
    train['k_folds'][valid_idx] = fold
    
## Check split samples
for i in range(n_splits):
    print(f"fold {i}: {len(train.query('k_folds==@i'))} samples")


# In[ ]:


## Hold-out validation
valid_fold = train.query(f'k_folds == 0').reset_index(drop=True)
train_fold = train.query(f'k_folds != 0').reset_index(drop=True)
print(len(train_fold), len(valid_fold))


# <a id ="4"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>4. Model</center></h1>

# <a id ="4.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.1 Dataset </center></h2>

# In[ ]:


def df_to_dataset(dataframe, num_columns, target=None,
                  shuffle=False, repeat=False,
                  batch_size=5, drop_remainder=False):
    df = dataframe.copy()
    if target is not None:
        labels = df.pop(target)
        data = {key: value[:, tf.newaxis] for key, value in df.items()}
        data = dict(data)
        
        column_indices = tf.range(start=0, limit=num_columns,
                                          delta=1, dtype='int64')
        column_indices = tf.expand_dims(column_indices, axis=0)
        column_indices = tf.repeat(column_indices, repeats=len(dataframe), axis=0)
        data['column_indices'] = column_indices
        
        ds = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        data = {key: value[:, tf.newaxis] for key, value in df.items()}
        data = dict(data)
        
        column_indices = tf.range(start=0, limit=num_columns,
                                          delta=1, dtype='int64')
        column_indices = tf.expand_dims(column_indices, axis=0)
        column_indices = tf.repeat(column_indices, repeats=len(dataframe), axis=0)
        data['column_indices'] = column_indices
        
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(batch_size)
    return ds


# In[ ]:


## Create datasets
num_columns = exp_config['num_columns']
batch_size = exp_config['batch_size']

train_ds = df_to_dataset(train_fold, num_columns,
                         target='Transported',
                         shuffle=True,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False)

valid_ds = df_to_dataset(valid_fold, num_columns,
                         target='Transported',
                         shuffle=False,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False)

## Display a batch sample
example = next(iter(train_ds))[0]
input_dtypes = {}
for key in example:
    input_dtypes[key] = example[key].dtype
    print(f'{key}, shape:{example[key].shape}, {example[key].dtype}')


# <a id ="4.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.2 Preprocessing Model </center></h2>

# In[ ]:


## After binning, all features are categorical.
numerical_columns = []
categorical_columns = ['Age', 'RoomService', 'FoodCourt',
                       'ShoppingMall', 'Spa', 'VRDeck',
                       'HomePlanet', 'CryoSleep',
                       'Destination', 'VIP', 
                       'Cabin_0', 'Cabin_1', 'Cabin_2']


# In[ ]:


## Preprocessing model inputs
def create_preprocess_inputs(numerical, categorical, num_columns, input_dtypes):
    preprocess_inputs = {}
    numerical_inputs = {key: layers.Input(shape=(1,),
                                          dtype=input_dtypes[key]) for key in numerical}
    categorical_inputs = {key: layers.Input(shape=(1,), 
                                            dtype=input_dtypes[key]) for key in categorical}
    preprocess_inputs.update(**numerical_inputs, **categorical_inputs)
    
    column_indices_inputs = layers.Input(shape=(num_columns, ),
                                         dtype='int64')
    preprocess_inputs['column_indices']  =column_indices_inputs
    return preprocess_inputs


preprocess_inputs = create_preprocess_inputs(numerical_columns,
                                             categorical_columns,
                                             num_columns,
                                             input_dtypes)
preprocess_inputs


# In[ ]:


## Create Preprocessing model
def create_preprocessing_model(numerical, categorical,
                               num_columns, input_dtypes, df):
    
    ## Create inputs
    preprocess_inputs = create_preprocess_inputs(numerical,
                                                 categorical,
                                                 num_columns,
                                                 input_dtypes)
    
    ## Preprocessing layers for numerical_features
    normalize_layers = {}
    for nc in numerical:
        normalize_layer = layers.Normalization(mean=df[nc].mean(),
                                               variance=df[nc].var())
        normalize_layers[nc] = normalize_layer
        
    ## Preprocessing layers for categorical_features
    lookup_layers = {}
    for cc in categorical:
        if input_dtypes[cc] is tf.string:
            lookup_layer = layers.StringLookup(vocabulary=df[cc].unique(),
                                               output_mode='int')
        elif input_dtypes[cc] is tf.int64:
            lookup_layer = layers.IntegerLookup(vocabulary=df[cc].unique(),
                                                output_mode='int')
        lookup_layers[cc] = lookup_layer
    
    ## Create outputs
    preprocess_outputs = {}
    for key in preprocess_inputs:
        if key in normalize_layers:
            output = normalize_layers[key](preprocess_intputs[key])
            preprocess_outputs[key] = output
        elif key in lookup_layers:
            output = lookup_layers[key](preprocess_inputs[key])
            preprocess_outputs[key] = output
        elif key is 'column_indices':
            preprocess_outputs[key] = preprocess_inputs[key]
            
    ## Create model
    preprocessing_model = tf.keras.Model(preprocess_inputs,
                                         preprocess_outputs)
    
    return preprocessing_model, lookup_layers


preprocessing_model, lookup_layers = create_preprocessing_model(numerical_columns,
                                                             categorical_columns,
                                                             num_columns,
                                                             input_dtypes,
                                                             train_fold)


# In[ ]:


## Apply the preprocessing model in tf.data.Dataset.map
train_ds = train_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

## Display a preprocessed input sample
example = next(train_ds.take(1).as_numpy_iterator())[0]
for key in example:
    print(f'{key}, shape:{example[key].shape}, {example[key].dtype}')


# <a id ="4.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.3 Training Model </center></h2>

# In[ ]:


## Training model inputs
def create_model_inputs(numerical, categorical, input_dtypes):
    model_inputs = {}
    
    normalized_inputs = {key: layers.Input(shape=(1,),
                                           dtype=input_dtypes[key]) for key in numerical}
    lookup_inputs = {key: layers.Input(shape=(1,),
                                       dtype='int64') for key in categorical}
    column_indices_inputs = layers.Input(shape=(len(categorical),),
                                        dtype='int64')
    
    model_inputs.update(**normalized_inputs, **lookup_inputs)
    model_inputs['column_indices'] = column_indices_inputs
    return model_inputs

model_inputs = create_model_inputs(numerical_columns,
                                   categorical_columns,
                                   input_dtypes)
model_inputs


# In[ ]:


## Create Embedding layers
def create_embedding_layers(model_inputs, numerical, categorical,
                            lookup_layers, emb_dim):
    numerical_feature_list = []
    encoded_categorical_feature_list = []
    
    for key in model_inputs:
        if key in numerical:
            numerical_feature_list.append(model_inputs[key])
        elif key in categorical:
            ## Create Embeddings for categorical features
            embedding = layers.Embedding(input_dim=lookup_layers[key].vocabulary_size(),
                                         output_dim=emb_dim)
            encoded_categorical_feature = embedding(model_inputs[key])
            encoded_categorical_feature_list.append(encoded_categorical_feature)
        elif key is 'column_indices':
            ## Create positional embedding (column embedding)
            column_embedding = layers.Embedding(
                input_dim=len(categorical), output_dim=emb_dim, 
                name='column_embedding')
            column_embeddings = column_embedding(model_inputs[key])
    
    if len(numerical_feature_list) != 0:
        numerical_features = tf.concat(numerical_feature_list, axis=1)
    else:
        numerical_features = tf.stack(numerical_feature_list)
    
    encoded_categorical_features = tf.concat(encoded_categorical_feature_list, axis=1)
    encoded_categorical_features = encoded_categorical_features + column_embeddings
    
    return numerical_features, encoded_categorical_features

cat_embedding_dim = model_config['cat_embedding_dim']
numerical_features, encoded_categorical_features = create_embedding_layers(model_inputs,
                                                                           numerical_columns,
                                                                           categorical_columns,
                                                                           lookup_layers,
                                                                           cat_embedding_dim)
numerical_features.shape, encoded_categorical_features.shape


# ### Tab Transformer
# 
# The TabTransformer architecture works as follows:
# 
# - All the categorical features are encoded as embeddings, using the same embedding_dims. This means that each value in each categorical feature will have its own embedding vector.
# 
# - A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
# 
# - The embedded categorical features are fed into a stack of Transformer blocks. Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
# 
# - The outputs of the final Transformer layer, which are the contextual embeddings of the categorical features, are concatenated with the input numerical features, and fed into a final MLP block.
# 
# <img src="https://raw.githubusercontent.com/keras-team/keras-io/master/examples/structured_data/img/tabtransformer/tabtransformer.png" width="500"/>
# 

# In[ ]:


## TabTransformer's module
def create_mlp(hidden_units, dropout_rates,
               activation, normalization_layer,
               name=None):
    mlp_layers = []
    for i, units in enumerate(hidden_units):
        mlp_layers.append(normalization_layer)
        mlp_layers.append(layers.Dense(units,
                                       activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rates[i]))
    return keras.Sequential(mlp_layers, name=name)


# In[ ]:


## Create TabTransformer model
def create_tabtransformer(num_transformer_blocks,
                          num_heads,
                          emb_dim,
                          tf_dropout_rates,
                          ff_dropout_rates,
                          mlp_dropout_rates,
                          mlp_hidden_units_factors,
                          numerical_columns,
                          categorical_columns,
                          num_columns,
                          input_dtypes,
                          lookup_layers,):
    
    model_inputs = create_model_inputs(numerical_columns,
                                       categorical_columns,
                                       input_dtypes)
    
    numerical_features, encoded_categorical_features = create_embedding_layers(model_inputs,
                                                                           numerical_columns,
                                                                           categorical_columns,
                                                                           lookup_layers,
                                                                           emb_dim)
    
    for block_idx in range(num_transformer_blocks):
        ## Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=emb_dim,
            dropout=tf_dropout_rates[block_idx],
            name=f'multi-head_attention_{block_idx}'
        )(encoded_categorical_features, encoded_categorical_features)
        ## Skip connection 1
        x = layers.Add(
            name=f'skip_connection1_{block_idx}'
        )([attention_output, encoded_categorical_features])
        ## Layer normalization 1
        x = layers.LayerNormalization(
            name=f'layer_norm1_{block_idx}', 
            epsilon=1e-6
        )(x)
        ## Feedforward
        feedforward_output = keras.Sequential([
            layers.Dense(emb_dim, activation=keras.activations.gelu),
            layers.Dropout(ff_dropout_rates[block_idx]),
        ], name=f'feedforward_{block_idx}'
        )(x)
        ## Skip_connection 2
        x = layers.Add(
            name=f'skip_connection2_{block_idx}'
        )([feedforward_output, x])
        ## Layer normalization 2
        encoded_categorical_features = layers.LayerNormalization(
            name=f'layer_norm2_{block_idx}', 
            epsilon=1e-6
        )(x)
        
    contextualized_categorical_features = layers.Flatten(
    )(encoded_categorical_features)
    
    ## Numerical features
    if len(numerical_columns) > 0:
        numerical_features = layers.LayerNormalization(
            name=f'numerical_norm', 
            epsilon=1e-6
        )(numerical_features)
        
        ## Concatenate categorical features with numerical features
        features = layers.Concatenate()([
            contextualized_categorical_features,
            numerical_features
        ])
    else:
        features = contextualized_categorical_features
        
    ## Final MLP
    mlp_hidden_units = [
        int(factor * features.shape[-1]) for factor in mlp_hidden_units_factors]
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rates=mlp_dropout_rates,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name='MLP'
    )(features)
    
    ## Add a sigmoid to cat the output from 0 to 1
    model_outputs = layers.Dense(
        units=1,
        activation='sigmoid',
        name='sigmoid'
    )(features)
    
    ## Create model
    training_model = keras.Model(inputs=model_inputs,
                                 outputs=model_outputs)
    
    return training_model
    
## Settings for TabTransformer
num_transformer_blocks = model_config['num_transformer_blocks']
num_heads = model_config['num_heads']
tf_dropout_rates = model_config['tf_dropout_rates']
ff_dropout_rates = model_config['ff_dropout_rates']
mlp_dropout_rates = model_config['mlp_dropout_rates']
mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

## Create TabTransformer
training_model = create_tabtransformer(num_transformer_blocks,
                                       num_heads,
                                       cat_embedding_dim,
                                       tf_dropout_rates,
                                       ff_dropout_rates,
                                       mlp_dropout_rates,
                                       mlp_hidden_units_factors,
                                       numerical_columns,
                                       categorical_columns,
                                       num_columns,
                                       input_dtypes,
                                       lookup_layers)


# In[ ]:


## model compile and build
lr = exp_config['learning_rate']
wd = exp_config['weight_decay']

optimizer = tfa.optimizers.AdamW(
    learning_rate=lr, 
    weight_decay=wd)

loss_fn = keras.losses.BinaryCrossentropy(
    from_logits=False, 
    label_smoothing=model_config['label_smoothing'])

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=['accuracy', keras.metrics.AUC()])

training_model.summary()


# <a id ="5"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>5. Model Training</center></h1>

# <a id ="5.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.1 Learning Rate Finder </center></h2>

# In[ ]:


class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = tf.constant((max_lr / min_lr) ** (1 / n_rounds))
        self.lrs = []
        self.losses = []
        
    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr
        
    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs['loss'])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True 
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)


# In[ ]:


min_lr = 1e-6
max_lr = 5e-2
lr_find_epochs = 1
lr_find_steps = 100
lr_find_batch_size = 512

lr_find = LRFind(min_lr, max_lr, lr_find_steps)
lr_find_ds = df_to_dataset(train_fold, num_columns,
                           target='Transported',
                           repeat=True,
                           batch_size=lr_find_batch_size)
lr_find_ds = lr_find_ds.map(lambda x, y: (preprocessing_model(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

training_model.fit(lr_find_ds,
                   steps_per_epoch=lr_find_steps,
                   epochs=lr_find_epochs,
                   callbacks=[lr_find])

plt.plot(lr_find.lrs, lr_find.losses)
plt.xscale('log')
plt.show()


# <a id ="5.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.2 Model Training </center></h2>

# In[ ]:


## Settings for Training
epochs = exp_config['train_epochs']
batch_size = exp_config['batch_size']
steps_per_epoch = len(train_fold)//batch_size

## Re-construct the model
training_model_config = training_model.get_config()
training_model = tf.keras.Model.from_config(training_model_config)

## Model compile
learning_rate = exp_config['learning_rate']
weight_decay = exp_config['weight_decay']

learning_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=learning_rate,
    decay_steps=epochs*steps_per_epoch, 
    alpha=0.0)

optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_schedule,
    weight_decay=weight_decay)

loss_fn = keras.losses.BinaryCrossentropy(
    from_logits=False, 
    label_smoothing=model_config['label_smoothing'])

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=['accuracy', keras.metrics.AUC()])


# In[ ]:


## Checkpoint callback
checkpoint_filepath = exp_config['checkpoint_filepath']
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, 
    save_weights_only=True, 
    monitor='val_loss', 
    mode='min', 
    save_best_only=True)

## Model training
history = training_model.fit(train_ds,
                  epochs=epochs,
                  shuffle=True,
                  validation_data=valid_ds,
                  callbacks=[model_checkpoint_callback])

## Load the best parameters
training_model.load_weights(checkpoint_filepath)


# In[ ]:


## Plot the train and valid losses
def plot_history(hist, title=None, valid=True):
    plt.figure(figsize=(7, 5))
    plt.plot(np.array(hist.index), hist['loss'], label='Train Loss')
    if valid:
        plt.plot(np.array(hist.index), hist['val_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show()
    
hist = pd.DataFrame(history.history)
plot_history(hist)


# <a id ="6"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>6. Inference</center></h1>

# <a id ="6.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>6.1 Finalize Model </center></h2>

# In[ ]:


## Finalize with all training data
if exp_config['finalize']:
    
    ## Create datasets    
    train_all_ds = df_to_dataset(train, num_columns,
                         target='Transported',
                         shuffle=True,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False)
    
    ## Create preprocessing model
    preprocessing_model, lookup_layers = create_preprocessing_model(numerical_columns,
                                                             categorical_columns,
                                                             num_columns,
                                                             input_dtypes,
                                                             train)
    
    ## Apply the preprocessing model in tf.data.Dataset.map
    train_all_ds = train_all_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    
    ## Re-construct the training model
    #training_model_config = training_model.get_config()
    #training_model = tf.keras.Model.from_config(training_model_config)
    training_model = create_tabtransformer(num_transformer_blocks,
                                       num_heads,
                                       cat_embedding_dim,
                                       tf_dropout_rates,
                                       ff_dropout_rates,
                                       mlp_dropout_rates,
                                       mlp_hidden_units_factors,
                                       numerical_columns,
                                       categorical_columns,
                                       num_columns,
                                       input_dtypes,
                                       lookup_layers)

    ## Model compile
    learning_rate = exp_config['learning_rate']
    weight_decay = exp_config['weight_decay']
    
    learning_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs*steps_per_epoch, 
        alpha=0.0)
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_schedule,
        weight_decay=weight_decay)
    
    loss_fn = keras.losses.BinaryCrossentropy(
        from_logits=False, 
        label_smoothing=model_config['label_smoothing'])
    
    training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=['accuracy', keras.metrics.AUC()])
    
    ## Model training
    final_hist = training_model.fit(train_all_ds, 
                                    epochs=epochs,
                                    shuffle=True)


# In[ ]:


## Plot the training loss
if exp_config['finalize']:
    final_hist = pd.DataFrame(final_hist.history)
    plot_history(final_hist, valid=False)


# <a id ="6.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>6.2 Test Inference </center></h2>

# In[ ]:


## Inference_model = preprocessing_model + training_model
inference_inputs = preprocessing_model.input
inference_outputs = training_model(preprocessing_model(inference_inputs))
inference_model = tf.keras.Model(inputs=inference_inputs,
                                 outputs=inference_outputs)


# In[ ]:


## Test Dataset
test_ds = df_to_dataset(test, num_columns,
                        target=None,
                        shuffle=False,
                        repeat=False,
                        batch_size=batch_size,
                        drop_remainder=False)


# In[ ]:


## Inference and submission
probas = inference_model.predict(test_ds)
probas = np.squeeze(probas)

preds = np.where(probas > 0.5, True, False)

submission_df['Transported'] = preds
submission_df.to_csv('submission.csv', index=False)
submission_df.head()


# <a id ="7"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>7. Cross Validation and Ensebmling</center></h1>

# In[ ]:


if exp_config['cross_validation']:
    ## Settings for Training
    epochs = exp_config['train_epochs']
    batch_size = exp_config['batch_size']
    steps_per_epoch = len(train)//batch_size
    
    submission_df['probas_mean'] = 0.
    
    ## Create cross validation samples
    for fold in range(exp_config['n_splits']):
        valid_fold = train.query(f'k_folds == {fold}').reset_index(drop=True)
        train_fold = train.query(f'k_folds != {fold}').reset_index(drop=True)
        
        ## Create datasets    
        train_ds = df_to_dataset(train_fold, num_columns,
                         target='Transported',
                         shuffle=True,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=True)
        
        valid_ds = df_to_dataset(valid_fold, num_columns,
                         target='Transported',
                         shuffle=False,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=True)
        
        ## Create preprocessing model
        preprocessing_model, lookup_layers = create_preprocessing_model(numerical_columns,
                                                             categorical_columns,
                                                             num_columns,
                                                             input_dtypes,
                                                             train_fold)
        
        ## Apply the preprocessing model in tf.data.Dataset.map
        train_ds = train_ds.map(lambda x, y: (preprocessing_model(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(lambda x, y: (preprocessing_model(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        ## Re-construct the training model
        training_model = create_tabtransformer(num_transformer_blocks,
                                       num_heads,
                                       cat_embedding_dim,
                                       tf_dropout_rates,
                                       ff_dropout_rates,
                                       mlp_dropout_rates,
                                       mlp_hidden_units_factors,
                                       numerical_columns,
                                       categorical_columns,
                                       num_columns,
                                       input_dtypes,
                                       lookup_layers)
        
        ## Model compile
        learning_rate = exp_config['learning_rate']
        weight_decay = exp_config['weight_decay']
        
        learning_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=epochs*steps_per_epoch, 
            alpha=0.0)
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_schedule,
            weight_decay=weight_decay)
        
        loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=False, 
            label_smoothing=model_config['label_smoothing'])
        
        training_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.AUC()])
        
        ## Checkpoint callback
        checkpoint_filepath = f"./tmp/model_{fold}/exp.ckpt"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, 
            save_weights_only=True, 
            monitor='val_loss', 
            mode='min', 
            save_best_only=True)
        
        ## Model training
        history = training_model.fit(train_ds,
                                     epochs=epochs,
                                     shuffle=True,
                                     validation_data=valid_ds,
                                     callbacks=[model_checkpoint_callback],
                                     verbose=0)
        
        ## Plot the train and valid losses
        hist = pd.DataFrame(history.history)
        plot_history(hist, title=f'fold: {fold}')
        
        ## Load the best parameters
        training_model.load_weights(checkpoint_filepath)
        
        ## Test Dataset 
        test_ds = df_to_dataset(test, num_columns,
                        target=None,
                        shuffle=False,
                        repeat=False,
                        batch_size=batch_size,
                        drop_remainder=False)
        test_ds = test_ds.map(lambda x: preprocessing_model(x),
                        num_parallel_calls=tf.data.AUTOTUNE)
        
        ## Inference
        probas = training_model.predict(test_ds)
        probas = np.squeeze(probas)
        submission_df[f'probas_{fold}'] = probas
        submission_df['probas_mean'] += probas
    
    ## Ensebmle the inferences of cvs
    submission_df['probas_mean'] /= exp_config['n_splits']
    probas_mean = submission_df['probas_mean'].values
    preds = np.where(probas_mean > 0.5, True, False)
    submission_df['Transported'] = preds
    
    ## Create Submission file
    for fold in range(exp_config['n_splits']):
        submission_df = submission_df.drop([f'probas_{fold}'], axis=1)
    submission_df = submission_df.drop(['probas_mean'], axis=1)
    submission_df.to_csv('submission_cv.csv', index=False)
    submission_df.head()

