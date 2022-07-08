#!/usr/bin/env python
# coding: utf-8

# ---
# # [House Prices - Advanced Regression Techniques][1]
# 
# - Goal: It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
# 
# ---
# #### **The aim of this notebook is to**
# - **1. Conduct Exploratory Data Analysis (EDA).**
# - **2. Apply Box-Cox Transformation on 'SalePrice'.**
# - **3. Use PyCaret for the entire ML pipeline.**
# - **4. Create and train a Deep Learning model with TensorFlow.**
# 
# ---
# **References:** Thanks to previous great codes and notebooks.
# 
# - [PyCaret Tutoricals][2]
#  - [Regression Tutorial - Level Beginner][3]
#  - [Regression Tutorial  - Level Intermediate][4]
#  
# **My Previous Notebooks:**
# - [SpaceshipTitanic: EDA + TabTransformer[TensorFlow]][5]
# 
# ---
# ### **If you find this notebook useful, please do give me an upvote. It helps me keep up my motivation.**
# #### **Also, I would appreciate it if you find any mistakes and help me correct them.**
# 
# ---
# [1]: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# [2]: https://pycaret.gitbook.io/docs/get-started/tutorials
# [3]: https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Beginner%20-%20REG101.ipynb
# [4]: https://github.com/pycaret/pycaret/blob/master/tutorials/Regression%20Tutorial%20Level%20Intermediate%20-%20REG102.ipynb
# [5]: https://www.kaggle.com/code/masatomurakawamm/spaceshiptitanic-eda-tabtransformer-tensorflow

# <h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>0. TABLE OF CONTENTS</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Settings</a></li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Data Loading</a></li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. Exploratory Data Analysis</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#3.1" class="list-group-item list-group-item-action">3.1 Feature Selection</a></li>
#             <li><a href="#3.2" class="list-group-item list-group-item-action">3.2 Target Distribution</a></li>
#             <li><a href="#3.3" class="list-group-item list-group-item-action">3.3 Numerical Features</a></li>
#             <li><a href="#3.4" class="list-group-item list-group-item-action">3.4 Categorical Feature</a></li>
#         </ul>
#     </li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. PyCaret</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 Setting up Environment</a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 Create Model</a></li>
#             <li><a href="#4.3" class="list-group-item list-group-item-action">4.3 Tune Model</a></li>
#             <li><a href="#4.4" class="list-group-item list-group-item-action">4.4 Plot Model</a></li>
#             <li><a href="#4.5" class="list-group-item list-group-item-action">4.5 Validate Model on Hold-out Sample</a></li>
#             <li><a href="#4.6" class="list-group-item list-group-item-action">4.6 Finalize Model and Inference</a></li>
#             <li><a href="#4.7" class="list-group-item list-group-item-action">4.7 Ensemble Models</a></li>
#         </ul>
#     </li>
#     <li><a href="#5" class="list-group-item list-group-item-action">5. Deep Learning</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#5.1" class="list-group-item list-group-item-action">5.1 Creating Dataset</a></li>
#             <li><a href="#5.2" class="list-group-item list-group-item-action">5.2 Preprocessing Model</a></li>
#             <li><a href="#5.3" class="list-group-item list-group-item-action">5.3 Training Model</a></li>
#             <li><a href="#5.4" class="list-group-item list-group-item-action">5.4 Model Training</a></li>
#             <li><a href="#5.5" class="list-group-item list-group-item-action">5.5 Inference</a></li>
#         </ul>
#     </li>
# </ul>

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

print('Import done!')


# In[ ]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


# In[ ]:


## Parameters
data_config = {'train_csv_path': '../input/house-prices-advanced-regression-techniques/train.csv',
               'test_csv_path': '../input/house-prices-advanced-regression-techniques/test.csv',
               'sample_submission_path': '../input/house-prices-advanced-regression-techniques/sample_submission.csv',
              }

exp_config = {'gpu': True,
              'n_splits': 5,
              'batch_size': 128,
              'learning_rate': 1e-3,
              'train_epochs': 100,
              'checkpoint_filepath': './tmp/model/exp.ckpt',
             }

model_config = {'emb_dim': 2,
                'model_units': [512, 128],
                'dropout_rates': [0.2, 0.2,],
               }

print('Parameters setted!')


# <a id ="2"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>2. Data Loading</center></h1>

# ---
# ### [File and Data Field Descriptions](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
# 
# - **train.csv** - the training set
# - **test.csv** - the test set
# - **data_description.txt** - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
# - **sample_submission.csv** - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms.
# 
# 
# ---
# ### [Submission & Evaluation](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)
# 
# - Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
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

# <a id ="3.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.1 Feature Selection</center></h2>

# In[ ]:


## Drop the columns which contain null values more than 500.
def feature_selection(dataframe):
    df = dataframe.copy()
    for column in df.columns:
        n_null = train_df[column].isnull().sum()
        if n_null > 500:
            df = df.drop([column], axis=1)
    return df

## Feature selection on train data
train = feature_selection(train_df)
print(len(train_df.columns), len(train.columns))


# In[ ]:


## Feature selection on test data
feature_list = list(train.columns)
feature_list.remove('SalePrice')
test = test_df[feature_list]


# In[ ]:


## Select numerical and categorical features
def get_numerical_categorical(df, feature_list, n_unique=False):
    numerical_features = []
    categorical_features = []
    for column in feature_list:
        if n_unique:
            if df[column].nunique() > n_unique:
                numerical_features.append(column)
            else:
                categorical_features.append(column)
        else:
            if df[column].dtypes == 'object': 
                categorical_features.append(column)
            else:
                numerical_features.append(column)
    return numerical_features, categorical_features

target = 'SalePrice'

## Features which has more than 30 unique values as numerical
numerical_features, categorical_features = get_numerical_categorical(train,
                                                                     feature_list,
                                                                     n_unique=30)
numerical_features.remove('Id')
print(len(numerical_features), len(categorical_features))


# In[ ]:


## Numerical features' dtype check 
for n in range(len(numerical_features)):
    print(numerical_features[n])
    print(train[numerical_features[n]].dtypes)
    print()


# In[ ]:


## Categoical features' unique values check
for n in range(len(categorical_features)):
    print(categorical_features[n])
    print(train[categorical_features[n]].unique())
    print()


# <a id ="3.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.2 Target Distribution</center></h2>

# In[ ]:


sns.histplot(x=target, data=train, kde=True)
plt.show()


# #### [Box-Cox Transformation](https://qiita.com/dyamaguc/items/b468ae66f9ce6ee89724) on 'SalePrice'

# In[ ]:


fig = plt.figure(figsize=(10, 8))

list_lambda = [-2, -1, -0.5, 0, 0.5, 1, 2]
for i, lmbda in enumerate(list_lambda):
    boxcox = sp.stats.boxcox(train[target], lmbda=lmbda)
    ax = fig.add_subplot(4, 2, i+1)
    sns.histplot(data=boxcox, kde=True, ax=ax)
    plt.title('lambda='+str(list_lambda[i]))
    plt.xlabel('SalePrice')
    
auto_boxcox, best_lambda = sp.stats.boxcox(train[target], lmbda=None)
ax = fig.add_subplot(4, 2, 8)
sns.histplot(data=auto_boxcox, kde=True, ax=ax)
plt.title('lambda=' + str(round(best_lambda, 2)))
plt.xlabel('SalePrice')
    
fig.tight_layout()


# In[ ]:


## Box-Cox Summary
fig = plt.figure(figsize=(12, 5))

## Box-Cox Transformation
auto_boxcox, best_lambda = sp.stats.boxcox(train[target], lmbda=None)
ax = fig.add_subplot(1, 2, 1)
sns.histplot(data=auto_boxcox, kde=True, ax=ax)
plt.title('Transformed')
plt.xlabel('SalePrice')

## Reverse Transformation
x = sp.special.inv_boxcox(auto_boxcox, best_lambda)
ax = fig.add_subplot(1, 2, 2)
sns.histplot(x=x, data=train, kde=True, ax=ax)
plt.title('Reverse Transformed')
plt.xlabel('SalePrice')

plt.show()


# <a id ="3.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.3 Numerical Features</center></h2>

# In[ ]:


## Distributions of numerical features
fig = plt.figure(figsize=(10, 18))
for i, nf in enumerate(numerical_features):
    ax = fig.add_subplot(6, 3, i+1)
    sns.histplot(train[nf], kde=True, ax=ax)
    plt.title(nf)
    plt.xlabel(None)
fig.tight_layout()


# In[ ]:


## Heatmap of correlation matrix
numerical_columns = numerical_features + ['SalePrice']
train_numerical = train[numerical_columns]

fig = px.imshow(train_numerical.corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0, 
                aspect='auto')
fig.update_layout(height=600, 
                  width=600,
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


# <a id ="3.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>3.4 Categorical Features</center></h2>

# In[ ]:


## Distributions of categorical features
fig = plt.figure(figsize=(10, 50))
for i, cf in enumerate(categorical_features):
    ax = fig.add_subplot(19, 3, i+1)
    sns.histplot(train[cf], kde=False, ax=ax)
    plt.title(cf)
    plt.xlabel(None)
fig.tight_layout()


# <a id ="4"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>4. PyCaret</center></h1>

# #### [PyCaret](https://pycaret.org/) is an open-source, low-code machine learning library in Python that automates machine learning workflows.

# In[ ]:


## Installing and importing dependencies
get_ipython().system('pip install -U -q pycaret --ignore-installed llvmlite')
get_ipython().system('pip install -U -q numba==0.53 --ignore-installed llvmlite')
get_ipython().system('pip install -U -q Pillow==9.1.0')

from pycaret.regression import *
print('Import done!')


# <a id ="4.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.1 Setting up Environment</center></h2>

# - The `setup()` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment.
# - When `setup()` is executed, PyCaret will automatically infer the data types for all features, and displays a table containing the features and their inferred data types. If all of the data types are correctly identified `enter` can be pressed to continue or `quit` can be typed to end the expriment. When `silent` parameter equals `True`, you can omit the manual check.
# - Also, you can pass directly `numeric_features` or `categorical_features` parameters.

# In[ ]:


silent = True

exp_reg123 = setup(data=train, 
                   target='SalePrice',
                   train_size=0.8,
                   #numeric_features=numerical_features,
                   numeric_imputation='mean',
                   #categorical_features=categorical_features,
                   categorical_imputation='constant',
                   handle_unknown_categorical=True,
                   ordinal_features=None,
                   date_features=None,
                   ignore_features=['Id'],
                   normalize=True,
                   transformation=False,
                   transform_target=True,
                   transform_target_method='box-cox',
                   combine_rare_levels=True,
                   rare_level_threshold=0.05,
                   remove_multicollinearity=True,
                   multicollinearity_threshold=0.95, 
                   bin_numeric_features=None,
                   log_experiment=False,
                   experiment_name='house_prices_exp1',
                   session_id=123,
                   use_gpu=exp_config['gpu'],
                   silent=silent) 


# <a id ="4.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.2 Create Model</center></h2>

# - The `compare_models()` function conducts training and evaluation over 15 models using cross validation.
# - The default `fold` parameter value is 10. To seve time, I setted the parameter `fold=5` for saving time.
# - By default, `compare_models()` return the best performing model, but can be used to return a list of top N models by using `n_select` parameter.
# - `exclude` parameter is used to block certain models.
# - To create each models, we can use `create_model()` function.

# In[ ]:


top3 = compare_models(fold=5,
                      n_select=3,
                      round=2,
                      exclude=['ransac'])


# In[ ]:


## Select the best model
best_model = top3[0]

for i in range(len(top3)):
    print(f'Top {i+1} Model: \n{top3[i]}\n\n' )


# In[ ]:


## Create the specific model.
catboost = create_model('catboost', fold=5, round=2)
et = create_model('et', fold=5, round=2, verbose=False)
rf = create_model('rf', fold=5, round=2, verbose=False)


# In[ ]:


models = [catboost, et, rf]

for model in models:
    print(model, '\n')


# <a id ="4.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.3 Tune Model</center></h2>

# -  The `tune_model()` function automatically tunes the hyperparameters of a model using Random Grid Search on a pre-defined search space.
# - We can change the metric for optimization by `optimize` parameter (default, Accuracy).
# - `n_iter` is the number of iterations within a random grid search (default value is 10). Increasing the value may improve the performance but will also increase the training time.

# In[ ]:


## It will take some time.
#tuned_best = tune_model(best_model, fold=5, n_iter=30)


# <a id ="4.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.4 Plot Model</center></h2>

# - The `plot_model()` function can be used to analyze the performance across different aspects such as Residuals Plot, Prediction Error, Feature Importance etc. 

# In[ ]:


plot_model(catboost, plot='parameter')


# In[ ]:


plot_model(catboost, plot='residuals') ## Default


# In[ ]:


plot_model(catboost, plot = 'error')


# In[ ]:


plot_model(catboost, plot='feature')


# - We can also use `evaluate_model()` function for further model analysis.

# In[ ]:


#evaluate_model(catboost)


# <a id ="4.5"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.5 Validate Model on Hold-out Sample</center></h2>

# -  Before finalizing the model, we can conduct the final check by predicting the hold-out set and reviewing the evaluation metrics (All of the evaluation metrics we have seen above are cross-validated results based on training set (80%) only).
# - Remaining 20% of data (hold-out samples) is used for `predict_model()` function.

# In[ ]:


predict_model(catboost)


# <a id ="4.6"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.6 Finalize Model and Inference</center></h2>

# - `finalize_model()` function fits the model onto the complete dataset including the hold-out sample. 

# In[ ]:


final_model = finalize_model(catboost)
print(final_model)


# In[ ]:


## Inference
test_predictions = predict_model(final_model, data=test)
submission_df['SalePrice'] = test_predictions['Label']
submission_df.to_csv('submission_pycaret.csv', index=False)
test_predictions


# <a id ="4.7"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>4.7 Ensemble Models</center></h2>

# - `blend_model()` function creates multiple models and then averages the individual predictions to form a final prediction. 

# In[ ]:


## Blend individual models
blender = blend_models(estimator_list=models, fold=5)


# In[ ]:


## Validate blended model
predict_model(blender)


# In[ ]:


## Finalize blended model
final_blender = finalize_model(blender)
print(final_blender)


# In[ ]:


## Inference with blended model
test_blend_predictions = predict_model(final_blender, data=test)
submission_df['SalePrice'] = test_blend_predictions['Label']
submission_df.to_csv('submission_blender.csv', index=False)
test_blend_predictions


# <a id ="5"></a><h1 style="background:#05445E; border:0; border-radius: 12px; color:#D3D3D3"><center>5. Deep Learning</center></h1>

# In[ ]:


## Import dependencies 
import sklearn
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


# <a id ="5.1"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.1 Creating Dataset</center></h2>

# In[ ]:


## Fill NaN in numerical columns with its median
train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median()) 

## Fill NaN in categorical columns with its mode
train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode().iloc[0])  

train.info()


# In[ ]:


## Box-Cox Transformation on target ('SalePrice')
auto_boxcox, best_lambda = sp.stats.boxcox(train[target], lmbda=None)
train['target_transformed'] = auto_boxcox

## Reverse Transformation
#x = sp.special.inv_boxcox(auto_boxcox, best_lambda)

## Box-Cox + Standardization
box_cox_mean = train['target_transformed'].mean()
box_cox_std = train['target_transformed'].std()
train['target_transformed_standardized'] = (train['target_transformed'] - box_cox_mean) / box_cox_std


# In[ ]:


## Train validation split
n_splits = exp_config['n_splits']
kf = KFold(n_splits=n_splits)
train['k_folds'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    train['k_folds'][valid_idx] = fold
    
for i in range(n_splits):
    print(f"fold {i}: {len(train.query('k_folds==@i'))} samples")


# In[ ]:


## Hold-out validation
valid_fold = train.query(f'k_folds == 0').reset_index(drop=True)
train_fold = train.query(f'k_folds != 0').reset_index(drop=True)
print(len(train_fold), len(valid_fold))


# In[ ]:


def df_to_dataset(dataframe, target=None,
                  shuffle=False, repeat=False,
                  batch_size=5, drop_remainder=False):
    df = dataframe.copy()
    if target is not None:
        labels = df.pop(target)
        data = {key: value[:, tf.newaxis] for key, value in df.items()}
        data = dict(data)
        ds = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        data = {key: value[:, tf.newaxis] for key, value in df.items()}
        data = dict(data)
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
batch_size = exp_config['batch_size']
target = 'target_transformed_standardized'

train_ds = df_to_dataset(train_fold, 
                         target=target,
                         shuffle=True,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=True)

valid_ds = df_to_dataset(valid_fold,
                         target=target,
                         shuffle=False,
                         repeat=False,
                         batch_size=batch_size,
                         drop_remainder=False)

example = next(iter(train_ds))[0]
input_dtypes = {}
for key in example:
    input_dtypes[key] = example[key].dtype
    print(f'{key}, shape:{example[key].shape}, {example[key].dtype}')


# <a id ="5.2"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.2 Preprocessing Model</center></h2>

# In[ ]:


def create_preprocessing_model(numerical, categorical, input_dtypes, df):
    ## Create input layers
    preprocess_inputs = {}
    for key in numerical:
        preprocess_inputs[key] = layers.Input(shape=(1,),
                                              dtype=input_dtypes[key])
    for key in categorical:
        preprocess_inputs[key] = layers.Input(shape=(1,),
                                              dtype=input_dtypes[key])
    
    ## Create preprocess layers
    normalize_layers = {}
    for key in numerical:
        normalize_layer = layers.Normalization(mean=df[key].mean(),
                                               variance=df[key].var())
        normalize_layers[key] = normalize_layer
        
    lookup_layers = {}
    for key in categorical:
        if input_dtypes[key] == tf.string:
            lookup_layer = layers.StringLookup(vocabulary=df[key].unique(),
                                               output_mode='int')
        elif input_dtypes[key] == tf.int64:
            lookup_layer = layers.IntegerLookup(vocabulary=df[key].unique(),
                                                output_mode='int')
        lookup_layers[key] = lookup_layer
        
    ## Create outputs
    preprocess_outputs = {}
    for key in preprocess_inputs:
        if key in normalize_layers:
            output = normalize_layers[key](preprocess_inputs[key])
            preprocess_outputs[key] = output
        elif key in lookup_layers:
            output = lookup_layers[key](preprocess_inputs[key])
            preprocess_outputs[key] = output
            
    ## Create model
    preprocessing_model = tf.keras.Model(preprocess_inputs,
                                         preprocess_outputs)
    
    return preprocessing_model, lookup_layers


# In[ ]:


## Create preprocessing model
preprocessing_model, lookup_layers = create_preprocessing_model(
    numerical_features, categorical_features, 
    input_dtypes, train_fold)

## Apply the preprocessing model in tf.data.Dataset.map
train_ds = train_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (preprocessing_model(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

## Display a preprocessed input sample
example = next(train_ds.take(1).as_numpy_iterator())[0]
for key in example:
    print(f'{key}, shape: {example[key].shape}, {example[key].dtype}')


# <a id ="5.3"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.3 Training Model</center></h2>

# In[ ]:


def create_training_model(numerical, categorical, input_dtypes, df,
                          lookup_layers, emb_dim=1,
                          model_units=[128,], 
                          dropout_rates=[0.2,]):
    ## Create input layers
    model_inputs = {}
    for key in numerical:
        model_inputs[key] = layers.Input(shape=(1,), 
                                         dtype='float32')
    for key in categorical:
        model_inputs[key] = layers.Input(shape=(1,), 
                                         dtype='int64')
    
    features = []
    
    for key in model_inputs:
        if key in numerical:
            features.append(model_inputs[key])
        elif key in categorical:
            ## Create embedding layers
            embedding = layers.Embedding(
                input_dim=lookup_layers[key].vocabulary_size(),
                output_dim=emb_dim)
            encoded_categorical = embedding(model_inputs[key])
            encoded_categorical = tf.squeeze(encoded_categorical, axis=1)
            features.append(encoded_categorical)
        
    ## Concatenate all features
    x = tf.concat(features, axis=1)
    
    for units, dropout_rate in zip(model_units, dropout_rates):
        feedforward = keras.Sequential([
            layers.Dense(units, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(dropout_rate),
        ])
        x = feedforward(x)
        
    final_layer = layers.Dense(units=1, activation=None)
    model_outputs = final_layer(x)
    
    training_model = tf.keras.Model(inputs=model_inputs,
                                    outputs=model_outputs)
    return training_model


# In[ ]:


## Create training model
emb_dim = model_config['emb_dim']
model_units = model_config['model_units']
dropout_rates = model_config['dropout_rates']

training_model = create_training_model(numerical_features,
                                       categorical_features,
                                       input_dtypes,
                                       train_fold,
                                       lookup_layers,
                                       emb_dim=emb_dim,
                                       model_units=model_units, 
                                       dropout_rates=dropout_rates)

## Model compile and build
lr = exp_config['learning_rate']
optimizer = keras.optimizers.Adam(learning_rate=lr)
loss_fn = keras.losses.MeanSquaredError()

training_model.compile(optimizer=optimizer,
                       loss=loss_fn,
                       metrics=[keras.metrics.mean_squared_error,])

training_model.summary()


# <a id ="5.4"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.4 Model Training</center></h2>

# In[ ]:


## Settings for Training
epochs = exp_config['train_epochs']
batch_size = exp_config['batch_size']
steps_per_epoch = len(train_fold)//batch_size 

## For saving the best model
checkpoint_filepath = exp_config['checkpoint_filepath']
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, 
    save_weights_only=True, 
    monitor='val_loss', 
    mode='min', 
    save_best_only=True)

## For the adjustment of learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    cooldown=10,
    min_lr=1e-5,
    verbose=1)

## Model training
history = training_model.fit(train_ds,
                             epochs=epochs,
                             shuffle=True,
                             validation_data=valid_ds,
                             callbacks=[model_checkpoint_callback,
                                        reduce_lr])

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


# <a id ="5.5"></a><h2 style="background:#75E6DA; border:0; border-radius: 12px; color:black"><center>5.5 Inference</center></h2>

# In[ ]:


## Inference_model = preprocessing_model + training_model
inference_inputs = preprocessing_model.input
inference_outputs = training_model(preprocessing_model(inference_inputs))
inference_model = tf.keras.Model(inputs=inference_inputs,
                                 outputs=inference_outputs)


# In[ ]:


## Fill NaN in numerical columns with its median
test[numerical_features] = test[numerical_features].fillna(train[numerical_features].median()) 
## Fill NaN in categorical columns with its mode
test[categorical_features] = test[categorical_features].fillna(train[categorical_features].mode().iloc[0])  

## Create test dataset
test_ds = df_to_dataset(test,
                        target=None,
                        shuffle=False,
                        repeat=False,
                        batch_size=batch_size,
                        drop_remainder=False,)


# In[ ]:


preds = inference_model.predict(test_ds)
preds = np.squeeze(preds)

def reverse_transformation(preds,
                           mean=box_cox_mean,
                           std=box_cox_std,
                           box_cox_lambda=best_lambda):
    x = (preds * std) + mean
    x = sp.special.inv_boxcox(x, box_cox_lambda)
    return x 

price_inference = reverse_transformation(preds)
submission_df['SalePrice'] = price_inference
submission_df.to_csv('submission_dnn.csv', index=False)
submission_df.head()


# In[ ]:




