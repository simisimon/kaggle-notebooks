#!/usr/bin/env python
# coding: utf-8

# 1. [Import Libraries](#1)
# 2. [Reading the Dataset](#2)
# 3. [Data Pre-processing](#3)
# 4. [Modeling](#4)
#     - 4.1. [Logistic Regression](#7)
#     - 4.2. [LightGBM](#8)
#     - 4.3. [Wrap-up and get the output](#9)
# 5. [Conclusions](#5)
# 6. [References](#6)

# # 1. Import Libraries <a id = 1> </a>

# In[ ]:


import gc

import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

import numpy as np 

import pandas as pd

from scipy import stats

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU

import warnings
warnings.filterwarnings('ignore')


# # 2. Reading the Dataset <a id = 2> </a>

# #### <a href = 'https://www.kaggle.com/c/santander-customer-transaction-prediction'>Link to the dataset in Kaggle.</a>

# In[ ]:


raw_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
raw_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
raw_train.head()


# In[ ]:


print(f'Shape of training dataset: {raw_train.shape}')
print(f'Shape of testing dataset: {raw_test.shape}')
print(20 * "-")
print(f'Number of NaN values in training dataset: {raw_train.isnull().values.sum()}')
print(f'Number of NaN values in testing dataset: {raw_test.isnull().values.sum()}')
print(20 * "-")
print(f"Number of training values that aren't float64: {sum(raw_train.iloc[:, 2:].dtypes != 'float64')}")
print(f"Number of testing values that aren't float64: {sum(raw_test.iloc[:, 2:].dtypes != 'float64')}")


# # Data Pre-processing <a id = 3> </a>

# In[ ]:


plt.figure(figsize = (22, 5))
sns.countplot(data = raw_train, x = 'target')
plt.xlabel('Target', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.title('Class distribution', fontsize = 20)


# In[ ]:


num_0, num_1 = np.bincount(raw_train['target'])
print(f'Number of samples belonging to class zero: {num_0} ({num_0 / raw_train.shape[0]:0.2%} of total)')
print(f'Number of samples belonging to class one: {num_1} ({num_1 / raw_train.shape[0]:0.2%} of total)')

del num_0, num_1
_ = gc.collect()


# In[ ]:


y = raw_train['target']
train_len = len(raw_train)
id_test = raw_test['ID_code']

merged_df = pd.concat([raw_train, raw_test], axis = 0).drop(columns = ['ID_code', 'target'])

del raw_train, raw_test
_ = gc.collect()

merged_df.head()


# In[ ]:


merged_minmax = MinMaxScaler().fit_transform(merged_df)
merged_minmax_df = pd.DataFrame(merged_minmax)
merged_minmax_df = merged_minmax_df.add_prefix('var_')


# In[ ]:


def stat_detector(df):
    
    '''
    argument:
        df: a DataFrame.
    outputs:
        outlier_mask: 0-1 mask; 0: not outlier, 1: is outlier.
        Three plot about mean values, std values and outliers for each feature. 
    '''
    plt.figure(figsize=(22, 10))
    
    ax1 = plt.subplot2grid((22, 22), (0, 0), rowspan = 9, colspan = 10)
    
    plt.title(f'Distribution of mean values for each feature', fontsize = 20)
    ax1 = sns.distplot(df.mean(), kde = True, bins = 50, color = 'red')
    plt.ylabel('density', fontsize = 15)

    ax2 = plt.subplot2grid((22, 22), (0, 12), rowspan = 9, colspan = 10)
    
    plt.title(f'Distribution of standard deviation values for each feature', fontsize = 20)
    ax2 = sns.distplot(df.std(), kde = True, bins = 50, color = 'green')
    plt.ylabel('density', fontsize = 15)

    ax3 = plt.subplot2grid((22, 22), (11, 0), rowspan = 11, colspan = 22)

    total_data = df.shape[0] * df.shape[1]

    outlier_mask = (np.abs(stats.zscore(df)) > 3.0) * 1
    sum_outlier_col = outlier_mask.sum()
    total_outlier = sum(sum_outlier_col)
    outlier_share = (total_outlier / total_data) * 100
    share_outlier_col = (sum_outlier_col / df.shape[0]) * 100

    ax3 = plt.axhline(y = outlier_share, color = 'red', linestyle = '--', label = 'Average')
    ax3 = share_outlier_col.plot.bar()
    plt.title(f'Percentage of outliers for each feature', fontsize = 20)
    plt.suptitle(f'total samples: {total_data:,}, total outliers: {total_outlier:,}, share of outliers from total data: {outlier_share:0.3}%',
                 y = 0.48, fontsize = 15)
    plt.xlabel('Features', fontsize = 15)
    plt.ylabel('% of outliers', fontsize = 15)
    plt.legend(['Average'], fontsize = 15)
    plt.xticks([])
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 3))

    return outlier_mask

_ = stat_detector(merged_minmax_df)


# In[ ]:


merged_minmax_df.head(5)


# In the first step, the dataset is scaled. Then, 64 features are added to the dataset in two steps. Data obtained from PCA is first added, followed by data received from Denoising Autoencoder. 
# 
# |method|Number of initial features|Number of features after transformation|
# |:-:|:-:|:-:|
# |Scaled dataset|200|200|
# |Scaled dataset + PCA|200|232|
# |Scaled dataset + PCA + DAE|232|264|
# 

# In[ ]:


merged_pca = PCA(n_components = 0.2, random_state = 0).fit_transform(merged_minmax)
merged_pca_scaled = MinMaxScaler().fit_transform(merged_pca)
merged_pca_df = pd.DataFrame(merged_pca_scaled)
merged_pca_df = merged_pca_df.add_prefix('pca_')
print(f'Shape of dataset after transformation using PCA: {merged_pca.shape}')

merged_minmax_pca_df = pd.concat([merged_minmax_df, merged_pca_df], axis = 1)
print(f'Shape of dataset after concatenating: {merged_minmax_pca_df.shape}\n')

del merged_minmax, merged_minmax_df, merged_pca, merged_pca_scaled #merged_minmax_df
_ = gc.collect()

merged_minmax_pca_df.head(2)


# In[ ]:


_ = stat_detector(merged_minmax_pca_df)


# In[ ]:


input_dim = merged_pca_df.shape[1] 

encoder_decoder = Sequential(
    [
        Dense(64, input_shape = (input_dim, )),
        BatchNormalization(),
        LeakyReLU(),
        Dense(32),
        BatchNormalization(),
        LeakyReLU(),
        Dense(16),
        Dense(32),
        BatchNormalization(),
        LeakyReLU(),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(),
        Dense(input_dim, activation = 'linear')
        
    ]
)

auto_encoder = Model(inputs = encoder_decoder.input, outputs = encoder_decoder.output)
auto_encoder.summary()


# In[ ]:


auto_encoder.compile(loss = 'mse', optimizer = 'adam')

epochs = 10
index_merged_pca = np.random.choice(np.arange(merged_pca_df.shape[0]), int(merged_pca_df.shape[0] / 4))

auto_encoder_history = auto_encoder.fit(
    merged_pca_df.iloc[index_merged_pca].values, #Can we add noise? + np.random.normal(0.5, 0.15, size = merged_pca_df.iloc[index_merged_pca].shape)
    merged_pca_df.iloc[index_merged_pca].values,
    batch_size = 32,
    epochs = epochs
)


# In[ ]:


merged_dae = auto_encoder.predict(merged_pca_df.values)
merged_dae_df = pd.DataFrame(merged_dae)
merged_dae_df = merged_dae_df.add_prefix('dae_')
merged_pca_dae = pd.concat([merged_minmax_pca_df, merged_dae_df], axis = 1)

del merged_dae, merged_pca_df, merged_dae_df, auto_encoder_history, index_merged_pca
_ = gc.collect()

merged_pca_dae.head(2)


# In[ ]:


_ = stat_detector(merged_pca_dae)


# In[ ]:


X_train, X_dev, y_train, y_dev = train_test_split(merged_pca_dae[:train_len], y, test_size = 0.3, shuffle = True, stratify = y)

stnd_scaler = StandardScaler().fit(X_train)
X_train_scaled = stnd_scaler.transform(X_train)
X_dev_scaled = stnd_scaler.transform(X_dev)
X_test_scaled = stnd_scaler.transform(merged_pca_dae[train_len:])

del merged_pca_dae, stnd_scaler
_ = gc.collect()


# # 4. Modeling <a id = 4> </a>

# ## 4.1. Logistic Regression <a id = 7> </a>

# In[ ]:


param_grid_lr = {
    'C': [0.001, 0.01]
}

lr = LogisticRegression(penalty = 'l2', solver = 'sag', class_weight = 'balanced', random_state = 0)

grid_lr = GridSearchCV(
    estimator = lr, 
    param_grid = param_grid_lr, 
    scoring='roc_auc',
    cv = 3, 
    refit = True, 
    n_jobs = -1
)

grid_lr.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_lr.best_params_}")


# In[ ]:


y_pred_lr = grid_lr.predict_proba(X_dev_scaled)[:, 1]
y_pred_lr_final = grid_lr.predict_proba(X_test_scaled)[:, 1]

print(f'AUC score for dev dataset using LogisticRegression: {roc_auc_score(y_dev, y_pred_lr):0.3f}')


# ## 4.2. LightGBM <a id = 8> </a>

# In[ ]:


lgb_model = lgb.LGBMClassifier(
    objective = 'binary',
    learning_rate = 0.07,
    num_leaves = 20,
    n_estimators = 500, 
    max_depth = 8,
    class_weight = 'balanced',
    subsample = 1,
    colsample_bytree = 1,
    metric = 'auc',
    device = 'gpu',
    gpu_platform_id = 0,
    gpu_device_id = 0,
    random_state = 0                          
)

param_grid_lgb = {
    'reg_alpha': [0.1, 1],
    'reg_lambda': [0.1, 1]
}

grid_lgb = GridSearchCV(
    estimator = lgb_model, 
    param_grid = param_grid_lgb,
    scoring='roc_auc',
    cv = 3, 
    refit = True, 
    n_jobs = -1
)

grid_lgb.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_lgb.best_params_}")


# In[ ]:


y_pred_lgb = grid_lgb.predict_proba(X_dev_scaled)[:, 1]
y_pred_lgb_final = grid_lgb.predict_proba(X_test_scaled)[:, 1]

print(f'AUC score for dev dataset using LightGBM: {roc_auc_score(y_dev, y_pred_lgb):0.3f}')


# ## 4.3. Wrap-up and get the output <a id = 9> </a>

# In[ ]:


AUC_df = pd.DataFrame(
    dict(
    model = ['Logistic Regression', 'LightGBM'],
    score = [roc_auc_score(y_dev, y_pred_lr), roc_auc_score(y_dev, y_pred_lgb)]
    )
)


plt.figure(figsize = (22, 5))
ax = sns.barplot(data = AUC_df, x = 'model', y = 'score')
ax.bar_label(ax.containers[0], fontsize = 15)
ax.set_xlabel('AUC-ROC score', fontsize = 15)
ax.set_ylabel('Model', fontsize = 15)
ax.set_ylim([0.7, 0.92])


# In[ ]:


output = pd.DataFrame(
    {
    'ID_code': [],
    'target': []   
    }
)

output['ID_code'] = id_test
output['target'] = y_pred_lgb_final

output


# In[ ]:


output.to_csv('./submission.csv', index = False)


# # 5. Conclusions <a id = 5> </a>

# It is assumed that the optimal error rate is 0.92 (best AUC-ROC). Considering that our best AUC-ROC is 0.89, there is a considerable bias that can be avoided.</br>
# Besides these, I also just scaled the dataset and trained it with the LightGBM model without any further pre-processing. I obtained an AUC-ROC score of 0.88! In other words, all this pre-processing on the notebook hasn't really improved the score.Besides these, I also just scaled the dataset and trained it with the LightGBM model without any further pre-processing. I obtained an AUC-ROC score of 0.88! In other words, all this pre-processing on the notebook hasn't really improved the score.

# # 6. References <a id = 6> </a>

# - <a href = 'https://ekamperi.github.io/machine%20learning/2021/01/21/encoder-decoder-model.html'>The encoder-decoder model as a dimensionality reduction technique</a>
