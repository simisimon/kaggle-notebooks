#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # [Google Cloud & NCAA® ML Competition 2020-NCAAM](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament)

# ## In this notebook, I'm just training some models (LGB, XGB etc.).
# ## There is a caveat that the top of LB is bad in this first stage of the competition.
# ## In the future, I plan to take into account a number of advices of experienced professionals and improve this kernel to clear the leaks of information and other problems.

# # Acknowledgements
# 
# This kernel uses such good kernels: 
# * [Merging FE & Prediction - xgb, lgb, logr, linr](https://www.kaggle.com/vbmokin/merging-fe-prediction-xgb-lgb-logr-linr)
# * [Basic Starter Kernel](https://www.kaggle.com/addisonhoward/basic-starter-kernel-ncaa-men-s-dataset-2019)
# * [2020 Basic Starter Kernel](https://www.kaggle.com/hiromoon166/2020-basic-starter-kernel)
# * [March Madness 2020 NCAAM EDA and baseline](https://www.kaggle.com/artgor/march-madness-2020-ncaam-eda-and-baseline)
# * [March Madness 2020 NCAAM:Simple Lightgbm on KFold](https://www.kaggle.com/ratan123/march-madness-2020-ncaam-simple-lightgbm-on-kfold)
# * [NCAAM2020: XGBoost + LightGBM K-Fold (Baseline)](https://www.kaggle.com/khoongweihao/ncaam2020-xgboost-lightgbm-k-fold-baseline)

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [My upgrade](#1)
#     -  [Commit now](#1.1)
#     -  [Previous commits: LGB](#1.2)
#     -  [Previous commits: Merging of LGB, XGB, Logistic Regression](#1.3)
# 1. [Import libraries](#2)
# 1. [Download data & FE](#3)
# 1. [Models tuning](#4)
#     -  [LGB](#4.1)
#     -  [XGB](#4.2)    
#     -  [Logistic Regression](#4.3)
# 1. [Showing Confusion Matrices](#5)
# 1. [Comparison and merging solutions](#6)
# 1. [Submission](#7)

# ## 1. My upgrade<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# LGB
lgb_num_leaves_max = 100
lgb_in_leaf = 75
lgb_lr = 0.007
lgb_bagging = 7

# XGB
xgb_max_depth = 40
xgb_min_child_weight = 75
xgb_lr = 0.0004
xgb_n_estimators = 4000

# Set weight of models
w_lgb = 0.85
w_xgb = 0.1
w_logreg = 1 - w_lgb - w_xgb
w_logreg


# ## 1.1. Commit now <a class="anchor" id="1.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **LGB**
# 
#     params_lgb = {'num_leaves': 100,
#                   'min_data_in_leaf': 75,
#                   'objective': 'binary',
#                   'max_depth': -1,
#                   'learning_rate': 0.007,
#                   "boosting_type": "gbdt",
#                   "bagging_seed": 7,
#                   "metric": 'logloss',
#                   "verbosity": -1,
#                   'random_state': 42,
#                  }

# **XGB**
# 
#     params_xgb = {'max_depth':40,
#                   'objective':'binary:logistic',
#                   'min_child_weight': 75,
#                   'learning_rate': 0.004,
#                   'eta'      :0.3,
#                   'subsample':0.8,
#                   'lambda '  :4,
#                   'eval_metric':'logloss',
#                   'n_estimators':4000,
#                   'colsample_bytree ':0.9,
#                   'colsample_bylevel':1
#                   }

# #### Set weight of models
#     w_lgb = 0.8
#     w_xgb = 0.15
#     w_logreg = 1 - w_lgb - w_xgb

# ## 1.2. Previous commits: LGB <a class="anchor" id="1.2"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 1
# 
#     params = {'num_leaves': 255,
#               'min_data_in_leaf': 100,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.29445**

# ### Commit 2
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 100,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.29445**

# ### Commit 6
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 50,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.14726**

# ### Commit 8
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.05038**

# ### Commit 9
# 
#     params = {'num_leaves': 63,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.005,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.08311**

# ### Commit 13
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.01,
#               "boosting_type": "gbdt",
#               "bagging_seed": 11,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.04872**

# ### Commit 14
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.01,
#               "boosting_type": "gbdt",
#               "bagging_seed": 7,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.04872**

# ### Commit 15
# 
#     params = {'num_leaves': 127,
#               'min_data_in_leaf': 10,
#               'objective': 'binary',
#               'max_depth': -1,
#               'learning_rate': 0.001,
#               "boosting_type": "gbdt",
#               "bagging_seed": 7,
#               "metric": 'logloss',
#               "verbosity": -1,
#               'random_state': 42,
#              }
#              
# **LB = 0.17908**

# ## 1.3. Previous commits: Merging of LGB, XGB, Logistic Regression <a class="anchor" id="1.3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Commit 20
# 
# **LGB**
# 
#     params_lgb = {'num_leaves': 127,
#                   'min_data_in_leaf': 10,
#                   'objective': 'binary',
#                   'max_depth': -1,
#                   'learning_rate': 0.0001,
#                   "boosting_type": "gbdt",
#                   "bagging_seed": 7,
#                   "metric": 'logloss',
#                   "verbosity": -1,
#                   'random_state': 42,
#                  }
# 
# **XGB**
# 
#     params_xgb = {'max_depth':63,
#                   'objective':'binary:logistic',
#                   'min_child_weight': 10,
#                   'learning_rate': 0.0001,
#                   'eta'      :0.3,
#                   'subsample':0.8,
#                   'lambda '  :4,
#                   'eval_metric':'logloss',
#                   'n_estimators':2000,
#                   'colsample_bytree ':0.9,
#                   'colsample_bylevel':1
#                   }
#                   
#     w_lgb = 0.48
#     w_xgb = 0.48
#     w_logreg = 1 - w_lgb - w_xgb
#     
#     y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred
#     
# 
# **LB = 0.34462**

# ### Commit 22
# 
# **LGB**
# 
#     params_lgb = {'num_leaves': 63,
#                   'min_data_in_leaf': 10,
#                   'objective': 'binary',
#                   'max_depth': -1,
#                   'learning_rate': 0.005,
#                   "boosting_type": "gbdt",
#                   "bagging_seed": 7,
#                   "metric": 'logloss',
#                   "verbosity": -1,
#                   'random_state': 42,
#                  }
# 
# **XGB**
# 
#     params_xgb = {'max_depth':63,
#                   'objective':'binary:logistic',
#                   'min_child_weight': 10,
#                   'learning_rate': 0.005,
#                   'eta'      :0.3,
#                   'subsample':0.8,
#                   'lambda '  :4,
#                   'eval_metric':'logloss',
#                   'n_estimators':2000,
#                   'colsample_bytree ':0.9,
#                   'colsample_bylevel':1
#                   }
#                   
#     w_lgb = 0.48
#     w_xgb = 0.48
#     w_logreg = 1 - w_lgb - w_xgb
#     
#     y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred
#     
# 
# **LB = 0.27677**

# ### Commit 27
# 
# #### LGB
# * lgb_num_leaves_max = 100
# * lgb_in_leaf = 75
# * lgb_lr = 0.007
# * lgb_bagging = 7
# 
# #### XGB
# * xgb_max_depth = 40
# * xgb_min_child_weight = 75
# * xgb_lr = 0.0004
# * xgb_n_estimators = 4000
# 
# #### Set weight of models
# * w_lgb = 0.8
# * w_xgb = 0.15
# * w_logreg = 1 - w_lgb - w_xgb    
# 
# **LB = 0.24733**

# ## 2. Import libraries <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import xgboost as xgb

import gc

import warnings
warnings.filterwarnings("ignore")


# ## 3. Download data & FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[ ]:


def get_seed(x):
    return int(x[1:3])

tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))


# In[ ]:


season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()


# In[ ]:


tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[ ]:


tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)


# In[ ]:


tourney_lose_result = tourney_win_result.copy()
tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']


# ## Prepare Training Data

# In[ ]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[ ]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
train_df = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
train_df


# # Preparing testing data

# In[ ]:


test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))


# In[ ]:


test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# ## 4. Model tuning <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X = train_df.drop('result', axis=1)
y = train_df.result


# ### 4.1 LGB <a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to:
# * [March Madness 2020 NCAAM EDA and baseline](https://www.kaggle.com/artgor/march-madness-2020-ncaam-eda-and-baseline)
# * [March Madness 2020 NCAAM:Simple Lightgbm on KFold](https://www.kaggle.com/ratan123/march-madness-2020-ncaam-simple-lightgbm-on-kfold)

# In[ ]:


params_lgb = {'num_leaves': lgb_num_leaves_max,
              'min_data_in_leaf': lgb_in_leaf,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': lgb_lr,
              "boosting_type": "gbdt",
              "bagging_seed": lgb_bagging,
              "metric": 'logloss',
              "verbosity": -1,
              'random_state': 42,
             }


# In[ ]:


NFOLDS = 10
folds = KFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)
y_preds_lgb = np.zeros(test_df.shape[0])
y_train_lgb = np.zeros(X.shape[0])
y_oof = np.zeros(X.shape[0])

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    
    y_train_lgb += clf.predict(X) / NFOLDS
    y_preds_lgb += clf.predict(test_df) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()


# ### 4.2 XGB <a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


params_xgb = {'max_depth': xgb_max_depth,
              'objective': 'binary:logistic',
              'min_child_weight': xgb_min_child_weight,
              'learning_rate': xgb_lr,
              'eta'      : 0.3,
              'subsample': 0.8,
              'lambda '  : 4,
              'eval_metric': 'logloss',
              'n_estimators': xgb_n_estimators,
              'colsample_bytree ': 0.9,
              'colsample_bylevel': 1
              }


# In[ ]:


# Thanks to https://www.kaggle.com/khoongweihao/ncaam2020-xgboost-lightgbm-k-fold-baseline
NFOLDS = 10
folds = KFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)

y_preds_xgb = np.zeros(test_df.shape[0])
y_train_xgb = np.zeros(X.shape[0])
y_oof_xgb = np.zeros(X.shape[0])

train_df_set = xgb.DMatrix(X)
test_set = xgb.DMatrix(test_df)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_set = xgb.DMatrix(X_train, y_train)
    val_set = xgb.DMatrix(X_valid, y_valid)
    
    clf = xgb.train(params_xgb, train_set, num_boost_round=2000, evals=[(train_set, 'train'), (val_set, 'val')], early_stopping_rounds=100, verbose_eval=100)
    
    y_train_xgb += clf.predict(train_df_set) / NFOLDS
    y_preds_xgb += clf.predict(test_set) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()


# ### 4.3 Logistic Regression <a class="anchor" id="4.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


test_df.head()


# In[ ]:


# Standardization for regression models
df = pd.concat([X, test_df], axis=0, sort=False).reset_index(drop=True)
df_log = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(df),
    columns=df.columns,
    index=df.index
)
train_log, test_log = df_log.iloc[:len(X),:], df_log.iloc[len(X):,:].reset_index(drop=True)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_log, y)
coeff_logreg = pd.DataFrame(train_log.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


# Eli5 visualization
eli5.show_weights(logreg)


# In[ ]:


y_logreg_train = logreg.predict(train_log)
y_logreg_pred = logreg.predict(test_log)


# ## 5. Showing Confusion Matrices <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    y_true = y_true.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


# Showing Confusion Matrix for LGB model
plot_cm(y_train_lgb, y, 'Confusion matrix for LGB model', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for XGB model
plot_cm(y_train_xgb, y, 'Confusion matrix for XGB model', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for Logistic Regression
plot_cm(y_logreg_train, y, 'Confusion matrix for Logistic Regression', figsize=(7,7))


# ## 6. Comparison and merging solutions <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Merging (blending) solutions
# 

# In[ ]:


# From the kernel https://www.kaggle.com/vbmokin/merging-fe-prediction-xgb-lgb-logr-linr
y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_logreg_pred


# ## 7. Submission <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to @nroman for the advices from post https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/discussion/131539
# 
# Later I will improve this block of code.

# In[ ]:


sub = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
MNCAATourneyCompactResults = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
sub = pd.concat([sub, sub['ID'].str.split('_', expand=True).rename(columns={0: 'Season', 1: 'Team1', 2: 'Team2'}).astype(np.int64)], axis=1)
merge = pd.merge(sub, MNCAATourneyCompactResults[['Season', 'WTeamID', 'LTeamID']], how='left', left_on=['Season', 'Team1', 'Team2'], right_on=['Season', 'WTeamID', 'LTeamID'])
sub.loc[~merge['WTeamID'].isnull(), 'Pred'] = 1
merge = pd.merge(sub, MNCAATourneyCompactResults[['Season', 'WTeamID', 'LTeamID']], how='left', left_on=['Season', 'Team2', 'Team1'], right_on=['Season', 'WTeamID', 'LTeamID'])
sub.loc[~merge['WTeamID'].isnull(), 'Pred'] = 0
sub = sub.drop(['Season', 'Team1', 'Team2'], axis=1)


# In[ ]:


sub['Pred'] = y_preds
sub.head()


# In[ ]:


sub.info()


# In[ ]:


sub['Pred'].hist()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
