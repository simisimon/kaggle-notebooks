#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datatable as dt

import os


# # VotingClassifier with only one CatBoost model
# 
# It's the 2nd notebook on the theme of single model voting, please read the foreword in this [notebook](https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837)
# 
# This time I used the same solution with CatBoost Classifier. The initial score with traditional method, cv=7 was **0.81751**
# 
# Let's try to increase it.

# In[ ]:


import copy
import time
import random

import warnings

from sklearn.preprocessing import RobustScaler

from catboost import CatBoostClassifier

from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def ht(df, n=2):
    display(df.head(n))
    display(df.tail(n))
    display(df.shape)
    
target = 'claim'


# In[ ]:


SEED = 2021

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')


# In[ ]:


y = train[target].copy()
features = train.columns.tolist()
features.remove('id')
features.remove(target)


# ## Preprocessing

# In[ ]:


train['n_missing'] = train[features].isna().sum(axis=1)
test['n_missing'] = test[features].isna().sum(axis=1)

train['std'] = train[features].std(axis=1)
test['std'] = test[features].std(axis=1)

n_missing = train['n_missing'].copy()

train[features] = train[features].fillna(train[features].mean())
test[features] = test[features].fillna(test[features].mean())

features += ['n_missing', 'std']

scaler = RobustScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

train.shape, test.shape


# > Thanks [mlanhenke](https://www.kaggle.com/mlanhenke) for **params** from this [notebook](https://www.kaggle.com/mlanhenke/tps-09-single-catboostclassifier-0-81676)

# In[ ]:


catb_params = {
    'iterations': 15585, 
    'objective': 'CrossEntropy', 
    'bootstrap_type': 'Bernoulli', 
    'od_wait': 1144, 
    'learning_rate': 0.023575206684596582, 
    'reg_lambda': 36.30433203563295, 
    'random_strength': 43.75597655616195, 
    'depth': 7, 
    'min_data_in_leaf': 11, 
    'leaf_estimation_iterations': 1, 
    'subsample': 0.8227911142845009,
    'devices' : '0',
    'verbose' : 0,
    'eval_metric': 'AUC'
}


# In[ ]:


cat_clf0 = CatBoostClassifier(**catb_params, random_state=17)
cat_clf1 = CatBoostClassifier(**catb_params, random_state=42)
cat_clf2 = CatBoostClassifier(**catb_params, random_state=2021)
cat_clf3 = CatBoostClassifier(**catb_params, random_state=31)
cat_clf4 = CatBoostClassifier(**catb_params, random_state=19)
cat_clf5 = CatBoostClassifier(**catb_params, random_state=77)
cat_clf6 = CatBoostClassifier(**catb_params, random_state=177)


# In[ ]:


if 'claim' in train.columns.tolist():
    y = train.pop('claim')
print(train.shape, test.shape)    

estimators=[('cat0', cat_clf0), 
            ('cat1', cat_clf1), 
            ('cat2', cat_clf2), 
            ('cat3', cat_clf3), 
            ('cat4', cat_clf4), 
            ('cat5', cat_clf5),
            ('cat6', cat_clf6), 
           ]

start = time.time()
print(f'fitting ...')
model = VotingClassifier(estimators=estimators, voting='soft', verbose=True)
model.fit(train, y)

print('predicting ...')
model_pred = model.predict_proba(test)[:, -1]

elapsed = time.time() - start
print(f'elapsed time: {elapsed:.2f}sec\n')


# In[ ]:


sample_solution = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')
sample_solution[target] = model_pred
ht(sample_solution)
sample_solution.to_csv('submission.csv', index=False)
print()
print('==================== R E A D Y ====================')


# The same method increased the score of LGBM from **0.81800** to **0.81837**, see the [notebook](https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837)

# #### Thanks for reading. Don't forget to upvote if you find it usefull.
