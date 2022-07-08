#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
import lightgbm as lgb


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-sep-2021/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-sep-2021/test.csv")
submission = pd.read_csv("/kaggle/input/tabular-playground-series-sep-2021/sample_solution.csv")


# In[ ]:


def check(df):
    col_list = train.columns.values
    rows = []
    for col in col_list:
        tmp = (col,
              train[col].dtype,
              train[col].isnull().sum(),
              train[col].count(),
              train[col].nunique(),
              train[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows) 
    df.columns = ['feature','dtype','nan','count','nunique','unique']
    return df

check(train)


# In[ ]:


total_cell = np.product(train.shape)
missing_values_count = train.isnull().sum()
total_missing = missing_values_count.sum()
percent_missing = (total_missing / total_cell)* 100

print(percent_missing, " % missing")


# In[ ]:


from sklearn.model_selection import KFold

train["kfold"] = -1
kf = KFold(n_splits = 10, shuffle=True, random_state = 0)
for fold,(train_index, valid_index) in enumerate(kf.split(X = train)):
    print(fold,train_index, valid_index)
    train.loc[valid_index, "kfold"] = fold
train.kfold.value_counts()


# In[ ]:


useful = [col for col in train.columns if col not in ('id','claim','kfold')]
test = test[useful]


# In[ ]:


param = {      
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "n_estimators" : 1000,
        "early_stopping_rounds" : 10,
    
        "feature_fraction" : 1.0,
        "num_leaves" : 94,
        "bagging_fraction": 0.6737009142690187,
        "bagging_freq": 1,
        "lambda_l1": 5.559056252126386, 
        "lambda_l2": 9.77312118560801,
        "min_child_samples" : 50
     }

score = []
test_pred = []
valid_pred = {}
for fold in range(10):
    
    X_train = train[train.kfold != fold].reset_index(drop = True)
    X_valid = train[train.kfold == fold].reset_index(drop = True)
    
    X_test = test.copy()
    valid_ids = X_valid.id.values.tolist()
    
    y_train = X_train.claim
    y_valid = X_valid.claim
    
    X_train = X_train[useful]
    X_valid = X_valid[useful]
    
    feature = list(X_train.columns[1:])
    
    X_train['n_missing'] = X_train[feature].isna().sum(axis = 1)
    X_train['std'] = X_train[feature].std(axis = 1)
    X_train['mean'] = X_train[feature].mean(axis = 1)
    X_train['median'] = X_train[feature].mean(axis = 1)
    X_train['kurt'] = X_train[feature].kurtosis(axis = 1)

    X_valid['n_missing'] = X_valid[feature].isna().sum(axis = 1)
    X_valid['std'] = X_valid[feature].std(axis = 1)
    X_valid['mean'] = X_valid[feature].mean(axis = 1)
    X_valid['median'] = X_valid[feature].mean(axis = 1)
    X_valid['kurt'] = X_valid[feature].kurtosis(axis = 1)

    X_test['n_missing'] = X_test[feature].isna().sum(axis = 1)
    X_test['std'] = X_test[feature].std(axis = 1)
    X_test['mean'] = X_test[feature].mean(axis = 1)
    X_test['median'] = X_train[feature].mean(axis = 1)
    X_test['kurt'] = X_test[feature].kurtosis(axis = 1)
    
    feature += ['n_missing','std','mean','median','kurt']
    X_train[feature] = X_train[feature].fillna(X_train[feature].mean())
    X_valid[feature] = X_valid[feature].fillna(X_valid[feature].mean())
    X_test[feature] = X_test[feature].fillna(X_test[feature].mean())
    
    scaler= StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_valid = pd.DataFrame(scaler.transform(X_valid))
    X_test = pd.DataFrame(scaler.transform(X_test))
                                
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid,reference = lgb.train)
    
    model = LGBMRegressor()
    model = lgb.train(param,   lgb_train, valid_sets = [lgb_valid],verbose_eval = 100)
                   
    pred_valid = model.predict(X_valid)
    valid_pred.update(dict(zip(valid_ids, pred_valid)))
    test_preds = model.predict(X_test)
    test_pred.append(test_preds)
    
    auc = roc_auc_score(y_valid, pred_valid)
    print(fold,auc)
    score.append(auc)
    
print(score)


# In[ ]:


valid_prediction = pd.DataFrame.from_dict(valid_pred, orient = "index").reset_index()
valid_prediction.columns = ['id', 'pred_23']
valid_prediction.to_csv("valid_pred_23.csv", index = False)


# In[ ]:


submission['claim'] = np.mean(np.column_stack(test_pred), axis = 1)
submission.to_csv("submission.csv", index = False)


# In[ ]:




