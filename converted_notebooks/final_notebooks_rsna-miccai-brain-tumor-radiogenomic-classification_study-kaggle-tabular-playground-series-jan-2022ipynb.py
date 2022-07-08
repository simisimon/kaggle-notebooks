#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## Split csv

# 

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-jan-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-jan-2021/test.csv')
sub = pd.read_csv('../input/tabular-playground-series-jan-2021/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


columns = [col for col in train.columns.to_list() if col not in ['id','target']]


# In[ ]:


data = train[columns]
target = train['target']


# ## XGBoost with Optuna 

# In[ ]:


def objective(trial, data=data, target=target):
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=777)
    param = {
        'tree_method': 'gpu_hist',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree' : trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample' : trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate' : trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators' : 10000,
        'max_depth' : trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
        'random_state' : trial.suggest_categorical('random_state', [777]),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBRegressor(**param)
    
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
    
    preds = model.predict(test_x)
    
    rmse = mean_squared_error(test_y, preds, squared=False)
    
    return rmse


# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[ ]:


study.trials_dataframe()


# ## Optuna visualization

# In[ ]:


optuna.visualization.plot_optimization_history(study)


# In[ ]:


optuna.visualization.plot_slice(study)


# In[ ]:


optuna.visualization.plot_contour(study, params=['alpha',
                                                'lambda',
                                                'subsample',
                                                'learning_rate'])


# In[ ]:


optuna.visualization.plot_param_importances(study)


# In[ ]:


optuna.visualization.plot_edf(study)


# ## create XGBoostRegressor model with the best hyperparameters

# In[ ]:


Best_trial = study.best_trial.params
Best_trial["n_estimators"], Best_trial["tree_method"] = 10000, 'gpu_hist'
Best_trial


# In[ ]:


preds = np.zeros(test.shape[0])
kf = KFold(n_splits=5, random_state=777, shuffle=True)
rmse=[]
n=0
for trn_idx, test_idx in kf.split(train[columns], train['target']):
    X_tr, X_val = train[columns].iloc[trn_idx], train[columns].iloc[test_idx]
    y_tr, y_val = train['target'].iloc[trn_idx], train['target'].iloc[test_idx]
    model = xgb.XGBRegressor(**Best_trial)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    preds += model.predict(test[columns])/kf.n_splits
    rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    print(f'fold: {n+1} ==> rmse: {rmse[n]}')
    n += 1


# In[ ]:


np.mean(rmse)


# In[ ]:


preds


# In[ ]:


len(preds)


# In[ ]:


sub['target']=preds
sub.to_csv('submission.csv', index=False)


# In[ ]:




