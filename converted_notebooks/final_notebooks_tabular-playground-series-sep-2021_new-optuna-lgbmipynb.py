#!/usr/bin/env python
# coding: utf-8

# # Contributions
# Please try out the following and let me know if it helps at all
# 1. Trying new feature engineering combinations: None of the notebooks I've looked at have used abs()+sum(), or sem()
# 
# ![image.png](attachment:2f8d1cf0-2eb2-46f0-b5cd-0c896fbc9877.png)
# ![image.png](attachment:30ec3830-9d45-401c-8686-be109661a809.png)
# 
# 2. Adjusting learning rate: You can save time as well as fine tune your models

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu
import lightgbm as lgb
import gc
from tqdm import tqdm

SEED = 0


# In[ ]:


train = pd.read_csv("/kaggle/input/tabular-playground-series-sep-2021/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/tabular-playground-series-sep-2021/test.csv", index_col='id')


# In[ ]:


features = [x for x in train.columns.values if x[0]=="f"]


# In[ ]:


train['n_missing'] = train[features].isna().sum(axis=1)
train['abs_sum'] = train[features].abs().sum(axis=1)
train['sem'] = train[features].sem(axis=1)
train['std'] = train[features].std(axis=1)
train['avg'] = train[features].mean(axis=1)
train['max'] = train[features].max(axis=1)
train['min'] = train[features].min(axis=1)

test['n_missing'] = test[features].isna().sum(axis=1)
test['abs_sum'] = test[features].abs().sum(axis=1)
test['sem'] = test[features].sem(axis=1)
test['std'] = test[features].std(axis=1)
test['avg'] = test[features].mean(axis=1)
test['max'] = test[features].max(axis=1)
test['min'] = test[features].min(axis=1)


# In[ ]:


# imputer = SimpleImputer(strategy="median")
# X = imputer.fit_transform(X)
# X_test = imputer.transform(X_test)


# Idea taken from www.kaggle.com/dlaststark/tps-sep-single-xgboost-model
# I have modified the choices using the following rationale:
# 1. Mean: normal distribution
# 2. Median: unimodal and skewed
# 3. Mode: all other cases

# In[ ]:


X = train.drop(["claim"], axis=1)
X_test = test
y = train["claim"]


# In[ ]:


scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(X), index=train.index, columns=test.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=test.index, columns=test.columns)


# In[ ]:


fill_value_dict = {
    'f1': 'Mean', 
    'f2': 'Median', 
    'f3': 'Median', 
    'f4': 'Median', 
    'f5': 'Mode', 
    'f6': 'Mean', 
    'f7': 'Median', 
    'f8': 'Median', 
    'f9': 'Median', 
    'f10': 'Median', 
    'f11': 'Mean', 
    'f12': 'Median', 
    'f13': 'Mean', 
    'f14': 'Median', 
    'f15': 'Mean', 
    'f16': 'Median', 
    'f17': 'Median', 
    'f18': 'Median', 
    'f19': 'Median', 
    'f20': 'Median', 
    'f21': 'Median', 
    'f22': 'Mean', 
    'f23': 'Mode', 
    'f24': 'Median', 
    'f25': 'Median', 
    'f26': 'Median', 
    'f27': 'Median', 
    'f28': 'Median', 
    'f29': 'Mode', 
    'f30': 'Median', 
    'f31': 'Median', 
    'f32': 'Median', 
    'f33': 'Median', 
    'f34': 'Mean', 
    'f35': 'Median', 
    'f36': 'Mean', 
    'f37': 'Median', 
    'f38': 'Median', 
    'f39': 'Median', 
    'f40': 'Mode', 
    'f41': 'Median', 
    'f42': 'Mode', 
    'f43': 'Mean', 
    'f44': 'Median', 
    'f45': 'Median', 
    'f46': 'Mean', 
    'f47': 'Mode', 
    'f48': 'Mean', 
    'f49': 'Mode', 
    'f50': 'Mode', 
    'f51': 'Median', 
    'f52': 'Median', 
    'f53': 'Median', 
    'f54': 'Mean', 
    'f55': 'Mean', 
    'f56': 'Mode', 
    'f57': 'Mean', 
    'f58': 'Median', 
    'f59': 'Median', 
    'f60': 'Median', 
    'f61': 'Median', 
    'f62': 'Median', 
    'f63': 'Median', 
    'f64': 'Median', 
    'f65': 'Mode', 
    'f66': 'Median', 
    'f67': 'Median', 
    'f68': 'Median', 
    'f69': 'Mean', 
    'f70': 'Mode', 
    'f71': 'Median', 
    'f72': 'Median', 
    'f73': 'Median', 
    'f74': 'Mode', 
    'f75': 'Mode', 
    'f76': 'Mean', 
    'f77': 'Mode', 
    'f78': 'Median', 
    'f79': 'Mean', 
    'f80': 'Median', 
    'f81': 'Mode', 
    'f82': 'Median', 
    'f83': 'Mode', 
    'f84': 'Median', 
    'f85': 'Median', 
    'f86': 'Median', 
    'f87': 'Median', 
    'f88': 'Median', 
    'f89': 'Median', 
    'f90': 'Mean', 
    'f91': 'Mode', 
    'f92': 'Median', 
    'f93': 'Median', 
    'f94': 'Median', 
    'f95': 'Median', 
    'f96': 'Median', 
    'f97': 'Mean', 
    'f98': 'Median', 
    'f99': 'Median', 
    'f100': 'Mode', 
    'f101': 'Median', 
    'f102': 'Median', 
    'f103': 'Median', 
    'f104': 'Median', 
    'f105': 'Median', 
    'f106': 'Median', 
    'f107': 'Median', 
    'f108': 'Median', 
    'f109': 'Mode', 
    'f110': 'Median', 
    'f111': 'Median', 
    'f112': 'Median', 
    'f113': 'Mean', 
    'f114': 'Median', 
    'f115': 'Median', 
    'f116': 'Mode', 
    'f117': 'Median', 
    'f118': 'Mean'
}


for col in tqdm(features):
    if fill_value_dict.get(col)=='Mean':
        fill_value = X[col].mean()
    elif fill_value_dict.get(col)=='Median':
        fill_value = X[col].median()
    elif fill_value_dict.get(col)=='Mode':
        fill_value = X[col].mode().iloc[0]
    
    X[col].fillna(fill_value, inplace=True)
    X_test[col].fillna(fill_value, inplace=True)


# In[ ]:


X = X.values
X_test = X_test.values


# In[ ]:


del test, train, scaler
gc.collect()


# In[ ]:


import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

def train_model_optuna(trial, X_train, X_valid, y_train, y_valid):
    """
    A function to train a model using different hyperparamerters combinations provided by Optuna. 
    Loss of validation data predictions is returned to estimate hyperparameters effectiveness.
    """
    preds = 0
       
    #A set of hyperparameters to optimize by optuna
    lgbm_params = {
                    "objective": trial.suggest_categorical("objective", ['binary']),
                    "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt']),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, step=0.001),
                    "n_estimators": trial.suggest_int("n_estimators", 35000, 45000),        
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, step=0.1),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 200.0, step=0.1),
                    "random_state": trial.suggest_categorical("random_state", [0]),
                    "bagging_seed": trial.suggest_categorical("bagging_seed", [0]),
                    "feature_fraction_seed": trial.suggest_categorical("feature_fraction_seed", [0]), 
                    "n_jobs": trial.suggest_categorical("n_jobs", [4]), 
                    "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.01),
                    "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step=0.001),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                    'metric': trial.suggest_categorical('metric', ['AUC'])
                    }
    
    # Model loading and training
    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric="auc",
              early_stopping_rounds=100,
              verbose=False)
  
    print(f"Number of boosting rounds: {model.best_iteration_}")
    oof = model.predict_proba(X_valid)[:, 1]
   
    return roc_auc_score(y_valid, oof)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Splitting data into train and valid folds using target bins for stratification\nX_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)\n# Setting optuna verbosity to show only warning messages\n# If the line is uncommeted each iteration results will be shown\n# optuna.logging.set_verbosity(optuna.logging.WARNING)\n\ntime_limit = 3600 * 7\n\nstudy = optuna.create_study(direction='maximize')\nstudy.optimize(lambda trial: train_model_optuna(trial,\n                                                X_train,\n                                                X_valid,\n                                                y_train, y_valid),\n               timeout=time_limit\n              )\n\n# Showing optimization results\nprint('Number of finished trials:', len(study.trials))\nprint('Best trial parameters:', study.best_trial.params)\nprint('Best score:', study.best_value)\n")

