#!/usr/bin/env python
# coding: utf-8

# # Imports and Defines

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip install lightautoml --user')
clear_output()

import numpy as np
import pandas as pd 
import random,os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

TRAIN_PATH = "../input/house-prices-advanced-regression-techniques/train.csv"
TEST_PATH = "../input/house-prices-advanced-regression-techniques/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/house-prices-advanced-regression-techniques/sample_submission.csv"
SUBMISSION_PATH = "submission.csv "

ID = "Id"
TARGET = "SalePrice"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()

def rmse(y_true, y_pred, **kwargs):
    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)

MODEL_TASK = Task('reg', metric = rmse)
MODEL_ROLES = {'target': TARGET}
MODEL_THREADS = 2
MODEL_FOLDS = 2
MODEL_TIMEOUT = 60*5


# # Preprocess Data

# In[ ]:


def autoPreProcess(train,test,DROP_COLS,TARGET):
    train = train.drop(DROP_COLS, axis = 1)
    test = test.drop(DROP_COLS, axis = 1)
    
    def checkNull_fillData(train,test):
        for col in train.columns:
            if len(train.loc[train[col].isnull() == True]) != 0:
                if train[col].dtype == "float64" or train[col].dtype == "int64":
                    train.loc[train[col].isnull() == True,col] = train[col].median()
                    test.loc[test[col].isnull() == True,col] = train[col].median()
                else:
                    train.loc[train[col].isnull() == True,col] = "Missing"
                    test.loc[test[col].isnull() == True,col] = "Missing"
        
    checkNull_fillData(train,test)
    
    str_list = [] 
    num_list = []
    for colname, colvalue in train.iteritems():
        if colname == TARGET:
            continue
            
        if type(colvalue[1]) == str:
            str_list.append(colname)
        else:
            num_list.append(colname)

    scaler = StandardScaler()
    train[num_list] = scaler.fit_transform(train[num_list])
    test[num_list] = scaler.transform(test[num_list])
    
    train_len = len(train)

    train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    train_test = pd.get_dummies(train_test, columns=str_list)
    
    train = train_test[:train_len]
    test = train_test[train_len:]

    test.drop(labels=[TARGET],axis = 1,inplace=True)
    
    return train,test

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
DROP_COLS = [ID]

train,test = autoPreProcess(train,test,DROP_COLS,TARGET)
train.head()


# # Mean Median Featrue Engineering

# In[ ]:


for col in train.columns:
    if col == ID or col == TARGET:
        continue
    
    mean_minus_median = train[col].mean() - train[col].median()
    mean_of_mean_median_sum =  (train[col].mean() + train[col].median())/2
    train["mean_minus_median"] = train[col] - mean_minus_median
    train["mean_of_mean_median_sum"] = train[col] - mean_of_mean_median_sum
    
    mean_minus_median = test[col].mean() - test[col].median()
    mean_of_mean_median_sum =  (test[col].mean() + test[col].median())/2
    test["mean_minus_median"] = test[col] - mean_minus_median
    test["mean_of_mean_median_sum"] = test[col] - mean_of_mean_median_sum


# # Build Model

# In[ ]:


automl = TabularAutoML(task = MODEL_TASK, timeout = MODEL_TIMEOUT, cpu_limit = MODEL_THREADS,
                       reader_params = {'n_jobs': MODEL_THREADS, 'cv': MODEL_FOLDS, 'random_state': SEED})
pred = automl.fit_predict(train, roles = MODEL_ROLES, verbose=3)

predict_data = pred.data[:, 0]
predict_data


# # Predict Data

# In[ ]:


pred_test = automl.predict(test)

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] = pred_test.data[:, 0]
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

