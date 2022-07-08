#!/usr/bin/env python
# coding: utf-8

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

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

TRAIN_PATH = "../input/spaceship-titanic/train.csv"
TEST_PATH = "../input/spaceship-titanic/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/spaceship-titanic/sample_submission.csv"
SUBMISSION_PATH = "submission.csv "

ID = "PassengerId"
TARGET = "Transported"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()

MODEL_TASK = Task('binary',metric = 'accuracy')
MODEL_ROLES = {'target': TARGET}
MODEL_THREADS = 2
MODEL_FOLDS = 2
MODEL_TIMEOUT = 60*5


# In[ ]:


def autoPreProcess(train,test,DROP_COLS,BOOL_COLS,TARGET):
    train_len = len(train)

    train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    train_test = train_test.drop(DROP_COLS, axis = 1)

    def checkNull_fillData(df):
        for col in df.columns:
            if len(df.loc[df[col].isnull() == True]) != 0:
                if df[col].dtype == "float64" or df[col].dtype == "int64":
                    df.loc[df[col].isnull() == True,col] = df[col].median()
                elif col in BOOL_COLS:
                    df.loc[df[col].isnull() == True,col] = df[col].mode()[0]
                else:
                    df.loc[df[col].isnull() == True,col] = "Missing"

    checkNull_fillData(train_test)
    
    for col in BOOL_COLS:
        train_test[col] =  train_test[col].astype(int)

    str_list = [] 
    num_list = []
    for colname, colvalue in train_test.iteritems():
        if colname == TARGET:
            continue
            
        if type(colvalue[1]) == str:
            str_list.append(colname)
        else:
            num_list.append(colname)

    train_test = pd.get_dummies(train_test, columns=str_list)

    scaler = StandardScaler()
    train_test[num_list] = scaler.fit_transform(train_test[num_list])
    
    train = train_test[:train_len]
    test = train_test[train_len:]

    test.drop(labels=[TARGET],axis = 1,inplace=True)
    
    return train,test

train = pd.read_csv(TRAIN_PATH)
train[TARGET] = train[TARGET].astype(int)

test = pd.read_csv(TEST_PATH)
DROP_COLS = ['PassengerId', 'Name', 'Cabin']
BOOL_COLS = ['CryoSleep','VIP']

train,test = autoPreProcess(train,test,DROP_COLS,BOOL_COLS,TARGET)
train.head()


# In[ ]:


automl = TabularAutoML(task = MODEL_TASK, timeout = MODEL_TIMEOUT, cpu_limit = MODEL_THREADS,
                       reader_params = {'n_jobs': MODEL_THREADS, 'cv': MODEL_FOLDS, 'random_state': SEED})
pred = automl.fit_predict(train, roles = MODEL_ROLES, verbose=3)

predict_data = pred.data[:, 0]
predict_data


# In[ ]:


pred_test = automl.predict(test)

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] =  (pred_test.data[:, 0] > 0.5).astype(bool)
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

