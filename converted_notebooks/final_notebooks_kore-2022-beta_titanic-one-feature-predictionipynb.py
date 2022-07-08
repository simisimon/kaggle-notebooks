#!/usr/bin/env python
# coding: utf-8

# # Imports and Defines

# In[ ]:


import numpy as np
import pandas as pd 
import random,os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "../input/titanic/train.csv"
TEST_PATH = "../input/titanic/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/titanic/gender_submission.csv"
SUBMISSION_PATH = "submission.csv "

ID = "PassengerId"
TARGET = "Survived"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


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
                    train.loc[train[col].isnull() == True,col] = train[col].mode()[0]
                    test.loc[test[col].isnull() == True,col] = train[col].mode()[0]
        
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
DROP_COLS = [ID,"Name","Ticket","Cabin"]

train,test = autoPreProcess(train,test,DROP_COLS,TARGET)
train.head()


# # Select One Column

# In[ ]:


FI = np.abs(train.corr()[TARGET]).sort_values()
df_FI = pd.DataFrame(FI)
LastFI= df_FI[::-1].T.drop([TARGET],axis=1)
LastFI.T


# In[ ]:


LastFI.columns[0]


# In[ ]:


ONE_FEATURE = LastFI.columns[0]


# In[ ]:


train[[ONE_FEATURE,TARGET]]


# In[ ]:


train_one = train[[ONE_FEATURE,TARGET]]
test_one = test[[ONE_FEATURE]]


# # Build Model using One Column

# In[ ]:


from IPython.display import clear_output
get_ipython().system('pip install lightautoml --user')
clear_output()

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco


# In[ ]:


MODEL_TASK = Task(name='binary',metric='accuracy')
MODEL_ROLES = {'target': TARGET}
MODEL_THREADS = 2
MODEL_FOLDS = 2
MODEL_TIMEOUT = 60*5


# In[ ]:


automl = TabularAutoML(task = MODEL_TASK, timeout = MODEL_TIMEOUT, cpu_limit = MODEL_THREADS,
                       reader_params = {'n_jobs': MODEL_THREADS, 'cv': MODEL_FOLDS, 'random_state': SEED})
pred = automl.fit_predict(train_one, roles = MODEL_ROLES, verbose=3)

predict_data = pred.data[:, 0]


# # Predict Data uising One Column Model

# In[ ]:


pred_test = automl.predict(test_one)

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] =  (pred_test.data[:, 0] > 0.5).astype(int)
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

